"""A3 motion: RAFT optical flow decomposed into a global (camera) component and a residual
(object) component -- guide §3.2.

For every consecutive frame pair we compute RAFT-large flow at half resolution, fit a
6-parameter affine motion model to the whole field by *robust* least squares (IRLS with
Tukey weights, so a large moving object does not drag the camera fit), and report

    global_mag   = mean |affine field|          (camera pan / zoom / rotate)
    residual_mag = mean |flow - affine field|   (object motion, jitter, non-rigid)
    total_mag    = mean |flow|
    dyn_deg      = VBench-style dynamic degree on the *total* flow (kept for comparability)
    residual_frac= residual_mag / (total_mag + eps)

All magnitudes are in pixels of the ORIGINAL resolution (flow is rescaled back), averaged
over frame pairs.  Jitter shows up as high residual with low temporal coherence, which the
temporal metrics (warp error, flicker) catch separately.
"""
import numpy as np, torch, torch.nn.functional as F
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

_raft = None


def get_raft(device="cuda"):
    global _raft
    if _raft is None:
        _raft = raft_large(weights=Raft_Large_Weights.C_T_SKHT_V2).to(device).eval()
    return _raft


@torch.no_grad()
def raft_flow(frames_u8, device="cuda", scale=0.5, bs=8, iters=12):
    """frames_u8 [F,H,W,3] -> flow [F-1, 2, h, w] (float32, in pixels of the *scaled* grid)."""
    x = torch.from_numpy(frames_u8).permute(0, 3, 1, 2).float().div(127.5).sub(1.0)
    h, w = x.shape[-2:]
    hs, ws = int(round(h * scale / 8)) * 8, int(round(w * scale / 8)) * 8
    x = F.interpolate(x, size=(hs, ws), mode="bilinear", align_corners=False)
    model = get_raft(device)
    out = []
    for i in range(0, x.shape[0] - 1, bs):
        a = x[i:i + bs + 1]
        f = model(a[:-1].to(device), a[1:].to(device), num_flow_updates=iters)[-1]
        out.append(f.float().cpu())
    return torch.cat(out), (h / hs, w / ws)


def fit_affine_irls(flow, n_iter=5, c=4.685):
    """flow [2,h,w] -> (affine field [2,h,w], params [2,3]).  Robust (Tukey biweight) LSQ."""
    _, h, w = flow.shape
    ys, xs = torch.meshgrid(torch.arange(h, dtype=torch.float32), torch.arange(w, dtype=torch.float32), indexing="ij")
    X = torch.stack([xs.flatten(), ys.flatten(), torch.ones(h * w)], 1)          # [N,3]
    Y = flow.reshape(2, -1).T                                                     # [N,2]
    wgt = torch.ones(h * w)
    for _ in range(n_iter):
        Xw = X * wgt[:, None]
        A = torch.linalg.lstsq(Xw, Y * wgt[:, None]).solution                     # [3,2]
        r = (Y - X @ A).norm(dim=1)
        s = 1.4826 * r.median() + 1e-6
        u = (r / (c * s)).clamp(max=1.0)
        wgt = (1 - u ** 2) ** 2
    field = (X @ A).T.reshape(2, h, w)
    return field, A.T


def dynamic_degree_vbench(mags, h, w):
    """VBench DynamicDegree: per pair, mean of the top-5% flow magnitudes; video is 'dynamic' if
    the mean over the largest half of the pairs exceeds 6*min(h,w)/256 (their 'count_num' rule
    simplified to the score threshold used in their code)."""
    top = []
    for m in mags:
        v = m.flatten()
        k = max(1, int(0.05 * v.numel()))
        top.append(v.topk(k).values.mean().item())
    top = sorted(top, reverse=True)
    score = float(np.mean(top[:max(1, len(top) // 2)]))
    thr = 6.0 * min(h, w) / 256.0
    return score, float(score > thr)


@torch.no_grad()
def motion_metrics(frames_u8, device="cuda", return_flow=False):
    flow_half, (sy, sx) = raft_flow(frames_u8, device)
    flow = flow_half.clone()
    flow[:, 0] *= sx; flow[:, 1] *= sy                     # back to original-resolution pixels
    F_, h, w = flow.shape[0], frames_u8.shape[1], frames_u8.shape[2]
    tot, glob, res, mags, zoom = [], [], [], [], []
    for i in range(F_):
        f = flow[i]
        field, A = fit_affine_irls(f)
        r = f - field
        m = f.norm(dim=0)
        mags.append(m)
        tot.append(m.mean().item()); glob.append(field.norm(dim=0).mean().item()); res.append(r.norm(dim=0).mean().item())
        zoom.append(0.5 * (A[0, 0] + A[1, 1]).item())      # isotropic scale change per frame (zoom)
    tot, glob, res = map(np.array, (tot, glob, res))
    dd_score, dd = dynamic_degree_vbench(mags, h, w)
    out = dict(
        flow_total=float(tot.mean()), flow_global=float(glob.mean()), flow_residual=float(res.mean()),
        residual_frac=float(res.mean() / (tot.mean() + 1e-6)),
        flow_total_p95=float(np.percentile(tot, 95)),
        # temporal coherence of motion: correlation of successive residual magnitudes (jitter -> low)
        residual_autocorr=float(np.corrcoef(res[:-1], res[1:])[0, 1]) if F_ > 2 and res.std() > 1e-6 else 1.0,
        zoom_abs=float(np.abs(zoom).mean()),
        dynamic_degree_score=dd_score, dynamic_degree=dd,
    )
    return (out, flow_half) if return_flow else out


if __name__ == "__main__":
    import sys, imageio
    fr = np.stack(imageio.mimread(sys.argv[1], memtest=False))
    print(motion_metrics(fr))
