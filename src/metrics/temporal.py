"""A5 temporal quality + image-quality side metrics (VBench-style, own implementations).

  subject_consistency    DINO ViT-B/16 CLS cosine (VBench recipe: mean of sim-to-first and sim-to-prev)
  background_consistency CLIP ViT-B/32 image-embedding cosine, same recipe
  temporal_flicker       VBench `temporal_flickering`: 1 - mean|I_t - I_{t+1}|/255 over frames
  warp_error             RAFT backward-warp L1 (t+1 -> t); low = coherent motion, high = jitter/flicker
  sharpness              mean Laplacian variance (no-reference sharpness proxy)
  colorfulness           Hasler-Süsstrunk colourfulness (over-saturation is a known DMD artefact)

NB: A3 (motion) and A5 are anti-correlated by construction -- a frozen video has perfect
consistency.  We report both and never aggregate them (guide §3.3 / §4).
"""
import numpy as np, torch, torch.nn.functional as F
from .motion import raft_flow

_m = {}


def _dino(device):
    if "dino" not in _m:
        from transformers import ViTModel, ViTImageProcessor
        _m["dino"] = (ViTModel.from_pretrained("facebook/dino-vitb16", dtype=torch.float16).eval().to(device),
                      ViTImageProcessor.from_pretrained("facebook/dino-vitb16"))
    return _m["dino"]


def _clip(device):
    if "clip" not in _m:
        from transformers import CLIPModel, CLIPProcessor
        _m["clip"] = (CLIPModel.from_pretrained("openai/clip-vit-base-patch32", dtype=torch.float16).eval().to(device),
                      CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"))
    return _m["clip"]


def _seq_consistency(feats):
    f = F.normalize(feats.float(), dim=-1)
    return float((((f[1:] * f[0:1]).sum(-1) + (f[1:] * f[:-1]).sum(-1)) / 2).mean())


@torch.no_grad()
def subject_consistency(frames, device="cuda", bs=32):
    m, pr = _dino(device)
    feats = []
    for i in range(0, len(frames), bs):
        x = pr(images=list(frames[i:i + bs]), return_tensors="pt").to(device)
        feats.append(m(pixel_values=x["pixel_values"].half()).last_hidden_state[:, 0])
    return _seq_consistency(torch.cat(feats))


@torch.no_grad()
def clip_image_embeds(frames, device="cuda", bs=32):
    m, pr = _clip(device)
    out = []
    for i in range(0, len(frames), bs):
        x = pr(images=list(frames[i:i + bs]), return_tensors="pt").to(device)
        out.append(m.get_image_features(pixel_values=x["pixel_values"].half()).pooler_output)
    return torch.cat(out)


@torch.no_grad()
def background_consistency(frames, device="cuda"):
    return _seq_consistency(clip_image_embeds(frames, device))


def temporal_flicker(frames):
    x = frames.astype(np.float32)
    return float(1.0 - np.abs(x[1:] - x[:-1]).mean() / 255.0)


@torch.no_grad()
def warp_error(frames, device="cuda", flow=None):
    if flow is None:
        flow, _ = raft_flow(frames, device)                   # flow at half res
    n, _, h, w = flow.shape
    t = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0
    t = F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)
    gy, gx = torch.meshgrid(torch.arange(h).float(), torch.arange(w).float(), indexing="ij")
    errs = []
    for i in range(n):
        xx = gx + flow[i, 0]; yy = gy + flow[i, 1]
        grid = torch.stack([2 * xx / (w - 1) - 1, 2 * yy / (h - 1) - 1], -1)[None]
        warped = F.grid_sample(t[i + 1:i + 2], grid, align_corners=True, padding_mode="border")
        errs.append((warped - t[i:i + 1]).abs().mean().item())
    return float(np.mean(errs))


def sharpness(frames, step=4):
    import cv2
    return float(np.mean([cv2.Laplacian(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var()
                          for f in frames[::step]]))


def colorfulness(frames, step=8):
    vals = []
    for f in frames[::step]:
        r, g, b = [f[..., i].astype(np.float32) for i in range(3)]
        rg, yb = r - g, 0.5 * (r + g) - b
        vals.append(np.sqrt(rg.std() ** 2 + yb.std() ** 2) + 0.3 * np.sqrt(rg.mean() ** 2 + yb.mean() ** 2))
    return float(np.mean(vals))


@torch.no_grad()
def temporal_metrics(frames, device="cuda", flow=None):
    return dict(subject_consistency=subject_consistency(frames, device),
                background_consistency=background_consistency(frames, device),
                temporal_flicker=temporal_flicker(frames),
                warp_error=warp_error(frames, device, flow),
                sharpness=sharpness(frames), colorfulness=colorfulness(frames))
