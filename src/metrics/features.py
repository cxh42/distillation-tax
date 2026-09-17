"""Video feature spaces for A1 (coverage) and A2 (diversity).

  videomae : MCG-NJU/videomae-base-finetuned-kinetics backbone, 16 frames @224, mean-pooled
             last hidden state (768-d).  Video-native, motion-aware.
  dinov2   : facebook/dinov2-large CLS per frame (16 uniformly spaced frames), averaged (1024-d).
             Appearance-centric; cached locally (no download).

Both are L2-normalised; distances are cosine.  Two spaces so that any coverage/diversity
conclusion can be shown not to depend on the choice of feature extractor (guide §9).
"""
import numpy as np, torch, torch.nn.functional as F

_models = {}


def _sample_idx(n, k=16):
    return np.linspace(0, n - 1, k).round().astype(int)


def _to_tensor(frames_u8, idx, size, mean, std):
    x = torch.from_numpy(frames_u8[idx]).permute(0, 3, 1, 2).float() / 255.0
    x = F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)
    m = torch.tensor(mean).view(1, 3, 1, 1); s = torch.tensor(std).view(1, 3, 1, 1)
    return (x - m) / s


@torch.no_grad()
def videomae_feat(frames_u8, device="cuda"):
    if "videomae" not in _models:
        from transformers import VideoMAEModel
        _models["videomae"] = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base-finetuned-kinetics",
                                                            dtype=torch.float16).to(device).eval()
    m = _models["videomae"]
    x = _to_tensor(frames_u8, _sample_idx(len(frames_u8), 16), (224, 224),
                   [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    h = m(pixel_values=x[None].to(device, torch.float16)).last_hidden_state[0].float()
    return F.normalize(h.mean(0), dim=0).cpu().numpy()


@torch.no_grad()
def dinov2_feat(frames_u8, device="cuda"):
    if "dinov2" not in _models:
        from transformers import AutoModel
        _models["dinov2"] = AutoModel.from_pretrained("facebook/dinov2-large", dtype=torch.float16).to(device).eval()
    m = _models["dinov2"]
    x = _to_tensor(frames_u8, _sample_idx(len(frames_u8), 16), (224, 224),
                   [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    cls = m(pixel_values=x.to(device, torch.float16)).last_hidden_state[:, 0].float()
    per_frame = F.normalize(cls, dim=1)
    return F.normalize(per_frame.mean(0), dim=0).cpu().numpy(), per_frame.cpu().numpy()


@torch.no_grad()
def all_features(frames_u8, device="cuda"):
    vm = videomae_feat(frames_u8, device)
    dv, dv_frames = dinov2_feat(frames_u8, device)
    return dict(videomae=vm, dinov2=dv, dinov2_frames=dv_frames)
