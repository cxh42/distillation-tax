"""Compare one cond-only forward pass: diffusers WanTransformer3DModel vs Wan-repo WanModel (same weights)."""
import os, sys, torch
torch.set_grad_enabled(False)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from wan_env import Runner, LAT_F, LAT_C, LAT_H, LAT_W
from diffusers import WanTransformer3DModel
R = Runner("T-nocfg")
p = "a dog running happily"
cond = R.te([p])
tr = WanTransformer3DModel.from_pretrained("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", subfolder="transformer", dtype=torch.bfloat16).cuda().eval()
g = torch.Generator("cuda").manual_seed(0)
lat = torch.randn([1, LAT_F, LAT_C, LAT_H, LAT_W], generator=g, device="cuda", dtype=torch.bfloat16)
for t in [999.0, 500.0]:
    ts = t * torch.ones([1, LAT_F], device="cuda", dtype=torch.float32)
    v_wan, _ = R.gen(lat, cond, ts)                                   # [1,F,C,H,W]
    v_dif = tr(hidden_states=lat.permute(0, 2, 1, 3, 4), timestep=torch.tensor([t], device="cuda"),
               encoder_hidden_states=cond["prompt_embeds"], return_dict=False)[0].permute(0, 2, 1, 3, 4)
    d = (v_wan.float() - v_dif.float())
    print(f"t={t}: wan std {v_wan.float().std():.4f} diff std {v_dif.float().std():.4f}  |diff| mean {d.abs().mean():.4f} max {d.abs().max():.3f}  corr {torch.corrcoef(torch.stack([v_wan.float().flatten(), v_dif.float().flatten()]))[0,1]:.4f}", flush=True)
