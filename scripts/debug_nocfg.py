import os, sys, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from wan_env import Runner, NEG, LAT_F, LAT_C, LAT_H, LAT_W, FlowUniPCMultistepScheduler
torch.set_grad_enabled(False)
R = Runner("T-nocfg")
p = "a dog running happily"
cond = R.te([p]); uncond = R.te([NEG])
g = torch.Generator("cuda").manual_seed(0)
noise = torch.randn([1, LAT_F, LAT_C, LAT_H, LAT_W], generator=g, device="cuda", dtype=torch.bfloat16)
for cfg in [1.0, 1.0001, 5.0]:
    sched = FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    sched.set_timesteps(10, device="cuda", shift=5.0)
    lat = noise.clone()
    for i, t in enumerate(sched.timesteps):
        ts = t * torch.ones([1, LAT_F], device="cuda", dtype=torch.float32)
        v_c, x0c = R.gen(lat, cond, ts)
        if cfg > 1.0:
            v_u, _ = R.gen(lat, uncond, ts); v = v_u + cfg * (v_c - v_u)
        else:
            v = v_c
        lat = sched.step(v.unsqueeze(0), t, lat.unsqueeze(0), return_dict=False)[0].squeeze(0)
        print(f"cfg={cfg} step {i} t={float(t):.0f} v.std={v.float().std():.3f} x0c.std={x0c.float().std():.3f} lat.std={lat.float().std():.3f} lat.mean={lat.float().mean():.3f}", flush=True)
    print(f"cfg={cfg} final lat std {lat.float().std():.3f}  channel means {lat.float().mean(dim=(0,1,3,4))[:4].tolist()}")
