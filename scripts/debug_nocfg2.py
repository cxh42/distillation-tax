import os, sys, torch, numpy as np
torch.set_grad_enabled(False)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from wan_env import Runner, write_mp4
R = Runner("T-nocfg")
p = "a dog running happily"
for seed in [1, 2]:
    fr = R.generate(p, seed)
    print(f"seed {seed}: pixel mean {fr.mean():.1f} std {fr.std():.1f}", flush=True)
    write_mp4(fr, f"/home/xinghao/Projects/CVPR_0/dtax/corpus/smoke/T-nocfg/P04_s{seed}.mp4")
