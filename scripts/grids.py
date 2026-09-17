"""Human-eye grids: for each prompt id, rows = configurations (fixed order), cols = frames.
   python scripts/grids.py --pids P04,P15,P26 --seed 0 [--tag pilotA]"""
import os, sys, argparse, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ORDER = ["T-full", "T-few", "T-nocfg", "T-cfg3", "T-cfg7.5", "AR-diff", "AR-diff-cfg3", "S-bi-causvid", "S-bi-fastwan", "S-causvid", "S-sf", "S-cf", "AR-diff-rcm", "S-bi-rcm", "S-rcm-tfdcm", "S-rcm-sfdmd"]
ap = argparse.ArgumentParser(); ap.add_argument("--pids", required=True); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--tag", default="pilotA")
a = ap.parse_args()
os.makedirs(os.path.join(ROOT, "figures", "grids"), exist_ok=True)
for pid in a.pids.split(","):
    vids, labs = [], []
    for v in ORDER:
        p = os.path.join(ROOT, "corpus", v, f"{pid}_s{a.seed}.mp4")
        if os.path.exists(p):
            vids.append(p); labs.append(v)
    out = os.path.join(ROOT, "figures", "grids", f"{a.tag}_{pid}_s{a.seed}.jpg")
    subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "sheet.py"), out, *vids, "--labels", ",".join(labs), "--w", "256"], check=True)
