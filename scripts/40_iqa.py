"""No-reference image-quality metrics on the corpus (VBench-style 'imaging quality' = MUSIQ, plus
CLIP-IQA, MANIQA and TOPIQ for robustness).  8 uniformly spaced frames per video, mean score.

  python scripts/40_iqa.py [--variants a,b] [--metrics musiq,clipiqa,maniqa,topiq_nr] [--seeds 0-7]
Writes results/<variant>/<stem>.iqa.json and merges the means into the per-video json under keys iqa_<metric>.
"""
import os, sys, json, argparse, time, numpy as np, imageio, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--variants", default=None); ap.add_argument("--metrics", default="musiq,clipiqa,maniqa,topiq_nr"); ap.add_argument("--seeds", default="0-7")
ap.add_argument("--n-frames", type=int, default=8)
args = ap.parse_args()
import pyiqa
dev = "cuda"
metrics = {m: pyiqa.create_metric(m, device=dev) for m in args.metrics.split(",")}
for m, M in metrics.items():
    print(m, "lower_better" if M.lower_better else "higher_better", flush=True)
s0, s1 = map(int, args.seeds.split("-"))
cdir = os.path.join(ROOT, "corpus")
variants = args.variants.split(",") if args.variants else sorted(d for d in os.listdir(cdir) if d != "smoke" and os.path.isdir(os.path.join(cdir, d)))
t0 = time.time(); n_done = 0
for v in variants:
    files = sorted(f for f in os.listdir(os.path.join(cdir, v)) if f.endswith(".mp4") and s0 <= int(f[:-4].split("_s")[1]) <= s1)
    rdir = os.path.join(ROOT, "results", v); os.makedirs(rdir, exist_ok=True)
    todo = [f for f in files if not os.path.exists(os.path.join(rdir, f[:-4] + ".iqa.json"))]
    print(f"[{v}] {len(todo)}/{len(files)} to do", flush=True)
    for f in todo:
        try:
            fr = np.stack(imageio.mimread(os.path.join(cdir, v, f), memtest=False))
        except Exception as e:
            print("  skip", f, repr(e)[:80]); continue
        idx = np.linspace(0, len(fr) - 1, args.n_frames).round().astype(int)
        x = torch.from_numpy(fr[idx]).permute(0, 3, 1, 2).float().div(255).to(dev)
        rec = {}
        with torch.no_grad():
            for m, M in metrics.items():
                rec["iqa_" + m] = float(M(x).float().mean())
        json.dump(rec, open(os.path.join(rdir, f[:-4] + ".iqa.json"), "w"))
        jp = os.path.join(rdir, f[:-4] + ".json")
        if os.path.exists(jp):
            j = json.load(open(jp)); j.update(rec); json.dump(j, open(jp, "w"), indent=1)
        n_done += 1
        if n_done % 100 == 0:
            print(f"  {n_done} videos, {time.time() - t0:.0f}s", flush=True)
print("IQA_DONE", n_done, f"{time.time() - t0:.0f}s", flush=True)
