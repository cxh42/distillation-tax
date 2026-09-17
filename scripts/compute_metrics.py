"""Offline metrics for a corpus directory (guide §3.5: generate once, measure offline).

  python scripts/compute_metrics.py --variant T-full [--corpus corpus] [--which motion,temporal,features,semantic]

Per video writes results/<variant>/<pid>_s<seed>.json (merged with existing keys) and
results/<variant>/<pid>_s<seed>.feat.npz (videomae / dinov2 features).  Idempotent per group.
"""
import os, sys, json, argparse, time, numpy as np, imageio
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from prompts_io import load_pilot_prompts, prompt_id   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True)
ap.add_argument("--corpus", default="corpus")
ap.add_argument("--results", default="results")
ap.add_argument("--which", default="motion,temporal,features,semantic")
ap.add_argument("--force", action="store_true")
args = ap.parse_args()
which = args.which.split(",")
cdir = os.path.join(ROOT, args.corpus, args.variant)
rdir = os.path.join(ROOT, args.results, args.variant)
os.makedirs(rdir, exist_ok=True)
plist = load_pilot_prompts()
pmap = {prompt_id(i): (p, tag) for i, (p, tag) in enumerate(plist)}

import torch
from metrics import motion, temporal, features, semantic

files = sorted(f for f in os.listdir(cdir) if f.endswith(".mp4"))
print(f"[{args.variant}] {len(files)} videos, groups {which}", flush=True)
t0 = time.time()
for n, f in enumerate(files):
    stem = f[:-4]; pid, seed = stem.split("_s")
    jpath = os.path.join(rdir, stem + ".json"); fpath = os.path.join(rdir, stem + ".feat.npz")
    rec = json.load(open(jpath)) if os.path.exists(jpath) else {}
    need = [g for g in which if args.force or not rec.get("_done_" + g)]
    if "features" in need and os.path.exists(fpath) and not args.force:
        need.remove("features")
    if not need:
        continue
    fr = np.stack(imageio.mimread(os.path.join(cdir, f), memtest=False))
    prompt, tag = pmap.get(pid, (None, None))
    rec.update(variant=args.variant, pid=pid, seed=int(seed), prompt=prompt, subset=tag, n_frames=int(fr.shape[0]))
    flow = None
    if "motion" in need:
        m, flow = motion.motion_metrics(fr, return_flow=True); rec.update(m); rec["_done_motion"] = True
    if "temporal" in need:
        rec.update(temporal.temporal_metrics(fr, flow=flow)); rec["_done_temporal"] = True
    if "features" in need:
        ft = features.all_features(fr)
        np.savez(fpath, **ft); rec["_done_features"] = True
    if "semantic" in need and prompt is not None:
        rec.update(semantic.semantic_metrics(fr, prompt, pid)); rec["_done_semantic"] = True
    json.dump(rec, open(jpath, "w"), indent=1)
    if n % 10 == 0 or n == len(files) - 1:
        print(f"  {n + 1}/{len(files)} {stem}  {time.time() - t0:.0f}s  " +
              " ".join(f"{k}={rec[k]:.3f}" for k in ("flow_total", "flow_residual", "subject_consistency", "clip_t", "vqa_yes") if k in rec), flush=True)
print("DONE metrics", args.variant, flush=True)
