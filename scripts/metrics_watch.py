"""Metrics watcher: polls corpus/<variant>/ for finished videos without results and computes
all metric groups, in parallel with generation.  Only starts a video when enough GPU memory
is free (generation of AR-diff peaks at ~20 GB; the metric models need ~11 GB).

  python scripts/metrics_watch.py [--variants a,b,c] [--min-free-gb 11] [--once]
"""
import os, sys, json, time, argparse, subprocess, numpy as np, imageio, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from prompts_io import load_pilot_prompts, prompt_id   # noqa: E402
from metrics import motion, temporal, features, semantic   # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--variants", default=None)
ap.add_argument("--min-free-gb", type=float, default=11.0)
ap.add_argument("--once", action="store_true")
ap.add_argument("--poll", type=int, default=60)
args = ap.parse_args()
pmap = {prompt_id(i): (p, tag) for i, (p, tag) in enumerate(load_pilot_prompts())}
GROUPS = ["motion", "temporal", "features", "semantic"]


def free_gb():
    """GPU memory available to *this* process = free + what it already holds (its own models)."""
    f, t = torch.cuda.mem_get_info()
    return (f + torch.cuda.memory_reserved()) / 1e9


def is_complete(path):
    """A video is 'finished' once it is listed in the manifest (written after the mp4)."""
    return True


def pending():
    out = []
    cdir = os.path.join(ROOT, "corpus")
    vs = args.variants.split(",") if args.variants else sorted(d for d in os.listdir(cdir) if d != "smoke")
    for v in vs:
        d = os.path.join(cdir, v)
        if not os.path.isdir(d):
            continue
        man = os.path.join(d, "manifest.jsonl")
        listed = set()
        if os.path.exists(man):
            for l in open(man):
                try:
                    r = json.loads(l); listed.add(f"{r['pid']}_s{r['seed']}")
                except Exception:
                    pass
        rdir = os.path.join(ROOT, "results", v)
        for stem in sorted(listed):
            j = os.path.join(rdir, stem + ".json")
            rec = json.load(open(j)) if os.path.exists(j) else {}
            if not all(rec.get("_done_" + g) for g in GROUPS):
                out.append((v, stem))
    return out


def process(v, stem):
    cdir = os.path.join(ROOT, "corpus", v); rdir = os.path.join(ROOT, "results", v); os.makedirs(rdir, exist_ok=True)
    jpath = os.path.join(rdir, stem + ".json"); fpath = os.path.join(rdir, stem + ".feat.npz")
    rec = json.load(open(jpath)) if os.path.exists(jpath) else {}
    fr = np.stack(imageio.mimread(os.path.join(cdir, stem + ".mp4"), memtest=False))
    pid, seed = stem.split("_s"); prompt, tag = pmap.get(pid, (None, None))
    rec.update(variant=v, pid=pid, seed=int(seed), prompt=prompt, subset=tag, n_frames=int(fr.shape[0]))
    flow = None
    if not rec.get("_done_motion"):
        m, flow = motion.motion_metrics(fr, return_flow=True); rec.update(m); rec["_done_motion"] = True
    if not rec.get("_done_temporal"):
        rec.update(temporal.temporal_metrics(fr, flow=flow)); rec["_done_temporal"] = True
    if not rec.get("_done_features"):
        np.savez(fpath, **features.all_features(fr)); rec["_done_features"] = True
    if not rec.get("_done_semantic") and prompt is not None:
        rec.update(semantic.semantic_metrics(fr, prompt, pid)); rec["_done_semantic"] = True
    json.dump(rec, open(jpath, "w"), indent=1)
    return rec


n_done = 0
while True:
    todo = pending()
    if not todo:
        if args.once:
            break
        time.sleep(args.poll); continue
    for v, stem in todo:
        while free_gb() < args.min_free_gb:
            time.sleep(30)
        t0 = time.time()
        try:
            rec = process(v, stem); n_done += 1
            print(f"[{time.strftime('%H:%M:%S')}] {v}/{stem} {time.time() - t0:.0f}s  flow_res={rec.get('flow_residual', 0):.2f} "
                  f"subj={rec.get('subject_consistency', 0):.3f} vqa={rec.get('vqa_yes', 0):.2f}  (done {n_done}, queue {len(todo)})", flush=True)
        except Exception as e:
            print(f"[{time.strftime('%H:%M:%S')}] ERROR {v}/{stem}: {e!r}", flush=True)
            time.sleep(10)
    if args.once and not pending():
        break
print("WATCH_DONE", flush=True)
