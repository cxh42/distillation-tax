"""Generate the corpus for one variant:  corpus/<variant>/P<idx>_s<seed>.mp4

  python scripts/generate.py --variant T-full --seeds 0-7 [--prompts 0-29] [--tag pilot]

Idempotent: existing files are skipped.  One process = one variant (weights loaded once).
Writes a timing line per video to stdout and corpus/<variant>/manifest.jsonl.
"""
import os, sys, json, time, argparse
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))


def parse_range(s, n):
    if s is None:
        return list(range(n))
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-"); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


ap = argparse.ArgumentParser()
ap.add_argument("--variant", required=True)
ap.add_argument("--seeds", default="0-7")
ap.add_argument("--prompts", default=None, help="indices into the pilot list, e.g. 0-29 or 3,7")
ap.add_argument("--outdir", default=None)
args = ap.parse_args()
# resolve BEFORE importing wan_env (which chdir's into third_party/Causal-Forcing)
outdir = os.path.abspath(args.outdir) if args.outdir else os.path.join(ROOT, "corpus", args.variant)

from prompts_io import load_pilot_prompts, prompt_id
import torch
from wan_env import Runner, write_mp4, VARIANTS

plist = load_pilot_prompts()
pidx = parse_range(args.prompts, len(plist))
seeds = parse_range(args.seeds, 16)
os.makedirs(outdir, exist_ok=True)
todo = [(i, s) for i in pidx for s in seeds
        if not os.path.exists(os.path.join(outdir, f"{prompt_id(i)}_s{s}.mp4"))]
print(f"[{args.variant}] {VARIANTS[args.variant]}  todo {len(todo)} / {len(pidx) * len(seeds)}", flush=True)
if not todo:
    sys.exit(0)

t0 = time.time()
R = Runner(args.variant)
print(f"loaded in {time.time() - t0:.0f}s; GPU {torch.cuda.memory_allocated() / 1e9:.1f} GB", flush=True)
man = open(os.path.join(outdir, "manifest.jsonl"), "a")
for n, (i, s) in enumerate(todo):
    prompt, tag = plist[i]
    t0 = time.time()
    frames = R.generate(prompt, s)
    dt = time.time() - t0
    path = os.path.join(outdir, f"{prompt_id(i)}_s{s}.mp4")
    write_mp4(frames, path)
    rec = dict(variant=args.variant, pid=prompt_id(i), seed=s, prompt=prompt, subset=tag,
               frames=int(frames.shape[0]), h=int(frames.shape[1]), w=int(frames.shape[2]), sec=round(dt, 1))
    man.write(json.dumps(rec, ensure_ascii=False) + "\n"); man.flush()
    print(f"  {n + 1}/{len(todo)} {rec['pid']} s{s} {dt:6.1f}s  peak {torch.cuda.max_memory_allocated() / 1e9:.1f}GB | {prompt[:60]}", flush=True)
print("DONE", args.variant, flush=True)
