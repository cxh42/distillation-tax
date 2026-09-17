"""2x2 factorial interaction (causalisation x distillation) with a prompt-paired bootstrap.

All four cells share prompts AND initial noise (same seed -> same latent noise), so the per-prompt
interaction  I_p = (S-causal_p - S-bi_p) - (AR-diff_p - T-full_p)  is a paired quantity; the CI is a
bootstrap over prompts of mean(I_p).  Diversity (A2) is per-prompt by construction; video-level axes
are averaged over the common seeds within each prompt.

  python scripts/32_interaction.py [--cells T-full,AR-diff,S-bi-causvid,S-causvid] [--seeds 0-7] [--n-boot 5000]
"""
import os, sys, json, argparse, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import analysis as A

ap = argparse.ArgumentParser()
ap.add_argument("--cells", default="T-full,AR-diff,S-bi-causvid,S-causvid")   # bi-50, causal-50, bi-dmd, causal-dmd
ap.add_argument("--seeds", default=None, help="restrict to these seeds (default: seeds common to all four cells)")
ap.add_argument("--space", default="videomae")
ap.add_argument("--n-boot", type=int, default=5000)
ap.add_argument("--tag", default="")
args = ap.parse_args()
T, C, B, D = args.cells.split(",")     # T=bi-50  C=causal-50  B=bi-dmd  D=causal-dmd

df, feats = A.load_results()
df = df[df.variant.isin([T, C, B, D])]
seeds = set.intersection(*[set(df[df.variant == v].seed) for v in (T, C, B, D)])
if args.seeds:
    a, b = args.seeds.split("-"); seeds &= set(range(int(a), int(b) + 1))
df = df[df.seed.isin(seeds)]
prompts = sorted(set.intersection(*[set(df[df.variant == v].pid) for v in (T, C, B, D)]))
df = df[df.pid.isin(prompts)]
print(f"cells {T},{C},{B},{D}  common seeds {sorted(seeds)}  prompts {len(prompts)}")

# per-prompt cell values for each axis (higher = better)
div = A.pairwise_diversity(df, feats, args.space)
div2 = A.pairwise_diversity(df, feats, "dinov2")
rows = {}
for ax, (metric, sign) in A.AXES.items():
    if metric == "recall":
        continue                                   # set-level; handled separately below
    src = div if metric == "pair_dist" else df
    piv = src.groupby(["pid", "variant"])[metric].mean().unstack() * sign
    rows[ax] = piv[[T, C, B, D]]
rows["A2_diversity_dinov2"] = div2.groupby(["pid", "variant"]).pair_dist.mean().unstack()[[T, C, B, D]]

rng = np.random.default_rng(0)
out = {}
print(f"\n{'axis':22s} {'causal-only':>12s} {'distill-only':>13s} {'both':>8s} {'interaction':>12s}   95% CI (paired bootstrap)   P(I>=0)")
for ax, piv in rows.items():
    t, c, b, d = [piv[v].values for v in (T, C, B, D)]
    Ip = (d - b) - (c - t)                         # per-prompt interaction
    n = len(Ip); idx = rng.integers(0, n, size=(args.n_boot, n))
    boot = Ip[idx].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5]); p_ge0 = float((boot >= 0).mean())
    out[ax] = dict(causal_only=float((c - t).mean()), distill_only=float((b - t).mean()), both=float((d - t).mean()),
                   interaction=float(Ip.mean()), lo=float(lo), hi=float(hi), p_ge0=p_ge0, n_prompts=n,
                   se_paired=float(Ip.std(ddof=1) / np.sqrt(n)))
    print(f"{ax:22s} {(c - t).mean():+12.4f} {(b - t).mean():+13.4f} {(d - t).mean():+8.4f} {Ip.mean():+12.4f}   [{lo:+.4f}, {hi:+.4f}]   {p_ge0:.4f}")

# A1 (set-level): jackknife over prompts of the interaction of recall
from metrics.coverage import prdc
def recall_of(ref_v, gen_v, pids):
    sub = df[df.pid.isin(pids)]
    R = sub[sub.variant == ref_v]; G = sub[sub.variant == gen_v]
    Rf = np.stack([feats[(ref_v, p, s)][args.space] for p, s in zip(R.pid, R.seed)])
    Gf = np.stack([feats[(gen_v, p, s)][args.space] for p, s in zip(G.pid, G.seed)])
    return prdc(Rf, Gf)["recall"]
def self_recall(v, pids):
    sub = df[(df.pid.isin(pids)) & (df.variant == v)]
    m = sub.seed.values % 2 == 0
    F = np.stack([feats[(v, p, s)][args.space] for p, s in zip(sub.pid, sub.seed)])
    return prdc(F[m], F[~m])["recall"]
def inter_recall(pids):
    t = self_recall(T, pids); c = recall_of(T, C, pids); b = recall_of(T, B, pids); d = recall_of(T, D, pids)
    return (d - b) - (c - t), (t, c, b, d)
I0, cells = inter_recall(prompts)
loo = np.array([inter_recall([p for p in prompts if p != q])[0] for q in prompts])
se = np.sqrt((len(prompts) - 1) / len(prompts) * ((loo - loo.mean()) ** 2).sum())
print(f"{'A1_coverage(recall)':22s} {cells[1]-cells[0]:+12.4f} {cells[2]-cells[0]:+13.4f} {cells[3]-cells[0]:+8.4f} {I0:+12.4f}   [{I0-1.96*se:+.4f}, {I0+1.96*se:+.4f}]   (jackknife)")
out["A1_coverage"] = dict(causal_only=cells[1]-cells[0], distill_only=cells[2]-cells[0], both=cells[3]-cells[0], interaction=I0, lo=I0-1.96*se, hi=I0+1.96*se)
json.dump(dict(cells=[T, C, B, D], seeds=sorted(seeds), n_prompts=len(prompts), axes=out),
          open(os.path.join(ROOT, "logs", f"interaction{args.tag}.json"), "w"), indent=1)
