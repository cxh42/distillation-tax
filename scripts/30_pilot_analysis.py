"""Stage-1 pilot analysis: tables + rank diagnostics + decomposition + figures.

  python scripts/30_pilot_analysis.py [--min-seeds 4] [--tag pilotA]

Works on whatever is in results/ (partial corpora allowed; variants with too few videos are dropped).
Writes logs/pilot_<tag>.json (all numbers) and figures/fig*_<tag>.png.
"""
import os, sys, json, argparse, warnings, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
import analysis as A   # noqa: E402
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30); pd.set_option("display.precision", 3)

ap = argparse.ArgumentParser()
ap.add_argument("--min-seeds", type=int, default=4)
ap.add_argument("--tag", default="pilot")
ap.add_argument("--ref", default="T-full")
ap.add_argument("--space", default="videomae")
args = ap.parse_args()

df, feats = A.load_results()
df = df[df._get_numeric_data().notna().all(axis=1) | True]
# keep only prompts x seeds complete enough
cnt = df.groupby("variant").size()
keep = cnt[cnt >= 30 * args.min_seeds].index.tolist()
df = A.add_adjusted_temporal(df[df.variant.isin(keep)].copy())
print("variants:", {v: int(cnt[v]) for v in keep})
if args.ref not in keep:
    sys.exit("reference variant not ready")

VID = ["flow_residual", "flow_global", "flow_total", "dynamic_degree", "residual_autocorr", "vqa_yes", "vqa_min_q", "vqascore", "clip_t",
       "warp_error", "warp_error_adj", "subject_consistency", "background_consistency", "temporal_flicker", "sharpness", "colorfulness"]
means = df.groupby("variant")[VID].mean()
sems = df.groupby("variant")[VID].sem()
print("\n=== per-variant means (video level) ===")
print(means.round(3).to_string())

div = A.pairwise_diversity(df, feats, args.space)
div_tab = div.groupby("variant").pair_dist.agg(["mean", "sem", "count"])
print("\n=== A2 diversity: mean pairwise cosine distance across seeds (%s) ===" % args.space)
print(div_tab.round(4).to_string())

cov = A.coverage_table(df, feats, args.ref, args.space)
print("\n=== A1 coverage vs %s (%s space, k=3) ===" % (args.ref, args.space))
print(cov.set_index("variant")[["precision", "precision_lo", "precision_hi", "recall", "recall_lo", "recall_hi", "density", "coverage"]].round(3).to_string())
cov2 = A.coverage_table(df, feats, args.ref, "dinov2")
print("\n=== A1 coverage, DINOv2 space (robustness) ===")
print(cov2.set_index("variant")[["precision", "recall", "density", "coverage"]].round(3).to_string())

T = A.tax_matrix(df, div, cov, args.ref)
print("\n=== tax matrix: standardised delta vs %s (negative = worse) ===" % args.ref)
print(T.round(2).to_string())
out = dict(variants=keep, means=means.to_dict(), sems=sems.to_dict(), diversity=div_tab.to_dict(),
           coverage=cov.to_dict("records"), coverage_dinov2=cov2.to_dict("records"), tax_matrix=T.to_dict())
for label, rows in [("all", T.index.tolist()), ("students", [v for v in T.index if v.startswith("S-")])]:
    if len(rows) >= 3:
        corr, spec = A.rank_diagnostics(T.loc[rows])
        print(f"\n=== rank diagnostics ({label}: {len(rows)} rows) ===")
        print("axis correlation across configs:\n", corr.round(2).to_string())
        print("eigenvalue share of the axis-correlation matrix:", np.round(spec, 3), " -> 2nd share = %.1f%%" % (100 * spec[1]))
        out[f"axis_corr_{label}"] = corr.to_dict(); out[f"sv_{label}"] = spec.tolist()

est = A.axis_estimates(df, div, cov, args.ref)
rr = A.rank_reversals(est, [v for v in est.variant.unique() if v != args.ref])
print(f"\n=== significant rank reversals among non-reference configs: {len(rr)} (config pairs x axis pairs, both |z|>1.96) ===")
if len(rr):
    print(rr.round(2).to_string(index=False))
rrs = A.rank_reversals(est, [v for v in est.variant.unique() if v.startswith("S-")])
print(f"=== ... among the 5 students only: {len(rrs)} ===")
out["rank_reversals"] = rr.to_dict("records"); out["rank_reversals_students"] = rrs.to_dict("records"); out["axis_estimates"] = est.to_dict("records")

# video-level axis correlation (within the student pool)
S = df[df.variant.str.startswith("S-")]
vl = S[["flow_residual", "vqa_yes", "warp_error"]].corr(method="spearman")
print("\n=== video-level Spearman among students (flow_residual, vqa_yes, warp_error) ===\n", vl.round(2).to_string())
out["video_level_corr_students"] = vl.to_dict()

# subset split: general vs distillation-sensitive
print("\n=== A3/A4 by subset (general vs sensitive) ===")
sub = df.assign(sens=df.subset != "general").groupby(["variant", "sens"])[["flow_residual", "vqa_yes", "warp_error"]].mean().unstack()
print(sub.round(3).to_string()); out["by_subset"] = {str(k): v for k, v in sub.to_dict().items()}

# decomposition chains
for chain in [("T-full", "T-few", "S-bi-causvid", "S-causvid"), ("T-full", "AR-diff", "S-causvid"), ("T-full", "S-bi-causvid", "S-causvid", "S-sf"), ("T-full", "S-bi-causvid", "S-cf")]:
    if all(c in keep for c in chain):
        D = A.decomposition(df, div, cov, chain)
        print("\n=== decomposition along", " -> ".join(chain), "===")
        print(D.round(3).to_string(index=False)); out.setdefault("decomposition", {})[" -> ".join(chain)] = D.to_dict("records")

# CFG curve
cfgs = {"T-nocfg": 1.0, "T-cfg3": 3.0, "T-full": 5.0, "T-cfg7.5": 7.5}
have = [v for v in cfgs if v in keep]
if len(have) >= 2:
    print("\n=== teacher CFG sweep ===")
    rows = []
    for v in have:
        c = cov.set_index("variant"); key = v + " (self split)" if v == args.ref else v
        rows.append(dict(cfg=cfgs[v], variant=v, pair_dist=div[div.variant == v].pair_dist.mean(), precision=c.loc[key, "precision"], recall=c.loc[key, "recall"],
                         flow_residual=means.loc[v, "flow_residual"], vqa_yes=means.loc[v, "vqa_yes"], warp_error=means.loc[v, "warp_error"]))
    cf = pd.DataFrame(rows).sort_values("cfg"); print(cf.round(3).to_string(index=False)); out["cfg_sweep"] = cf.to_dict("records")

json.dump(out, open(os.path.join(ROOT, "logs", f"pilot_{args.tag}.json"), "w"), indent=1, default=float)
df.to_csv(os.path.join(ROOT, "logs", f"pilot_{args.tag}_videos.csv"), index=False)
div.to_csv(os.path.join(ROOT, "logs", f"pilot_{args.tag}_diversity.csv"), index=False)
print("\nsaved logs/pilot_%s.json" % args.tag)
