"""Figures from logs/pilot_<tag>.json (+ the per-video / diversity CSVs written by 30_pilot_analysis.py).

  fig1_motion_vs_temporal_<tag>.png   A3 vs A5 per config (the anti-correlation VBench totals hide)
  fig2_decomposition_<tag>.png        additive tax decomposition per axis, standardised units, with CIs
  fig3_pareto_<tag>.png               precision vs recall, and diversity vs semantic, per config
  fig4_cfg_sweep_<tag>.png            teacher diversity / precision / recall vs CFG with the students' positions
Colours: one hue per *model* (weights); teacher sampler variants share the teacher hue with different markers.
"""
import os, sys, json, argparse, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="pilot"); args = ap.parse_args()
J = json.load(open(os.path.join(ROOT, "logs", f"pilot_{args.tag}.json")))
df = pd.read_csv(os.path.join(ROOT, "logs", f"pilot_{args.tag}_videos.csv"))
div = pd.read_csv(os.path.join(ROOT, "logs", f"pilot_{args.tag}_diversity.csv"))
FIG = os.path.join(ROOT, "figures"); os.makedirs(FIG, exist_ok=True)

# validated categorical palette (dataviz reference), fixed order, one hue per model
HUE = {"T": "#2a78d6", "AR-diff": "#eb6834", "S-bi-causvid": "#1baf7a", "S-bi-fastwan": "#eda100",
       "S-causvid": "#e87ba4", "S-sf": "#008300", "S-cf": "#4a3aa7", "rcm": "#e34948"}
MARK = {"T-full": "o", "T-few": "s", "T-nocfg": "^", "T-cfg3": "v", "T-cfg7.5": "D", "AR-diff": "o", "AR-diff-cfg3": "v", "AR-diff-cfg7.5": "D",
        "S-bi-causvid": "o", "S-bi-fastwan": "o", "S-causvid": "o", "S-sf": "o", "S-cf": "o",
        "AR-diff-rcm": "o", "S-bi-rcm": "s", "S-rcm-tfdcm": "^", "S-rcm-sfdmd": "D"}
ORDER = ["T-full", "T-few", "T-nocfg", "T-cfg3", "T-cfg7.5", "AR-diff", "AR-diff-cfg3", "AR-diff-cfg7.5", "S-bi-causvid", "S-bi-fastwan", "S-causvid", "S-sf", "S-cf",
         "AR-diff-rcm", "S-bi-rcm", "S-rcm-tfdcm", "S-rcm-sfdmd"]
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"


def hue(v):
    if v.startswith("T-"):
        return HUE["T"]
    if "rcm" in v:
        return HUE["rcm"]
    return HUE.get(v.replace("-cfg3", "").replace("-cfg7.5", ""), "#888")


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TXT2, labelsize=8); ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.xaxis.label.set_color(TXT2); ax.yaxis.label.set_color(TXT2); ax.title.set_color(TXT)


def variants_present():
    return [v for v in ORDER if v in J["variants"]]


means = pd.DataFrame(J["means"]); sems = pd.DataFrame(J["sems"])

# ---------------------------------------------------------------- fig1: A3 vs A5
fig, ax = plt.subplots(figsize=(6.4, 4.6), dpi=150)
for v in variants_present():
    x, y = means.loc[v, "flow_residual"], means.loc[v, "warp_error"]
    ax.errorbar(x, y, xerr=sems.loc[v, "flow_residual"], yerr=sems.loc[v, "warp_error"], fmt=MARK[v], color=hue(v),
                ms=7, mec="white", mew=1.2, elinewidth=1, capsize=0)
    ax.annotate(v, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=TXT)
ax.set_xlabel("A3 object motion: residual flow (px / frame)  →  more motion")
ax.set_ylabel("A5 temporal quality: warp error  ↓  better")
ax.set_title("Motion vs temporal quality per configuration (mean ± s.e.m.)", fontsize=10)
style(ax); fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig1_motion_vs_temporal_{args.tag}.png")); plt.close(fig)

# ---------------------------------------------------------------- fig2: decomposition
if "decomposition" in J:
    chains = list(J["decomposition"].keys())
    fig, axes = plt.subplots(1, len(chains), figsize=(4.6 * len(chains), 4.4), dpi=150, squeeze=False)
    for ax, ch in zip(axes[0], chains):
        D = pd.DataFrame(J["decomposition"][ch])
        axes_ = list(dict.fromkeys(D.axis)); steps = list(dict.fromkeys(D.step))
        sd = {a: df[{"A3_motion": "flow_residual", "A4_semantic": "vqa_yes", "A5_temporal": "warp_error"}.get(a, "flow_residual")].std() for a in axes_}
        sd["A2_diversity"] = div.pair_dist.std(); sd["A1_coverage"] = 0.05
        w = 0.8 / len(steps)
        for i, st in enumerate(steps):
            sub = D[D.step == st]
            vals = [sub[sub.axis == a].tax.iloc[0] / sd[a] for a in axes_]
            err = [[(sub[sub.axis == a].tax.iloc[0] - sub[sub.axis == a].lo.iloc[0]) / sd[a] for a in axes_],
                   [(sub[sub.axis == a].hi.iloc[0] - sub[sub.axis == a].tax.iloc[0]) / sd[a] for a in axes_]]
            err = np.nan_to_num(np.array(err))
            col = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"][i]
            ax.bar(np.arange(len(axes_)) + i * w - 0.4 + w / 2, vals, w * 0.92, color=col, yerr=err, ecolor=TXT2, capsize=2, label=st, lw=0)
        ax.axhline(0, color=TXT2, lw=0.8)
        ax.set_xticks(range(len(axes_))); ax.set_xticklabels([a.replace("_", "\n") for a in axes_], fontsize=8)
        ax.set_ylabel("tax  (σ units; + = lost along this step)"); ax.set_title(ch, fontsize=9); ax.legend(fontsize=7, frameon=False)
        style(ax)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig2_decomposition_{args.tag}.png")); plt.close(fig)

# ---------------------------------------------------------------- fig3: pareto panels
cov = pd.DataFrame(J["coverage"]).set_index("variant")
divm = pd.DataFrame(J["diversity"])
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), dpi=150)
ax = axes[0]
for v in variants_present():
    key = v + " (self split)" if v == "T-full" else v
    if key not in cov.index:
        continue
    x, y = cov.loc[key, "recall"], cov.loc[key, "precision"]
    ax.errorbar(x, y, xerr=[[x - cov.loc[key, "recall_lo"]], [cov.loc[key, "recall_hi"] - x]],
                yerr=[[y - cov.loc[key, "precision_lo"]], [cov.loc[key, "precision_hi"] - y]], fmt=MARK[v], color=hue(v), ms=7, mec="white", mew=1.2, elinewidth=1)
    ax.annotate(v if v != "T-full" else "T-full (self split)", (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=TXT)
ax.set_xlabel("A1 recall vs T-full manifold  →  more of the teacher's modes covered"); ax.set_ylabel("precision  →  samples inside the teacher manifold")
ax.set_title("Coverage / fidelity (VideoMAE space, k=3, 95% CI over prompts)", fontsize=10); style(ax)
ax = axes[1]
for v in variants_present():
    if v not in divm.index:
        continue
    x, y = divm.loc[v, "mean"], means.loc[v, "vqa_yes"]
    ax.errorbar(x, y, xerr=divm.loc[v, "sem"], yerr=sems.loc[v, "vqa_yes"], fmt=MARK[v], color=hue(v), ms=7, mec="white", mew=1.2, elinewidth=1)
    ax.annotate(v, (x, y), xytext=(6, 4), textcoords="offset points", fontsize=8, color=TXT)
ax.set_xlabel("A2 cross-seed diversity (pairwise cosine distance)"); ax.set_ylabel("A4 semantic adherence (BLIP-2 P(yes))")
ax.set_title("Diversity vs prompt adherence", fontsize=10); style(ax)
fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig3_pareto_{args.tag}.png")); plt.close(fig)

# ---------------------------------------------------------------- fig4: CFG sweep
if "cfg_sweep" in J and len(J["cfg_sweep"]) >= 2:
    cf = pd.DataFrame(J["cfg_sweep"]).sort_values("cfg")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), dpi=150)
    for ax, m, lab in zip(axes, ["pair_dist", "recall", "precision"], ["A2 diversity (pairwise dist.)", "A1 recall vs T-full", "precision"]):
        ax.plot(cf.cfg, cf[m], "-o", color=HUE["T"], lw=2, ms=7, mec="white", mew=1.2, label="teacher, 50 steps")
        for c, val in zip(cf.cfg, cf[m]):
            ax.annotate(f"{val:.3f}", (c, val), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=7, color=TXT2)
        for v in variants_present():
            if v.startswith("S-") or v.startswith("AR"):
                y = divm.loc[v, "mean"] if m == "pair_dist" else (cov.loc[v, m] if v in cov.index else np.nan)
                ax.axhline(y, color=hue(v), lw=1.2, ls="--"); ax.annotate(v, (7.6, y), fontsize=7, color=hue(v), va="center")
        ax.set_xlabel("teacher CFG"); ax.set_ylabel(lab); ax.set_xticks([1, 3, 5, 7.5]); ax.set_xlim(0.5, 9.2); style(ax)
    axes[0].set_title("Where do the students sit on the teacher's CFG curve?", fontsize=10, loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, f"fig4_cfg_sweep_{args.tag}.png")); plt.close(fig)
print("figures written to", FIG)
