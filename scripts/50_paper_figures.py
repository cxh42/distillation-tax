"""Paper figures (PDF + PNG) from data/ and logs/.

  fig_teaser.jpg        four 2x2 cells x 4 seeds, middle frame (the collapse is visible)
  fig_interaction.pdf   A2 / A5 2x2 cells and interaction across the CFG band and the second lab
  fig_recipes.pdf       diversity vs object motion for every student cell; ladders drawn as arrows
  fig_perframe.pdf      per-frame cross-seed diversity (DINOv2) for the 2x2 + rCM cells
  fig_iqa_human.pdf     four IQA metrics vs human quality BT score
  fig_pareto.pdf        recall vs precision, and the teacher CFG curve with students
"""
import os, sys, json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
OUT = os.path.join(ROOT, "paper", "figures"); os.makedirs(OUT, exist_ok=True)
S = pd.read_csv(os.path.join(ROOT, "data", "per_variant_summary.csv"), index_col=0)
TXT, TXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.labelsize": 8, "legend.fontsize": 7, "pdf.fonttype": 42})
NAMES = {"T-full": "Teacher (bi, 50-step)", "AR-diff": "Causal, 50-step (CF lab)", "AR-diff-rcm": "Causal, 50-step (NV lab)",
         "S-bi-causvid": "Bi + DMD (CausVid)", "S-causvid": "Causal + DMD, TF (CausVid)", "S-sf": "Causal + DMD, SR (Self-Forcing)",
         "S-cf": "Causal + DMD, SR (Causal-Forcing)", "S-bi-rcm": "Bi + CM (rCM)", "S-rcm-tfdcm": "Causal + CM, TF (Causal-rCM)",
         "S-rcm-sfdmd": "Causal + CM + SR-DMD (Causal-rCM)", "S-bi-fastwan": "Bi + DMD (FastWan)", "S-cf-cd": "Causal + CM (CF lab)",
         "S-cf-ode": "Causal + ODE-init (CF lab)", "S-causvid-ode": "Causal + ODE-init (CausVid)", "S-sf-sid": "Causal + SiD, SR", "S-sf-sid2": "Causal + SiD-v2, SR",
         "S-sf-gan": "Causal + GAN, SR", "S-cf-fw": "Causal + DMD, SR, chunk=1", "S-rcm-tfdcm-fw": "Causal + CM, TF, chunk=1", "S-rcm-sfdmd-fw": "Causal + CM + SR-DMD, chunk=1"}


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TXT2); ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name + ".pdf"), bbox_inches="tight"); fig.savefig(os.path.join(OUT, name + ".png"), dpi=200, bbox_inches="tight"); plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------ teaser
def teaser(pid="P21", seeds=(0, 1, 2, 3), w=208):
    import imageio
    from PIL import Image, ImageDraw
    cells = [("T-full", "Teacher: bidirectional, 50 steps"), ("AR-diff", "Causal only: autoregressive, 50 steps, undistilled"),
             ("S-bi-causvid", "Distilled only: bidirectional DMD"), ("S-causvid", "Causal x distilled: autoregressive DMD  ->  collapse")]
    h = int(w * 480 / 832); rows = []
    for v, lab in cells:
        row = Image.new("RGB", (w * len(seeds) + 3 * (len(seeds) - 1), h + 16), "white")
        ImageDraw.Draw(row).text((2, 2), lab, fill="black")
        for j, s in enumerate(seeds):
            fr = imageio.mimread(os.path.join(ROOT, "corpus", v, f"{pid}_s{s}.mp4"), memtest=False)
            row.paste(Image.fromarray(fr[len(fr) // 2]).resize((w, h)), (j * (w + 3), 16))
        rows.append(row)
    sheet = Image.new("RGB", (rows[0].width, sum(r.height + 4 for r in rows)), "white"); y = 0
    for r in rows:
        sheet.paste(r, (0, y)); y += r.height + 4
    sheet.save(os.path.join(OUT, "fig_teaser.jpg"), quality=90); print("wrote fig_teaser (", pid, ")")


# ------------------------------------------------------------------ interaction
def interaction():
    J = {k: json.load(open(os.path.join(ROOT, "logs", f"interaction_{k}.json"))) for k in ["band3", "8seed_cfg5", "band7.5", "8seed_rcm", "8seed_rcm_sf"]}
    labels = ["CausVid\nCFG 3", "CausVid\nCFG 5 (8 s.)", "CausVid\nCFG 7.5", "Causal-rCM\nTF-dCM", "Causal-rCM\n+SR-DMD"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), gridspec_kw=dict(wspace=0.35))
    for ax, axis, title in zip(axes, ["A2_diversity", "A5_temporal"], ["A2  cross-seed diversity (VideoMAE)", "A5  temporal coherence  (-warp error)"]):
        x = np.arange(len(labels)); w = 0.2
        for i, (k, col, lab) in enumerate([("causal_only", ORANGE, "causal only"), ("distill_only", AQUA, "distilled only"), ("both", MAGENTA, "both"), ("interaction", BLUE, "interaction")]):
            vals = [J[t]["axes"][axis][k] for t in J]
            err = None
            if k == "interaction":
                err = [[J[t]["axes"][axis]["interaction"] - J[t]["axes"][axis]["lo"] for t in J], [J[t]["axes"][axis]["hi"] - J[t]["axes"][axis]["interaction"] for t in J]]
            ax.bar(x + (i - 1.5) * w, vals, w * 0.92, color=col, yerr=err, ecolor=TXT, capsize=2, lw=0, label=lab)
        ax.axhline(0, color=TXT2, lw=0.7); ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6.5); ax.set_title(title); style(ax)
    axes[0].set_ylabel("change vs teacher (+ = better)")
    axes[0].legend(frameon=False, ncol=2, loc="lower left", fontsize=6)
    save(fig, "fig_interaction")


# ------------------------------------------------------------------ recipes: diversity vs motion
def recipes():
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    groups = {"teacher": (["T-full"], BLUE, "o"), "causal undistilled": (["AR-diff", "AR-diff-rcm"], ORANGE, "o"),
              "bi + DMD": (["S-bi-causvid", "S-bi-fastwan"], AQUA, "s"), "bi + CM": (["S-bi-rcm"], AQUA, "^"),
              "causal + DMD/SiD (TF init)": (["S-causvid", "S-causvid-ode"], MAGENTA, "s"),
              "causal + DMD/SiD (self-rollout)": (["S-sf", "S-cf", "S-sf-sid", "S-sf-sid2", "S-rcm-sfdmd", "S-cf-fw", "S-rcm-sfdmd-fw"], GREEN, "s"),
              "causal + GAN": (["S-sf-gan"], GREEN, "D"),
              "causal + CM / ODE-init (no DMD)": (["S-rcm-tfdcm", "S-cf-cd", "S-cf-ode", "S-rcm-tfdcm-fw"], RED, "^")}
    for lab, (vs, col, mk) in groups.items():
        vs = [v for v in vs if v in S.index]
        ax.scatter(S.loc[vs, "flow_residual"], S.loc[vs, "div_videomae"], c=col, marker=mk, s=28, edgecolors="white", linewidths=0.8, label=lab, zorder=3)
        for v in vs:
            ax.annotate(v, (S.loc[v, "flow_residual"], S.loc[v, "div_videomae"]), xytext=(3, 2), textcoords="offset points", fontsize=5.5, color=TXT2)
    for a, b in [("S-cf-cd", "S-cf-ode"), ("S-cf-ode", "S-cf"), ("S-causvid-ode", "S-causvid"), ("S-rcm-tfdcm", "S-rcm-sfdmd")]:
        ax.annotate("", xy=(S.loc[b, "flow_residual"], S.loc[b, "div_videomae"]), xytext=(S.loc[a, "flow_residual"], S.loc[a, "div_videomae"]),
                    arrowprops=dict(arrowstyle="->", color=TXT2, lw=0.8, shrinkA=4, shrinkB=4))
    ax.axhline(S.loc["T-full", "div_videomae"], color=BLUE, lw=0.7, ls="--"); ax.axvline(S.loc["T-full", "flow_residual"], color=BLUE, lw=0.7, ls="--")
    ax.set_xlabel("A3  object motion: residual flow (px / frame)"); ax.set_ylabel("A2  cross-seed diversity (VideoMAE)")
    ax.set_xlim(0.6, 6.6); ax.legend(frameon=False, loc="lower right", fontsize=6); style(ax)
    save(fig, "fig_recipes")


# ------------------------------------------------------------------ per-frame diversity
def perframe():
    import analysis as A
    df, feats = A.load_results(); df = df[df.seed < 8]
    fig, ax = plt.subplots(figsize=(3.6, 2.6))
    for v, col, ls in [("T-full", BLUE, "-"), ("AR-diff", ORANGE, "-"), ("AR-diff-rcm", ORANGE, "--"), ("S-bi-causvid", AQUA, "-"), ("S-bi-rcm", AQUA, "--"),
                       ("S-causvid", MAGENTA, "-"), ("S-sf", GREEN, "-"), ("S-rcm-sfdmd", GREEN, "--"), ("S-rcm-tfdcm", RED, "-"), ("S-cf-cd", RED, "--")]:
        g = df[df.variant == v]; curves = []
        for pid, gg in g.groupby("pid"):
            F = np.stack([feats[(v, pid, s)]["dinov2_frames"] for s in gg.seed if (v, pid, s) in feats]); n = F.shape[0]; iu = np.triu_indices(n, 1)
            curves.append([(1 - F[:, t] @ F[:, t].T)[iu].mean() for t in range(F.shape[1])])
        c = np.array(curves).mean(0); ax.plot(np.linspace(0, 80, len(c)), c, color=col, ls=ls, lw=1.4, label=v)
    ax.axvspan(0, 12, color=GRID, alpha=0.6, lw=0); ax.text(1, ax.get_ylim()[0] + 0.005, "chunk 1", fontsize=6, color=TXT2)
    ax.set_xlabel("pixel frame index"); ax.set_ylabel("cross-seed diversity (DINOv2, per frame)"); ax.legend(frameon=False, ncol=2, fontsize=5.5); style(ax)
    save(fig, "fig_perframe")


# ------------------------------------------------------------------ IQA vs human
def iqa_human():
    bt = json.load(open(os.path.join(ROOT, "logs", "human_eval_bt.json")))
    cells = ["T-full", "S-sf", "S-bi-causvid", "AR-diff", "S-causvid", "S-rcm-tfdcm"]
    fig, axes = plt.subplots(1, 4, figsize=(7.4, 2.3), gridspec_kw=dict(wspace=0.45))
    for ax, (m, lab) in zip(axes, [("iqa_musiq", "MUSIQ (VBench imaging quality)"), ("iqa_clipiqa", "CLIP-IQA"), ("iqa_maniqa", "MANIQA"), ("iqa_topiq_nr", "TOPIQ")]):
        x = [bt["quality"][c] for c in cells]; y = S.loc[cells, m]
        ax.scatter(x, y, c=[BLUE if c == "T-full" else MAGENTA if c == "S-causvid" else TXT2 for c in cells], s=26, edgecolors="white", zorder=3)
        for c, xi, yi in zip(cells, x, y):
            ax.annotate(c, (xi, yi), xytext=(3, 2), textcoords="offset points", fontsize=5.5, color=TXT2)
        from scipy import stats
        rho = stats.spearmanr(x, y)[0]; ax.set_title(f"{lab}\n$\\rho$ vs human = {rho:+.2f}", fontsize=7)
        ax.set_xlabel("human quality (BT score)"); ax.set_xlim(-2.6, 4.6); style(ax)
    axes[0].set_ylabel("automatic score (higher = 'better')")
    save(fig, "fig_iqa_human")


# ------------------------------------------------------------------ pareto + CFG
def pareto():
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.8))
    ax = axes[0]
    show = ["T-full", "AR-diff", "AR-diff-rcm", "S-bi-causvid", "S-bi-fastwan", "S-bi-rcm", "S-causvid", "S-sf", "S-cf", "S-rcm-tfdcm", "S-rcm-sfdmd", "S-cf-cd", "S-sf-gan", "T-few"]
    for v in show:
        col = BLUE if v.startswith("T-") else ORANGE if v.startswith("AR") else RED if ("tfdcm" in v or "cf-cd" in v or "bi-rcm" in v) else MAGENTA if v == "S-causvid" else GREEN if v in ("S-sf", "S-cf", "S-rcm-sfdmd", "S-sf-gan") else AQUA
        ax.errorbar(S.loc[v, "recall"], S.loc[v, "precision"], xerr=[[S.loc[v, "recall"] - S.loc[v, "recall_lo"]], [S.loc[v, "recall_hi"] - S.loc[v, "recall"]]],
                    yerr=[[S.loc[v, "precision"] - S.loc[v, "precision_lo"]], [S.loc[v, "precision_hi"] - S.loc[v, "precision"]]], fmt="o", color=col, ms=4, mec="white", mew=0.8, elinewidth=0.7)
        ax.annotate(v, (S.loc[v, "recall"], S.loc[v, "precision"]), xytext=(3, 2), textcoords="offset points", fontsize=5.5, color=TXT2)
    ax.set_xlabel("A1  recall of the teacher manifold (k-NN, VideoMAE)"); ax.set_ylabel("precision"); style(ax)
    ax = axes[1]
    cf = [("T-nocfg", 1), ("T-cfg3", 3), ("T-full", 5), ("T-cfg7.5", 7.5)]
    ax.plot([c for _, c in cf], [S.loc[v, "div_videomae"] for v, _ in cf], "-o", color=BLUE, lw=1.6, ms=5, mec="white", label="teacher, 50 steps")
    ax.plot([3, 5, 7.5], [S.loc[v, "div_videomae"] for v in ["AR-diff-cfg3", "AR-diff", "AR-diff-cfg7.5"]], "-o", color=ORANGE, lw=1.6, ms=5, mec="white", label="causal, 50 steps")
    for v, col in [("S-bi-rcm", RED), ("S-rcm-tfdcm", RED), ("S-bi-causvid", AQUA), ("S-sf", GREEN), ("S-rcm-sfdmd", GREEN), ("S-causvid", MAGENTA)]:
        ax.axhline(S.loc[v, "div_videomae"], color=col, lw=0.9, ls="--"); ax.annotate(v, (7.9, S.loc[v, "div_videomae"]), fontsize=5.5, color=col, va="center")
    ax.set_xlabel("teacher CFG"); ax.set_ylabel("A2  cross-seed diversity"); ax.set_xlim(0.5, 9.6); ax.set_xticks([1, 3, 5, 7.5]); ax.legend(frameon=False); style(ax)
    save(fig, "fig_pareto")


if __name__ == "__main__":
    teaser(); interaction(); recipes(); perframe(); iqa_human(); pareto()
