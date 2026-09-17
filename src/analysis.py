"""Stage-1 rank analysis and stage-2 tax decomposition (guide §2.2, §5).

Inputs: results/<variant>/*.json (+ .feat.npz).  Produces a tidy per-video table, the
method x axis matrix of standardised deltas vs T-full, its axis-correlation matrix and
singular-value spectrum, and the additive tax decomposition with bootstrap CIs.

Axis primary metrics (one per axis, no aggregation across axes -- guide §3.3):
  A1 coverage   : recall  (k-NN manifold, VideoMAE space; DINOv2 as robustness check)  [set-level]
  A2 diversity  : mean pairwise cosine distance across the 8 seeds of a prompt (VideoMAE) [prompt-level]
  A3 motion     : flow_residual (object motion; flow_global reported alongside)          [video-level]
  A4 semantic   : vqa_yes (BLIP-2 per-prompt questions; clip_t alongside)               [video-level]
  A5 temporal   : warp_error, sign-flipped so that higher = better                       [video-level]
"""
import os, json, glob, numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AXES = {  # name -> (metric, sign)  sign=+1 higher is better
    "A1_coverage": ("recall", +1), "A2_diversity": ("pair_dist", +1), "A3_motion": ("flow_residual", +1),
    "A4_semantic": ("vqa_yes", +1), "A5_temporal": ("warp_error", -1),
}


def load_results(results_dir=os.path.join(ROOT, "results")):
    rows, feats = [], {}
    for j in glob.glob(os.path.join(results_dir, "*", "*.json")):
        r = json.load(open(j)); r = {k: v for k, v in r.items() if not k.startswith("_")}
        r.pop("vqa_per_q", None)
        rows.append(r)
        fp = j[:-5] + ".feat.npz"
        if os.path.exists(fp):
            feats[(r["variant"], r["pid"], r["seed"])] = dict(np.load(fp))
    return pd.DataFrame(rows), feats


def pairwise_diversity(df, feats, space="videomae"):
    """A2: per (variant, prompt) mean pairwise cosine distance over seeds."""
    out = []
    for (v, pid), g in df.groupby(["variant", "pid"]):
        F = np.stack([feats[(v, pid, s)][space] for s in g.seed if (v, pid, s) in feats])
        if len(F) < 2:
            continue
        D = 1 - F @ F.T
        iu = np.triu_indices(len(F), 1)
        out.append(dict(variant=v, pid=pid, subset=g.subset.iloc[0], n=len(F), pair_dist=float(D[iu].mean())))
    return pd.DataFrame(out)


def coverage_table(df, feats, ref="T-full", space="videomae", k=3, n_boot=None):
    """k-NN precision/recall of every variant against the reference.  k-NN metrics are set-size
    sensitive, so each comparison uses the seeds common to the reference and the variant, and the
    reference is subsampled to those seeds (a 4-seed variant is compared with a 4-seed reference)."""
    from metrics.coverage import prdc_jackknife as prdc_ci
    R = df[df.variant == ref]
    out = []
    for v, g in df.groupby("variant"):
        common = sorted(set(R.seed.unique()) & set(g.seed.unique()))
        if len(g) < 4 * 30:          # smoke-only variants (1 video) cannot support k-NN metrics
            continue
        Rs = R[R.seed.isin(common)]; g = g[g.seed.isin(common)]
        Rf = np.stack([feats[(ref, p, s)][space] for p, s in zip(Rs.pid, Rs.seed)]); Rg = Rs.pid.values
        G = np.stack([feats[(v, p, s)][space] for p, s in zip(g.pid, g.seed)]); Gg = g.pid.values
        if v == ref:   # noise floor: split the reference by seed parity
            m = Rs.seed.values % 2 == 0
            res = prdc_ci(Rf[m], Rf[~m], Rg[m], Rg[~m], k)
            v = ref + " (self split)"
        else:
            res = prdc_ci(Rf, G, Rg, Gg, k)
        row = dict(variant=v, space=space, n_seeds=len(common))
        for mname, d in res.items():
            row[mname] = d["value"]; row[mname + "_lo"] = d["lo"]; row[mname + "_hi"] = d["hi"]
        out.append(row)
    return pd.DataFrame(out)


def add_adjusted_temporal(df):
    """warp_error_adj: residual of log(warp_error) after a pooled linear fit on log(1+flow_total),
    i.e. temporal quality at matched motion (secondary to the raw A5 primary)."""
    x = np.log1p(df.flow_total.values); y = np.log(df.warp_error.values + 1e-6)
    a, b = np.polyfit(x, y, 1)
    df = df.copy(); df["warp_error_adj"] = y - (a * x + b)
    return df


def video_level_means(df, metrics):
    return df.groupby("variant")[metrics].agg(["mean", "sem"])


def tax_matrix(df, div, cov, ref="T-full", variants=None):
    """Rows = variants, cols = axes; value = (variant - ref) / pooled std of the axis at video
    (or prompt / set) level, sign-corrected so negative = worse than T-full."""
    variants = variants or [v for v in df.variant.unique() if v != ref]
    M = {}
    for ax, (metric, sign) in AXES.items():
        if metric == "recall":
            base = cov.set_index("variant")
            sd = 0.05                                   # set-level: use a fixed scale (recall is a fraction)
            ref_val = base.loc[ref + " (self split)", "recall"] if ref + " (self split)" in base.index else 1.0
            M[ax] = {v: sign * (base.loc[v, "recall"] - ref_val) / sd for v in variants if v in base.index}
        elif metric == "pair_dist":
            sd = div.pair_dist.std()
            ref_val = div[div.variant == ref].pair_dist.mean()
            M[ax] = {v: sign * (div[div.variant == v].pair_dist.mean() - ref_val) / sd for v in variants}
        else:
            sd = df[metric].std()
            ref_val = df[df.variant == ref][metric].mean()
            M[ax] = {v: sign * (df[df.variant == v][metric].mean() - ref_val) / sd for v in variants}
    return pd.DataFrame(M).loc[variants]


def rank_diagnostics(T):
    """Axis-correlation matrix across configurations and the eigenvalue share of that correlation
    matrix (scale-free: each axis z-scored across configs first).  A rank-1 tax has one dominant
    eigenvalue; the pre-registered C2 criterion is 2nd share > 20% or |r| generally < 0.7."""
    X = T.values.astype(float)
    corr = np.corrcoef(X.T)
    ev = np.sort(np.linalg.eigvalsh(np.nan_to_num(corr)))[::-1]
    return pd.DataFrame(corr, index=T.columns, columns=T.columns), ev / ev.sum()


def axis_estimates(df, div, cov, ref="T-full"):
    """Per (variant, axis): value, standard error, sign-corrected so higher = better."""
    from scipy import stats
    rows = []
    for v in df.variant.unique():
        for ax, (metric, sign) in AXES.items():
            if metric == "recall":
                c = cov.set_index("variant"); key = v + " (self split)" if v == ref else v
                if key not in c.index:
                    continue
                val, se = c.loc[key, "recall"], (c.loc[key, "recall_hi"] - c.loc[key, "recall_lo"]) / 3.92
            elif metric == "pair_dist":
                x = div[div.variant == v].pair_dist; val, se = x.mean(), x.sem()
            else:
                x = df[df.variant == v][metric]; val, se = x.mean(), x.sem()
            rows.append(dict(variant=v, axis=ax, value=sign * val, se=se))
    return pd.DataFrame(rows)


def rank_reversals(est, variants=None, z=1.96):
    """Count (pair of configs, pair of axes) with a *significant* rank reversal: config i beats j on
    axis a and j beats i on axis b, both differences > z * SE.  Any such reversal is direct evidence
    that the tax is not a single scalar."""
    E = est.pivot(index="variant", columns="axis", values="value"); S = est.pivot(index="variant", columns="axis", values="se")
    variants = variants or E.index.tolist(); axes = E.columns.tolist()
    out = []
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            a_, b_ = variants[i], variants[j]
            d = (E.loc[a_] - E.loc[b_]); sd = np.sqrt(S.loc[a_] ** 2 + S.loc[b_] ** 2); zs = d / sd
            for x in range(len(axes)):
                for y in range(x + 1, len(axes)):
                    if zs.iloc[x] > z and zs.iloc[y] < -z or zs.iloc[x] < -z and zs.iloc[y] > z:
                        out.append(dict(config_a=a_, config_b=b_, axis_x=axes[x], z_x=float(zs.iloc[x]), axis_y=axes[y], z_y=float(zs.iloc[y])))
    return pd.DataFrame(out)


def decomposition(df, div, cov, chain=("T-full", "T-few", "S-bi-causvid", "S-causvid"), n_boot=500, seed=0):
    """Additive tax decomposition along a chain, per axis, with bootstrap over prompts.
       total = chain[0] - chain[-1] = sum of successive differences."""
    rng = np.random.default_rng(seed)
    prompts = sorted(df.pid.unique())

    def axis_values(sub_df, sub_div, sub_cov):
        vals = {}
        for ax, (metric, sign) in AXES.items():
            if metric == "recall":
                base = sub_cov.set_index("variant")["recall"] if sub_cov is not None else None
                vals[ax] = {v: sign * float(base.get(v if v != chain[0] else v + " (self split)", np.nan)) for v in chain} if base is not None else None
            elif metric == "pair_dist":
                vals[ax] = {v: sign * sub_div[sub_div.variant == v].pair_dist.mean() for v in chain}
            else:
                vals[ax] = {v: sign * sub_df[sub_df.variant == v][metric].mean() for v in chain}
        return vals

    point = axis_values(df, div, cov)
    boots = {ax: [] for ax in AXES}
    for _ in range(n_boot):
        ps = rng.choice(prompts, size=len(prompts), replace=True)
        bdf = pd.concat([df[df.pid == p] for p in ps]); bdiv = pd.concat([div[div.pid == p] for p in ps])
        bv = axis_values(bdf, bdiv, None)
        for ax in AXES:
            if bv[ax] is not None:
                boots[ax].append([bv[ax][v] for v in chain])
    rows = []
    for ax in AXES:
        pv = point[ax]
        if pv is None:
            continue
        steps = [(chain[i], chain[i + 1], pv[chain[i]] - pv[chain[i + 1]]) for i in range(len(chain) - 1)]
        B = np.array(boots[ax]) if boots[ax] else None
        for i, (a, b, d) in enumerate(steps):
            lo = hi = np.nan
            if B is not None and len(B):
                bd = B[:, i] - B[:, i + 1]; lo, hi = np.percentile(bd, [2.5, 97.5])
            rows.append(dict(axis=ax, step=f"{a} -> {b}", tax=d, lo=lo, hi=hi,
                             total=pv[chain[0]] - pv[chain[-1]]))
    return pd.DataFrame(rows)
