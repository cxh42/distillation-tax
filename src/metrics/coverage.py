"""A1 mode coverage: k-NN manifold precision / recall (Kynkäänniemi et al. 2019) and
density / coverage (Naeem et al. 2020) between a reference set (T-full samples) and a
variant's samples, in a video feature space.  Cosine distance on L2-normalised features.

    precision : fraction of variant samples that fall inside the reference manifold  (fidelity)
    recall    : fraction of reference samples that fall inside the variant manifold  (mode coverage)
    density   : like precision but counts how many reference balls contain the sample (robust to outliers)
    coverage  : fraction of reference samples whose k-NN ball contains >=1 variant sample (robust recall)

The expected distillation signature (guide §3.1) is precision ~flat, recall/coverage down.
Sets are pooled over prompts (8 seeds x 30 prompts = 240 per config); CIs come from a
leave-one-prompt-out jackknife (a with-replacement bootstrap is invalid for k-NN radii).
"""
import numpy as np, torch


def _cdist(a, b):
    a = torch.as_tensor(a, dtype=torch.float32); b = torch.as_tensor(b, dtype=torch.float32)
    return (1.0 - a @ b.T).clamp(min=0)          # cosine distance, features already unit-norm


def prdc(ref, gen, k=3):
    ref, gen = np.asarray(ref), np.asarray(gen)
    drr = _cdist(ref, ref); dgg = _cdist(gen, gen); dgr = _cdist(gen, ref)   # [n,n] [m,m] [m,n]
    r_ref = drr.kthvalue(k + 1, dim=1).values                                # k-th NN excl. self
    r_gen = dgg.kthvalue(k + 1, dim=1).values
    inside_ref = dgr <= r_ref[None, :]                                       # gen i inside ref-ball j
    precision = inside_ref.any(dim=1).float().mean().item()
    recall = (dgr.T <= r_gen[None, :]).any(dim=1).float().mean().item()      # ref j inside gen-ball i
    density = inside_ref.float().sum(dim=1).mean().item() / k
    coverage = (dgr.min(dim=0).values <= r_ref).float().mean().item()
    return dict(precision=precision, recall=recall, density=density, coverage=coverage)


def prdc_jackknife(ref, gen, ref_groups, gen_groups, k=3):
    """Leave-one-prompt-out jackknife CI.  (A bootstrap *with replacement* over prompts is invalid
    for k-NN manifold metrics: duplicated groups collapse the k-th-NN radii.)"""
    ref, gen = np.asarray(ref), np.asarray(gen)
    ref_groups, gen_groups = np.asarray(ref_groups), np.asarray(gen_groups)
    groups = np.unique(np.concatenate([ref_groups, gen_groups]))
    point = prdc(ref, gen, k)
    loo = {m: [] for m in point}
    for g in groups:
        r = prdc(ref[ref_groups != g], gen[gen_groups != g], k)
        for m in r:
            loo[m].append(r[m])
    n = len(groups)
    out = {}
    for m in point:
        v = np.array(loo[m]); se = np.sqrt((n - 1) / n * ((v - v.mean()) ** 2).sum())
        out[m] = dict(value=point[m], lo=point[m] - 1.96 * se, hi=point[m] + 1.96 * se, se=se)
    return out


prdc_bootstrap = None   # removed (invalid for k-NN metrics), see prdc_jackknife


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    a = rng.normal(size=(240, 64)); a /= np.linalg.norm(a, axis=1, keepdims=True)
    b = rng.normal(size=(240, 64)); b /= np.linalg.norm(b, axis=1, keepdims=True)
    print("same distribution:", prdc(a, b))
    c = a[:120] + 0.01 * rng.normal(size=(120, 64)); c /= np.linalg.norm(c, axis=1, keepdims=True)
    c = np.concatenate([c, c])
    print("mode-dropped (half the modes, duplicated):", prdc(a, c))
