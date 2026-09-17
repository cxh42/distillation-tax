# Mode-Seeking Meets Autoregression: a training-free factorial diagnosis of diversity collapse in few-step video generators

Everything behind the paper in [`paper/main.pdf`](paper/main.pdf): a $2\times2$ factorial design
({bidirectional, causal} × {50-step, few-step distilled}) assembled from **public Wan2.1-1.3B checkpoints**
of two labs, 20 student recipes, 5,405 videos, a five-axis non-aggregated evaluation protocol, a blind
pairwise human study and four no-reference IQA metrics. Nothing is trained.

| finding | where |
|---|---|
| The cross-seed **diversity loss is an interaction** of distillation × causal architecture (−0.063 [−0.092, −0.036]), sign-stable across teacher CFG 3/5/7.5 and feature spaces | `paper/` §4.1, `RESULTS.md` §3–4, `logs/interaction_*.json` |
| The interaction belongs to the **objective**: absent under consistency distillation, halved under GAN, present under DMD/SiD; self-rollout training halves it | §4.2, `RESULTS.md` §3.4, §3.8, §5 |
| More than half of the loss is incurred at the **ODE-regression init** stage, before DMD; it is fully present in the **first chunk**; chunk size does not modulate it (failed prediction, reported) | §4.3, `RESULTS.md` §4.5, §5 |
| Causalisation costs temporal coherence, **distillation refunds it**; camera motion is taxed more than object motion; prompt adherence is untaxed; oversaturation comes with causal fine-tuning, not DMD | §4.4 |
| Four frame-level **IQA metrics are inverted** w.r.t. human quality judgement on distilled video | §4.5, `RESULTS.md` §6–7 |

`RESULTS.md` is the lab notebook: pre-registration (§0, written before any result), every intermediate
reading, the two GPU incidents, and the final numbers. It is more detailed than the paper and in Chinese.

## Layout

```
paper/            main.tex, refs.bib, figures/ (fig_*.pdf|png), main.pdf
RESULTS.md        decision memo / lab notebook (Chinese)
data/             per_variant_summary.csv, per_video_metrics.csv (5,405 rows), per-prompt diversity CSVs
logs/             interaction_*.json (paired bootstrap), pilot_*.json, iqa_table.csv, human_eval_bt.json, round3_table.csv
results/<cell>/   per-video metric JSON (+ .iqa.json); features (.npz) are not in git — regenerate with scripts/metrics_watch.py
figures/          analysis figures and human-eye contact sheets; human_eval/ (pairs.json = the 90 blind judgements, pairs_key.json)
prompts/          general.txt, distill_sensitive.txt, vqa_questions.json (all pre-registered), README.md
configs/          common_generation.yaml, variants/README.md
src/              wan_env.py (single inference harness for every cell), metrics/*, analysis.py
scripts/          generate.py, metrics_watch.py, 30_pilot_analysis.py, 32_interaction.py, 40_iqa.py, 50_paper_figures.py,
                  human_eval_server.py (pairwise blind UI), setup_third_party.sh, dl_all_ckpts.sh, convert_diffusers_to_wan.py
third_party/      patches/ only; the four upstream repos are cloned at pinned commits by scripts/setup_third_party.sh
```

The video corpus (16 GB, `corpus/<cell>/P<prompt>_s<seed>.mp4`) and checkpoints (~120 GB) are not in
git; `scripts/dl_all_ckpts.sh` downloads every checkpoint used and `scripts/generate.py` regenerates any
cell deterministically (same seed → same initial noise).

## Reproduce

```bash
python -m venv .venv && . .venv/bin/activate          # torch>=2.7 + CUDA 12.8, diffusers, transformers, pyiqa, ...
bash scripts/setup_third_party.sh                       # Causal-Forcing (patched), Self-Forcing, CausVid, rcm at pinned commits
bash scripts/dl_all_ckpts.sh                            # ~120 GB from HF
python scripts/encode_prompts.py                        # UMT5-XXL embeddings for all prompts -> prompts/prompt_emb.pt
python scripts/generate.py --variant T-full --seeds 0-7 # one cell; see src/wan_env.py:VARIANTS for all names
python scripts/metrics_watch.py --once                  # motion / temporal / features / semantic metrics for every video
python scripts/40_iqa.py                                # MUSIQ, CLIP-IQA, MANIQA, TOPIQ
python scripts/30_pilot_analysis.py --tag final         # tables, rank diagnostics, CFG sweep
python scripts/32_interaction.py --seeds 0-7            # prompt-paired 2x2 interaction with bootstrap CI
python scripts/50_paper_figures.py                      # paper figures
python scripts/human_eval_server.py                     # blind pairwise UI at http://127.0.0.1:8765
```

Hardware used: one RTX 5090 (32 GB), ~130 GPU-hours in total. The metrics watcher must be off while the
50-step causal cells generate (they need ~20 GB).

## Cells

See `configs/variants/README.md` (verbatim `VARIANTS` table) and Table 1 of the paper. Every cell runs
through the same code path (`third_party/Causal-Forcing` + our sdpa patch), the same VAE, the same cached
prompt embeddings and the same 81×480×832 setting.

## Citation

```
@article{modeseeking2026,
  title  = {Mode-Seeking Meets Autoregression: A Training-Free Factorial Diagnosis of Diversity Collapse in Few-Step Video Generators},
  year   = {2026},
  note   = {preprint}
}
```
