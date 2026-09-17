# data/ — consolidated tables (everything the paper quotes)

All numbers use seeds 0–7 (8 seeds × 30 prompts = 240 videos per cell; the CFG-band and `T-few`
cells have seeds 0–3). Cell names are the keys of `src/wan_env.py:VARIANTS`
(`T-*` teacher family, `AR-diff*` causal undistilled, `S-*` students).

## per_variant_summary.csv (one row per cell)
| column | meaning |
|---|---|
| `n` | videos in the cell |
| `flow_residual`, `flow_global` | A3: mean RAFT flow magnitude (px/frame at 480×832) of the residual (object) / robust-affine global (camera) field |
| `dynamic_degree` | VBench-style dynamic-degree indicator (mean over videos) |
| `vqa_yes`, `clip_t` | A4: BLIP-2 P(yes) on the prompt's pre-registered questions (mean over questions × 8 frames); CLIP ViT-L/14 text–frame cosine |
| `warp_error` | A5: RAFT backward-warp L1 between consecutive frames (lower = more coherent) |
| `subject_consistency` | DINO ViT-B/16 CLS cosine consistency (VBench recipe) |
| `sharpness`, `colorfulness` | Laplacian variance; Hasler–Süsstrunk colourfulness (over-saturation ↑) |
| `div_videomae`, `div_videomae_sem`, `div_dinov2` | A2: mean pairwise cosine distance across the seeds of a prompt, averaged over prompts (VideoMAE-base K400 features / DINOv2-L frame-mean features) |
| `recall`, `recall_lo/hi`, `precision`, `precision_lo/hi` | A1: k-NN manifold recall/precision (k=3) of the cell against `T-full`, VideoMAE space, seeds matched to the reference; 95% leave-one-prompt-out jackknife CI |
| `recall_dinov2`, `precision_dinov2` | same in DINOv2 space |
| `iqa_musiq`, `iqa_clipiqa`, `iqa_maniqa`, `iqa_topiq_nr` | no-reference IQA (pyiqa), mean of 8 uniformly spaced frames |

## per_video_metrics.csv (one row per video, 5,405 rows)
Same metric definitions at video level, plus: `flow_total`, `flow_total_p95`, `residual_frac`,
`residual_autocorr` (lag-1 autocorrelation of per-pair residual magnitude; jitter ↓), `zoom_abs`,
`dynamic_degree_score` (the VBench top-5 % flow statistic before thresholding), `background_consistency`
(CLIP B/32), `temporal_flicker` (1 − mean |ΔI|/255), `vqascore` (BLIP-2 P(yes) to "Does this image show
{prompt}?"), `vqa_min_q` (hardest question), `vqa_per_q` (JSON list). `subset` is `general` or the
distillation-sensitive category (`fast`, `multi`, `rare`, `count`, `physics`).

## per_prompt_diversity_{videomae,dinov2}.csv
Per (cell, prompt): `pair_dist` = mean pairwise cosine distance over that prompt's seeds, `n` = seeds.
These are the paired observations behind every A2 confidence interval.

## Where the rest lives
* `logs/interaction_*.json` — 2×2 main effects and interaction per axis with prompt-paired bootstrap CIs (`8seed_cfg5` = headline; `band3`, `band7.5` = CFG band; `8seed_rcm`, `8seed_rcm_sf` = NVIDIA cells).
* `logs/pilot_pilotB_v2.json` — full analysis dump (means, coverage tables, tax matrix, axis-correlation spectrum, rank reversals, CFG sweep).
* `logs/human_eval_bt.json` — Bradley–Terry scores from the 90 pairwise judgements (`figures/human_eval/pairs.json`, key in `pairs_key.json`).
* `logs/iqa_table.csv`, `logs/round3_table.csv` — IQA per cell; round-3 (chunk / loss / stage) table.
* `results/<cell>/<pid>_s<seed>.json` — the raw per-video record; `.iqa.json` the IQA part; `.feat.npz` (release asset) the VideoMAE / DINOv2 / per-frame DINOv2 features.
