# Prompt sets (pre-registered 2026-09-13, before any generation)

* `general.txt` — 15 prompts taken verbatim from the VBench prompt list (`tas/data/vbench_all.txt`),
  chosen by hand *before* running anything to cover the VBench dimensions: subject consistency
  (animals), human action, scene / object class, temporal flickering (still scenes) and
  dynamic degree. Mix of static (still frame / tranquil tableau), slow (walk, read, drink) and
  dynamic (run, take off, turn a corner) content.
* `distill_sensitive.txt` — 15 new prompts (tab-separated `tag<TAB>prompt`) written to hit the regimes
  prior work reports as most affected by few-step distillation (guide §3.7):
  `fast` fast large-amplitude motion (4), `multi` multi-object interaction (4),
  `rare` rare / compositional combinations (3), `count` object-count conservation (3),
  `physics` a physical event with a clear before/after (1).

Rule (guide §9): the two sets are fixed now and are not edited after seeing results.
Negative prompt for CFG runs: the official Wan2.1 Chinese negative prompt (see `src/wan_env.py:NEG`).
