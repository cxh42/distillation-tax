#!/bin/bash
# Minimal set still needed for the conclusions (2026-09-17 06:10):
#   C loss axis   : S-sf-sid, S-sf-sid2, S-sf-gan          (does a non-DMD / other mode-seeking loss show the interaction?)
#   CM 2nd lab    : S-cf-cd                                (consistency distillation immune -- replicate in the Causal-Forcing lab)
#   ODE-init cells: S-cf-ode, S-causvid-ode                (is the collapse already present before DMD?)
# Everything else (frame-wise multi-step cells, sCM, warp-4step, DF, 16 seeds, CFG-band seeds 4-7) is cut.
cd /home/xinghao/Projects/CVPR_0/dtax
PY=/home/xinghao/Projects/CVPR_0/.venv/bin/python
run() { echo "=== $1 seeds $2 $(date '+%m-%d %H:%M')"; $PY -u scripts/generate.py --variant $1 --seeds $2 ${3:+--prompts $3} 2>&1 | grep --line-buffered -v "kv_cache\[\|it/s\]" ; }
until [ $(wc -l < corpus/S-rcm-tfdcm-fw/manifest.jsonl) -ge 240 ]; do sleep 20; done
systemctl --user stop dtax-chain9; sleep 3; pkill -f "scripts/generate.p[y]"; sleep 5
echo "=== chain11 start $(date '+%m-%d %H:%M')"
for v in S-sf-sid S-sf-sid2 S-sf-gan S-cf-cd S-cf-ode S-causvid-ode; do run $v 0-7; done
echo "=== GEN_DONE $(date '+%m-%d %H:%M')"
until [ $(ls results/S-causvid-ode/*.json 2>/dev/null | wc -l) -ge 240 ]; do sleep 60; done
systemctl --user stop dtax-metrics-c41 2>/dev/null
echo CHAIN_DONE
