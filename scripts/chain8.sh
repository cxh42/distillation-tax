#!/bin/bash
# Minimal tail: only what the paper still needs from the GPU -- the CFG=1 point (T-nocfg seeds 0-3) and its metrics.
cd /home/xinghao/Projects/CVPR_0/dtax
PY=/home/xinghao/Projects/CVPR_0/.venv/bin/python
echo "=== T-nocfg seeds 0-3 $(date '+%m-%d %H:%M')"
$PY -u scripts/generate.py --variant T-nocfg --seeds 0-3 2>&1 | grep --line-buffered -v "it/s\]"
echo "=== waiting for metrics $(date '+%m-%d %H:%M')"
until [ $(ls results/T-nocfg/*.json 2>/dev/null | wc -l) -ge 120 ]; do sleep 30; done
systemctl --user stop dtax-metrics-c31 2>/dev/null
echo "=== GPU idle $(date '+%m-%d %H:%M')"
echo CHAIN_DONE
