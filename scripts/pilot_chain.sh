#!/bin/bash
# Stage-1 pilot corpus.  Pass A: seeds 0-3 for all variants (incl. the CFG sweep points).
# Pass B: seeds 4-7 for the core decomposition variants.  Idempotent (generate.py skips existing).
cd /home/xinghao/Projects/CVPR_0/dtax
PY=/home/xinghao/Projects/CVPR_0/.venv/bin/python
run() { echo "=== $1 seeds $2 $(date '+%m-%d %H:%M')"; $PY -u scripts/generate.py --variant $1 --seeds $2 2>&1 | grep -v "kv_cache\[\|it/s\]" ; }
for v in S-sf S-cf S-causvid S-bi-causvid S-bi-fastwan T-few T-full AR-diff T-nocfg T-cfg3 T-cfg7.5; do run $v 0-3; done
echo "=== PASS_A_DONE $(date '+%m-%d %H:%M')"
for v in S-sf S-cf S-causvid S-bi-causvid S-bi-fastwan T-few T-full AR-diff T-nocfg; do run $v 4-7; done
echo "=== PASS_B_DONE $(date '+%m-%d %H:%M')"
echo CHAIN_DONE
