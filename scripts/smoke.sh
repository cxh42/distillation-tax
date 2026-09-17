#!/bin/bash
# Stage-0 acceptance: one video (prompt 4 = "a dog running happily", seed 0) per configuration.
cd /home/xinghao/Projects/CVPR_0/dtax
PY=/home/xinghao/Projects/CVPR_0/.venv/bin/python
for v in S-sf S-cf S-causvid S-bi-causvid S-bi-fastwan T-few AR-diff T-nocfg T-full; do
  echo "=== $v $(date +%H:%M:%S)"
  $PY -u scripts/generate.py --variant $v --seeds 0 --prompts 4 --outdir corpus/smoke/$v 2>&1 | grep -v "kv_cache\[" | tail -4
done
echo SMOKE_DONE
