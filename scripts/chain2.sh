#!/bin/bash
# Replacement chain (after AR-diff OOM'd next to the metrics watcher):
#   wait for T-nocfg pass A -> stop main chain -> AR-diff seeds 0-7 with the watcher paused ->
#   restart watcher -> CFG sweep points -> pass B for the remaining variants -> AR-diff-cfg3 subset.
cd /home/xinghao/Projects/CVPR_0/dtax
PY=/home/xinghao/Projects/CVPR_0/.venv/bin/python
run() { echo "=== $1 seeds $2 $(date '+%m-%d %H:%M')"; $PY -u scripts/generate.py --variant $1 --seeds $2 ${3:+--prompts $3} 2>&1 | grep --line-buffered -v "kv_cache\[\|it/s\]" ; }
until [ $(wc -l < corpus/T-nocfg/manifest.jsonl 2>/dev/null || echo 0) -ge 120 ]; do sleep 60; done
echo "=== T-nocfg pass A complete; taking over from dtax-gen $(date '+%m-%d %H:%M')"
systemctl --user stop dtax-gen; sleep 5
systemctl --user stop dtax-metrics; sleep 5
run AR-diff 0-7
systemctl --user start dtax-metrics-b 2>/dev/null || \
  systemd-run --user --unit=dtax-metrics-b --collect -p WorkingDirectory=/home/xinghao/Projects/CVPR_0/dtax \
    --setenv=https_proxy=http://127.0.0.1:7897 --setenv=http_proxy=http://127.0.0.1:7897 --setenv=PYTHONUNBUFFERED=1 \
    /bin/bash -c "$PY -u scripts/metrics_watch.py --min-free-gb 13 >> logs/metrics_watch.log 2>&1"
for v in T-cfg3 T-cfg7.5; do run $v 0-3; done
echo "=== PASS_A_DONE $(date '+%m-%d %H:%M')"
for v in S-sf S-cf S-causvid S-bi-causvid S-bi-fastwan T-few T-full T-nocfg; do run $v 4-7; done
echo "=== PASS_B_DONE $(date '+%m-%d %H:%M')"
run AR-diff-cfg3 0-1 0-29
echo CHAIN_DONE
