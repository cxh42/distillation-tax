#!/bin/bash
# Priority order set by the user (2026-09-14 02:10):
#   1. AR-diff seeds 0-7 (watcher paused: it needs ~20 GB)      -> decides the causalisation story
#   2. pass B (seeds 4-7) for the 5 students, then T-few, T-full  -> student-internal rank reversals
#   3. CFG sweep: T-nocfg 0-7, T-cfg3 0-3, T-cfg7.5 0-3           -> A2 narrative
#   4. AR-diff-cfg3 subset
cd /home/xinghao/Projects/CVPR_0/dtax
PY=/home/xinghao/Projects/CVPR_0/.venv/bin/python
run() { echo "=== $1 seeds $2 $(date '+%m-%d %H:%M')"; $PY -u scripts/generate.py --variant $1 --seeds $2 ${3:+--prompts $3} 2>&1 | grep --line-buffered -v "kv_cache\[\|it/s\]" ; }
start_watcher() {
  systemd-run --user --unit=dtax-metrics-$1 --collect -p WorkingDirectory=/home/xinghao/Projects/CVPR_0/dtax \
    --setenv=https_proxy=http://127.0.0.1:7897 --setenv=http_proxy=http://127.0.0.1:7897 --setenv=PYTHONUNBUFFERED=1 \
    /bin/bash -c "$PY -u scripts/metrics_watch.py --min-free-gb 13 >> logs/metrics_watch.log 2>&1"
}
run AR-diff 0-7
start_watcher b
for v in S-sf S-cf S-causvid S-bi-causvid S-bi-fastwan T-few T-full; do run $v 4-7; done
echo "=== PASS_B_DONE $(date '+%m-%d %H:%M')"
run T-nocfg 0-7
for v in T-cfg3 T-cfg7.5; do run $v 0-3; done
echo "=== CFG_SWEEP_DONE $(date '+%m-%d %H:%M')"
run AR-diff-cfg3 0-1 0-29
echo CHAIN_DONE
