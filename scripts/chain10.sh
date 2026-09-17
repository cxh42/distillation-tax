#!/bin/bash
# Everything that is still runnable on this machine, after chain9:
#   1. 16-seed completion of the primary 2x2 multi-step cells (AR-diff, T-full seeds 8-15)
#   2. 16-seed completion of the remaining students
#   3. remaining teacher-family cells (T-few 4-7, S-bi-fastwan 4-7, T-nocfg 4-7, T-cfg3/7.5 4-7, AR-diff-cfg3/7.5 4-7)
cd /home/xinghao/Projects/CVPR_0/dtax
PY=/home/xinghao/Projects/CVPR_0/.venv/bin/python
run() { echo "=== $1 seeds $2 $(date '+%m-%d %H:%M')"; $PY -u scripts/generate.py --variant $1 --seeds $2 ${3:+--prompts $3} 2>&1 | grep --line-buffered -v "kv_cache\[\|it/s\]" ; }
W=50
watcher_on()  { W=$((W+1)); systemd-run --user --unit=dtax-metrics-c$W --collect -p WorkingDirectory=/home/xinghao/Projects/CVPR_0/dtax \
    --setenv=https_proxy=http://127.0.0.1:7897 --setenv=http_proxy=http://127.0.0.1:7897 --setenv=PYTHONUNBUFFERED=1 \
    /bin/bash -c "$PY -u scripts/metrics_watch.py --min-free-gb 13 >> logs/metrics_watch.log 2>&1"; }
watcher_off() { systemctl --user stop dtax-metrics-c$W 2>/dev/null; sleep 5; }
until grep -q CHAIN_DONE logs/chain9.log; do sleep 60; done
echo "=== chain10 start $(date '+%m-%d %H:%M')"
watcher_on
for v in S-bi-causvid S-sf S-cf S-bi-fastwan S-bi-rcm S-rcm-tfdcm S-rcm-sfdmd; do run $v 8-15; done
run S-bi-fastwan 4-7
run T-few 4-7
run T-full 8-15
run T-nocfg 4-7
run T-cfg3 4-7
run T-cfg7.5 4-7
watcher_off
run AR-diff 8-15
run AR-diff-cfg3 4-7
run AR-diff-cfg7.5 4-7
watcher_on
echo "=== ALL_GEN_DONE $(date '+%m-%d %H:%M')"
until [ $(ls results/AR-diff-cfg7.5/*.json 2>/dev/null | wc -l) -ge 240 ]; do sleep 60; done
watcher_off
echo CHAIN_DONE
