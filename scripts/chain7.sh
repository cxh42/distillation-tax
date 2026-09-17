#!/bin/bash
# Resume after the second Xid-79 (2026-09-16 10:40).  Cheap 16-seed student cells first; the two long
# multi-step 16-seed cells (AR-diff, T-full seeds 8-15) are deferred to the end and only run if still wanted.
cd /home/xinghao/Projects/CVPR_0/dtax
PY=/home/xinghao/Projects/CVPR_0/.venv/bin/python
run() { echo "=== $1 seeds $2 $(date '+%m-%d %H:%M')"; $PY -u scripts/generate.py --variant $1 --seeds $2 ${3:+--prompts $3} 2>&1 | grep --line-buffered -v "kv_cache\[\|it/s\]" ; }
W=30
watcher_on()  { W=$((W+1)); systemd-run --user --unit=dtax-metrics-c$W --collect -p WorkingDirectory=/home/xinghao/Projects/CVPR_0/dtax \
    --setenv=https_proxy=http://127.0.0.1:7897 --setenv=http_proxy=http://127.0.0.1:7897 --setenv=PYTHONUNBUFFERED=1 \
    /bin/bash -c "$PY -u scripts/metrics_watch.py --min-free-gb 13 >> logs/metrics_watch.log 2>&1"; }
watcher_off() { systemctl --user stop dtax-metrics-c$W 2>/dev/null; sleep 5; }
watcher_on
for v in S-bi-causvid S-sf S-cf S-bi-fastwan S-bi-rcm S-rcm-tfdcm S-rcm-sfdmd; do run $v 8-15; done
run T-few 4-7
run S-bi-fastwan 4-7
run T-nocfg 0-7
echo "=== CHEAP_DONE $(date '+%m-%d %H:%M')"
if [ -f STOP_BEFORE_LONG ]; then echo "STOP_BEFORE_LONG present; skipping AR-diff/T-full seeds 8-15"; echo CHAIN_DONE; exit 0; fi
watcher_off
run AR-diff 8-15
watcher_on
run T-full 8-15
echo "=== SEEDS16_DONE $(date '+%m-%d %H:%M')"
echo CHAIN_DONE
