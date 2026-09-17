#!/bin/bash
# Re-prioritised after the 2x2 interaction became the headline (2026-09-14 16:00):
#   0. let chain3 finish the S-causvid / S-bi-causvid seeds 4-7 cells, then take over
#   1. CFG-matched pair at CFG 3 (DMD's own real-score guidance is 3-3.5): AR-diff-cfg3 + T-cfg3, seeds 0-3
#   2. T-full seeds 4-7  (8-seed reference cell)
#   3. CFG-matched pair at 7.5: AR-diff-cfg7.5 + T-cfg7.5, seeds 0-3        -> sensitivity band
#   4. seeds 8-15 for the four 2x2 cells (students first, cheap), then T-few/S-bi-fastwan/T-nocfg remainders
# AR-diff runs need ~20 GB -> the metrics watcher is stopped around them and restarted after.
cd /home/xinghao/Projects/CVPR_0/dtax
PY=/home/xinghao/Projects/CVPR_0/.venv/bin/python
run() { echo "=== $1 seeds $2 $(date '+%m-%d %H:%M')"; $PY -u scripts/generate.py --variant $1 --seeds $2 ${3:+--prompts $3} 2>&1 | grep --line-buffered -v "kv_cache\[\|it/s\]" ; }
W=0
watcher_on()  { W=$((W+1)); systemd-run --user --unit=dtax-metrics-c$W --collect -p WorkingDirectory=/home/xinghao/Projects/CVPR_0/dtax \
    --setenv=https_proxy=http://127.0.0.1:7897 --setenv=http_proxy=http://127.0.0.1:7897 --setenv=PYTHONUNBUFFERED=1 \
    /bin/bash -c "$PY -u scripts/metrics_watch.py --min-free-gb 13 >> logs/metrics_watch.log 2>&1"; }
watcher_off() { systemctl --user stop dtax-metrics-c$W 2>/dev/null; sleep 5; }
until [ $(wc -l < corpus/S-bi-causvid/manifest.jsonl 2>/dev/null || echo 0) -ge 240 ]; do sleep 60; done
echo "=== taking over from chain3 $(date '+%m-%d %H:%M')"
systemctl --user stop dtax-chain3; sleep 3; systemctl --user stop dtax-metrics-b; sleep 5
run AR-diff-cfg3 0-3
watcher_on
run T-cfg3 0-3
run T-full 4-7
watcher_off
run AR-diff-cfg7.5 0-3
watcher_on
run T-cfg7.5 0-3
echo "=== CFG_BAND_DONE $(date '+%m-%d %H:%M')"
for v in S-causvid S-bi-causvid S-sf S-cf S-bi-fastwan; do run $v 8-15; done
watcher_off
run AR-diff 8-15
watcher_on
run T-full 8-15
echo "=== SEEDS16_DONE $(date '+%m-%d %H:%M')"
run T-few 4-7
run S-bi-fastwan 4-7
run T-nocfg 0-7
echo CHAIN_DONE
