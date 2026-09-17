#!/bin/bash
# Takes over after AR-diff-cfg3 (chain4) finishes.  Priority: CFG-3 pair -> second within-lab 2x2 (rCM) ->
# T-full 8 seeds -> CFG-7.5 pair -> 16 seeds for the primary 2x2 -> remainders.
cd /home/xinghao/Projects/CVPR_0/dtax
PY=/home/xinghao/Projects/CVPR_0/.venv/bin/python
run() { echo "=== $1 seeds $2 $(date '+%m-%d %H:%M')"; $PY -u scripts/generate.py --variant $1 --seeds $2 ${3:+--prompts $3} 2>&1 | grep --line-buffered -v "kv_cache\[\|it/s\]" ; }
W=10
watcher_on()  { W=$((W+1)); systemd-run --user --unit=dtax-metrics-c$W --collect -p WorkingDirectory=/home/xinghao/Projects/CVPR_0/dtax \
    --setenv=https_proxy=http://127.0.0.1:7897 --setenv=http_proxy=http://127.0.0.1:7897 --setenv=PYTHONUNBUFFERED=1 \
    /bin/bash -c "$PY -u scripts/metrics_watch.py --min-free-gb 13 >> logs/metrics_watch.log 2>&1"; }
watcher_off() { systemctl --user stop dtax-metrics-c$W 2>/dev/null; sleep 5; }
until [ $(wc -l < corpus/AR-diff-cfg3/manifest.jsonl 2>/dev/null || echo 0) -ge 120 ]; do sleep 60; done
echo "=== taking over from chain4 $(date '+%m-%d %H:%M')"
systemctl --user stop dtax-chain4; sleep 3; systemctl --user stop dtax-metrics-c1 2>/dev/null; sleep 5
# smoke the two causal rCM variants first (1 video each) so a port bug surfaces immediately
run S-rcm-tfdcm 0 4
run S-rcm-sfdmd 0 4
watcher_on
run T-cfg3 0-3
for v in S-bi-rcm S-rcm-tfdcm S-rcm-sfdmd; do run $v 0-7; done
watcher_off
run AR-diff-rcm 0-7
watcher_on
echo "=== RCM_2x2_DONE $(date '+%m-%d %H:%M')"
run T-full 4-7
watcher_off
run AR-diff-cfg7.5 0-3
watcher_on
run T-cfg7.5 0-3
echo "=== CFG_BAND_DONE $(date '+%m-%d %H:%M')"
for v in S-causvid S-bi-causvid S-sf S-cf S-bi-fastwan S-bi-rcm S-rcm-tfdcm S-rcm-sfdmd; do run $v 8-15; done
watcher_off
run AR-diff 8-15
watcher_on
run T-full 8-15
echo "=== SEEDS16_DONE $(date '+%m-%d %H:%M')"
run T-few 4-7
run S-bi-fastwan 4-7
run T-nocfg 0-7
echo CHAIN_DONE
