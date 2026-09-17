#!/bin/bash
# Round 3: (B) chunk-size axis, (C) loss axis, extra causal cells.  Few-step cells at 8 seeds first (cheap),
# then the three multi-step causal-undistilled cells at 4 seeds (watcher off: ~20 GB each).
cd /home/xinghao/Projects/CVPR_0/dtax
PY=/home/xinghao/Projects/CVPR_0/.venv/bin/python
run() { echo "=== $1 seeds $2 $(date '+%m-%d %H:%M')"; $PY -u scripts/generate.py --variant $1 --seeds $2 ${3:+--prompts $3} 2>&1 | grep --line-buffered -v "kv_cache\[\|it/s\]" ; }
W=40
watcher_on()  { W=$((W+1)); systemd-run --user --unit=dtax-metrics-c$W --collect -p WorkingDirectory=/home/xinghao/Projects/CVPR_0/dtax \
    --setenv=https_proxy=http://127.0.0.1:7897 --setenv=http_proxy=http://127.0.0.1:7897 --setenv=PYTHONUNBUFFERED=1 \
    /bin/bash -c "$PY -u scripts/metrics_watch.py --min-free-gb 13 >> logs/metrics_watch.log 2>&1"; }
watcher_off() { systemctl --user stop dtax-metrics-c$W 2>/dev/null; sleep 5; }
until grep -q ALL_DONE logs/dl_round3.log; do sleep 30; done
# smoke: one video per new few-step variant so a port bug surfaces immediately
for v in S-cf-fw S-rcm-sfdmd-fw S-rcm-tfdcm-fw S-sf-sid S-sf-gan S-cf-cd S-cf-ode S-causvid-ode S-causvid-w4 S-rcm-tfscm S-rcm-sfdmd-scm S-sf-sid2; do run $v 0 4; done
echo "=== SMOKE_DONE $(date '+%m-%d %H:%M')"
watcher_on
for v in S-cf-fw S-rcm-sfdmd-fw S-rcm-tfdcm-fw S-sf-sid S-sf-gan S-cf-cd S-cf-ode S-causvid-ode S-causvid-w4 S-rcm-tfscm S-rcm-sfdmd-scm S-sf-sid2; do run $v 0-7; done
echo "=== FEWSTEP_DONE $(date '+%m-%d %H:%M')"
watcher_off
for v in AR-diff-rcm-df AR-diff-fw AR-diff-rcm-fw; do run $v 0-3; done
watcher_on
echo "=== MULTISTEP_DONE $(date '+%m-%d %H:%M')"
until [ $(ls results/AR-diff-rcm-fw/*.json 2>/dev/null | wc -l) -ge 120 ]; do sleep 60; done
watcher_off
echo CHAIN_DONE
