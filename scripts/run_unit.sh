#!/bin/bash
# run_unit.sh <unit-name> <logfile> <python script + args...>   -- detached, survives editor crashes
UNIT=$1; LOG=$2; shift 2
systemd-run --user --unit=$UNIT --collect -p WorkingDirectory=/home/xinghao/Projects/CVPR_0/dtax \
  --setenv=https_proxy=http://127.0.0.1:7897 --setenv=http_proxy=http://127.0.0.1:7897 \
  --setenv=PYTHONUNBUFFERED=1 \
  /bin/bash -c "/home/xinghao/Projects/CVPR_0/.venv/bin/python -u $* > $LOG 2>&1"
