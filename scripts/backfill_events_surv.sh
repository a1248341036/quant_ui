#!/usr/bin/env bash
# 等待复权因子回填完成后，串行补拉 events + surv（低并发）。
# 由 systemd-run 以后台 unit 方式运行，断 SSH 不影响。
set -u

PY=/home/ubuntu/stock-analyzer/local_venv/bin/python
cd /home/ubuntu/quant/quant_ui || exit 1

echo "[$(date +%H:%M:%S)] 等待 backfill-adj 完成"
while systemctl is-active --quiet backfill-adj; do
    sleep 30
done

echo "[$(date +%H:%M:%S)] backfill-adj 完成，开始 --events（分红/解禁/改名）"
"$PY" scripts/sync_postgres.py --events --workers 2 --sleep 0.3
echo "[$(date +%H:%M:%S)] --events rc=$?"

echo "[$(date +%H:%M:%S)] 开始 --surv（业绩预告/快报/调研）"
"$PY" scripts/sync_postgres.py --surv --workers 2 --sleep 0.3
echo "[$(date +%H:%M:%S)] --surv rc=$?"

echo "[$(date +%H:%M:%S)] events/surv 全部完成"
