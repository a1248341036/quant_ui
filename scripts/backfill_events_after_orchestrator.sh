#!/usr/bin/env bash
# 等全量因子回填覆盖完整、且 run_full_backfill.sh 编排结束后，
# 再串行补拉 events（分红/解禁/改名）。surv/fina/index-weight 由编排负责。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

PY="${QUANT_UI_PYTHON:-$HOME/stock-analyzer/local_venv/bin/python}"

factor_done() {
    "$PY" - <<PYEOF >/tmp/factor_cov.txt 2>/dev/null
import sys
sys.path.insert(0, "$PROJECT_ROOT")
from core.pg import get_conn
with get_conn() as c, c.cursor() as cur:
    cur.execute(
        "SELECT count(*) FILTER (WHERE adj_factor IS NOT NULL AND adj_factor<>0), count(*) FROM stock_daily"
    )
    done, total = cur.fetchone()
print(done, total)
sys.exit(0 if total > 0 and done >= total else 1)
PYEOF
}

echo "[$(date +%H:%M:%S)] 等待因子回填覆盖完整..."
while ! factor_done; do
    read -r done total < /tmp/factor_cov.txt
    echo "[$(date +%H:%M:%S)] 因子覆盖 $done/$total，继续等待"
    sleep 120
done
read -r done total < /tmp/factor_cov.txt
echo "[$(date +%H:%M:%S)] 因子覆盖完整: $done/$total"

echo "[$(date +%H:%M:%S)] 等待 run_full_backfill.sh 编排结束..."
while systemctl is-active --quiet backfill-orchestrator; do
    sleep 60
done
echo "[$(date +%H:%M:%S)] 编排结束，开始 --events（分红/解禁/改名）"
"$PY" scripts/sync_postgres.py --events --workers 2 --sleep 0.3
echo "[$(date +%H:%M:%S)] --events rc=$?"

echo "[$(date +%H:%M:%S)] events 补拉完成"
