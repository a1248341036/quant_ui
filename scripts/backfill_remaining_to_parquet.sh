#!/usr/bin/env bash
# 补齐剩余 Tushare 数据（fina 零星 / events / surv / index_weight / report_rc）
# 并导出全部 PG 表到 Parquet。由 systemd-run 后台运行，断 SSH 不影响。
set -u

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

LOG=./data/backfill_remaining_parquet.log
PY="${QUANT_UI_PYTHON:-$HOME/stock-analyzer/local_venv/bin/python}"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG"; }

log "=== 开始：补齐剩余 Tushare 数据并导出 Parquet ==="

# 1. 生成缺失代码清单（fina/events/surv）
"$PY" - "$PROJECT_ROOT" >> "$LOG" 2>&1 <<'PYEOF'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
from core.pg import get_conn

groups = {
    "fina": ["fina_indicator", "income", "balancesheet", "cashflow"],
    "events": ["dividend", "share_float", "namechange"],
    "surv": ["stk_surv", "forecast", "express"],
}
with get_conn() as conn, conn.cursor() as cur:
    cur.execute("SELECT ts_code FROM stock_basic ORDER BY ts_code")
    all_codes = {r[0] for r in cur.fetchall()}
    print(f"stock_basic total: {len(all_codes)}", flush=True)
    for group, tables in groups.items():
        missing = set(all_codes)
        for t in tables:
            cur.execute(f"SELECT DISTINCT ts_code FROM {t}")
            missing -= {r[0] for r in cur.fetchall()}
        path = Path(f"/tmp/{group}_missing.txt")
        path.write_text("\n".join(sorted(missing)) + ("\n" if missing else ""), encoding="utf-8")
        print(f"missing {group}: {len(missing)} -> {path}", flush=True)
PYEOF

run_step() {
    local name="$1"; shift
    log "=== $name 开始 ==="
    "$PY" scripts/sync_postgres.py "$@" >> "$LOG" 2>&1
    local rc=$?
    log "=== $name 结束 rc=$rc ==="
    return "$rc"
}

run_step fina --fina --codes-file /tmp/fina_missing.txt --workers 2 --sleep 0.3
run_step events --events --codes-file /tmp/events_missing.txt --workers 2 --sleep 0.3
run_step surv --surv --codes-file /tmp/surv_missing.txt --workers 2 --sleep 0.3
run_step index-weight --index-weight --sleep 0.3
run_step report-rc --report-rc --sleep 0.3

log "=== 开始导出 Parquet ==="
"$PY" scripts/export_pg_to_parquet.py --tables all >> "$LOG" 2>&1
log "=== 导出结束 rc=$? ==="
log "=== 全部完成 ==="
