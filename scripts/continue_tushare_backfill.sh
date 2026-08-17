#!/usr/bin/env bash
# 续跑未完成的 Tushare 补拉：fina 缺失 -> surv 缺失 -> events 缺失 -> index_weight -> report_rc
# 全部幂等：fina 走 upsert，surv/events 按股票 DELETE+INSERT。
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT" || exit 1

LOG=./data/backfill_continue.log
PY="${QUANT_UI_PYTHON:-$HOME/stock-analyzer/local_venv/bin/python}"

log() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }

run_one() {
    local name="$1"; shift
    log "=== $name 开始 ==="
    "$PY" scripts/sync_postgres.py "$@" >> "$LOG" 2>&1
    log "=== $name 结束 rc=$? ==="
}

log "=== 续跑 Tushare 补拉开始 ==="
log "fina missing: $(wc -l < /tmp/fina_missing.txt 2>/dev/null || echo 0) 只"
run_one fina --fina --codes-file /tmp/fina_missing.txt --workers 2 --sleep 0.3
log "surv missing: $(wc -l < /tmp/surv_missing.txt 2>/dev/null || echo 0) 只"
run_one surv --surv --codes-file /tmp/surv_missing.txt --workers 3 --sleep 0.3
log "events missing: $(wc -l < /tmp/events_missing.txt 2>/dev/null || echo 0) 只"
run_one events --events --codes-file /tmp/events_missing.txt --workers 2 --sleep 0.3
run_one index-weight --index-weight --sleep 0.3
run_one report-rc --report-rc --sleep 0.3
log "=== 续跑 Tushare 补拉全部结束 ==="
