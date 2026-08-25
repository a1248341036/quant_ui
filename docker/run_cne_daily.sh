#!/bin/bash
# ─────────────────────────────────────────────
# run_cne_daily.sh — CNE 数据湖每日流水线（Linux/Docker）
# 对标 scripts/run_cne_daily.ps1（Windows 版），功能一致：
#   1. 按 wave 依赖顺序依次执行 cne run daily --group <name>
#   2. stale 补抓（可选，由 STALE_RETRY 环境变量控制）
#   3. meta 备份 + staging 清理
#
# gate wave（core/finalize）失败 → exit 1；soft wave 失败只告警。
# 非交易日 cne 内部跳过，exit 0。
#
# 用法: ./docker/run_cne_daily.sh [--trade-date YYYY-MM-DD] [--quiet]
# ─────────────────────────────────────────────
set -u

CNE_ROOT="/app/CNEquity"
CONFIG="${CNE_ROOT}/configs/cnequity.quant_dataset.toml"
LOG_DIR="${CNE_ROOT}/data/cnequity/logs"
BACKUP_DIR="${CNE_ROOT}/data/cnequity/backups"

WAVES=(core fundamentals events capital macro_risk signals research finalize)
GATE_WAVES=(core finalize)
STALE_RETRY="${STALE_RETRY:-1}"
STALE_DELAY_SEC="${STALE_DELAY_SEC:-1800}"

TRADE_DATE=""
QUIET=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --trade-date) TRADE_DATE="$2"; shift 2 ;;
        --quiet) QUIET="--quiet"; shift ;;
        *) echo "unknown arg: $1"; exit 2 ;;
    esac
done

mkdir -p "$LOG_DIR" "$BACKUP_DIR"
STAMP=$(date +%Y%m%d)
LOG_FILE="${LOG_DIR}/daily-${STAMP}.log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

is_gate() {
    local w="$1"
    for g in "${GATE_WAVES[@]}"; do [[ "$g" == "$w" ]] && return 0; done
    return 1
}

DATE_ARGS=()
[[ -n "$TRADE_DATE" ]] && DATE_ARGS=(--trade-date "$TRADE_DATE")

log "==== CNE daily pipeline start $(date '+%Y-%m-%d %H:%M:%S') trade_date=${TRADE_DATE:-today} ===="

# reconcile: clean up crashed runs from previous invocations
log "--- reconcile ---"
cd "$CNE_ROOT"
cne clean --reconcile-runs --dry-run --config "$CONFIG" >> "$LOG_FILE" 2>&1 || true

FAILED_GATES=()
FAILED_SOFT=()

for wave in "${WAVES[@]}"; do
    log "--- wave: $wave ---"
    cne run daily --group "$wave" $QUIET "${DATE_ARGS[@]}" --config "$CONFIG" 2>&1 | tee -a "$LOG_FILE"
    rc=${PIPESTATUS[0]}
    if [ $rc -eq 0 ]; then
        log "wave $wave OK"
    else
        kind="soft"; is_gate "$wave" && kind="gate"
        log "wave $wave FAILED (exit=$rc, kind=$kind)"
        if is_gate "$wave"; then
            FAILED_GATES+=("$wave")
        else
            FAILED_SOFT+=("$wave")
        fi
    fi
done

# ── stale retry ──
if [ "$STALE_RETRY" = "1" ] && [ ${#FAILED_GATES[@]} -eq 0 ]; then
    log "--- stale probe ---"
    cne status --datasets --config "$CONFIG" >> "$LOG_FILE" 2>&1
    if [ $? -eq 0 ]; then
        log "nothing stale — no retry needed"
    else
        log "something is stale; waiting ${STALE_DELAY_SEC}s before retry"
        sleep "$STALE_DELAY_SEC"
        log "--- stale retry ---"
        cne run daily --stale-only $QUIET "${DATE_ARGS[@]}" --config "$CONFIG" 2>&1 | tee -a "$LOG_FILE"
        rc=${PIPESTATUS[0]}
        if [ $rc -eq 0 ]; then
            log "stale retry OK"
        else
            log "stale retry FAILED"
            FAILED_SOFT+=("stale-retry")
        fi
    fi
fi

# ── health check ──
log "--- health check ---"
cne audit --full --config "$CONFIG" 2>&1 | tee -a "$LOG_FILE"

# ── meta backup ──
META_DIR="${CNE_ROOT}/data/cnequity/meta"
if [ -d "$META_DIR" ]; then
    TS=$(date +%Y%m%d-%H%M%S)
    tar -czf "${BACKUP_DIR}/meta-${TS}.tar.gz" -C "$(dirname "$META_DIR")" "$(basename "$META_DIR")" 2>/dev/null
    # retention 14 days
    find "$BACKUP_DIR" -name 'meta-*.tar.gz' -mtime +14 -delete 2>/dev/null
    log "backup: ${BACKUP_DIR}/meta-${TS}.tar.gz"
fi

# ── staging cleanup ──
cne clean --config "$CONFIG" >> "$LOG_FILE" 2>&1 || true

# ── summary ──
log "---- summary ----"
log "  gate failures: ${FAILED_GATES[*]:-none}"
log "  soft failures: ${FAILED_SOFT[*]:-none}"

if [ ${#FAILED_GATES[@]} -gt 0 ]; then
    log "==== DONE — GATE FAILED: ${FAILED_GATES[*]} ===="
    exit 1
fi
log "==== DONE — gate OK ===="
exit 0
