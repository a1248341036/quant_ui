#!/usr/bin/env bash
# 全量补数编排：Tushare 日线（daily/daily_basic/adj_factor）串行拉全市场历史；
# 完成后排队 fina/surv/index_weight。全部 nice + ionice + 内存上限。
# wait_unit 会检查退出码，失败自动重试（防代理 SSL 抖动）。
set -u
cd /home/ubuntu/quant/quant_ui || exit 1

LOG=./data/full_backfill.log
PY=/home/ubuntu/stock-analyzer/local_venv/bin/python
MAX_RETRY=3

log() { echo "[$(date '+%H:%M:%S')] $*" >> "$LOG"; }

run_unit() {
    local unit="$1"; shift
    sudo systemd-run --unit="$unit" \
        --working-directory=$(pwd) \
        --property=User=ubuntu \
        --property=Nice=10 --property=IOSchedulingClass=idle \
        --property=MemoryHigh=900M --property=MemoryMax=1.2G \
        --property=StandardOutput=append:$(pwd)/data/${unit}.log \
        --property=StandardError=append:$(pwd)/data/${unit}.log \
        "$PY" scripts/sync_postgres.py "$@"
}

wait_unit() {
    local unit="$1"
    while systemctl is-active --quiet "$unit.service"; do
        log "等待 $unit 完成..."
        sleep 60
    done
    local result
    result=$(systemctl show "$unit.service" -p Result --value 2>/dev/null)
    log "$unit 结束 result=$result"
    systemctl reset-failed "$unit.service" 2>/dev/null || true
    [[ "$result" == "success" ]]
}

run_with_retry() {
    local unit="$1"; shift
    local i
    for i in $(seq 1 "$MAX_RETRY"); do
        run_unit "$unit" "$@"
        log "已启动 $unit（第 $i 次）"
        if wait_unit "$unit"; then
            return 0
        fi
        log "$unit 第 $i 次失败，10s 后重试..."
        sleep 10
    done
    log "$unit 重试 $MAX_RETRY 次仍失败，中止编排"
    exit 1
}

log "=== 开始全量补数编排 ==="

# Tushare 日线全量：从断点续跑，按交易日拉 daily + daily_basic + adj_factor
run_with_retry tushare-daily --daily-since 20150624 --daily-workers 4 --sleep 0.2
log "tushare-daily 已完成"

run_with_retry fina-sync --fina --workers 3 --sleep 0.3
log "fina-sync 已完成"

run_with_retry surv-sync --surv --workers 3 --sleep 0.3
log "surv-sync 已完成"

run_with_retry index-weight --index-weight --sleep 0.3
log "index-weight 已完成"

log "=== 全部补数任务完成 ==="
