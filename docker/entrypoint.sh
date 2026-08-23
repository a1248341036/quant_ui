#!/bin/bash
set -e

# ─────────────────────────────────────────────
# entrypoint.sh — 容器启动入口
# 1. 加载 .env 环境变量（存在才 source，缺失不阻塞）
# 2. 创建必要的运行时目录（volume 挂载后可能是空目录）
# 3. 传递控制权给 supervisord
# ─────────────────────────────────────────────

echo "[entrypoint] quant_ui container starting..."

# 加载 /app/.env（如果存在；compose 用 env_file 注入时此文件可不存在）
if [ -f /app/.env ]; then
    echo "[entrypoint] loading /app/.env"
    set -a
    # 容错：.env 中含特殊字符时避免整段失败
    source /app/.env || echo "[entrypoint] warning: /app/.env has lines that failed to parse"
    set +a
fi

# ── 数据目录结构（volume 挂载后可能是空目录，按 core/store.py 约定补齐）──
mkdir -p /app/data/stock
mkdir -p /app/data/etf
mkdir -p /app/data/fund
mkdir -p /app/data/db
mkdir -p /app/data/pg_parquet
mkdir -p /app/data/quant_dataset
mkdir -p /app/data/qweave
mkdir -p /app/data/backup
mkdir -p /app/data/alphaagent

# CNEquity 数据湖根（compose 与 /app/data 共用同一 volume）
mkdir -p /app/CNEquity/data/quant_dataset/_cnequity/{meta,staging,curated,derived}

# data_status 运行时状态目录（compose 通过 QUANT_DATA_STATUS_DIR 指向此处，
# state.json / tasks.json / logs 随数据 volume 持久化，容器重建不丢）
mkdir -p /app/data/data_status/logs
mkdir -p /app/labs
mkdir -p /app/results
mkdir -p /app/artifacts/alphaagent
mkdir -p /app/logs/factor_mining/ui

# supervisor 日志目录
mkdir -p /var/log/supervisor

# 首次启动提示
if [ ! -f /app/data/stock/panel.parquet ] && [ -z "$(ls -A /app/data/stock 2>/dev/null)" ]; then
    echo "[entrypoint] WARNING: no existing data found under /app/data."
    echo "[entrypoint] First-time setup: copy your existing data/ into the volume,"
    echo "[entrypoint] or run: docker exec quant_ui python scripts/refresh_data.py"
fi

echo "[entrypoint] handing off to supervisord..."
exec "$@"