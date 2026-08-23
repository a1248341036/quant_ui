# ─────────────────────────────────────────────
# quant_ui Dockerfile — 单容器全栈（可移植）
# python:3.12-slim (Debian) + supervisor + supercronic
#
# 镜像内容：
#   - quant_ui 后端（FastAPI :17891，托管 Vue 前端 dist）
#   - data_status 数据看板（:8001）
#   - CNEquity 数据湖（cne serve :8787，读取同一数据 volume）
#   - 定时任务（supercronic，Asia/Shanghai 时区）
#   - alphaagent 因子挖掘（可选，依赖 .env 中 LLM 配置）
#
# 数据/结果/日志全部走 volume（/app/data、/app/labs、/app/results、
# /app/artifacts、/app/logs），镜像内不含任何业务数据。
# ─────────────────────────────────────────────

# ---- Stage 1: Python 依赖层（缓存友好）----
FROM python:3.12-slim AS deps

# 系统编译依赖（lightgbm/scipy/psycopg/CNEquity build 需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装 quant_ui 主依赖（利用 Docker 层缓存）
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# alphaagent 依赖（agentscope/openai，LLM 因子挖掘用）
# 注意：backend/routers/alphaagent.py 在 main.py 顶层 import alphaagent.*，
# 且 alphaagent.factor.mining 强依赖 agentscope —— 安装失败会导致 API 无法启动，
# 因此这里不允许静默失败。
COPY requirements-alphaagent.txt .
RUN pip install --no-cache-dir -r requirements-alphaagent.txt

# CNEquity 数据湖包（源码安装，保证 CLI/包与仓库同版本）
# .dockerignore 已排除 CNEquity/data|.venv|.git|tests|docs 等，仅源码进入上下文
COPY CNEquity/ /app/CNEquity-src/
RUN pip install --no-cache-dir /app/CNEquity-src \
    && rm -rf /app/CNEquity-src

# ---- Stage 2: 前端构建产物确认（已有 dist/，无需 node）----
# quant_ui 前端 dist/ 与 CNEquity dashboard 静态文件已随源码提交，
# 镜像直接拷贝，不引入 node 构建阶段。

# ---- Stage 3: 运行层 ----
FROM python:3.12-slim AS runtime

# supervisord + supercronic + curl(健康检查) + tzdata(时区) + procps(ps)
RUN apt-get update && apt-get install -y --no-install-recommends \
    supervisor \
    curl \
    tzdata \
    procps \
    && rm -rf /var/lib/apt/lists/*

# 安装 supercronic（多架构：amd64/arm64 各自 SHA1 校验）
# 官方发布页 https://github.com/aptible/supercronic/releases
# 校验值来自 v0.2.30 release 页的安装说明（官方 sha1sum）
# TARGETARCH 仅在 buildx/buildkit 下注入，普通 docker build 时用 uname 兜底
ARG SUPERCRONIC_VERSION=v0.2.30
ARG TARGETARCH
ARG SUPERCRONIC_SHA1_AMD64=9f27ad28c5c57cd133325b2a66bba69ba2235799
ARG SUPERCRONIC_SHA1_ARM64=d5e02aa760b3d434bc7b991777aa89ef4a503e49
RUN set -eux; \
    ARCH="${TARGETARCH:-$(uname -m)}"; \
    case "$ARCH" in \
        amd64|x86_64) SUPERCRONIC="supercronic-linux-amd64"; EXPECT="${SUPERCRONIC_SHA1_AMD64}";; \
        arm64|aarch64) SUPERCRONIC="supercronic-linux-arm64"; EXPECT="${SUPERCRONIC_SHA1_ARM64}";; \
        *) echo "Unsupported arch: $ARCH" >&2; exit 1;; \
    esac; \
    curl -fsSLo /usr/local/bin/supercronic \
        "https://github.com/aptible/supercronic/releases/download/${SUPERCRONIC_VERSION}/${SUPERCRONIC}"; \
    echo "${EXPECT}  /usr/local/bin/supercronic" | sha1sum -c -; \
    chmod +x /usr/local/bin/supercronic

# 时区：A 股所有定时任务按 Asia/Shanghai 运行
ENV TZ=Asia/Shanghai
RUN ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 从 deps 层拷贝已安装的 Python 包
COPY --from=deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=deps /usr/local/bin /usr/local/bin

WORKDIR /app

# ── 拷贝项目代码（数据/结果/产物通过 volume 挂载，不进镜像）──
COPY backend/    backend/
COPY core/       core/
COPY alphaagent/ alphaagent/
COPY data_status/ data_status/
COPY strategies/ strategies/
COPY scripts/    scripts/
COPY static/dist static/dist
COPY db/         db/

# CNEquity：仅保留 configs/（cne serve 与 core/cne_reader.py 读取）
# cnequity 包本体（含 serve 静态资源）已由 deps 层 pip 安装到 site-packages。
# 说明：docker-compose.yml 额外以只读 bind 挂载整个 ./CNEquity:/app/CNEquity:ro，
# 覆盖此处拷贝的 configs/，保证运行时配置与宿主机源码一致且只读。
COPY CNEquity/configs/ CNEquity/configs/

# docker 配置文件
COPY docker/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY docker/crontab          /app/docker/crontab
COPY docker/entrypoint.sh    /app/docker/entrypoint.sh
RUN chmod +x /app/docker/entrypoint.sh

# 环境变量默认值（运行时通过 .env 或 -e 覆盖）
ENV QUANT_UI_DATA_DIR=/app/data \
    QUANT_DATA_SOURCE=pg_parquet \
    QUANT_USE_CNE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CNE_TOKEN=dev-only-change-me

# 数据/产物目录（volume 挂载点）
RUN mkdir -p /app/data /app/labs /app/results /app/artifacts /app/logs

# 运行时状态目录（data_status 的 state.json/tasks.json/logs 落在此，volume 持久化）
RUN mkdir -p /app/data/data_status

VOLUME ["/app/data", "/app/labs", "/app/results", "/app/artifacts", "/app/logs"]

EXPOSE 17891 8001 8787

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -sf http://localhost:17891/api/health || exit 1

ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf", "-n"]