#!/usr/bin/env bash
# 受限回测包装：把回测命令放进独立 systemd transient unit，并套上 cgroup 内存上限，
# 防止回测单进程把整机（3.6G RAM）打满触发全局 OOM。
#
# 用法：
#   scripts/run_backtest.sh python backend/lab_runner.py --config ... --out ...
#   scripts/run_backtest.sh python scripts/parameter_sweep.py ...
#   MEMORY_HIGH=2G MEMORY_MAX=2.5G scripts/run_backtest.sh python ...
#
# 默认 MemoryHigh=2G（软限制，接近时开始回收/换页），MemoryMax=2.5G（硬限制，
# 达到后先换页，仍无法满足则被 systemd OOM 杀掉）。需要免密 sudo。
set -uo pipefail

MEMORY_HIGH="${MEMORY_HIGH:-2G}"
MEMORY_MAX="${MEMORY_MAX:-2.5G}"

usage() {
    echo "用法: $0 <命令...>" >&2
    echo "环境变量: MEMORY_HIGH（软限制，默认 2G）、MEMORY_MAX（硬限制，默认 2.5G）" >&2
    exit 2
}

[[ $# -eq 0 ]] && usage

if ! command -v sudo >/dev/null 2>&1 || ! sudo -n true 2>/dev/null; then
    echo "需要免密 sudo 权限（systemd-run 需要 root 创建带内存上限的 unit）" >&2
    exit 2
fi

# 把调用者环境变量显式透传给 unit（systemd-run 默认不继承完整环境）
setenv_args=()
while IFS= read -r -d '' line; do
    case "$line" in
        BASH_FUNC_*|SHELLOPTS=*|PWD=*) continue ;;
    esac
    setenv_args+=(--setenv="$line")
done < <(env -0)

exec sudo --preserve-env systemd-run --pipe --wait --collect --quiet \
    --property="User=$(id -un)" \
    --property="MemoryHigh=$MEMORY_HIGH" \
    --property="MemoryMax=$MEMORY_MAX" \
    --working-directory="$(pwd)" \
    "${setenv_args[@]}" \
    -- "$@"
