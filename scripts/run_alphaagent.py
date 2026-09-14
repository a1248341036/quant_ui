"""Run the upstream AlphaAgent AgentScope miner with the active model config."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

# 挖掘子进程可能从任意 cwd 被 spawn（如监控脚本用绝对路径拉起），
# 必须在顶层 import alphaagent 之前先把仓库根加入 sys.path，
# 否则会 ModuleNotFoundError: No module named 'alphaagent'
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# load_codex_provider 上移到 alphaagent/core/llm_provider.py，
# 此处 re-export 保持挖掘链路 import 路径不变
from alphaagent.core.llm_provider import load_codex_provider  # noqa: F401
UPSTREAM_ENTRY = ROOT / "scripts" / "alphaagent_factor_mining.py"
DEFAULT_PANEL = "cne://"  # 从 CNE 数据湖实时构建
DEFAULT_FACTORLIB = ROOT / "artifacts" / "alphaagent" / "factorzoo" / "production_main"


def _install_fs_guard() -> None:
    """挖掘子进程只读只写守卫：禁止删除因子库/研究记忆/轨迹日志（ALPHA_FS_GUARD=0 关闭）。"""
    sys.path.insert(0, str(ROOT))
    try:
        from alphaagent.core.fs_guard import install_fs_guard

        install_fs_guard()
    except Exception as exc:  # noqa: BLE001  守卫失效不阻断挖掘，但必须喊出来
        print(f"[fs_guard] 未安装，进程无删除防护运行: {exc}", file=sys.stderr)


def main() -> int:
    _install_fs_guard()
    load_codex_provider()
    args = sys.argv[1:]
    if "--panel" not in args:
        args += ["--panel", str(DEFAULT_PANEL)]
    if "--factorlib" not in args:
        args += ["--factorlib", str(DEFAULT_FACTORLIB)]
    sys.argv = [str(UPSTREAM_ENTRY), *args]
    runpy.run_path(str(UPSTREAM_ENTRY), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
