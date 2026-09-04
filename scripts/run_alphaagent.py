"""Run the upstream AlphaAgent AgentScope miner with the active model config."""

from __future__ import annotations

import os
import runpy
import sys
import tomllib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM_ENTRY = ROOT / "scripts" / "alphaagent_factor_mining.py"
DEFAULT_PANEL = "cne://"  # 从 CNE 数据湖实时构建
DEFAULT_FACTORLIB = ROOT / "artifacts" / "alphaagent" / "factorzoo" / "production_main"


def load_codex_provider() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
    except Exception:
        pass
    if (os.getenv("ALPHA_LLM_PROVIDER") or "").lower() != "codex":
        return
    path = Path(os.getenv("CODEX_CONFIG", Path.home() / ".codex" / "config.toml"))
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    provider_name = config["model_provider"]
    provider = config["model_providers"][provider_name]
    token = provider.get("experimental_bearer_token") or config.get("experimental_bearer_token")
    if not token:
        raise RuntimeError(
            f"Codex provider '{provider_name}' has no bearer token in provider or top-level config"
        )
    os.environ["OPENAI_API_KEY"] = str(token)
    os.environ["OPENAI_API_BASE"] = str(provider["base_url"]).rstrip("/")
    os.environ["MODEL"] = str(config["model"])


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
