#!/usr/bin/env python3
"""对比实验 launcher：用与 quant_ui 相同的 LLM（~/.codex/config.toml）启动原版 AlphaAgent 挖掘。

- 注入 OPENAI_API_KEY / OPENAI_API_BASE / MODEL（本地 codex 代理，与 quant_ui 同模型同代理）
- 窗口/标签对齐 quant_ui technical 口径：train 2020~2022 / val 2023~2024 / label_1d_close_to_close
- 盲测段 2025-01-01 起两侧均不可见（面板只导出 2020~2024）

用法:
  .venv/Scripts/python.exe scripts/run_original_alphaagent.py [--max-turns 16] [--tag compare1]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path

QUANT_UI_ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_ROOT = Path(r"D:\Quant\AlphaAgent")


def load_codex_env() -> None:
    path = Path(os.getenv("CODEX_CONFIG", Path.home() / ".codex" / "config.toml"))
    with path.open("rb") as handle:
        config = tomllib.load(handle)
    provider = config["model_providers"][config["model_provider"]]
    token = provider.get("experimental_bearer_token") or config.get("experimental_bearer_token")
    if not token:
        raise RuntimeError(f"codex provider '{config['model_provider']}' 无 bearer token")
    os.environ["OPENAI_API_KEY"] = str(token)
    os.environ["OPENAI_API_BASE"] = str(provider["base_url"]).rstrip("/")
    os.environ["MODEL"] = str(config["model"])


def main() -> int:
    load_codex_env()
    args = sys.argv[1:]
    tag = "run"
    if "--tag" in args:
        i = args.index("--tag")
        tag = args[i + 1]
        del args[i : i + 2]
    out_dir = QUANT_UI_ROOT / "artifacts" / "alphaagent" / "comparison" / f"original_{tag}"

    cli = [
        ORIGINAL_ROOT / ".venv" / "Scripts" / "python.exe",
        str(ORIGINAL_ROOT / "scripts" / "factor_mining_agentscope.py"),
        "--train-start", "2020-01-01", "--train-end", "2022-12-31",
        "--val-start", "2023-01-01", "--val-end", "2024-12-31",
        "--label-col", "label_1d_close_to_close",
        "--log-dir", str(out_dir / "logs"),
    ] + args

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[launcher] model={os.environ['MODEL']} base={os.environ['OPENAI_API_BASE']}")
    print(f"[launcher] out_dir={out_dir}")
    return subprocess.call([str(c) for c in cli], cwd=str(ORIGINAL_ROOT), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
