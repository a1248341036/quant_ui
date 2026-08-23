#!/usr/bin/env python3
"""LLM 股票因子挖掘 CLI。

示例：
  uv sync --extra mining
  uv run python scripts/factor_mining.py --panel artifacts/panel/panel_1d.parquet

环境变量（仓库根 .env）：OPENAI_API_KEY、OPENAI_API_BASE、MODEL。
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None  # type: ignore[assignment,misc]

from alphaagent.core.paths import FACTORZOO_DIR, PANEL_PATH  # noqa: E402
from alphaagent.factor.mining import MiningConfig, run_factor_mining  # noqa: E402
from alphaagent.factor.mining.context import StockEvalContext  # noqa: E402
from alphaagent.factor.mining.seed_factors import build_user_message_with_seed_factors  # noqa: E402
from alphaagent.factor.types import DEFAULT_LABEL_COL  # noqa: E402


def _load_env() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM 股票因子挖掘")
    p.add_argument("--panel", default=str(PANEL_PATH))
    p.add_argument("--train-start", default="2019-01-01")
    p.add_argument("--train-end", default="2021-12-31")
    p.add_argument("--val-start", default="2022-01-01")
    p.add_argument("--val-end", default="2024-12-31")
    p.add_argument("--label-col", default=DEFAULT_LABEL_COL)
    p.add_argument(
        "--no-fundamentals",
        action="store_true",
        help="不载入基本面列(funda_*)，省内存；prompt 也会隐藏基本面字段",
    )
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--max-turns", type=int, default=10)
    p.add_argument("--max-tool-calls-per-round", type=int, default=8)
    p.add_argument("--max-tool-workers", type=int, default=4)
    p.add_argument(
        "--max-parallel-eval",
        type=int,
        default=None,
        help="同时进行的 train/val 评估上限；不传则读环境变量 MAX_PARALLEL_EVAL（默认 1）。建议与 --max-tool-workers 匹配",
    )
    p.add_argument("--min-tool-call-rounds", type=int, default=3)
    p.add_argument("--log-dir", default="logs/factor_mining")
    p.add_argument(
        "--user-message",
        default="请在训练集上提出并迭代多个多行因子表达式，再于验证集上检验泛化；目标为提高 abs(IC)/RANKIC 与 ICIR，并兼顾月度稳健性。",
    )
    p.add_argument("--user-file", type=Path, help="从文件读取 user 消息（覆盖 --user-message）")
    p.add_argument(
        "--seed-factor",
        dest="seed_factors",
        action="append",
        nargs="+",
        default=[],
        metavar="PATH",
        help="初始种子因子 .dsl 路径，可重复指定；单次可跟多个路径，如 --seed-factor a.dsl b.dsl",
    )
    p.add_argument("--no-operator-catalog", action="store_true")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--factorlib", type=Path, default=None, help=f"默认 {FACTORZOO_DIR}")
    p.add_argument("--no-submit", action="store_true")
    p.add_argument("--max-cs-corr", type=float, default=0.8)
    p.add_argument("--similar-top-k", type=int, default=3)
    p.add_argument("--ingest-overwrite", action="store_true")
    return p.parse_args()


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def main() -> int:
    _load_env()
    args = _parse_args()

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("错误：未设置 OPENAI_API_KEY，请在 .env 中配置", file=sys.stderr)
        return 2

    model = os.environ.get("MODEL")
    if not model:
        print("错误：未设置 MODEL，请在 .env 中配置", file=sys.stderr)
        return 2

    try:
        from openai import OpenAI
    except ImportError:
        print("错误：请安装 openai（uv sync --extra mining）", file=sys.stderr)
        return 2

    base_url = os.environ.get("OPENAI_API_BASE")
    client = OpenAI(api_key=api_key, **({"base_url": base_url} if base_url else {}))

    user_message = args.user_file.read_text(encoding="utf-8") if args.user_file else args.user_message
    seed_paths = [Path(p) for batch in args.seed_factors for p in batch]
    if seed_paths:
        try:
            user_message = build_user_message_with_seed_factors(
                user_message, seed_paths, repo_root=ROOT
            )
        except FileNotFoundError as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 2

    config = MiningConfig(
        eval=StockEvalContext(
            panel_path=_resolve(args.panel),
            train_start=args.train_start,
            train_end=args.train_end,
            val_start=args.val_start,
            val_end=args.val_end,
            label_col=args.label_col,
            include_fundamentals=not args.no_fundamentals,
        ),
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_turns=args.max_turns,
        max_tool_calls_per_round=args.max_tool_calls_per_round,
        max_tool_workers=args.max_tool_workers,
        max_parallel_eval=args.max_parallel_eval,
        min_tool_call_rounds_before_allow_stop=args.min_tool_call_rounds,
        factorlib_path=_resolve(str(args.factorlib)) if args.factorlib else None,
        enable_submit=not args.no_submit,
        max_cs_corr=args.max_cs_corr,
        similar_top_k=args.similar_top_k,
        ingest_overwrite=args.ingest_overwrite,
    )

    out = run_factor_mining(
        config,
        user_message,
        client=client,
        log_dir=args.log_dir,
        include_operator_catalog=not args.no_operator_catalog,
        verbose=not args.quiet,
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
