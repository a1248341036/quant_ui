#!/usr/bin/env python3
"""LLM 股票因子挖掘 CLI（AgentScope 版，终端流式输出）。

与 scripts/factor_mining.py 使用相同的 system prompt 与 eval/submit 工具语义；
模型思考、回复与工具结果通过 AgentScope reply_stream 实时打印。

示例：
  uv sync --extra mining
  uv run python scripts/factor_mining_agentscope.py \\
    --panel artifacts/panel/panel_1d.parquet
  uv run python scripts/factor_mining_agentscope.py \\
    --panel artifacts/panel/panel_1d.parquet \\
    --seed-factor examples/factors/ma20_dev.dsl \\
    --user-message "在种子因子基础上继续优化"

环境变量（仓库根 .env）：OPENAI_API_KEY、OPENAI_API_BASE、MODEL。
"""

from __future__ import annotations

import argparse
import asyncio
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
from alphaagent.factor.mining import MiningConfig  # noqa: E402
from alphaagent.factor.mining.agentscope_run import run_factor_mining_agentscope  # noqa: E402
from alphaagent.factor.mining.context import StockEvalContext  # noqa: E402
from alphaagent.factor.mining.research_spec import load_research_spec, research_policy_prompt  # noqa: E402
from alphaagent.factor.mining.seed_factors import build_user_message_with_seed_factors  # noqa: E402
from alphaagent.factor.types import (
    DEFAULT_LABEL_COL,
    DEFAULT_TEST_START,
    DEFAULT_TRAIN_END,
    DEFAULT_TRAIN_START,
    DEFAULT_VAL_END,
    DEFAULT_VAL_START,
)  # noqa: E402


_help_test_end = (
    "测试段右端（留出测试段终审，只报告不拦截）；"
    "缺省时动态解析数据源最新交易日"
)


def _load_env() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LLM 股票因子挖掘（AgentScope 流式 CLI）")
    p.add_argument("--panel", default="cne://")
    p.add_argument(
        "--asset-type",
        default="stock",
        choices=["stock", "etf"],
        help="资产类型：stock（默认）走股票 panel；etf 走场内 ETF panel（etf_bars，无市值字段，amount 已是元）",
    )
    p.add_argument("--train-start", default=DEFAULT_TRAIN_START)
    p.add_argument("--train-end", default=DEFAULT_TRAIN_END)
    p.add_argument("--val-start", default=DEFAULT_VAL_START)
    p.add_argument("--val-end", default=DEFAULT_VAL_END)
    p.add_argument("--test-start", default=DEFAULT_TEST_START)
    p.add_argument("--test-end", default=None, help=_help_test_end)
    p.add_argument("--label-col", default=None, help="评估 label 列；缺省时取 ResearchSpec.recommended_label_col")
    p.add_argument(
        "--no-fundamentals",
        action="store_true",
        help="不载入基本面列(funda_*)，省内存；prompt 也会隐藏基本面字段（适合只挖价量因子）",
    )
    p.add_argument("--temperature", type=float, default=None)
    p.add_argument("--max-tokens", type=int, default=16384)  # hy3 thinking 需 >8K
    p.add_argument(
        "--max-turns",
        type=int,
        default=5,
        help=(
            "外层重进 agent 的次数上限（每次 = 一整轮 ReAct，模型自愿停手才回外层）；"
            "同时间接决定 ReAct 内循环上限 max(max_turns*max_tool_calls_per_round, max_turns, 20)。"
            "注意：它无法打断进行中的单次 reply_stream，实际运行长度主要由内循环上限与模型何时停手决定"
        ),
    )
    p.add_argument("--max-tool-calls-per-round", type=int, default=8)
    p.add_argument("--max-tool-workers", type=int, default=4)
    p.add_argument(
        "--population-max",
        type=int,
        default=24,
        help="种群批量筛选(propose_population)单轮候选上限；0=关闭路径B",
    )
    p.add_argument(
        "--max-parallel-eval",
        type=int,
        default=None,
        help="同时进行的 train/val 评估上限；不传则读环境变量 MAX_PARALLEL_EVAL（默认 1）。建议与 --max-tool-workers 匹配",
    )
    p.add_argument("--min-tool-call-rounds", type=int, default=3)
    p.add_argument("--log-dir", default="logs/factor_mining")
    p.add_argument(
        "--research-memory-file",
        type=Path,
        default=ROOT / "artifacts" / "alphaagent" / "research_memory.db",
        help="跨会话因子研究记忆；传空路径可在调用方禁用",
    )
    p.add_argument(
        "--research-spec-file",
        type=Path,
        default=None,
        help="ResearchSpec JSON；控制搜索、评估、Reviewer、记忆与交付策略",
    )
    p.add_argument(
        "--control-file",
        type=Path,
        default=None,
        help="运行中追加用户指令的 JSONL 队列文件（由 Web UI 使用）",
    )
    p.add_argument(
        "--user-message",
        default="请在训练集上提出并迭代多个多行因子表达式，再于验证集上检验泛化；目标为提高 abs(IC)/RANKIC 与 ICIR，并兼顾月度稳健性。",
    )
    p.add_argument("--user-file", type=Path, help="从文件读取 user 消息（覆盖 --user-message）")
    p.add_argument(
        "--resume-context-file",
        type=Path,
        default=None,
        help="历史研究上下文摘要，追加到本轮用户指令前（由 Web UI 使用）",
    )
    p.add_argument(
        "--seed-factor",
        dest="seed_factors",
        action="append",
        nargs="+",
        default=[],
        metavar="PATH",
        help="初始种子因子 .dsl 路径，可重复指定；单次可跟多个路径，如 --seed-factor a.dsl b.dsl",
    )
    p.add_argument("--no-operator-catalog", action="store_true", help="不在 system prompt 中注入算子清单")
    p.add_argument("--quiet", action="store_true", help="不在终端流式打印（仍写 JSONL 日志）")
    p.add_argument("--factorlib", type=Path, default=None, help=f"factorzoo 根目录（默认 {FACTORZOO_DIR}）")
    p.add_argument("--no-reviewer", action="store_true", help="禁用提交前 FactorReviewer 子 Agent（仅调试）")
    p.add_argument("--max-cs-corr", type=float, default=0.8, help="submit 截面去重 |corr| 上限")
    p.add_argument("--similar-top-k", type=int, default=3, help="查重失败时返回的最相似因子数")
    p.add_argument("--ingest-overwrite", action="store_true", help="submit 时覆盖已存在 factor_id")
    return p.parse_args()


def _resolve(path: str) -> Path | str:
    # CNE is a logical data-source URI, not a filesystem path.
    if path.lower().startswith("cne://"):
        return path
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
        import agentscope  # noqa: F401
    except ImportError:
        print("错误：请安装 agentscope（uv sync --extra mining）", file=sys.stderr)
        return 2

    user_message = args.user_file.read_text(encoding="utf-8") if args.user_file else args.user_message
    try:
        research_spec = load_research_spec(args.research_spec_file)
    except ValueError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    memory_path = _resolve(str(args.research_memory_file)) if args.research_memory_file else None
    if args.resume_context_file and args.resume_context_file.exists():
        history = args.resume_context_file.read_text(encoding="utf-8").strip()
        if history:
            user_message = (
                "以下是同一因子研究会话的已完成历史。必须基于已有候选、评估结果和失败原因继续，"
                "不要重复已经完成的无效尝试。\n\n"
                f"{history}\n\n"
                "用户现在追加的研究指令：\n"
                f"{user_message}"
            )
    seed_paths = [Path(p) for batch in args.seed_factors for p in batch]
    if seed_paths:
        try:
            user_message = build_user_message_with_seed_factors(
                user_message, seed_paths, repo_root=ROOT
            )
        except FileNotFoundError as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 2
    base_url = os.environ.get("OPENAI_API_BASE")

    label_col = args.label_col or str(research_spec.get("recommended_label_col") or "") or DEFAULT_LABEL_COL
    # 测试段右端：未显式传值时动态解析数据源最新交易日（默认值 None）
    if args.test_end:
        test_end: str | None = args.test_end
    else:
        from alphaagent.factor.window_config import resolve_test_end

        test_end = resolve_test_end(asset_type=args.asset_type)

    config = MiningConfig(
        eval=StockEvalContext(
            panel_path=_resolve(args.panel),
            train_start=args.train_start,
            train_end=args.train_end,
            val_start=args.val_start,
            val_end=args.val_end,
            test_start=args.test_start,
            test_end=test_end,
            label_col=label_col,
            include_fundamentals=not args.no_fundamentals,
            asset_type=args.asset_type,
        ),
        model=model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_turns=args.max_turns,
        max_tool_calls_per_round=args.max_tool_calls_per_round,
        max_tool_workers=args.max_tool_workers,
        population_max=args.population_max,
        max_parallel_eval=args.max_parallel_eval,
        min_tool_call_rounds_before_allow_stop=args.min_tool_call_rounds,
        factorlib_path=_resolve(str(args.factorlib)) if args.factorlib else None,
        enable_submit=True,
        enable_reviewer=not args.no_reviewer and research_spec["review_policy"]["enabled"],
        research_spec=research_spec,
        max_cs_corr=research_spec["delivery_policy"]["candidate"]["max_abs_corr"],
        similar_top_k=args.similar_top_k,
        ingest_overwrite=args.ingest_overwrite,
    )

    out = asyncio.run(
        run_factor_mining_agentscope(
            config,
            user_message,
            api_key=api_key,
            base_url=base_url,
            log_dir=args.log_dir,
            include_operator_catalog=not args.no_operator_catalog,
            extra_instructions=research_policy_prompt(research_spec),
            verbose=not args.quiet,
            control_file=args.control_file,
            research_memory_path=memory_path,
        )
    )
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
