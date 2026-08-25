"""因子挖掘编排入口：建会话 → 拼 prompt → 跑轨迹循环。"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from alphaagent.factor.mining.env_settings import resolve_max_parallel_eval
from alphaagent.factor.mining.schemas import SessionCreateRequest
from alphaagent.factor.mining.service import StockEvalService
from alphaagent.factor.mining.config import MiningConfig
from alphaagent.factor.mining.console import ConsolePrinter
from alphaagent.factor.mining.loop import run_trajectory
from alphaagent.factor.mining.operators import list_operator_names
from alphaagent.factor.mining.prompts import build_system_prompt
from alphaagent.factor.mining.submit import FactorSubmitService, default_factorlib_path
from alphaagent.factor.mining.tools import FactorEvalTools


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def run_factor_mining(
    config: MiningConfig,
    user_message: str,
    *,
    client: Any,
    log_dir: str | Path = "logs/factor_mining",
    include_operator_catalog: bool = True,
    extra_instructions: str = "",
    extra_body: dict[str, Any] | None = None,
    service: StockEvalService | None = None,
    verbose: bool = False,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    service = service or StockEvalService(
        max_parallel_eval=resolve_max_parallel_eval(config.max_parallel_eval),
    )
    root = repo_root or _repo_root()
    ctx = config.eval
    session_resp = service.create_session(
        SessionCreateRequest(
            panel_path=str(ctx.panel_path),
            train_start=ctx.train_start,
            train_end=ctx.train_end,
            val_start=ctx.val_start,
            val_end=ctx.val_end,
            label_col=ctx.label_col,
            include_fundamentals=ctx.include_fundamentals,
        )
    )

    submit_service: FactorSubmitService | None = None
    lib_path = (config.factorlib_path or default_factorlib_path(root)).resolve()
    registry_path = config.registry_path or lib_path / "mining_delivered_registry.json"
    expr_dir = config.expr_dir or lib_path / "expressions"
    submit_service = FactorSubmitService(
        service,
        factorlib_path=lib_path,
        registry_path=registry_path if registry_path.is_absolute() else root / registry_path,
        expr_dir=expr_dir if expr_dir.is_absolute() else root / expr_dir,
        repo_root=root,
        max_cs_corr=config.max_cs_corr,
        similar_top_k=config.similar_top_k,
        overwrite=config.ingest_overwrite,
        auto_realign_panel=config.auto_realign_panel,
    )

    tools = FactorEvalTools(service, session_resp.session_id, submit_service=submit_service)
    system_prompt = build_system_prompt(
        include_operator_catalog=include_operator_catalog,
        extra_instructions=extra_instructions,
        label_col=ctx.label_col,
        include_fundamentals=ctx.include_fundamentals,
    )

    printer = ConsolePrinter() if verbose else None
    if printer is not None:
        printer.session_start(config.model, len(list_operator_names()))

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_jsonl = log_dir / f"run_{stamp}.jsonl"

    messages = run_trajectory(
        client=client,
        model=config.model,
        system_prompt=system_prompt,
        user_message=user_message,
        tools=tools,
        log_jsonl=log_jsonl,
        max_turns=config.max_turns,
        max_tool_calls_per_round=config.max_tool_calls_per_round,
        max_tool_workers=config.max_tool_workers,
        min_tool_call_rounds_before_allow_stop=config.min_tool_call_rounds_before_allow_stop,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        extra_body=extra_body,
        printer=printer,
    )

    return {
        "session_id": session_resp.session_id,
        "n_messages": len(messages),
        "log_jsonl": str(log_jsonl),
        "messages_snapshot": str(log_jsonl.with_suffix(".messages.json")),
        "summary": str(log_jsonl.with_suffix(".summary.json")),
    }
