"""AgentScope 版因子挖掘编排：复用 prompt/工具上下文，终端流式输出。"""

from __future__ import annotations

import gzip
import hashlib
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agentscope.agent import Agent, ContextConfig, ReActConfig
from agentscope.credential import OpenAICredential
from agentscope.message import UserMsg
from agentscope.model import OpenAIChatModel
from agentscope.permission import PermissionContext, PermissionMode
from agentscope.state import AgentState
from agentscope.workspace import LocalWorkspace

from alphaagent.factor.mining.audit import build_manifest, canonical_hash
from alphaagent.factor.mining.env_settings import resolve_max_parallel_eval
from alphaagent.factor.mining.schemas import SessionCreateRequest
from alphaagent.factor.mining.service import StockEvalService
from alphaagent.factor.mining.agentscope_tools import build_factor_eval_toolkit, context_to_openai_messages
from alphaagent.factor.mining.cli_stream import MiningStreamObserver, stream_to_cli
from alphaagent.factor.mining.config import MiningConfig
from alphaagent.factor.mining.console import ConsolePrinter, ensure_utf8_stream
from alphaagent.factor.mining.loop import _NUDGE, _submit_record
from alphaagent.factor.mining.operators import list_operator_names
from alphaagent.factor.mining.prompts import build_system_prompt
from alphaagent.factor.mining.submit import FactorSubmitService, default_factorlib_path
from alphaagent.factor.mining.tools import FactorEvalTools
from alphaagent.factor.mining.factor_reviewer import FactorReviewer
from alphaagent.factor.mining.research_memory import ResearchMemoryStore
from alphaagent.factor.evaluation.profile import resolve_profiles
from core import factor_categories

_NUDGE_MSG = _NUDGE


def _client_kwargs() -> dict[str, Any]:
    """Return extra kwargs for the OpenAI AsyncClient.

    Some third-party relay providers (e.g. okmcode behind Cloudflare) block
    requests whose ``User-Agent`` contains "OpenAI/Python".  Override it
    with a neutral value so the request goes through CC Switch unmodified.
    """
    return {
        "default_headers": {"User-Agent": "quant-ui/1.0"},
    }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _build_model(
    config: MiningConfig,
    *,
    api_key: str,
    base_url: str | None,
    extra_body: dict[str, Any] | None,
) -> OpenAIChatModel:
    params: dict[str, Any] = {
        "max_tokens": config.max_tokens,
        "parallel_tool_calls": True,
    }
    if config.temperature is not None:
        params["temperature"] = config.temperature
    return OpenAIChatModel(
        credential=OpenAICredential(api_key=api_key, base_url=base_url),
        model=config.model,
        parameters=OpenAIChatModel.Parameters(**params),
        stream=True,
        extra_body=extra_body,
        client_kwargs=_client_kwargs(),
        max_retries=config.model_max_retries,
        retry_delay=config.model_retry_delay,
    )


async def create_mining_agent(
    *,
    config: MiningConfig,
    system_prompt: str,
    factor_tools: FactorEvalTools,
    workspace: LocalWorkspace,
    api_key: str,
    base_url: str | None,
    extra_body: dict[str, Any] | None,
    reviewer: FactorReviewer | None = None,
    interaction_policy: dict[str, Any] | None = None,
) -> Agent:
    toolkit = build_factor_eval_toolkit(
        factor_tools,
        max_workers=config.max_tool_workers,
        reviewer=reviewer,
        interaction_policy=interaction_policy,
        population_max=config.population_max,
    )
    react_iters = max(config.max_turns * config.max_tool_calls_per_round, config.max_turns, 20)
    return Agent(
        name="FactorMiner",
        system_prompt=system_prompt,
        model=_build_model(config, api_key=api_key, base_url=base_url, extra_body=extra_body),
        toolkit=toolkit,
        offloader=workspace,
        react_config=ReActConfig(max_iters=react_iters),
        state=AgentState(
            permission_context=PermissionContext(mode=PermissionMode.BYPASS),
        ),
        context_config=ContextConfig(
            trigger_ratio=0.75,
            reserve_ratio=0.15,
            tool_result_limit=5000,
        ),
    )


async def run_factor_mining_agentscope(
    config: MiningConfig,
    user_message: str,
    *,
    api_key: str,
    base_url: str | None = None,
    log_dir: str | Path = "logs/factor_mining",
    include_operator_catalog: bool = True,
    extra_instructions: str = "",
    extra_body: dict[str, Any] | None = None,
    service: StockEvalService | None = None,
    verbose: bool = True,
    repo_root: Path | None = None,
    control_file: Path | None = None,
    research_memory_path: Path | None = None,
) -> dict[str, Any]:
    """AgentScope 版挖掘入口：与 run_factor_mining 配置一致，CLI 流式输出。"""
    service = service or StockEvalService(
        max_parallel_eval=resolve_max_parallel_eval(config.max_parallel_eval),
        profiles=resolve_profiles(config.research_spec),
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
    research_mode = (config.research_spec or {}).get("research_mode", "technical")
    lib_path = (config.factorlib_path or factor_categories.production_dir(research_mode)).resolve()
    registry_path = config.registry_path or factor_categories.production_registry_path(research_mode)
    expr_dir = config.expr_dir or factor_categories.production_expr_dir(research_mode)
    submit_service = FactorSubmitService(
        service,
        factorlib_path=lib_path,
        registry_path=registry_path if Path(registry_path).is_absolute() else root / Path(registry_path),
        expr_dir=expr_dir if Path(expr_dir).is_absolute() else root / Path(expr_dir),
        repo_root=root,
        research_mode=research_mode,
        max_cs_corr=config.max_cs_corr,
        delivery_policy=(config.research_spec or {}).get("delivery_policy"),
        similar_top_k=config.similar_top_k,
        overwrite=config.ingest_overwrite,
    )

    factor_tools = FactorEvalTools(service, session_resp.session_id, submit_service=submit_service)
    system_prompt = build_system_prompt(
        include_operator_catalog=include_operator_catalog,
        extra_instructions=extra_instructions,
        label_col=ctx.label_col,
        include_fundamentals=ctx.include_fundamentals,
        panel_columns=session_resp.available_columns,
        population_max=config.population_max,
        research_spec=config.research_spec,
    )

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_jsonl = log_dir / f"run_{stamp}.jsonl"
    artifact_dir = log_dir / f"artifacts_{stamp}"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    factorlib_for_manifest = (config.factorlib_path or default_factorlib_path(root)).resolve()
    manifest = build_manifest(
        root=root,
        config=config,
        user_message=user_message,
        research_spec=config.research_spec or {},
        panel_path=ctx.panel_path,
        factorlib_path=factorlib_for_manifest,
        model=config.model,
    )
    (log_dir / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    memory_store = ResearchMemoryStore(research_memory_path) if research_memory_path is not None else None
    workspace_dir = log_dir / f"agentscope_workspace_{stamp}"
    workspace = LocalWorkspace(workdir=str(workspace_dir))
    await workspace.initialize()

    factor_tools = FactorEvalTools(service, session_resp.session_id, submit_service=submit_service)
    system_prompt = build_system_prompt(
        include_operator_catalog=include_operator_catalog,
        extra_instructions=extra_instructions,
        label_col=ctx.label_col,
        include_fundamentals=ctx.include_fundamentals,
        panel_columns=session_resp.available_columns,
        population_max=config.population_max,
        research_spec=config.research_spec,
    )

    # Windows 控制台默认 GBK：模型/工具输出含 emoji 时会中断会话，统一转 UTF-8 容错。
    ensure_utf8_stream(sys.stdout)
    ensure_utf8_stream(sys.stderr)
    printer = ConsolePrinter(stream=sys.stderr) if verbose else None
    if printer is not None:
        printer.session_start(config.model, len(list_operator_names()))

    def _emit(event: str, payload: dict[str, Any]) -> None:
        record = {"ts": _now(), "event": event, **payload}
        if event == "tool_results":
            raw = json.dumps(record.get("results", []), ensure_ascii=False, default=str)
            if len(raw.encode("utf-8")) > 12_000:
                artifact_name = f"tool_results_{record.get('turn', 0)}_{hashlib.sha1(raw.encode()).hexdigest()[:10]}.json.gz"
                artifact_path = artifact_dir / artifact_name
                with gzip.open(artifact_path, "wt", encoding="utf-8") as artifact:
                    artifact.write(raw)
                record["results"] = [{
                    "tool_call_id": row.get("tool_call_id"),
                    "name": row.get("name"),
                    "elapsed_seconds": row.get("elapsed_seconds"),
                    "result_ref": str(artifact_path.relative_to(log_dir)),
                    "result_sha256": hashlib.sha256(raw.encode("utf-8")).hexdigest(),
                } for row in record.get("results", [])]
                record["results_externalized"] = True
        with log_jsonl.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            f.flush()

    # ── Numba JIT 预热：在 Agent 启动前预编译慢算子，避免首次评估超时 ──
    _emit("jit_warmup_start", {"turn": 0})
    try:
        from alphaagent.dsl.core.jit_warmup import warmup_numba_jit
        warmup_results = warmup_numba_jit()
        _emit("jit_warmup_done", {"turn": 0, "results": warmup_results})
    except Exception as exc:
        _emit("jit_warmup_failed", {"turn": 0, "error": str(exc)})

    # ── 管道自检：Agent 开工前验证生产库/索引契约/evaluate/submit 全链路，
    #    快速失败而非让 LLM 中途反复重试同一基础设施错误（历史 run 一半时间
    #    浪费在 factorlib_not_initialized / MultiIndex.rows 的重复重试上）。──
    if submit_service is not None:
        from alphaagent.factor.mining.preflight import PreflightError, preflight_summary, run_preflight
        _emit("preflight_start", {"turn": 0})
        try:
            preflight = run_preflight(
                service=service,
                session_id=session_resp.session_id,
                submit_service=submit_service,
            )
            _emit("preflight_done", {"turn": 0, "checks": preflight.checks, "duration_s": preflight.duration_s})
            if printer is not None:
                printer.info(preflight_summary(preflight))
        except PreflightError as exc:
            _emit("preflight_failed", {"turn": 0, "error": str(exc)})
            if printer is not None:
                printer.error(f"启动自检失败，终止挖掘：\n{exc}")
            raise
        except Exception as exc:  # noqa: BLE001
            _emit("preflight_failed", {"turn": 0, "error": f"{type(exc).__name__}: {exc}"})
            raise

    started_at = _now()
    _emit("session_start", {
        "schema_version": 2,
        "run_id": log_dir.name,
        "started_at": started_at,
        "model": config.model,
        "max_turns": config.max_turns,
        "framework": "agentscope",
        "manifest": "run_manifest.json",
    })
    _emit("user_message", {"turn": 0, "content": user_message})

    # 写 run_meta.json：与 backend/alphaagent_service.save_meta 同 schema，
    # 让 CLI 直启的 run 也能被后端 hydrate_runs 恢复进前端任务列表。
    try:
        (log_dir / "run_meta.json").write_text(
            json.dumps(
                {
                    "run_id": log_dir.name,
                    "created_at": started_at,
                    "params": {"user_message": user_message},
                    "parent_run_id": None,
                    "title": user_message.strip().replace("\n", " ")[:32] or log_dir.name,
                    "archived": False,
                    "pinned": False,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    except OSError:
        pass

    tool_call_rounds = 0
    outer_turn = 0
    tool_call_rows: list[dict[str, Any]] = []
    submit_records: list[dict[str, Any]] = []
    usage_total: dict[str, int] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "calls": 0,
    }
    end_reason = "no_tool_calls"
    pending = user_message
    turn_limit = config.max_turns
    control_offset = 0

    def _review_emit(event: str, payload: dict[str, Any]) -> None:
        _emit(event, payload)
        if event != "reviewer_usage":
            return
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_input_tokens",
            "cache_creation_input_tokens",
        ):
            usage_total[key] += int(payload.get(key, 0) or 0)
        usage_total["calls"] += 1
        _emit("usage_total", {"turn": payload.get("turn", 0), **usage_total})

    reviewer = (
        FactorReviewer(
            config=config,
            api_key=api_key,
            base_url=base_url,
            extra_body=extra_body,
            workspace=workspace,
            emit=_review_emit,
        )
        if config.enable_reviewer
        else None
    )

    agent = await create_mining_agent(
        config=config,
        system_prompt=system_prompt,
        factor_tools=factor_tools,
        workspace=workspace,
        api_key=api_key,
        base_url=base_url,
        extra_body=extra_body,
        reviewer=reviewer,
        interaction_policy=(config.research_spec or {}).get("interaction_policy"),
    )

    def _take_control_messages() -> list[str]:
        nonlocal control_offset
        if control_file is None or not control_file.exists():
            return []
        messages: list[str] = []
        try:
            with control_file.open("r", encoding="utf-8") as handle:
                handle.seek(control_offset)
                lines = handle.readlines()
                control_offset = handle.tell()
        except OSError:
            return []
        for line in lines:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = item.get("content") if isinstance(item, dict) else None
            if isinstance(content, str) and content.strip():
                messages.append(content.strip())
        return messages

    def _dynamic_memory_context() -> str:
        """Retrieve compact evidence at every outer turn, not only at startup."""
        if memory_store is None:
            return ""
        policy = (config.research_spec or {}).get("memory_policy") or {}
        limit = int(policy.get("dynamic_retrieve_limit", 6))
        if limit <= 0:
            return ""

        query = memory_store.query_for_attempts(
            user_message,
            tool_call_rows,
            max_recent_attempts=8,
            max_expression_chars=1200,
        )
        return memory_store.context_for(
            query,
            limit=limit,
            include_rejected=bool(policy.get("include_rejected_paths", True)),
            prefer_orthogonal=bool(policy.get("prefer_orthogonal_to_approved", True)),
            include_expression=bool(policy.get("include_expression", True)),
            max_expression_chars=int(policy.get("max_expression_chars", 320)),
            enable_factor_retrieval=bool(policy.get("enable_factor_retrieval", False)),
        )

    def _queued_prompt(messages: list[str]) -> str:
        return "用户在当前研究会话追加了指令，请优先结合已有评估结果执行：\n" + "\n\n".join(messages)

    while outer_turn < turn_limit:
        if outer_turn == 0:
            # 清空批次历史，避免跨 run 污染
            FactorEvalTools._batch_history.clear()
        if reviewer is not None:
            reviewer.current_turn = outer_turn
        if printer is not None:
            printer.turn(outer_turn)

        observer = MiningStreamObserver(printer=printer, emit=_emit, turn=outer_turn)

        def _on_stream_emit(event: str, payload: dict[str, Any]) -> None:
            _emit(event, payload)
            if event == "usage":
                for key in (
                    "input_tokens",
                    "output_tokens",
                    "cache_input_tokens",
                    "cache_creation_input_tokens",
                ):
                    usage_total[key] += int(payload.get(key, 0) or 0)
                usage_total["calls"] += 1
                _emit("usage_total", {"turn": outer_turn, **usage_total})
            if event != "tool_results":
                return
            for row in payload.get("results") or []:
                res = row.get("result") if isinstance(row.get("result"), dict) else {}
                args_raw = row.get("arguments_raw")
                args_obj: dict[str, Any] = {}
                if isinstance(args_raw, dict):
                    args_obj = args_raw
                elif isinstance(args_raw, str):
                    try:
                        parsed_args = json.loads(args_raw)
                        args_obj = parsed_args if isinstance(parsed_args, dict) else {}
                    except json.JSONDecodeError:
                        args_obj = {}
                args_hash = canonical_hash(args_raw or "")
                summary_metrics = res.get("summary") if isinstance(res.get("summary"), dict) else {}
                if not summary_metrics.get("ic"):
                    metrics = res.get("metrics") if isinstance(res.get("metrics"), dict) else {}
                    cross_sectional = metrics.get("cross_sectional_core")
                    if isinstance(cross_sectional, dict) and cross_sectional:
                        summary_metrics = cross_sectional
                    elif "ic" in metrics:
                        summary_metrics = metrics
                tool_call_rows.append(
                    {
                        "turn": outer_turn,
                        "tool_call_id": row.get("tool_call_id"),
                        "name": row.get("name"),
                        "arguments_sha256": args_hash,
                        "elapsed_seconds": row.get("elapsed_seconds"),
                        "ok": bool(res.get("ok")),
                        "factor_name": args_obj.get("factor_name"),
                        "expression_sha256": canonical_hash(args_obj.get("multi_line_expr")) if args_obj.get("multi_line_expr") else None,
                        "split": res.get("split"),
                        "metrics": {key: summary_metrics.get(key) for key in ("ic", "icir", "rank_ic", "factor_coverage", "coverage") if summary_metrics.get(key) is not None},
                        "error_type": res.get("error_type"),
                        "error": str(res.get("error") or res.get("skipped_reason") or "")[:500],
                    }
                )
                if row.get("name") == "submit_factor":
                    submit_records.append(
                        _submit_record(
                            turn=outer_turn,
                            arguments_raw=row.get("arguments_raw"),
                            result=res,
                        )
                    )
                if memory_store is not None:
                    memory_entry = memory_store.record_tool_result(
                        run_id=log_dir.name,
                        row=row,
                    )
                    if memory_entry is not None:
                        _emit(
                            "research_memory_updated",
                            {
                                "turn": outer_turn,
                                "factor_name": memory_entry["factor_name"],
                                "verdict": memory_entry["verdict"],
                                "conclusion": memory_entry["conclusion"],
                            },
                        )

        observer.emit = _on_stream_emit

        if outer_turn > 0 or pending != user_message:
            _emit("user_message", {"turn": outer_turn, "content": pending})

        turn_memory = _dynamic_memory_context()
        agent_prompt = pending
        if turn_memory:
            agent_prompt = f"{turn_memory}\n\n# 当前研究任务 / 最新反馈\n{pending}"
            retrieved_count = sum(
                1 for line in turn_memory.splitlines() if line.startswith("- [")
            )
            _emit("research_memory_retrieved", {
                "turn": outer_turn,
                "entry_count": retrieved_count,
                "recent_attempt_count": len(tool_call_rows),
            })

        user_msg = UserMsg(name="user", content=agent_prompt)
        try:
            had_tools = await stream_to_cli(
                agent,
                user_msg,
                show_thinking=True,
                auto_confirm=True,
                observer=observer,
                quiet=not verbose,
            )
        except Exception as exc:
            end_reason = "error"
            _emit(
                "session_error",
                {
                    "turn": outer_turn,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=20),
                },
            )
            break

        queued_messages = _take_control_messages()
        if queued_messages:
            pending = _queued_prompt(queued_messages)
            _emit("continuation_accepted", {"turn": outer_turn, "count": len(queued_messages)})
            outer_turn += 1
            # A live user instruction earns one extra outer turn, bounded so
            # an unattended UI cannot extend the session indefinitely.
            turn_limit = min(config.max_turns + 20, turn_limit + 1)
            if outer_turn >= turn_limit:
                end_reason = "max_turns_reached"
                break
            continue

        if had_tools:
            tool_call_rounds += 1

        if not had_tools:
            if tool_call_rounds < config.min_tool_call_rounds_before_allow_stop:
                _emit("nudge", {"turn": outer_turn, "tool_call_rounds": tool_call_rounds})
                pending = _NUDGE_MSG
                outer_turn += 1
                continue
            end_reason = "no_tool_calls"
            break

        outer_turn += 1
        if outer_turn >= turn_limit:
            end_reason = "max_turns_reached"
            break

        # ── 轮间反思注入：结构化汇总 + 方向建议 ──
        turn_rows = [r for r in tool_call_rows if r.get("turn") == outer_turn - 1 and r.get("name") in ("evaluate_factor", "eval_on_train_set", "eval_on_val_set")]
        if turn_rows:
            reflection_lines = ["[轮间反思] 上一轮评估汇总："]
            for r in turn_rows:
                m = r.get("metrics", {})
                ic = m.get("ic")
                icir = m.get("icir")
                status = "PASS" if r.get("ok") and ic is not None and abs(ic) >= 0.02 else "fail"
                ic_str = f"{ic:+.4f}" if ic is not None else "N/A"
                icir_str = f"{icir:+.3f}" if icir is not None else "N/A"
                reflection_lines.append(f"  [{status}] {r.get('factor_name','?')}: IC={ic_str} ICIR={icir_str}")
            # 同质化检测
            exprs_this_turn = [r.get("expression_sha256") for r in turn_rows if r.get("expression_sha256")]
            if exprs_this_turn:
                from collections import Counter
                common = Counter(exprs_this_turn).most_common(1)[0]
                if common[1] >= 2:
                    reflection_lines.append("⚠ 警告: 本轮有重复表达式！请在下一轮尝试完全不同的信号维度。")
            # IC 趋势
            all_ics = [r.get("metrics", {}).get("ic") for r in tool_call_rows if r.get("metrics", {}).get("ic") is not None]
            if len(all_ics) >= 3:
                recent_ics = all_ics[-5:]
                avg_ic = sum(recent_ics) / len(recent_ics)
                reflection_lines.append(f"📊 累计 {len(all_ics)} 个因子评估，近5个平均IC={avg_ic:+.4f}")
                if all(abs(ic) < 0.02 for ic in recent_ics):
                    reflection_lines.append("⚠ 近5个因子IC全部低于0.02，当前方向可能无效。请：① 完全换一个信号族 ② 尝试不同时间窗口 ③ 考虑交互信号")
            # 方向建议
            explored_dims = set()
            for r in turn_rows:
                expr_hash = r.get("expression_sha256", "")
                # 粗略检测信号维度
                fname = str(r.get("factor_name", "")).lower()
                if "reversal" in fname or "pctchange" in fname:
                    explored_dims.add("reversal")
                if "momentum" in fname or "ma" in fname:
                    explored_dims.add("momentum")
                if "volume" in fname or "amount" in fname:
                    explored_dims.add("volume")
                if "vol" in fname and "volume" not in fname:
                    explored_dims.add("volatility")
                if "vwap" in fname:
                    explored_dims.add("vwap")
                if "chip" in fname:
                    explored_dims.add("chip")
                if "overnight" in fname or "gap" in fname:
                    explored_dims.add("overnight")
            if len(explored_dims) <= 1 and len(all_ics) >= 3:
                reflection_lines.append(f"💡 已探索维度: {explored_dims or {'unknown'}}。未探索: 动量(TS_MEAN($ret,N)), 波动率(TS_STD($ret,N)), 量价关系(TS_CORR), 隔夜跳空($adj_open vs prev_close), VWAP偏离, 筹码(CHIP_*). 请选一个未尝试的维度。")
            reflection_lines.append("请基于以上分析，发起下一轮 evaluate_factor 调用（建议并行2-3条不同维度的假设）。")
            pending = "\n".join(reflection_lines)

            # ── 批次蒸馏：从本轮结果提炼模式记忆（改进二）──
            if memory_store is not None:
                batch_for_distill = [
                    {
                        "factor_name": r.get("factor_name", ""),
                        "expression": str(r.get("arguments_raw", "")),
                        "metrics": r.get("metrics", {}),
                    }
                    for r in turn_rows
                ]
                try:
                    distilled = memory_store.distill_batch_patterns(
                        run_id=log_dir.name,
                        turn=outer_turn - 1,
                        batch_results=batch_for_distill,
                    )
                    if distilled:
                        _emit("patterns_distilled", {
                            "turn": outer_turn - 1,
                            "pattern_ids": distilled,
                            "count": len(distilled),
                        })
                except Exception as exc:
                    _emit("patterns_distill_error", {
                        "turn": outer_turn - 1,
                        "error": str(exc),
                    })
        # ── end: if turn_rows ──
    else:
        end_reason = "max_turns_reached"

    if printer is not None:
        ok = sum(1 for r in tool_call_rows if r.get("ok"))
        printer.session_end(end_reason, ok, len(tool_call_rows))

    train_attempts = {r.get("expression_sha256") for r in tool_call_rows if r.get("name") == "eval_on_train_set" and r.get("expression_sha256")}
    val_attempts = {r.get("expression_sha256") for r in tool_call_rows if r.get("name") == "eval_on_val_set" and r.get("expression_sha256")}
    candidate_stored = sum(1 for row in submit_records if row.get("candidate_stored"))
    production_stored = sum(1 for row in submit_records if row.get("stored"))
    failure_counts: dict[str, int] = {}
    for row in tool_call_rows:
        if not row.get("ok"):
            code = str(row.get("error_type") or "tool_failed")
            failure_counts[code] = failure_counts.get(code, 0) + 1
    train_by_expr = {r.get("expression_sha256"): r for r in tool_call_rows if r.get("name") == "eval_on_train_set" and r.get("expression_sha256")}
    val_by_expr = {r.get("expression_sha256"): r for r in tool_call_rows if r.get("name") == "eval_on_val_set" and r.get("expression_sha256")}
    matched_audits: list[dict[str, Any]] = []
    for expr_hash in sorted(train_by_expr.keys() & val_by_expr.keys()):
        train_metrics = train_by_expr[expr_hash].get("metrics", {})
        val_metrics = val_by_expr[expr_hash].get("metrics", {})
        train_ic = train_metrics.get("ic")
        val_ic = val_metrics.get("ic")
        if isinstance(train_ic, (int, float)) and isinstance(val_ic, (int, float)):
            matched_audits.append({
                "expression_sha256": expr_hash,
                "train_ic": train_ic,
                "val_ic": val_ic,
                "ic_gap": round(train_ic - val_ic, 8),
                "ic_retention": round(val_ic / train_ic, 6) if train_ic else None,
                "sign_match": (train_ic == 0 or val_ic == 0 or (train_ic > 0) == (val_ic > 0)),
                "overfit_suspected": bool(train_ic and abs(train_ic) >= 0.015 and (not val_ic or abs(val_ic) < 0.01 or (train_ic > 0) != (val_ic > 0))),
            })
    overfit_suspected = any(row["overfit_suspected"] for row in matched_audits)
    if production_stored:
        outcome, success = "production_factor", True
    elif candidate_stored:
        outcome, success = "candidate_only", False
    elif end_reason == "error":
        outcome, success = "error", False
    elif end_reason == "max_turns_reached":
        outcome, success = "interrupted", False
    else:
        outcome, success = "no_candidate", False

    messages = context_to_openai_messages(agent.state.context)
    snapshot = log_jsonl.with_suffix(".messages.json.gz")
    with gzip.open(snapshot, "wt", encoding="utf-8") as handle:
        json.dump(messages, handle, ensure_ascii=False, separators=(",", ":"), default=str)

    times = [r["elapsed_seconds"] for r in tool_call_rows if r.get("elapsed_seconds") is not None]
    submitted_factors = [r for r in submit_records if r.get("stored")]
    submit_failures = [r for r in submit_records if not r.get("stored")]
    summary = {
        "schema_version": 2,
        "run_id": log_dir.name,
        "log_jsonl": str(log_jsonl),
        "framework": "agentscope",
        "agent_session_id": agent.state.session_id,
        "outcome": outcome,
        "success": success,
        "termination_reason": end_reason,
        "tool_calls": {
            "count": len(tool_call_rows),
            "ok": sum(1 for r in tool_call_rows if r.get("ok")),
            "failed": sum(1 for r in tool_call_rows if not r.get("ok")),
            "elapsed_seconds_total": round(sum(times), 4) if times else 0.0,
        },
        "candidate_funnel": {
            "unique_train_evaluated": len(train_attempts),
            "unique_val_evaluated": len(val_attempts),
            "candidate_stored": candidate_stored,
            "production_stored": production_stored,
            "train_to_val_rate": round(len(val_attempts) / len(train_attempts), 4) if train_attempts else None,
            "val_to_production_rate": round(production_stored / len(val_attempts), 4) if val_attempts else None,
        },
        "failure_counts": failure_counts,
        "overfit_audit": {
            "status": "partial",
            "note": "本轮仅记录 train/val 候选覆盖与结果；锁定测试集仍需独立盲测。",
            "selection_bias_warning": bool(val_attempts and len(val_attempts) > 1),
            "matched_candidates": matched_audits,
            "overfit_suspected": overfit_suspected,
        },
        "usage": usage_total,
        "submitted_factors": submitted_factors,
        "submit_failures": submit_failures,
        "messages_snapshot": str(snapshot),
        "manifest": str(log_dir / "run_manifest.json"),
    }
    summary_path = log_jsonl.with_suffix(".summary.json")
    summary_text = json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n"
    summary_path.write_text(summary_text, encoding="utf-8")
    (log_dir / "run_summary.json").write_text(summary_text, encoding="utf-8")
    _emit("run_summary", summary)
    _emit("session_end", {"turn": outer_turn, "reason": end_reason})

    return {
        "session_id": session_resp.session_id,
        "agent_session_id": agent.state.session_id,
        "n_messages": len(messages),
        "log_jsonl": str(log_jsonl),
        "messages_snapshot": str(snapshot),
        "summary": str(log_jsonl.with_suffix(".summary.json")),
        "framework": "agentscope",
    }
