"""多轮因子挖掘主循环：OpenAI Chat Completions + 并行工具调用 + JSONL 轨迹。"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alphaagent.factor.mining.console import ConsolePrinter
from alphaagent.factor.mining.tools import FactorEvalTools

_NUDGE = (
    "[local] 请继续推进，不要停在解释或征询下一步；"
    "请直接基于当前上下文发起原生 tool_calls（建议并行多条 evaluate_factor，profile_id=train_screen）。"
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _chat_with_retry(client: Any, req: dict[str, Any], *, max_retries: int = 4, backoff: float = 2.0) -> Any:
    last: BaseException | None = None
    for attempt in range(max_retries + 1):
        try:
            return client.chat.completions.create(**req)
        except BaseException as exc:  # noqa: BLE001
            last = exc
            status = getattr(exc, "status_code", None)
            if status in (400, 401, 404, 422) or attempt == max_retries:
                raise
            time.sleep(min(60.0, backoff * (2**attempt)))
    assert last is not None  # pragma: no cover
    raise last


def _tool_calls_payload(msg: Any) -> list[dict[str, Any]]:
    if not getattr(msg, "tool_calls", None):
        return []
    return [
        {
            "id": tc.id,
            "type": "function",
            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
        }
        for tc in msg.tool_calls
    ]


def _parse_tool_arguments(arguments_raw: Any) -> dict[str, Any]:
    if isinstance(arguments_raw, str):
        try:
            arguments_raw = json.loads(arguments_raw) if arguments_raw.strip() else {}
        except json.JSONDecodeError:
            arguments_raw = {}
    return arguments_raw if isinstance(arguments_raw, dict) else {}


def _submit_record(*, turn: int, arguments_raw: Any, result: dict[str, Any]) -> dict[str, Any]:
    args = _parse_tool_arguments(arguments_raw)
    return {
        "turn": turn,
        "ok": bool(result.get("ok")),
        "stored": bool(result.get("stored")),
        "candidate_stored": bool(result.get("candidate_stored")),
        "factor_id": result.get("factor_id") or args.get("factor_name"),
        "factor_name": result.get("factor_name") or args.get("factor_name"),
        "comment": result.get("comment") or args.get("comment"),
        "multi_line_expr": args.get("multi_line_expr"),
        "metrics": result.get("metrics"),
        "skipped_reason": result.get("skipped_reason"),
        "error": result.get("error"),
        "error_type": result.get("error_type"),
        "registry_path": result.get("registry_path"),
        "dsl_path": result.get("dsl_path"),
        "similarity": result.get("similarity"),
    }


def _dispatch_parallel(tools: FactorEvalTools, tool_calls: list[Any], *, max_workers: int) -> list[dict[str, Any]]:
    def _one(tc: Any) -> dict[str, Any]:
        t0 = time.perf_counter()
        try:
            out = tools.dispatch(tc.function.name, tc.function.arguments)
        except Exception as e:  # noqa: BLE001
            out = {"ok": False, "error": f"{type(e).__name__}: {e}", "error_type": type(e).__name__}
        return {
            "tool_call_id": tc.id,
            "name": tc.function.name,
            "arguments_raw": tc.function.arguments,
            "result": out,
            "elapsed_seconds": round(time.perf_counter() - t0, 4),
        }

    if len(tool_calls) <= 1 or max_workers <= 1:
        return [_one(tc) for tc in tool_calls]

    workers = min(max_workers, len(tool_calls))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        by_id = {row["tool_call_id"]: row for row in ex.map(_one, tool_calls)}
    return [by_id[tc.id] for tc in tool_calls]


def run_trajectory(
    *,
    client: Any,
    model: str,
    system_prompt: str,
    user_message: str,
    tools: FactorEvalTools,
    log_jsonl: Path,
    max_turns: int = 16,
    max_tool_calls_per_round: int = 8,
    max_tool_workers: int = 4,
    min_tool_call_rounds_before_allow_stop: int = 3,
    temperature: float | None = None,
    max_tokens: int = 8192,
    extra_body: dict[str, Any] | None = None,
    printer: ConsolePrinter | None = None,
) -> list[dict[str, Any]]:
    schemas = tools.schemas()
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
    ]

    log_path = Path(log_jsonl)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    def _emit(event: str, payload: dict[str, Any]) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": _now(), "event": event, **payload}, ensure_ascii=False, default=str) + "\n")

    _emit("session_start", {"model": model, "max_turns": max_turns})
    tool_call_rows: list[dict[str, Any]] = []
    submit_records: list[dict[str, Any]] = []
    tool_call_rounds = 0

    for turn in range(max_turns):
        if printer is not None:
            printer.turn(turn)
        req: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": schemas,
            "tool_choice": "auto",
            "max_tokens": max_tokens,
        }
        if temperature is not None:
            req["temperature"] = temperature
        if extra_body:
            req["extra_body"] = extra_body

        _emit("llm_request", {
            "turn": turn,
            "model": model,
            "message_count": len(messages),
            "tool_count": len(schemas),
        })
        resp = _chat_with_retry(client, req)
        msg = resp.choices[0].message
        reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
        tc_payload = _tool_calls_payload(msg)
        _emit(
            "assistant",
            {"turn": turn, "content": msg.content, "reasoning": reasoning, "tool_calls": tc_payload},
        )

        if printer is not None:
            printer.assistant(msg.content, reasoning)

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": msg.content}
        if tc_payload:
            assistant_msg["tool_calls"] = tc_payload
        messages.append(assistant_msg)

        if not msg.tool_calls:
            if tool_call_rounds < min_tool_call_rounds_before_allow_stop:
                messages.append({"role": "user", "content": _NUDGE})
                _emit("nudge", {"turn": turn, "tool_call_rounds": tool_call_rounds})
                continue
            _emit("session_end", {"turn": turn, "reason": "no_tool_calls"})
            if printer is not None:
                printer.session_end("no_tool_calls", sum(1 for r in tool_call_rows if r.get("ok")), len(tool_call_rows))
            break

        tool_call_rounds += 1
        tcs = list(msg.tool_calls)[:max_tool_calls_per_round]
        rows = _dispatch_parallel(tools, tcs, max_workers=max_tool_workers)
        _emit("tool_results", {"turn": turn, "results": rows})

        for r in rows:
            res = r["result"] if isinstance(r.get("result"), dict) else {}
            if printer is not None:
                printer.tool_result(r["name"], r.get("arguments_raw"), res, r.get("elapsed_seconds"))
            tool_call_rows.append(
                {"turn": turn, "name": r["name"], "elapsed_seconds": r.get("elapsed_seconds"), "ok": res.get("ok")}
            )
            if r["name"] == "submit_factor":
                submit_records.append(_submit_record(turn=turn, arguments_raw=r.get("arguments_raw"), result=res))
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": r["tool_call_id"],
                    "content": tools.result_to_content(res),
                }
            )
    else:
        _emit("session_end", {"turn": max_turns - 1, "reason": "max_turns_reached"})
        if printer is not None:
            printer.session_end("max_turns_reached", sum(1 for r in tool_call_rows if r.get("ok")), len(tool_call_rows))

    snapshot = log_path.with_suffix(".messages.json")
    snapshot.write_text(json.dumps(messages, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    times = [r["elapsed_seconds"] for r in tool_call_rows if r.get("elapsed_seconds") is not None]
    submitted_factors = [r for r in submit_records if r.get("stored")]
    submit_failures = [r for r in submit_records if not r.get("stored")]
    summary = {
        "log_jsonl": str(log_path),
        "tool_calls": {
            "count": len(tool_call_rows),
            "ok": sum(1 for r in tool_call_rows if r.get("ok")),
            "elapsed_seconds_total": round(sum(times), 4) if times else 0.0,
        },
        "submitted_factors": submitted_factors,
        "submit_failures": submit_failures,
        "messages_snapshot": str(snapshot),
    }
    log_path.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    _emit("run_summary", summary)
    return messages
