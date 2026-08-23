"""AgentScope 挖掘会话的终端流式输出（参考 AQRA examples/agentscope/first_agent.py）。"""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TextIO

from agentscope.agent import Agent
from agentscope.event import ConfirmResult, EventType, UserConfirmResultEvent
from agentscope.message import UserMsg

_INDENT = "    "
_USE_COLOR = sys.stdout.isatty()
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _ansi(code: str, text: str) -> str:
    if not _USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def _strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class CliRunLogger:
    """将 CLI 输出写入 run 目录的 cli.log。"""

    def __init__(
        self,
        *,
        cli_log_path: TextIO | None = None,
        on_reply_end: Callable[[], None] | None = None,
    ) -> None:
        self._cli_file = cli_log_path
        self.on_reply_end = on_reply_end

    def write_plain(self, text: str) -> None:
        if self._cli_file is not None:
            self._cli_file.write(text)
            self._cli_file.flush()

    def write_line(self, line: str = "") -> None:
        self.write_plain(line + "\n")

    def close(self) -> None:
        if self._cli_file is not None:
            self._cli_file.close()


def _tag(
    label: str,
    *,
    detail: str = "",
    color: str = "1;36",
    logger: CliRunLogger | None = None,
    stream: TextIO | None = None,
) -> None:
    line = f"[{label}]"
    if detail:
        line = f"{line} {detail}"
    if logger:
        logger.write_line(line)
    out = stream or sys.stdout
    print(_ansi(color, line), file=out, flush=True)


def _prefix_body_delta(delta: str, *, need_leading_indent: bool) -> tuple[str, bool]:
    if not delta:
        return "", need_leading_indent
    out: list[str] = []
    parts = delta.split("\n")
    for i, part in enumerate(parts):
        if i > 0:
            out.append(f"\n{_INDENT}")
        if need_leading_indent or i > 0:
            out.append(_INDENT)
            need_leading_indent = False
        out.append(part)
    return _ansi("2", "".join(out)), need_leading_indent


def _print_body_delta(
    delta: str,
    *,
    need_leading_indent: bool,
    logger: CliRunLogger | None = None,
    stream: TextIO | None = None,
    use_color: bool = True,
) -> bool:
    if use_color:
        chunk, need_leading_indent = _prefix_body_delta(delta, need_leading_indent=need_leading_indent)
    else:
        chunk, need_leading_indent = delta, need_leading_indent
    if logger and chunk:
        logger.write_plain(_strip_ansi(chunk) if use_color else chunk)
    out = stream or sys.stdout
    print(chunk, end="", file=out, flush=True)
    return need_leading_indent


@dataclass
class _PendingToolCall:
    name: str
    arguments: str = ""
    result_parts: list[str] = field(default_factory=list)
    started_at: float = field(default_factory=time.perf_counter)


class MiningStreamObserver:
    """挖掘流式事件观察者：按 tool_call_id 跟踪并行工具调用，并流式落盘 agent 输出。"""

    def __init__(
        self,
        *,
        printer: Any | None = None,
        emit: Callable[[str, dict], None] | None = None,
        turn: int = 0,
    ) -> None:
        self.printer = printer
        self.emit = emit
        self.turn = turn
        self.had_tool_calls = False
        self.tool_call_count = 0
        self._pending: dict[str, _PendingToolCall] = {}
        self._thinking_parts: list[str] = []
        self._text_parts: list[str] = []
        self.completed_texts: list[str] = []
        self._logged_tool_calls: set[str] = set()

    def _emit_block(self, event: str, content: str) -> None:
        if self.emit is None or not content:
            return
        self.emit(event, {"turn": self.turn, "content": content})

    def on_thinking_start(self) -> None:
        self._thinking_parts.clear()

    def on_thinking_delta(self, delta: str) -> None:
        if delta:
            self._thinking_parts.append(delta)

    def on_thinking_end(self) -> None:
        self._emit_block("agent_thinking", "".join(self._thinking_parts))
        self._thinking_parts.clear()

    def on_text_delta(self, delta: str) -> None:
        if delta:
            self._text_parts.append(delta)

    def on_text_end(self) -> None:
        content = "".join(self._text_parts)
        if content:
            self.completed_texts.append(content)
        self._emit_block("assistant_message", content)
        self._text_parts.clear()

    def on_tool_call_start(self, tool_call_id: str, name: str) -> None:
        self.had_tool_calls = True
        self.tool_call_count += 1
        self._pending[tool_call_id] = _PendingToolCall(name=name)
        if self.emit is not None:
            self.emit(
                "tool_call_start",
                {"turn": self.turn, "tool_call_id": tool_call_id, "name": name},
            )

    def on_tool_call_delta(self, tool_call_id: str, delta: str) -> None:
        pending = self._pending.get(tool_call_id)
        if pending is not None and delta:
            pending.arguments += delta

    def on_tool_call_ready(self, tool_call_id: str) -> None:
        """工具参数流式接收完毕、即将执行时落盘 assistant tool_call。"""
        if self.emit is None or tool_call_id in self._logged_tool_calls:
            return
        pending = self._pending.get(tool_call_id)
        if pending is None:
            return
        self._logged_tool_calls.add(tool_call_id)
        self.emit(
            "assistant_tool_call",
            {
                "turn": self.turn,
                "tool_call_id": tool_call_id,
                "name": pending.name,
                "arguments_raw": pending.arguments,
            },
        )

    def on_tool_result_delta(self, tool_call_id: str, delta: str) -> None:
        pending = self._pending.get(tool_call_id)
        if pending is not None and delta:
            pending.result_parts.append(delta)

    def on_tool_result_end(self, tool_call_id: str) -> None:
        pending = self._pending.pop(tool_call_id, None)
        if pending is None:
            return

        elapsed = round(time.perf_counter() - pending.started_at, 4)
        raw = "".join(pending.result_parts)
        try:
            result = json.loads(raw) if raw.strip() else {}
        except json.JSONDecodeError:
            result = {"ok": False, "error": raw[:500]}

        if self.printer is not None:
            self.printer.tool_result(
                pending.name,
                pending.arguments,
                result if isinstance(result, dict) else {},
                elapsed,
            )

        if self.emit is not None:
            self.emit(
                "tool_results",
                {
                    "turn": self.turn,
                    "results": [
                        {
                            "tool_call_id": tool_call_id,
                            "name": pending.name,
                            "arguments_raw": pending.arguments,
                            "result": result,
                            "elapsed_seconds": elapsed,
                        }
                    ],
                },
            )

    def on_model_call_end(self, event: Any) -> None:
        """Persist per-model-call token usage for the UI and run summary."""
        if self.emit is None:
            return
        self.emit(
            "usage",
            {
                "turn": self.turn,
                "input_tokens": int(getattr(event, "input_tokens", 0) or 0),
                "output_tokens": int(getattr(event, "output_tokens", 0) or 0),
                "cache_input_tokens": int(getattr(event, "cache_input_tokens", 0) or 0),
                "cache_creation_input_tokens": int(
                    getattr(event, "cache_creation_input_tokens", 0) or 0
                ),
            },
        )
        # ── 如果本轮 LLM 没有输出任何 thinking 或 text，但有 tool_call，
        #    说明是"直接发 tool_call 无解释"黑盒模式。补一条结构化意图事件。 ──
        if not self._thinking_parts and not self._text_parts and self.had_tool_calls:
            # 从 pending tool calls 构造简要意图说明
            intent_lines = []
            for tc_id, pending in self._pending.items():
                args_raw = pending.arguments
                # 尝试解析 factor_name
                factor_name = "?"
                expr_preview = ""
                try:
                    import json as _json
                    parsed = _json.loads(args_raw)
                    factor_name = parsed.get("factor_name", "?")
                    expr = parsed.get("multi_line_expr", "")
                    # 取第一行有效表达式
                    for line in expr.split("\n"):
                        line = line.strip()
                        if line and not line.startswith("#"):
                            expr_preview = line[:100]
                            break
                except Exception:
                    pass
                intent_lines.append(f"  -> {pending.name}({factor_name}): {expr_preview}")
            content = "LLM 直接发起 tool_call（无 thinking/text 输出）:\n" + "\n".join(intent_lines)
            self.emit("agent_thinking", {"turn": self.turn, "content": content})


async def stream_to_cli(
    agent: Agent,
    user_msg: UserMsg,
    *,
    show_thinking: bool = True,
    auto_confirm: bool = True,
    logger: CliRunLogger | None = None,
    observer: MiningStreamObserver | None = None,
    quiet: bool = False,
    stream: TextIO | None = None,
) -> bool:
    """将 agent.reply_stream 事件流式打印到终端。

    Returns:
        本轮是否发生过至少一次工具调用。
    """
    pending: UserMsg | UserConfirmResultEvent = user_msg
    inject_comment: str | None = None
    body_needs_indent = False
    had_tool_calls = False
    out = stream or sys.stdout
    use_printer = observer is not None and observer.printer is not None

    while True:
        next_input: UserConfirmResultEvent | None = None

        async for event in agent.reply_stream(pending):
            match event.type:
                case EventType.MODEL_CALL_END:
                    if observer is not None:
                        observer.on_model_call_end(event)
                case EventType.THINKING_BLOCK_START:
                    if observer is not None:
                        observer.on_thinking_start()
                    if show_thinking and not quiet:
                        print(file=out, flush=True)
                        if logger:
                            logger.write_line()
                        _tag("思考", color="1;35", logger=logger, stream=out)
                        body_needs_indent = True
                case EventType.THINKING_BLOCK_DELTA:
                    if observer is not None:
                        observer.on_thinking_delta(event.delta)
                    if show_thinking and not quiet:
                        body_needs_indent = _print_body_delta(
                            event.delta,
                            need_leading_indent=body_needs_indent,
                            logger=logger,
                            stream=out,
                        )
                case EventType.THINKING_BLOCK_END:
                    if observer is not None:
                        observer.on_thinking_end()
                    if show_thinking and not quiet:
                        print(file=out, flush=True)
                        if logger:
                            logger.write_line()
                    body_needs_indent = False
                case EventType.TEXT_BLOCK_DELTA:
                    if observer is not None:
                        observer.on_text_delta(event.delta)
                    if not quiet:
                        if logger and event.delta:
                            logger.write_plain(event.delta)
                        print(event.delta, end="", file=out, flush=True)
                case EventType.TEXT_BLOCK_END:
                    if observer is not None:
                        observer.on_text_end()
                    if not quiet:
                        print(file=out, flush=True)
                        if logger:
                            logger.write_line()
                case EventType.TOOL_CALL_START:
                    had_tool_calls = True
                    if observer is not None:
                        observer.on_tool_call_start(event.tool_call_id, event.tool_call_name)
                    elif not quiet:
                        print(file=out, flush=True)
                        if logger:
                            logger.write_line()
                        _tag("工具", detail=event.tool_call_name, color="1;36", logger=logger, stream=out)
                    body_needs_indent = False
                case EventType.TOOL_CALL_DELTA:
                    if observer is not None:
                        observer.on_tool_call_delta(event.tool_call_id, event.delta)
                case EventType.TOOL_RESULT_START:
                    if observer is not None:
                        observer.on_tool_call_ready(event.tool_call_id)
                    if not quiet and not use_printer:
                        _tag("结果", detail=event.tool_call_name, color="1;33", logger=logger, stream=out)
                    body_needs_indent = True
                case EventType.TOOL_RESULT_TEXT_DELTA:
                    if observer is not None:
                        observer.on_tool_result_delta(event.tool_call_id, event.delta)
                    if not quiet and not use_printer:
                        body_needs_indent = _print_body_delta(
                            event.delta,
                            need_leading_indent=body_needs_indent,
                            logger=logger,
                            stream=out,
                        )
                case EventType.TOOL_RESULT_END:
                    if observer is not None:
                        observer.on_tool_result_end(event.tool_call_id)
                    if not quiet and not use_printer:
                        print(file=out, flush=True)
                        if logger:
                            logger.write_line()
                    body_needs_indent = False
                    if inject_comment:
                        agent.state.context.append(UserMsg(name="user", content=inject_comment))
                        inject_comment = None
                case EventType.REQUIRE_USER_CONFIRM:
                    if auto_confirm:
                        next_input = UserConfirmResultEvent(
                            reply_id=event.reply_id,
                            confirm_results=[
                                ConfirmResult(confirmed=True, tool_call=tc)
                                for tc in event.tool_calls
                            ],
                        )
                    elif not quiet:
                        for tool_call in event.tool_calls:
                            print(file=out, flush=True)
                            if logger:
                                logger.write_line()
                            _tag(
                                "待确认",
                                detail=f"{tool_call.name}: {tool_call.input}",
                                color="1;31",
                                logger=logger,
                                stream=out,
                            )
                        answer = input("确认执行? [y/N，可附加说明]: ").strip()
                        if logger:
                            logger.write_line(f"确认执行? [y/N，可附加说明]: {answer}")
                        confirmed = answer.lower() in ("y", "yes", "yes ")
                        inject_comment = None if confirmed or not answer else answer
                        next_input = UserConfirmResultEvent(
                            reply_id=event.reply_id,
                            confirm_results=[
                                ConfirmResult(confirmed=confirmed, tool_call=tc)
                                for tc in event.tool_calls
                            ],
                        )
                case EventType.EXCEED_MAX_ITERS:
                    if not quiet:
                        print(file=out, flush=True)
                        _tag("警告", detail="达到最大迭代次数", color="1;31", logger=logger, stream=out)
                case EventType.REPLY_END:
                    if logger and logger.on_reply_end:
                        logger.on_reply_end()

        if next_input is None:
            break
        pending = next_input

    if observer is not None:
        observer.had_tool_calls = had_tool_calls
    return had_tool_calls
