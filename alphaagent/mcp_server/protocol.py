# -*- coding: utf-8 -*-
"""MCP over stdio：换行分隔的 JSON-RPC 2.0，手写（与 `CNEquity/.../mcp_server/protocol.py` 同构）。

**为什么不用 `mcp` SDK**：与 CNEquity 同一理由——本地 stdio server 只需要 `initialize` /
`tools/list` / `tools/call` 三个方法，SDK 会带来 cryptography/pyjwt/opentelemetry 等一串传递依赖；
本项目已有 httpx 栈，不值得再引一套。若将来需要 sampling / roots / HTTP transport 再换 SDK。

**stdout 是线路**：任何 stray `print` 都会污染协议流，`serve_stdio` 启动前把日志指向 stderr，
除 `_send` 外不向 stdout 写任何东西。
"""
from __future__ import annotations

import json
import logging
import sys
from typing import Any, TextIO

from alphaagent.mcp_server.catalog import DESCRIPTORS, HANDLERS
from alphaagent.mcp_server.tools import ServerContext, ToolError

logger = logging.getLogger(__name__)

SERVER_NAME = "alphaagent"
PROTOCOL_VERSION = "2025-06-18"

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603

_INSTRUCTIONS = (
    "AlphaAgent A股因子挖掘/评估/交付工具面（与真实 run 同源的服务层与门槛）。"
    "推荐流程：① get_thresholds 看本档生效门槛；② list_fields + describe_operator 确认可用字段/算子；"
    "③ memory_search 查历史死路与成功骨架，避免重复探索；④ precheck_expression 静态预检；"
    "⑤ eval_batch 批量评估（一次可评数十个表达式，返回逐门 PASS/FAIL）；"
    "⑥ 过门后 dry_run_delivery（只读预检，含库内相关性/stage_one/stage_two）再 submit_factor（写，需 confirm=true）；"
    "⑦ 探过的因子用 memory_record 写回研究记忆（写，需 confirm=true），否则系统记忆看不见直驱探索。"
    "注意：换手指标按交付调仓频率重算，勿跨档比较；session/panel 由服务器内部串行复用。"
)


def _server_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    for pkg in ("alphaagent", "quant-ui"):
        try:
            return version(pkg)
        except PackageNotFoundError:
            continue
    return "0"


def _text_result(payload: Any, *, is_error: bool = False) -> dict:
    return {
        "content": [{"type": "text", "text": json.dumps(payload, ensure_ascii=False, default=str)}],
        "isError": is_error,
    }


def dispatch(ctx: ServerContext, method: str, params: dict) -> dict:
    """处理一个请求方法；协议级故障抛 `_RpcError`。"""
    if method == "initialize":
        return {
            "protocolVersion": params.get("protocolVersion") or PROTOCOL_VERSION,
            "capabilities": {"tools": {}},
            "serverInfo": {"name": SERVER_NAME, "version": _server_version()},
            "instructions": _INSTRUCTIONS,
        }
    if method == "ping":
        return {}
    if method == "tools/list":
        return {"tools": DESCRIPTORS}
    if method == "tools/call":
        return _call_tool(ctx, params)
    raise _RpcError(METHOD_NOT_FOUND, f"unknown method {method!r}")


def _call_tool(ctx: ServerContext, params: dict) -> dict:
    name = params.get("name")
    handler = HANDLERS.get(name)
    if handler is None:
        raise _RpcError(INVALID_PARAMS, f"unknown tool {name!r}")

    arguments = params.get("arguments") or {}
    if not isinstance(arguments, dict):
        raise _RpcError(INVALID_PARAMS, "arguments must be an object")

    # 工具失败以 result(isError=true) 返回而非 JSON-RPC error：这样 agent 能读到
    # 具体原因（如 "confirm=true"）并自我修正，而不是被客户端当成传输故障吞掉。
    try:
        return _text_result(handler(ctx, **arguments))
    except ToolError as exc:
        return _text_result({"error": str(exc)}, is_error=True)
    except TypeError as exc:
        return _text_result({"error": f"bad arguments for {name}: {exc}"}, is_error=True)
    except Exception as exc:  # noqa: BLE001 — 单次坏调用不应终结会话
        logger.exception("tool %s failed", name)
        return _text_result({"error": f"{type(exc).__name__}: {exc}"}, is_error=True)


class _RpcError(Exception):
    def __init__(self, code: int, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def handle_message(ctx: ServerContext, message: dict) -> dict | None:
    """把一条已解码消息变成响应；通知（无 id）返回 None。"""
    request_id = message.get("id")
    method = message.get("method")
    if not isinstance(method, str):
        if request_id is None:
            return None
        return {"jsonrpc": "2.0", "id": request_id,
                "error": {"code": INVALID_REQUEST, "message": "missing method"}}

    params = message.get("params") or {}
    if not isinstance(params, dict):
        params = {}
    if request_id is None:  # notifications/initialized 等：规范禁止回复
        return None
    try:
        return {"jsonrpc": "2.0", "id": request_id, "result": dispatch(ctx, method, params)}
    except _RpcError as exc:
        return {"jsonrpc": "2.0", "id": request_id,
                "error": {"code": exc.code, "message": exc.message}}
    except Exception as exc:  # noqa: BLE001 — 保持会话存活
        logger.exception("dispatch failed for %s", method)
        return {"jsonrpc": "2.0", "id": request_id,
                "error": {"code": INTERNAL_ERROR, "message": f"{type(exc).__name__}: {exc}"}}


def _send(out: TextIO, message: Any) -> None:
    out.write(json.dumps(message, ensure_ascii=False, default=str) + "\n")
    out.flush()


def serve_stdio(
    ctx: ServerContext,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> None:
    """从 stdin 读请求、往 stdout 写响应，直到 EOF。"""
    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout

    for line in stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except json.JSONDecodeError as exc:
            _send(stdout, {"jsonrpc": "2.0", "id": None,
                           "error": {"code": PARSE_ERROR, "message": str(exc)}})
            continue
        if isinstance(message, list):  # 批量请求
            replies = [r for r in (handle_message(ctx, m) for m in message) if r is not None]
            if replies:
                _send(stdout, replies)
            continue
        if not isinstance(message, dict):
            _send(stdout, {"jsonrpc": "2.0", "id": None,
                           "error": {"code": INVALID_REQUEST, "message": "message must be an object"}})
            continue
        reply = handle_message(ctx, message)
        if reply is not None:
            _send(stdout, reply)
    ctx.close()
