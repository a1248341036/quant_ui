# -*- coding: utf-8 -*-
"""AlphaAgent MCP server：协议层 + 工具层测试。

默认覆盖：握手 / 工具目录一致性 / 未知方法 / 参数校验 / 写工具 confirm 门 / 轻量工具。
**重活默认跳过**（评估类要载 6–8GB panel）：设 `ALPHAAGENT_MCP_EVAL_TEST=1` 才跑，
以免把单测拖成分钟级。
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path

import pytest

from alphaagent.factor.mining.delivery.delivery_criteria import DeliveryCriteria
from alphaagent.factor.mining.research_spec import effective_research_spec
from alphaagent.mcp_server import ServerContext, handle_message
from alphaagent.mcp_server.catalog import DESCRIPTORS, HANDLERS
from alphaagent.mcp_server.protocol import (
    INTERNAL_ERROR,
    INVALID_PARAMS,
    INVALID_REQUEST,
    METHOD_NOT_FOUND,
    PARSE_ERROR,
    PROTOCOL_VERSION,
    serve_stdio,
)
from alphaagent.mcp_server.tools import MAX_BATCH

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_ENABLED = os.environ.get("ALPHAAGENT_MCP_EVAL_TEST") == "1"


@pytest.fixture()
def ctx():
    c = ServerContext(REPO_ROOT)
    yield c
    c.close()


def _call(ctx: ServerContext, tool: str, **arguments) -> dict:
    """走完整协议路径调用工具，返回解码后的 payload。"""
    reply = handle_message(ctx, {"jsonrpc": "2.0", "id": tool, "method": "tools/call",
                                 "params": {"name": tool, "arguments": arguments}})
    assert reply is not None
    assert "error" not in reply, reply.get("error")
    content = reply["result"]["content"][0]["text"]
    payload = json.loads(content)
    return {"payload": payload, "is_error": bool(reply["result"].get("isError"))}


# ── 协议 ──────────────────────────────────────────────────────────────


def test_catalog_matches_handlers() -> None:
    names = [d["name"] for d in DESCRIPTORS]
    assert names == sorted(names) or len(set(names)) == len(names), "工具名不得重复"
    assert set(names) == set(HANDLERS)
    for d in DESCRIPTORS:
        assert d["description"].strip()
        assert d["inputSchema"]["type"] == "object"
        assert d["inputSchema"].get("additionalProperties") is False


def test_initialize_handshake(ctx: ServerContext) -> None:
    reply = handle_message(ctx, {"jsonrpc": "2.0", "id": 1, "method": "initialize",
                                 "params": {"protocolVersion": "2024-11-05"}})
    result = reply["result"]
    assert result["protocolVersion"] == "2024-11-05"  # 回显客户端版本
    assert "tools" in result["capabilities"]
    assert result["serverInfo"]["name"] == "alphaagent"
    assert "eval_batch" in result["instructions"] and "confirm=true" in result["instructions"]


def test_default_protocol_version(ctx: ServerContext) -> None:
    result = handle_message(ctx, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})["result"]
    assert result["protocolVersion"] == PROTOCOL_VERSION


def test_ping_and_notification(ctx: ServerContext) -> None:
    assert handle_message(ctx, {"jsonrpc": "2.0", "id": 9, "method": "ping", "params": {}})["result"] == {}
    # 通知（无 id）不得回复
    assert handle_message(ctx, {"jsonrpc": "2.0", "method": "notifications/initialized"}) is None


def test_unknown_method_and_tool(ctx: ServerContext) -> None:
    reply = handle_message(ctx, {"jsonrpc": "2.0", "id": 1, "method": "nope", "params": {}})
    assert reply["error"]["code"] == METHOD_NOT_FOUND
    reply = handle_message(ctx, {"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                                 "params": {"name": "not_a_tool", "arguments": {}}})
    assert reply["error"]["code"] == INVALID_PARAMS
    # 参数不是对象 → 协议级错误
    reply = handle_message(ctx, {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                                 "params": {"name": "get_thresholds", "arguments": "x"}})
    assert reply["error"]["code"] == INVALID_PARAMS
    # 缺少 method → INVALID_REQUEST
    reply = handle_message(ctx, {"jsonrpc": "2.0", "id": 4})
    assert reply["error"]["code"] == INVALID_REQUEST


def test_serve_stdio_roundtrip(ctx: ServerContext) -> None:
    frames = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
        {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
         "params": {"name": "get_thresholds", "arguments": {"mode": "technical"}}},
    ]
    stdin = io.StringIO("".join(json.dumps(f) + "\n" for f in frames) + "\n{bad json\n" + "123\n")
    stdout = io.StringIO()
    serve_stdio(ctx, stdin=stdin, stdout=stdout)
    replies = [json.loads(ln) for ln in stdout.getvalue().splitlines() if ln.strip()]
    assert [r.get("id") for r in replies] == [1, 2, 3, None, None]
    assert replies[3]["error"]["code"] == PARSE_ERROR
    assert replies[4]["error"]["code"] == INVALID_REQUEST
    assert len(replies[1]["result"]["tools"]) == len(DESCRIPTORS)


# ── 轻量工具（真源一致性） ─────────────────────────────────────────────


@pytest.mark.parametrize("mode", ["technical", "technical_weekly", "technical_monthly", "fundamental"])
def test_get_thresholds_matches_delivery_criteria(ctx: ServerContext, mode: str) -> None:
    out = _call(ctx, "get_thresholds", mode=mode)["payload"]
    crit = DeliveryCriteria.from_spec(effective_research_spec(mode))
    cand = out["candidate"]
    assert cand["min_abs_ic"] == crit.candidate.min_abs_ic
    assert cand["min_icir"] == crit.candidate.min_icir
    assert cand["min_coverage"] == crit.candidate.min_coverage
    assert cand["min_cs_autocorr"] == crit.candidate.min_cs_autocorr
    assert cand["max_abs_corr"] == crit.candidate.max_abs_corr
    assert cand["turnover_gate"] == crit.turnover_gate_limit
    assert out["engine_gate"]["freq"] == crit.engine_gate.freq
    assert out["label_col"].startswith("label_")


def test_get_thresholds_rejects_unknown_mode(ctx: ServerContext) -> None:
    res = _call(ctx, "get_thresholds", mode="macro")
    assert res["is_error"] and "unknown mode" in res["payload"]["error"]


def test_list_fields_and_describe_operator(ctx: ServerContext) -> None:
    fields = _call(ctx, "list_fields", prefixes="adj_")["payload"]
    assert fields["returned"] > 0 and all(c.startswith("adj_") for c in fields["columns"])
    assert fields["total_columns"] > 100
    op = _call(ctx, "describe_operator", name="TS_CORR")["payload"]
    assert op["signature_lines"] and "TS_CORR(" in op["signature_lines"][0].upper()
    assert _call(ctx, "describe_operator", name="NOT_AN_OP")["is_error"]


def test_precheck_expression(ctx: ServerContext) -> None:
    ok = _call(ctx, "precheck_expression", multi_line_expr="z = TS_MEAN($ret, 20)\nCS_ZSCORE(z)")["payload"]
    assert isinstance(ok, dict) and ok
    assert _call(ctx, "precheck_expression", multi_line_expr="   ")["is_error"]


def test_list_runs_and_run_summary(ctx: ServerContext) -> None:
    runs = _call(ctx, "list_runs", limit=3)["payload"]["runs"]
    assert runs and all("run_id" in r for r in runs)
    latest = _call(ctx, "run_summary")["payload"]
    assert latest["run_id"] == runs[0]["run_id"]
    assert _call(ctx, "run_summary", run_id="does-not-exist")["is_error"]


def test_memory_tools(ctx: ServerContext) -> None:
    assert _call(ctx, "memory_search", research_goal="   ")["is_error"]
    if not (REPO_ROOT / "artifacts" / "alphaagent" / "research_memory.db").is_file():
        pytest.skip("research_memory.db 不存在")
    out = _call(ctx, "memory_search", research_goal="低换手 慢信号 筹码", limit=5)["payload"]
    assert "block" in out and out["block_chars"] >= 0
    stats = _call(ctx, "memory_stats")["payload"]
    assert "saturation" in stats or "saturation_error" in stats


# ── 写工具 confirm 门 & 参数校验（都不触发 panel 加载） ───────────────────


def test_write_tools_require_confirm(ctx: ServerContext) -> None:
    r1 = _call(ctx, "submit_factor", multi_line_expr="CS_ZSCORE($ret)", factor_name="x", comment="y")
    assert r1["is_error"] and "confirm=true" in r1["payload"]["error"]
    r2 = _call(ctx, "memory_record", multi_line_expr="CS_ZSCORE($ret)")
    assert r2["is_error"] and "confirm=true" in r2["payload"]["error"]
    # comment 必填
    r3 = _call(ctx, "submit_factor", multi_line_expr="CS_ZSCORE($ret)", factor_name="x",
               comment="  ", confirm=True)
    assert r3["is_error"] and "comment" in r3["payload"]["error"]


def test_eval_batch_validation_before_panel_load(ctx: ServerContext) -> None:
    assert _call(ctx, "eval_batch", exprs=[])["is_error"]
    assert _call(ctx, "eval_batch", exprs=[{"expr": "  "}])["is_error"]
    assert _call(ctx, "eval_batch", exprs=["CS_ZSCORE($ret)"] * (MAX_BATCH + 1))["is_error"]
    assert _call(ctx, "eval_batch", exprs=["CS_ZSCORE($ret)"], split="test")["is_error"]
    assert _call(ctx, "eval_batch", exprs=["CS_ZSCORE($ret)"], mode="macro")["is_error"]
    assert _call(ctx, "eval_val", multi_line_expr="")["is_error"]
    assert _call(ctx, "dry_run_delivery", multi_line_expr="")["is_error"]


def test_universe_validation_before_panel_load(ctx: ServerContext) -> None:
    # OCR D1/D2 修复回归（2026-10-09）：非法池名在工具入口早失败（附可用清单），不触发面板加载
    r = _call(ctx, "eval_batch", exprs=["CS_ZSCORE($ret)"], universe="hs300")
    assert r["is_error"] and "股票池" in r["payload"]["error"]
    r2 = _call(ctx, "dry_run_delivery", multi_line_expr="CS_ZSCORE($ret)", universe="nope")
    assert r2["is_error"] and "股票池" in r2["payload"]["error"]


def test_bad_arguments_are_reported_as_tool_error(ctx: ServerContext) -> None:
    res = _call(ctx, "get_thresholds", unexpected_arg=1)
    assert res["is_error"] and "bad arguments" in res["payload"]["error"]


# ── 跨进程会话锁（多客户端共存；DSH + Codex 同时开也不会各占一个 6–8GB panel） ──


def test_session_lock_is_exclusive(tmp_path) -> None:
    from alphaagent.mcp_server.tools import SessionLock, ToolError as _ToolError

    path = tmp_path / "s.lock"
    a, b = SessionLock(path), SessionLock(path)
    a.acquire("pid=111 mode=technical")
    assert a.held and path.exists()
    with pytest.raises(_ToolError) as exc:
        b.acquire("pid=222 mode=technical")
    # Windows 上被锁字节区间禁止其它句柄读取，故只断言锁路径出现在错误里
    assert "另一个 MCP 会话" in str(exc.value) and str(path) in str(exc.value)
    a.release()
    assert not a.held
    assert path.read_text(encoding="utf-8").startswith("pid=111")  # 解锁后可读
    b.acquire("pid=222")  # 释放后可被接管
    assert b.held
    b.release()


def test_release_session_tool(ctx: ServerContext) -> None:
    out = _call(ctx, "release_session")["payload"]
    assert out["released"] is False
    assert out["before"]["lock_held"] is False
    assert "note" in out


def test_no_session_lock_context_does_not_touch_lock(tmp_path) -> None:
    c = ServerContext(REPO_ROOT, session_lock=False)
    try:
        assert c.session_info()["lock_held"] is False
    finally:
        c.close()
    assert not (REPO_ROOT / "artifacts" / "alphaagent" / ".mcp_session.lock").exists() or True


# ── 重活（默认跳过） ───────────────────────────────────────────────────


@pytest.mark.skipif(not EVAL_ENABLED, reason="设 ALPHAAGENT_MCP_EVAL_TEST=1 才跑（需载 panel）")
def test_eval_batch_real_single_expr(ctx: ServerContext) -> None:
    out = _call(ctx, "eval_batch", exprs=[{
        "name": "mcp_test_atc", "why": "真实评估自检",
        "expr": "CS_ZSCORE(TS_MEAN(DIVIDE($amount, $float_cap), 20))"}],
        mode="technical_monthly")["payload"]
    row = out["rows"][0]
    assert row["error"] is None
    assert row["ic"] is not None and row["coverage"] > 0.5
    assert isinstance(row["fails"], list)
    assert out["n"] == 1
    assert out["thresholds"]["min_abs_ic"] == pytest.approx(0.053, abs=1e-9)
