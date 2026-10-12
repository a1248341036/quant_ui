# -*- coding: utf-8 -*-
"""ov_memory_search（codex 直驱 OpenViking 冷路径记忆）工具测试。

背景：agentscope 路径 run 启动已注入 OpenViking 跨 session 记忆，但 codex
直驱（MCP）只有 SQLite memory_search。本工具补上 OpenViking 语义检索入口，
让 harness 查记忆阶段能看到跨 session 策略教训 + run 研究总结 + 死路族。
"""
from __future__ import annotations

from alphaagent.mcp_server import ServerContext
from alphaagent.mcp_server import tools
from alphaagent.mcp_server.catalog import DESCRIPTORS, HANDLERS


def _fake_ctx(tmp_path):
    return ServerContext(tmp_path)


class _FakeStore:
    """mock OVStore：记录调用参数，返回固定块。"""

    def __init__(self, lessons="### 跨 session 策略记忆\n- dead_families：vwap 族已饱和",
                 rag="### 研报知识\n- EPA 因子构建方法……"):
        self.lessons = lessons
        self.rag = rag
        self.lesson_calls: list[dict] = []
        self.rag_calls: list[dict] = []

    def retrieve_lessons(self, *, research_mode, query_text, limit, max_chars):
        self.lesson_calls.append(
            {"research_mode": research_mode, "query_text": query_text,
             "limit": limit, "max_chars": max_chars})
        return self.lessons

    def retrieve_report_knowledge(self, *, research_mode, query_text, limit, max_chars):
        self.rag_calls.append(
            {"research_mode": research_mode, "query_text": query_text,
             "limit": limit, "max_chars": max_chars})
        return self.rag


def test_catalog_registered():
    names = [d["name"] for d in DESCRIPTORS]
    assert "ov_memory_search" in names
    assert "ov_memory_search" in HANDLERS
    desc = next(d for d in DESCRIPTORS if d["name"] == "ov_memory_search")
    schema = desc["inputSchema"]
    assert "research_goal" in schema.get("properties", {})
    assert schema.get("required") == []  # 全部可选（research_goal 留空走档位模板）


def test_ov_memory_search_returns_lessons(tmp_path, monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(tools, "_ov_store", lambda: store)
    ctx = _fake_ctx(tmp_path)
    out = tools.ov_memory_search(ctx, research_goal="vwap 族为什么不该再挖",
                                 research_mode="technical")
    assert isinstance(out, dict)
    assert out["block"] == store.lessons


def test_ov_memory_search_passes_query_and_mode(tmp_path, monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(tools, "_ov_store", lambda: store)
    ctx = _fake_ctx(tmp_path)
    out = tools.ov_memory_search(
        ctx, research_goal="vwap 族为什么不该再挖", research_mode="technical_monthly",
        limit=7, max_chars=1200)
    assert out["available"] is True
    assert out["block"] == store.lessons
    assert "vwap" in out["block"]
    assert store.lesson_calls == [{
        "research_mode": "technical_monthly", "query_text": "vwap 族为什么不该再挖",
        "limit": 7, "max_chars": 1200}]
    assert "report_block" not in out  # 默认不查研报


def test_ov_memory_search_include_report_knowledge(tmp_path, monkeypatch):
    store = _FakeStore()
    monkeypatch.setattr(tools, "_ov_store", lambda: store)
    ctx = _fake_ctx(tmp_path)
    out = tools.ov_memory_search(ctx, research_mode="technical", include_report_knowledge=True)
    assert out["report_block"] == store.rag
    assert out["report_block_chars"] == len(store.rag)
    assert store.rag_calls and store.rag_calls[0]["limit"] <= 4


def test_ov_unavailable_degrades_gracefully(tmp_path, monkeypatch):
    def _none():
        return None

    monkeypatch.setattr(tools, "_ov_store", _none)
    ctx = _fake_ctx(tmp_path)
    out = tools.ov_memory_search(ctx, research_goal="anything")
    assert out["available"] is False
    assert out["block"] == ""
    assert "降级" in out["note"]


def test_ov_store_factory_uses_env_endpoint(tmp_path, monkeypatch):
    """_ov_store 优先读 OV_ENDPOINT，缺省 127.0.0.1:1933（与 ov_store 默认一致）。"""
    import alphaagent.factor.mining.memory.ov_store as ovs

    seen = {}

    class _CaptureStore(ovs.OVStore):
        def __init__(self, endpoint="", **kw):
            seen["endpoint"] = endpoint
            # 不真连 OV（懒初始化），直接缓存一个假 client
            self.endpoint = endpoint
            self.inject_max_chars = kw.get("inject_max_chars", 2400)
            self._client = object()

    monkeypatch.setattr(ovs, "OVStore", _CaptureStore)
    monkeypatch.delenv("OV_ENDPOINT", raising=False)
    tools._ov_store()
    assert seen["endpoint"] == "http://127.0.0.1:1933"

    monkeypatch.setenv("OV_ENDPOINT", "http://127.0.0.1:9999")
    tools._ov_store()
    assert seen["endpoint"] == "http://127.0.0.1:9999"
