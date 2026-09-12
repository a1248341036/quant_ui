# -*- coding: utf-8 -*-
"""provider 兼容层回归：tool_choice 归一 + 流中断传输错误重试。

背景（2026-09-10 deepseek-v4.1-flash 实测）：
- ContextConfig 压缩走 generate_structured_output → 强制 tool_choice=<工具名>，
  deepseek 思考模式 400 拒绝（agentscope 有 400 后降级 auto 的兜底，但每次压缩
  都白撞一次 400）；
- CC Switch 代理流式断连抛 httpx2.RemoteProtocolError，不在 OpenAIChatModel
  可重试集合里 → 整个挖掘 run 被判 error 终止（turn 0 未完即死，白烧 130 万
  input token）。
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx2
import pytest
from agentscope.model import OpenAIChatModel
from agentscope.tool import ToolChoice

from alphaagent.factor.mining.infra.provider_compat import (
    ProviderSafeChatModel,
    _tool_choice_auto_enabled,
)
from alphaagent.factor.mining.infra import usage_capture as usage_capture_mod
from alphaagent.factor.mining.infra.usage_capture import UsageCapturedChatModel

_PATCHES: list[tuple[type, object | None]] = []


def _patch_parent_stub(captured: list) -> None:
    """把 OpenAIChatModel._call_api_with_structured_output 换成捕获桩。"""

    async def stub(self, model_name, messages, structured_model, tool_choice=None, **kwargs):
        captured.append(tool_choice)
        return {"ok": True}

    _PATCHES.append((OpenAIChatModel, OpenAIChatModel.__dict__.get("_call_api_with_structured_output")))
    OpenAIChatModel._call_api_with_structured_output = stub


def teardown_function() -> None:
    while _PATCHES:
        cls, original = _PATCHES.pop()
        if original is not None:
            cls._call_api_with_structured_output = original
        else:
            delattr(cls, "_call_api_with_structured_output")


def _model() -> ProviderSafeChatModel:
    """免网络的模型实例：只测覆写逻辑，不触发 __init__ 的客户端构建。"""
    return object.__new__(ProviderSafeChatModel)


def _run(model, tool_choice):
    return asyncio.run(model._call_api_with_structured_output(
        model_name="m",
        messages=[],
        structured_model={"type": "object"},
        tool_choice=tool_choice,
    ))


class TestStructuredToolChoiceNormalization:
    def test_none_downgraded_to_auto(self):
        captured: list = []
        _patch_parent_stub(captured)
        _run(_model(), None)
        assert captured[-1] is not None and captured[-1].mode == "auto"

    def test_forced_tool_name_downgraded_to_auto(self):
        """复现实测 400 场景：压缩强制 tool_choice=<工具名> → 归一为 auto。"""
        captured: list = []
        _patch_parent_stub(captured)
        _run(_model(), ToolChoice(mode="generate_structured_output"))
        assert captured[-1].mode == "auto"

    def test_required_downgraded_to_auto(self):
        captured: list = []
        _patch_parent_stub(captured)
        _run(_model(), ToolChoice(mode="required"))
        assert captured[-1].mode == "auto"

    def test_auto_passes_through(self):
        captured: list = []
        _patch_parent_stub(captured)
        _run(_model(), ToolChoice(mode="auto"))
        assert captured[-1].mode == "auto"

    def test_kill_switch_passes_forced_choice_through(self, monkeypatch):
        monkeypatch.setenv("ALPHA_TOOL_CHOICE_AUTO", "0")
        assert _tool_choice_auto_enabled() is False
        captured: list = []
        _patch_parent_stub(captured)
        _run(_model(), None)
        assert captured[-1] is None  # 未归一，保持 agentscope 原生行为


class TestTransportRetry:
    def test_retryable_set_includes_stream_transport_errors(self):
        retryable = ProviderSafeChatModel._get_retryable_exceptions()
        for exc in (httpx2.RemoteProtocolError, httpx2.ReadError, httpx2.ConnectError):
            assert exc in retryable

    def test_parent_retryable_kept(self):
        for exc in UsageCapturedChatModel._get_retryable_exceptions():
            assert exc in ProviderSafeChatModel._get_retryable_exceptions()

    def test_remote_protocol_error_is_retryable_instance(self):
        """复现实测死因：流中断异常必须命中可重试集合（否则 run 直接判死）。"""
        err = httpx2.RemoteProtocolError(
            "peer closed connection without sending complete message body"
        )
        assert isinstance(err, ProviderSafeChatModel._get_retryable_exceptions())


class TestStreamInterruptRetry:
    """2026-09-12（run 42254d3990f0）真实死因回归测试。

    agentscope 基类 ``max_retries`` 只重试请求建立；流式响应**中途**断连
    （``httpx2.RemoteProtocolError: incomplete chunked read``）发生在
    ``async for chunk`` 迭代内，旧实现仅在 ``_get_retryable_exceptions`` 把异常
    加入集合（不生效）→ 断流直接冒泡终止整个 run。本组测试锁定新的
    ``_retry_stream`` 行为：断流后重发同一次请求续流。
    """

    def _model(self) -> ProviderSafeChatModel:
        m = object.__new__(ProviderSafeChatModel)
        m.max_retries = 2
        m.retry_delay = 0.0
        m.model = "test-model"
        return m

    def test_interrupt_retries_and_recovers(self, monkeypatch):
        """断流一次 → 重发请求续流，产出全部 chunk，不再抛异常。"""
        retried: list = []
        seen = {"n": 0}

        async def good_stream(elements):
            for e in elements:
                yield e

        async def stub_call(self, model_name, messages, tools=None, tool_choice=None, **kw):
            # 模拟 OpenAIChatModel._call_api：async def 返回 async generator 对象
            # （原实现 return self._parse_stream_response(...)，不 await）——
            # 重发路径本次成功，不再断流。
            retried.append((model_name, messages))
            return good_stream([1, 2, 3])

        monkeypatch.setattr(usage_capture_mod.OpenAIChatModel, "_call_api", stub_call)
        m = self._model()

        async def break_once():
            # 只断一次：第一条迭代前抛异常
            if seen["n"] == 0:
                seen["n"] += 1
                raise httpx2.RemoteProtocolError("incomplete chunked read")
            async for e in good_stream([1, 2, 3]):
                yield e

        async def run():
            out = []
            gen = m._retry_stream("m", ["msg"], None, None, {}, break_once())
            async for chunk in gen:
                out.append(chunk)
            return out

        out = asyncio.run(run())
        assert out == [1, 2, 3]
        assert len(retried) == 1  # 断流后重发了一次请求

    def test_retry_exhausted_raises(self, monkeypatch):
        """每次重发都断流 → 重试耗尽后抛原始异常（不再静默/不再冒泡别的）。"""
        calls = {"n": 0}

        async def always_broken(elements):
            raise httpx2.RemoteProtocolError("always broken")
            yield elements  # pragma: no cover

        async def stub_call(self, model_name, messages, tools=None, tool_choice=None, **kw):
            calls["n"] += 1
            return always_broken([1])

        monkeypatch.setattr(usage_capture_mod.OpenAIChatModel, "_call_api", stub_call)
        m = self._model()

        async def run():
            gen = m._retry_stream(
                "m", ["msg"], None, None, {}, always_broken([1]),
            )
            async for _ in gen:
                pass

        with pytest.raises(httpx2.RemoteProtocolError):
            asyncio.run(run())
        # 初始流断 1 次后，重发 max_retries 次都断流 → 重发路径只走 max_retries 次
        assert calls["n"] == m.max_retries

    def test_non_retryable_raises_without_retry(self, monkeypatch):
        """非传输类异常（如 TypeError）不重试直接抛。"""
        retried = {"n": 0}

        async def bad_type_err(elements):
            raise TypeError("bad arg")
            yield elements  # pragma: no cover

        async def stub_call(self, model_name, messages, tools=None, tool_choice=None, **kw):
            retried["n"] += 1
            return bad_type_err([1])

        monkeypatch.setattr(usage_capture_mod.OpenAIChatModel, "_call_api", stub_call)
        m = self._model()

        async def run():
            async for _ in m._retry_stream("m", ["msg"], None, None, {}, bad_type_err([1])):
                pass

        with pytest.raises(TypeError):
            asyncio.run(run())
        assert retried["n"] == 0  # 未触发重发


class TestBuildModelWiring:
    def test_build_model_returns_provider_safe_model(self):
        from alphaagent.factor.mining.agent.agentscope_run import _build_model

        config = SimpleNamespace(
            model="deepseek/deepseek-v4.1-flash",
            max_tokens=16384,
            temperature=None,
            model_max_retries=2,
            model_retry_delay=0.0,
        )
        model = _build_model(
            config,
            api_key="sk-test",
            base_url="http://127.0.0.1:9/v1",
            extra_body=None,
        )
        assert isinstance(model, ProviderSafeChatModel)
        assert isinstance(model, UsageCapturedChatModel)  # usage 捕获行为保留

    def test_reviewer_model_returns_provider_safe_model(self):
        from alphaagent.factor.mining.agent.factor_reviewer import FactorReviewer

        config = SimpleNamespace(
            model="deepseek/deepseek-v4.1-flash",
            max_tokens=8192,
            temperature=None,
            model_max_retries=2,
            model_retry_delay=0.0,
            research_spec={},
        )
        reviewer = FactorReviewer.__new__(FactorReviewer)
        reviewer.config = config
        reviewer.api_key = "sk-test"
        reviewer.base_url = "http://127.0.0.1:9/v1"
        reviewer.extra_body = None
        reviewer.usage_bridge = None
        assert isinstance(reviewer._model(), ProviderSafeChatModel)
