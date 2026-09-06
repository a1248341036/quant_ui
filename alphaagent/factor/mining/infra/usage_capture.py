"""模型层 token usage 捕获：补齐 agentscope 事件边界丢失的缓存命中字段。

背景：挖掘轨迹的 ``usage`` 事件里 ``cache_input_tokens`` 恒为 0。排查结论是
agentscope 的 OpenAIChatModel 已正确把 ``usage.prompt_tokens_details.cached_tokens``
解析成 ``ChatUsage.cache_input_tokens``（流式 + 非流式两条路径都覆盖），但
``Agent`` 构造 ``ModelCallEndEvent`` 时只拷贝了 input/output 两个字段，缓存字段
在事件边界被丢弃 → ``MiningStreamObserver.on_model_call_end`` 的
``getattr(event, "cache_input_tokens", 0)`` 永远拿 0。

修法：用 ``UsageCapturedChatModel`` 包装模型，在 API 调用出口直接截获完整
``ChatUsage``，经 ``UsageBridge`` 一对一递给 observer 补齐 cache 字段。
事件缺字段的兜底路径保留（未来 agentscope 补齐字段后 wrapper 不再是必需）。
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Callable

from agentscope.model import ChatResponse, OpenAIChatModel


class UsageBridge:
    """单次模型调用 → 单个 MODEL_CALL_END 事件的一对一 usage 传递。

    ``record`` 由模型包装器在每次 API 调用出口调用；``pop_latest`` 由 observer
    在收到对应 MODEL_CALL_END 事件时调用。主循环与 Reviewer 各持一个独立
    bridge，避免工具分派期交叉调用互相串号。
    """

    def __init__(self) -> None:
        self._last: Any = None

    def record(self, usage: Any) -> None:
        self._last = usage

    def pop_latest(self) -> Any:
        usage, self._last = self._last, None
        return usage


class UsageCapturedChatModel(OpenAIChatModel):
    """OpenAIChatModel 包装：把每次 API 调用的 ChatUsage 原样递给 listener。

    流式路径在 usage chunk（``include_usage=True`` 的终块）到达时捕获；
    非流式路径在返回的 ChatResponse 上捕获。除此之外不改变任何行为——
    chunk 原样转发，重试语义与父类一致（失败重试不会触发 listener）。
    """

    def __init__(
        self,
        *args: Any,
        usage_listener: Callable[[Any], None] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._usage_listener = usage_listener

    def _notify(self, usage: Any) -> None:
        if usage is not None and self._usage_listener is not None:
            self._usage_listener(usage)

    async def _call_api(
        self,
        model_name: str,
        messages: list,
        tools: list[dict] | None = None,
        tool_choice: Any = None,
        **generate_kwargs: Any,
    ) -> ChatResponse | AsyncGenerator[ChatResponse, None]:
        result = await super()._call_api(
            model_name,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            **generate_kwargs,
        )
        if isinstance(result, ChatResponse):
            self._notify(result.usage)
            return result
        return self._capture_stream(result)

    async def _capture_stream(
        self, gen: AsyncGenerator[ChatResponse, None]
    ) -> AsyncGenerator[ChatResponse, None]:
        async for chunk in gen:
            if getattr(chunk, "usage", None) is not None:
                self._notify(chunk.usage)
            yield chunk
