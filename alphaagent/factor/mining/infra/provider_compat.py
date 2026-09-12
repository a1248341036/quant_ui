"""provider 兼容层：修复第三方/中转 provider 的两类断 run 问题。

2026-09-10 deepseek-v4.1-flash（CC Switch 中转）实测两个问题：

1. **forced tool_choice 被拒（400）**：ContextConfig 上下文压缩走
   ``generate_structured_output``，agentscope 会强制 ``tool_choice=<工具名>``
   （``model/_base.py``：``if tool_choice is None: tool_choice = ToolChoice(mode=func_name)``），
   deepseek 思考模式返回 400 ``Thinking mode does not support this tool_choice``。
   OpenAIChatModel 自带"400 后降级 auto 重试"的兜底，但每次压缩都白撞一次 400
   （多一次往返 + 告警噪音）。本类在发起前就把强制选择归一为 ``auto``。

2. **流式断连杀死整个 run**：CC Switch 代理中断流式响应时，openai 客户端以
   ``httpx2.RemoteProtocolError`` 形态抛出——它不在 ``OpenAIChatModel`` 的
   可重试集合（仅 openai 的 ConnectionError/Timeout/RateLimit/InternalServerError）
   里，于是绕过重试直达 run 循环，``end_reason="error"`` 终止整个挖掘。
   本类把流中断类传输错误加入可重试集合（重试 = 重发同一次 completion 请求，
   半截响应未入上下文，无副作用）。

开关：``ALPHA_TOOL_CHOICE_AUTO=0`` 关闭 tool_choice 归一（排查问题时用）；
传输错误重试恒开（对任何 provider 都是标准语义）。
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, AsyncGenerator

from agentscope.model import ChatResponse
from agentscope.tool import ToolChoice

from alphaagent.factor.mining.infra.usage_capture import UsageCapturedChatModel

logger = logging.getLogger(__name__)

def _tool_choice_auto_enabled() -> bool:
    return os.environ.get("ALPHA_TOOL_CHOICE_AUTO", "1") not in {"0", "false", "False"}


class ProviderSafeChatModel(UsageCapturedChatModel):
    """UsageCapturedChatModel + provider 兼容（tool_choice 归一 / 流式断连重试）。

    2026-09-12 实证（run 42254d3990f0）：agentscope 基类的 ``max_retries`` 重试
    只包裹 ``_call_api`` 的**请求建立**；流式响应**中途**断连（CC Switch 中转
    以 ``httpx2.RemoteProtocolError: incomplete chunked read`` 形态抛出）发生在
    ``async for chunk`` 迭代里，不在重试循环内，直接冒泡终止整个 run。
    本类因此在流式 generator 层实现**真正的断流重试**：迭代中捕获可重试传输异常
    后，用同一套参数重发同一次 completion 请求续流（半截响应未入 agent 上下文，
    重发无副作用——旧实现只把错误加入异常集合而不生效，见 git 记录）。
    """

    @classmethod
    def _get_retryable_exceptions(cls) -> tuple[type[Exception], ...]:
        base = super()._get_retryable_exceptions()
        try:  # 懒导入：httpx2 是 openai 客户端的传输层依赖
            import httpx2  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return base
        return (*base, httpx2.RemoteProtocolError, httpx2.ReadError, httpx2.ConnectError)

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
            return result
        # 流式：包断流重试（agentscope 基类 max_retries 不覆盖流式中途断连）
        return self._retry_stream(
            model_name,
            messages,
            tools=tools,
            tool_choice=tool_choice,
            generate_kwargs=generate_kwargs,
            first=result,
        )

    async def _retry_stream(
        self,
        model_name: str,
        messages: list,
        tools: list[dict] | None,
        tool_choice: Any,
        generate_kwargs: dict[str, Any],
        first: AsyncGenerator[ChatResponse, None],
    ) -> AsyncGenerator[ChatResponse, None]:
        """迭代流式响应，中途断连用同一套参数重发请求续流。

        半截响应被丢弃（未 yield 到 agentscope 累加器，入上下文无副作用）。
        """
        retryable = self._get_retryable_exceptions()
        gen: AsyncGenerator[ChatResponse, None] = first
        for attempt in range(self.max_retries + 1):
            try:
                async for chunk in gen:
                    yield chunk
                return
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001
                if not isinstance(exc, retryable):
                    raise
                if attempt >= self.max_retries:
                    logger.warning(
                        "模型 %s 流式响应断连重试全部失败（%d 次）: %s",
                        self.model, self.max_retries + 1, exc,
                    )
                    raise
                logger.warning(
                    "模型 %s 流式响应断连（第 %d/%d 次重试）: %s —— %.1fs 后重发同一次请求续流",
                    self.model, attempt + 1, self.max_retries + 1, exc, self.retry_delay,
                )
                await asyncio.sleep(self.retry_delay)
                gen = await super()._call_api(
                    model_name,
                    messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    **generate_kwargs,
                )
                if isinstance(gen, ChatResponse):
                    yield gen
                    return

    async def _call_api_with_structured_output(
        self,
        model_name: str,
        messages: list,
        structured_model: Any,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> Any:
        # 归一：deepseek 思考模式等 provider 拒绝 "required"/指定工具名；
        # auto 语义由 system prompt + 工具 schema 保证，压缩调用不受影响。
        if _tool_choice_auto_enabled() and (
            tool_choice is None or getattr(tool_choice, "mode", "auto") not in ("auto", "none")
        ):
            tool_choice = ToolChoice(mode="auto")
        return await super()._call_api_with_structured_output(
            model_name=model_name,
            messages=messages,
            structured_model=structured_model,
            tool_choice=tool_choice,
            **kwargs,
        )
