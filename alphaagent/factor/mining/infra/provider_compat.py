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

import os
from typing import Any

from agentscope.tool import ToolChoice

from alphaagent.factor.mining.infra.usage_capture import UsageCapturedChatModel


def _tool_choice_auto_enabled() -> bool:
    return os.environ.get("ALPHA_TOOL_CHOICE_AUTO", "1") not in {"0", "false", "False"}


class ProviderSafeChatModel(UsageCapturedChatModel):
    """UsageCapturedChatModel + provider 兼容（tool_choice 归一 / 传输错误重试）。"""

    @classmethod
    def _get_retryable_exceptions(cls) -> tuple[type[Exception], ...]:
        base = super()._get_retryable_exceptions()
        try:  # 懒导入：httpx2 是 openai 客户端的传输层依赖
            import httpx2  # noqa: PLC0415
        except Exception:  # noqa: BLE001
            return base
        return (*base, httpx2.RemoteProtocolError, httpx2.ReadError, httpx2.ConnectError)

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
