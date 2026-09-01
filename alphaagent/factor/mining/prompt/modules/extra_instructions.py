# -*- coding: utf-8 -*-
"""模块 14 · extra_instructions：用户额外指令（启动消息透传，非空时挂载）。"""

NAME = "extra_instructions"
TITLE = "用户额外指令"
ORDER = 140
REQUIRED = False
SEP_BEFORE = "\n\n"


def enabled(ctx) -> bool:  # noqa: ANN001
    return bool(ctx.extra.get("extra_instructions", "").strip())


def render(ctx) -> str:  # noqa: ANN001
    return ctx.extra.get("extra_instructions", "").strip()
