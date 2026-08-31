"""提示词模块框架：把系统提示词拆成可按需启用的插件模块。

设计：
- ``PromptModule`` 声明一个提示词板块（名称、启用条件、渲染函数）；
- ``assemble_system_prompt`` 按注册顺序拼接启用模块的输出；
- 启用与否由 ``PromptContext``（label 列、基本面开关、panel 实际列等运行时事实）决定。

新增板块只需定义一个 ``PromptModule`` 并加入模块列表，不改框架。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)


@dataclass
class PromptContext:
    """一次挖掘会话的提示词装配上下文（运行时事实，而非配置副本）。"""

    label_col: str = "label_1d_open_to_open"
    include_operator_catalog: bool = True
    include_fundamentals: bool = True
    # panel 加载后的真实列集合；None 表示未知（保持旧静态行为）
    panel_columns: frozenset[str] | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptModule:
    name: str
    title: str
    enabled: Callable[[PromptContext], bool]
    render: Callable[[PromptContext], str]


def assemble_system_prompt(modules: list[PromptModule], ctx: PromptContext) -> str:
    """按序拼接启用模块；返回 (文本, 装配报告)。"""
    parts: list[str] = []
    report: list[dict[str, Any]] = []
    for module in modules:
        on = bool(module.enabled(ctx))
        chars = 0
        if on:
            text = module.render(ctx)
            if text.strip():
                parts.append(text.strip())
                chars = len(text)
        report.append({"module": module.name, "title": module.title,
                       "enabled": on, "chars": chars})
    logger.info(
        "system prompt assembled: %s",
        ", ".join(f"{m['module']}={'on' if m['enabled'] else 'off'}({m['chars']})" for m in report),
    )
    return "\n\n".join(parts), report


def summarize_report(report: list[dict[str, Any]]) -> dict[str, Any]:
    """装配报告摘要：总字符数与启用/停用清单，便于日志与前端展示。"""
    enabled = [m["module"] for m in report if m["enabled"]]
    disabled = [m["module"] for m in report if not m["enabled"]]
    return {
        "total_chars": sum(m["chars"] for m in report),
        "enabled": enabled,
        "disabled": disabled,
        "modules": report,
    }
