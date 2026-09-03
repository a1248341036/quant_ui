"""提示词模块框架：把系统提示词拆成可按需启用的插件模块。

设计：
- ``PromptModule`` 声明一个提示词板块（名称、顺序、启用条件、渲染函数）；
- ``assemble_system_prompt`` 按 order 稳定排序拼接启用模块的输出；
- 启用与否由 ``PromptContext``（运行时事实：label 列、panel 实际列、research_spec、
  数据面聚焦等）决定，而非静态配置。

新增板块 = 在 ``prompt/modules/`` 定义一个 ``PromptModule`` 并加入
``prompt.modules.DEFAULT_MODULES``，不改框架、不改其它模块。

装配完整性保障：
- 渲染文本中残留未替换的 ``{{VAR}}`` 占位符 → 记入 report 的 ``unresolved`` 字段并
  log warning（历史上 {{EVENT_DISCLOSURE_SECTION}} 残留过空分隔线，靠此拦截）；
- ``required=True`` 的模块渲染为空 → report 标 ``required_empty=True`` 并 log warning。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

# 未替换占位符：{{NAME}}（允许被模块自行清理的 {{OPTIONAL:NAME}} 两段式写法）
_PLACEHOLDER_RE = re.compile(r"\{\{[A-Z_][A-Z0-9_]*\}\}")


@dataclass
class PromptContext:
    """一次挖掘会话的提示词装配上下文（运行时事实，而非配置副本）。"""

    label_col: str = "label_1d_open_to_open"
    include_operator_catalog: bool = True
    include_fundamentals: bool = True
    # panel 加载后的真实列集合；None 表示未知（保持旧静态行为：字段族全注入）
    panel_columns: frozenset[str] | None = None
    # 资产类型文案（'stock'/'etf'）
    asset_type: str = "stock"
    # 每模式研究规范（delivery gates / 数据面聚焦 / prompt_policy 均从这里读）
    research_spec: dict[str, Any] | None = None
    # 种群批量模式候选上限；0 = 未启用
    population_max: int = 0
    # 数据面聚焦（用户多选，与 expressions.FACET_DEFS 对齐）；空 = 未启用
    focus_facets: tuple[str, ...] = ()
    # 分阶段动态注入：full=全量(默认), explore=探索, deepen=深耕, deliver=交付
    prompt_phase: str = "full"
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptModule:
    """一个可挂载的提示词板块。

    - ``order``：装配顺序（小在前）；同序按名称稳定排序；
    - ``sep_before``：与前一模块之间的分隔符（第一个模块忽略）；默认 ``"\\n\\n"``，
      需要 ``---`` 分隔线的板块声明 ``"\\n\\n---\\n\\n"``；
    - ``required``：True 且渲染为空 → 装配报告标 ``required_empty`` 并告警
      （核心语义板块用 required，装饰性板块用 False）；
    - ``enabled``：按运行时事实启停；``render``：返回板块 markdown 文本（纯内容，
      不含前后分隔符）。
    """

    name: str
    title: str
    order: int
    render: Callable[[PromptContext], str]
    enabled: Callable[[PromptContext], bool] = lambda ctx: True
    required: bool = False
    sep_before: str = "\n\n"


def _find_placeholders(text: str) -> list[str]:
    return sorted(set(_PLACEHOLDER_RE.findall(text or "")))


def assemble_system_prompt(modules: list[PromptModule], ctx: PromptContext) -> tuple[str, list[dict[str, Any]]]:
    """按 order 稳定排序拼接启用模块；返回 (文本, 装配报告)。

    拼接：相邻启用模块之间用后一模块的 ``sep_before`` 连接；首模块忽略。
    报告行：{module, title, order, enabled, chars, placeholders, required_empty}。
    """
    parts: list[str] = []
    report: list[dict[str, Any]] = []
    for module in sorted(modules, key=lambda m: (m.order, m.name)):
        on = bool(module.enabled(ctx))
        chars = 0
        placeholders: list[str] = []
        required_empty = False
        if on:
            text = module.render(ctx)
            placeholders = _find_placeholders(text)
            if text.strip():
                if parts:
                    parts.append(module.sep_before)
                parts.append(text.strip())
                chars = len(text)
            elif module.required:
                required_empty = True
        if placeholders:
            logger.warning(
                "prompt module '%s' rendered with unresolved placeholders: %s",
                module.name, ", ".join(placeholders),
            )
        if required_empty:
            logger.warning(
                "required prompt module '%s' rendered empty — 核心语义板块缺失", module.name,
            )
        report.append({
            "module": module.name,
            "title": module.title,
            "order": module.order,
            "enabled": on,
            "chars": chars,
            "placeholders": placeholders,
            "required_empty": required_empty,
        })
    logger.info(
        "system prompt assembled (%d chars): %s",
        sum(m["chars"] for m in report),
        ", ".join(f"{m['module']}={'on' if m['enabled'] else 'off'}({m['chars']})" for m in report),
    )
    return "".join(parts), report


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
