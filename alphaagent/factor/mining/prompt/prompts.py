"""股票因子挖掘 system prompt（插件化装配）。

板块清单与注册表见 ``prompt/modules/__init__.py``（DEFAULT_MODULES）；
框架（PromptModule / PromptContext / assemble_system_prompt）见 ``prompt_modules``。

新增板块 = 在 ``prompt/modules/`` 新增一个模块文件并注册一行，不改本文件。
板块启用与否由运行时事实（panel 实际列、research_spec、基本面开关等）决定。
"""

from __future__ import annotations

import logging
from typing import Any

from alphaagent.factor.types import DEFAULT_LABEL_COL
from alphaagent.factor.mining.prompt.prompt_modules import (
    PromptContext,
    assemble_system_prompt,
)
from alphaagent.factor.mining.prompt.modules import DEFAULT_MODULES
# 字段族列清单保持可导入（插件数据覆盖登记用）
from alphaagent.factor.mining.prompt.modules.data_fields import (  # noqa: F401
    EVENT_FACE_PANEL_COLUMNS,
    FF_PANEL_COLUMNS,
    FORECAST_PANEL_COLUMNS,
    HOLDER_PANEL_COLUMNS,
)

logger = logging.getLogger(__name__)


def build_system_prompt(
    *,
    include_operator_catalog: bool = True,
    extra_instructions: str = "",
    label_col: str = DEFAULT_LABEL_COL,
    include_fundamentals: bool = True,
    panel_columns: list[str] | None = None,
    population_max: int = 0,
    research_spec: dict[str, Any] | None = None,
    asset_type: str = "stock",
    focus_facets: list[str] | tuple[str, ...] | None = None,
    prompt_phase: str = "full",
    max_tool_calls_per_round: int = 8,
    model_name: str = "",
) -> str:
    """按模块注册表装配系统提示词；返回最终文本。

    板块启用与否由运行时事实（panel 实际列、基本面开关、种群模式、数据面聚焦、
    用户额外指令、分阶段注入策略、挖掘模型名）决定；装配报告（每模块 on/off +
    字符数 + 占位符残留）由 ``assemble_system_prompt`` 返回，需要时用
    ``build_system_prompt_with_report`` 获取。

    ``prompt_phase`` 控制分阶段动态注入：
    - ``"full"``（默认）：全量装配，向后兼容；
    - ``"explore"``：探索阶段，裁剪冷门算子/IC 形态学/中性化/交付等模块；
    - ``"deepen"``：深耕阶段，恢复全部约束；
    - ``"deliver"``：交付阶段，全量 + 交付模块。

    ``model_name`` 供 S4 model_adaptation 模块消费；空串时该模块不渲染。
    """
    text, _ = build_system_prompt_with_report(
        include_operator_catalog=include_operator_catalog,
        extra_instructions=extra_instructions,
        label_col=label_col,
        include_fundamentals=include_fundamentals,
        panel_columns=panel_columns,
        population_max=population_max,
        research_spec=research_spec,
        asset_type=asset_type,
        focus_facets=focus_facets,
        prompt_phase=prompt_phase,
        max_tool_calls_per_round=max_tool_calls_per_round,
        model_name=model_name,
    )
    return text


def build_system_prompt_with_report(
    *,
    include_operator_catalog: bool = True,
    extra_instructions: str = "",
    label_col: str = DEFAULT_LABEL_COL,
    include_fundamentals: bool = True,
    panel_columns: list[str] | None = None,
    population_max: int = 0,
    research_spec: dict[str, Any] | None = None,
    asset_type: str = "stock",
    focus_facets: list[str] | tuple[str, ...] | None = None,
    prompt_phase: str = "full",
    max_tool_calls_per_round: int = 8,
    model_name: str = "",
) -> tuple[str, list[dict[str, Any]]]:
    """装配系统提示词并返回 ``(text, module_report)``（无模块级可变全局）。"""
    cols = frozenset(panel_columns) if panel_columns is not None else None
    # 消融白名单：research_spec.prompt_policy.field_family_scope（P6a 价量字段族消融）
    pp = (research_spec or {}).get("prompt_policy") or {}
    family_scope_raw = pp.get("field_family_scope")
    family_scope = frozenset(family_scope_raw) if family_scope_raw else None
    ctx = PromptContext(
        label_col=label_col,
        include_operator_catalog=include_operator_catalog,
        include_fundamentals=include_fundamentals,
        panel_columns=cols,
        field_family_scope=family_scope,
        asset_type=asset_type,
        research_spec=research_spec,
        population_max=population_max,
        focus_facets=tuple(focus_facets or ()),
        prompt_phase=prompt_phase,
        max_tool_calls_per_round=max_tool_calls_per_round,
        extra_instructions=extra_instructions or "",
        model_name=model_name or "",
    )

    return assemble_system_prompt(DEFAULT_MODULES, ctx)


# 兼容只读快照：最后一次 build 的装配报告（测试消费）。不再被 build 就地修改，
# 避免并发 run 互相覆盖；新代码请用 build_system_prompt_with_report。
last_assembly_report: list[dict[str, Any]] = []
