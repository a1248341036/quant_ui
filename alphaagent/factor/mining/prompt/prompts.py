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
) -> str:
    """按模块注册表装配系统提示词；返回最终文本。

    板块启用与否由运行时事实（panel 实际列、基本面开关、种群模式、数据面聚焦、
    用户额外指令）决定；装配报告（每模块 on/off + 字符数 + 占位符残留）写入
    ``last_assembly_report``。
    """
    cols = frozenset(panel_columns) if panel_columns is not None else None
    ctx = PromptContext(
        label_col=label_col,
        include_operator_catalog=include_operator_catalog,
        include_fundamentals=include_fundamentals,
        panel_columns=cols,
        asset_type=asset_type,
        research_spec=research_spec,
        population_max=population_max,
        focus_facets=tuple(focus_facets or ()),
        extra={"extra_instructions": extra_instructions or ""},
    )

    text, module_report = assemble_system_prompt(DEFAULT_MODULES, ctx)

    last_assembly_report.clear()
    last_assembly_report.extend(module_report)
    return text


last_assembly_report: list[dict[str, Any]] = []
