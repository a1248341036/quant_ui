# -*- coding: utf-8 -*-
"""rotation_guide prompt 模块测试：默认关闭、开启后渲染（P1）。"""

from __future__ import annotations

from alphaagent.factor.mining.prompt.modules import DEFAULT_MODULES
from alphaagent.factor.mining.prompt.prompt_modules import PromptContext, assemble_system_prompt


def _module():
    for module in DEFAULT_MODULES:
        if module.name == "rotation_guide":
            return module
    raise AssertionError("rotation_guide 未注册到 DEFAULT_MODULES")


def test_default_disabled_renders_nothing() -> None:
    module = _module()
    ctx = PromptContext(research_spec={})
    assert module.enabled(ctx) is False
    text, report = assemble_system_prompt(DEFAULT_MODULES, ctx)
    assert "行业/风格轮动因子构建引导" not in text
    row = next(r for r in report if r["module"] == "rotation_guide")
    assert row["enabled"] is False and row["chars"] == 0


def test_enabled_renders_rotation_guide() -> None:
    module = _module()
    ctx = PromptContext(research_spec={"rotation_policy": {"enabled": True}})
    assert module.enabled(ctx) is True
    text, report = assemble_system_prompt(DEFAULT_MODULES, ctx)
    assert "行业/风格轮动因子构建引导" in text
    assert "行业内 alpha" in text
    assert "行业暴露" in text
    assert "CS_NEUTRALIZE(信号, $industry_sw_l1)" in text
    row = next(r for r in report if r["module"] == "rotation_guide")
    assert row["enabled"] is True and row["chars"] > 0


def test_order_between_neutralization_and_tool_contracts() -> None:
    names = [module.name for module in DEFAULT_MODULES]
    assert names.index("neutralization_guide") < names.index("rotation_guide") < names.index("tool_contracts")
