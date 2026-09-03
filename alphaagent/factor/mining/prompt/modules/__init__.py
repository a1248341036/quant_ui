# -*- coding: utf-8 -*-
"""提示词模块注册表：DEFAULT_MODULES = 装配系统提示词的有序板块清单。

新增板块 = 新建 `modules/<name>.py`（定义 NAME/TITLE/ORDER/REQUIRED/render[/enabled]）
并在下方列表注册一行；不改框架、不改其它模块。

插件模块可经 ``register_prompt_module`` 在运行时挂载（order 决定插入位置）。
"""

from __future__ import annotations

from alphaagent.factor.mining.prompt.prompt_modules import PromptModule

from . import (
    behavior_rules as _behavior_rules_mod,
    core_identity as _core_identity_mod,
    data_calibration as _data_calibration_mod,
    data_fields as _data_fields_mod,
    delivery_interface as _delivery_interface_mod,
    delivery_submission as _delivery_submission_mod,
    extra_instructions as _extra_mod,
    facet_focus as _facet_focus_mod,
    ic_robustness as _ic_robustness_mod,
    market_mechanisms as _market_mechanisms_mod,
    multi_period as _multi_period_mod,
    neutralization_guide as _neutralization_mod,
    operator_catalog as _operator_catalog_mod,
    population_mode as _population_mod,
    strategy_tracks as _strategy_tracks_mod,
    tool_contracts as _tool_contracts_mod,
    tool_examples as _tool_examples_mod,
)


def _phase_enabled(module, base_enabled):  # noqa: ANN001
    """组合阶段过滤与模块自身 enabled：PHASES 为空 → 全阶段启用。"""
    phases = getattr(module, "PHASES", None)
    if phases is None:
        return base_enabled

    def _check(ctx):  # noqa: ANN001
        if not base_enabled(ctx):
            return False
        phase = getattr(ctx, "prompt_phase", "full")
        # full 始终启用；否则要求当前阶段在 PHASES 集合中
        return phase == "full" or phase in phases

    return _check


def _static(module) -> PromptModule:  # noqa: ANN001
    """RAW 切片型模块（渲染固定文本）。"""
    base = lambda ctx: True  # noqa: E731
    return PromptModule(
        name=module.NAME, title=module.TITLE, order=module.ORDER,
        render=lambda ctx, _m=module: _m.RAW, required=module.REQUIRED,
        enabled=_phase_enabled(module, base),
        sep_before=module.SEP_BEFORE,
    )


def _dynamic(module) -> PromptModule:  # noqa: ANN001
    """render(ctx) 动态渲染型模块。"""
    base_enabled = getattr(module, "enabled", lambda ctx: True)
    return PromptModule(
        name=module.NAME, title=module.TITLE, order=module.ORDER,
        render=module.render,
        enabled=_phase_enabled(module, base_enabled),
        required=module.REQUIRED,
        sep_before=module.SEP_BEFORE,
    )


DEFAULT_MODULES: list[PromptModule] = [
    _static(_core_identity_mod),
    _dynamic(_strategy_tracks_mod),
    _dynamic(_delivery_interface_mod),
    _dynamic(_data_calibration_mod),
    _dynamic(_data_fields_mod),
    _dynamic(_market_mechanisms_mod),
    _static(_multi_period_mod),
    _dynamic(_operator_catalog_mod),
    _static(_neutralization_mod),
    _dynamic(_tool_contracts_mod),
    _static(_ic_robustness_mod),
    _static(_delivery_submission_mod),
    _static(_behavior_rules_mod),
    _dynamic(_tool_examples_mod),
    _dynamic(_population_mod),
    _dynamic(_facet_focus_mod),
    _dynamic(_extra_mod),
]


def register_prompt_module(module: PromptModule, *, replace: bool = False) -> None:
    """运行时挂载一个插件模块（order 决定插入位置；同名默认拒绝，replace=True 覆盖）。"""
    for i, existing in enumerate(DEFAULT_MODULES):
        if existing.name == module.name:
            if not replace:
                raise ValueError(f"prompt_module_already_registered:{module.name}")
            DEFAULT_MODULES[i] = module
            return
    DEFAULT_MODULES.append(module)
    DEFAULT_MODULES.sort(key=lambda m: (m.order, m.name))


def unregister_prompt_module(name: str) -> bool:
    """卸载一个模块；返回是否真的移除。"""
    before = len(DEFAULT_MODULES)
    DEFAULT_MODULES[:] = [m for m in DEFAULT_MODULES if m.name != name]
    return len(DEFAULT_MODULES) < before
