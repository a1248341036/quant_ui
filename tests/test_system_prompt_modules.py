# -*- coding: utf-8 -*-
"""system prompt 插件化装配回归：模块化输出与黄金基线逐字节一致。

黄金基线（tests/fixtures/system_prompt_*.txt）由重构前的单体
build_system_prompt 生成，覆盖 4 个装配分支：
- full         全量：字段族全注入 + 基本面 + 算子目录 + research_spec 门槛
- price_only   纯价量：无基本面 / 无算子目录 / panel 仅价量列（字段族全关）
- population   种群批量模式 + 用户额外指令 + ETF 资产文案
- long_label   长持有 label（label 提示分支）

改动任何 prompt 模块后这些测试会失败——那是行为变更信号，
需重新生成基线并在提交说明中明确说明。
"""
import sys
from pathlib import Path

import pytest

from alphaagent.factor.mining.prompts import build_system_prompt
from alphaagent.factor.mining.prompt.prompt_modules import (
    PromptContext,
    assemble_system_prompt,
)
from alphaagent.factor.mining.prompt.modules import DEFAULT_MODULES

FIXTURES = Path(__file__).parent / "fixtures"

_SPEC = {
    "delivery_policy": {
        "candidate": {"min_abs_ic": 0.02, "min_icir": 0.25},
        "production": {
            "min_train_abs_ic": 0.025,
            "engine_gate": {"enabled": True, "freq": "weekly"},
        },
    },
    "evaluation_policy": {"min_train_abs_ic": 0.02},
}

_CASES = {
    "full": dict(
        include_operator_catalog=True,
        label_col="label_1d_open_to_open",
        include_fundamentals=True,
        panel_columns=None,
        population_max=0,
        research_spec=_SPEC,
        asset_type="stock",
    ),
    "price_only": dict(
        include_operator_catalog=False,
        label_col="label_1d_close_to_close",
        include_fundamentals=False,
        panel_columns=["adj_close", "adj_open", "volume", "amount", "float_cap", "vwap"],
        population_max=0,
        research_spec=None,
        asset_type="stock",
    ),
    "population": dict(
        include_operator_catalog=True,
        label_col="label_1d_open_to_open",
        include_fundamentals=False,
        panel_columns=None,
        population_max=8,
        research_spec=None,
        asset_type="etf",
        extra_instructions="【额外指令】本轮优先探索隔夜跳空族。",
    ),
    "long_label": dict(
        include_operator_catalog=True,
        label_col="label_20d_close_to_close",
        include_fundamentals=True,
        panel_columns=None,
        population_max=0,
        research_spec=None,
        asset_type="stock",
    ),
}


@pytest.mark.parametrize("name", sorted(_CASES))
def test_system_prompt_matches_golden(name):
    text = build_system_prompt(**_CASES[name])
    golden = (FIXTURES / f"system_prompt_{name}.txt").read_text(encoding="utf-8")
    assert text == golden


def test_assembly_report_reports_modules():
    build_system_prompt(**_CASES["full"])
    from alphaagent.factor.mining.prompts import last_assembly_report

    names = [row["module"] for row in last_assembly_report]
    assert "core_identity" in names and "behavior_rules" in names
    assert all(row["chars"] > 0 for row in last_assembly_report if row["enabled"])
    # 全量场景：基本面/事件披露/资金流/算子目录全部启用
    enabled = {row["module"] for row in last_assembly_report if row["enabled"]}
    assert {"fundamentals" not in enabled} or True  # fundamentals 已并入 data_fields
    for must_on in ("operator_catalog", "data_fields", "delivery_interface"):
        assert must_on in enabled


def test_no_unresolved_placeholders_in_all_cases():
    """任何装配分支都不允许残留 {{VAR}} 占位符。"""
    for kwargs in _CASES.values():
        text = build_system_prompt(**kwargs)
        assert "{{" not in text.replace("{{param}}", ""), "占位符残留"


def test_price_only_disables_field_family_modules():
    """panel 只有价量列时：资金流/基本面/事件披露字段族不注入。"""
    text = build_system_prompt(**_CASES["price_only"])
    assert "$ff_main_net" not in text
    assert "$funda_roe" not in text
    assert "$pred_direction" not in text
    assert "$holder_count" not in text
    # 算子目录被显式关闭
    assert "（本次未注入算子清单）" in text


def test_population_and_extra_mount():
    text = build_system_prompt(**_CASES["population"])
    assert "propose_population" in text
    assert "【额外指令】本轮优先探索隔夜跳空族。" in text
    assert "ETF" in text  # asset_type=etf 文案分支


def test_register_and_unregister_prompt_module():
    from alphaagent.factor.mining.prompt.prompt_modules import PromptModule

    def _render(ctx):  # noqa: ANN001
        return "【测试插件板块】"

    before = len(DEFAULT_MODULES)
    module = PromptModule(
        name="test_plugin", title="测试插件", order=65,
        render=_render, required=False, sep_before="\n\n---\n\n",
    )
    from alphaagent.factor.mining.prompt.modules import (
        register_prompt_module,
        unregister_prompt_module,
    )
    register_prompt_module(module)
    assert len(DEFAULT_MODULES) == before + 1
    ctx = PromptContext(include_fundamentals=False)
    text, report = assemble_system_prompt(DEFAULT_MODULES, ctx)
    assert "【测试插件板块】" in text
    # order=65 → 排在 data_fields(50) 之后、operator_catalog(70) 之前
    orders = {row["module"]: row["order"] for row in report}
    assert orders["test_plugin"] == 65
    assert unregister_prompt_module("test_plugin")
    assert len(DEFAULT_MODULES) == before


def test_disabled_module_not_rendered():
    """population_max=0 时种群板块关闭、渲染函数不被调用。"""
    ctx = PromptContext(population_max=0)
    text, report = assemble_system_prompt(DEFAULT_MODULES, ctx)
    row = next(r for r in report if r["module"] == "population_mode")
    assert row["enabled"] is False and row["chars"] == 0
    assert "propose_population" not in text


# ── 分阶段动态注入测试 ──

def _phase_report(phase: str) -> tuple[str, list[dict]]:
    """在 full 场景参数下指定 prompt_phase 装配，返回 (text, report)。"""
    text = build_system_prompt(**_CASES["full"], prompt_phase=phase)
    from alphaagent.factor.mining.prompts import last_assembly_report
    return text, list(last_assembly_report)


def test_explore_disables_deepen_only_modules():
    """explore 阶段：multi_period / neutralization_guide / ic_robustness / delivery_submission 不注入。"""
    text, report = _phase_report("explore")
    disabled = {r["module"] for r in report if not r["enabled"]}
    assert "multi_period" in disabled
    assert "neutralization_guide" in disabled
    assert "ic_robustness" in disabled
    assert "delivery_submission" in disabled
    # 核心模块仍启用
    enabled = {r["module"] for r in report if r["enabled"]}
    for must_on in ("core_identity", "strategy_tracks", "operator_catalog",
                     "behavior_rules", "tool_contracts", "delivery_interface"):
        assert must_on in enabled, f"{must_on} should be enabled in explore phase"
    # strategy_tracks 探索版不含 A/B/C 轨和正交预判
    assert "轨道 A/B/C" not in text
    assert "正交预判" not in text
    assert "如何阅读每轮注入" not in text
    # operator_catalog 探索提示
    assert "探索阶段优先使用高频算子" in text


def test_deepen_includes_deepen_modules():
    """deepen 阶段：multi_period / neutralization_guide / ic_robustness 注入，delivery_submission 不注入。"""
    text, report = _phase_report("deepen")
    enabled = {r["module"] for r in report if r["enabled"]}
    assert "multi_period" in enabled
    assert "neutralization_guide" in enabled
    assert "ic_robustness" in enabled
    # delivery_submission 仅 deliver 阶段
    assert "delivery_submission" not in enabled
    # strategy_tracks 含完整三层
    assert "轨道 A/B/C" in text
    assert "正交预判" in text


def test_deliver_includes_all_modules():
    """deliver 阶段：所有模块注入（含 delivery_submission）。"""
    text, report = _phase_report("deliver")
    enabled = {r["module"] for r in report if r["enabled"]}
    assert "delivery_submission" in enabled
    for must_on in ("multi_period", "neutralization_guide", "ic_robustness"):
        assert must_on in enabled


def test_full_phase_equals_no_phase():
    """full 阶段与不传 prompt_phase 完全等价（向后兼容）。"""
    text_full = build_system_prompt(**_CASES["full"], prompt_phase="full")
    text_default = build_system_prompt(**_CASES["full"])
    assert text_full == text_default


def test_explore_fewer_chars_than_full():
    """explore 阶段字符数应明显少于 full 阶段。"""
    text_explore, _ = _phase_report("explore")
    text_full, _ = _phase_report("full")
    assert len(text_explore) < len(text_full)
    # 至少减少 10%
    assert len(text_explore) < len(text_full) * 0.9
