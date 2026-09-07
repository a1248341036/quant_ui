# -*- coding: utf-8 -*-
"""工具错误分类测试：纯文本输出 → error_type 前缀分类（no_ok 桶细分）。"""

from alphaagent.factor.mining.infra.cli_stream import _classify_error_text


def test_interaction_contract_rejected():
    text = (
        "⛔ 交互契约拦截: 检测到结构化交互算子 ['DIVERGENCE_RANK']，但缺少 interaction 契约。\n"
        "建议: 补充 interaction_type...\n(表达式: abc...)"
    )
    assert _classify_error_text(text) == "InteractionContractRejected"


def test_preflight_rejected():
    text = "⛔ 预审拦截: 纯市值因子: 直接使用 $float_cap 无 LOG 变换…"
    assert _classify_error_text(text) == "PreflightRejected"
    assert _classify_error_text("⛔ 纯市值因子: ...") == "PreflightRejected"


def test_duplicate_exact_eval():
    text = "⛔ 重复评估拦截：该表达式与历史条目 xxx 逐字相同…"
    assert _classify_error_text(text) == "DuplicateExactEval"


def test_tool_arguments_validation():
    text = "Input validation failed for tool 'evaluate_factor': 'profile_id' is a required property"
    assert _classify_error_text(text) == "ToolArgumentsError"


def test_eval_timeout():
    text = "评估超时（>900s），算子可能首次 JIT 编译或计算量过大，已自动跳过"
    assert _classify_error_text(text) == "EvalTimeout"


def test_unknown_stays_none():
    # 未分类文本返回 None（保持 no_ok 兜底，暴露新形态）
    assert _classify_error_text("Some brand new failure mode") is None
    assert _classify_error_text("") is None
