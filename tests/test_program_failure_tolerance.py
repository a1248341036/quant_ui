# -*- coding: utf-8 -*-
"""程序失败容错三件套回归测试（2026-10-03）。

数据依据（当夜 14 run，eval_error 占全部评估 29.9%）：
  ① `interaction_missing_fields:economic_mechanism(>=20 chars)` ×17 —— 契约机制字段缺失被硬拦
  ② `expected_strong_side` 非法：'high_decile' / 'D10低换手端' / None —— 散文回退失效
  ③ `未定义的名称 'TS_FILL_NAN'` / `AND() takes 2 positional arguments but 3 were given` /
     `表达式引用了不可用字段: $turnover_rate` —— 报错不可行动，模型只能盲试
"""
from __future__ import annotations

import pytest

from alphaagent.dsl.eval import _problem_from_exception, compile_multi_line_factor
from alphaagent.factor.mining.eval.prediction import _canon_side, normalize_prediction
from alphaagent.factor.mining.interactions import lint_expression_interaction

# ══════════════ ① 契约机制自动补齐 ══════════════

_EXPR = "GATED_SIGNAL(CS_ZSCORE($ret), CS_ZSCORE($amount))"
_CONTRACT_NO_MECH = {
    "interaction_type": "gated_signal",
    "base_signal": "动量",
    "condition_signal": "成交额",
}


def test_missing_mechanism_autofilled_not_blocked():
    spec, warn, blocked = lint_expression_interaction(
        _EXPR, dict(_CONTRACT_NO_MECH), policy={"auto_fill_missing_contract": True})
    assert blocked is None, "机制缺失不应再硬拦（与信号补全同口径）"
    assert spec is not None
    assert "自动补全" in spec["economic_mechanism"]
    assert warn and "economic_mechanism" in warn


def test_short_mechanism_also_autofilled():
    short = dict(_CONTRACT_NO_MECH, economic_mechanism="太短")
    spec, warn, blocked = lint_expression_interaction(
        _EXPR, short, policy={"auto_fill_missing_contract": True})
    assert blocked is None and spec is not None
    assert len(spec["economic_mechanism"]) >= 20


def test_switch_off_restores_hard_block():
    _, _, blocked = lint_expression_interaction(
        _EXPR, dict(_CONTRACT_NO_MECH), policy={"auto_fill_missing_contract": False})
    assert blocked is not None
    assert "interaction_missing_fields" in str(blocked.get("error") or "")


def test_valid_mechanism_unchanged():
    good = dict(_CONTRACT_NO_MECH, economic_mechanism="量价齐升反映资金一致做多，门控后信号更纯净")
    spec, warn, blocked = lint_expression_interaction(
        _EXPR, good, policy={"auto_fill_missing_contract": True})
    assert blocked is None and spec is not None
    assert "自动补全" not in spec["economic_mechanism"]
    assert not (warn and "economic_mechanism" in warn)


# ══════════════ ② expected_strong_side 归一化（散文回退修复） ══════════════

@pytest.mark.parametrize("raw,expect", [
    ("high_decile", "high_factor"),
    ("high decile", "high_factor"),
    ("high-decile", "high_factor"),
    ("D10低换手端", "high_factor"),
    ("高因子端(D10=低换手)", "high_factor"),
    ("D1~D3", "low_factor"),
    ("D8-D10", "high_factor"),
    ("D4-D7", "middle"),
    ("neutral", "middle"),
    ("top_decile", "high_factor"),
    ("middle", "middle"),
])
def test_side_canonicalization(raw, expect):
    assert _canon_side(raw) == expect


@pytest.mark.parametrize("raw", ["", "??", None])
def test_illegal_side_still_rejected(raw):
    assert _canon_side(raw) is None


def test_normalize_prediction_accepts_llm_wordings():
    for raw, expect in (("high_decile", "high_factor"), ("D10低换手端", "high_factor")):
        out = normalize_prediction({"expected_shape": "monotonic_increasing",
                                    "expected_strong_side": raw, "expected_sign": 1})
        assert out is not None, f"{raw} 仍被拒"
        assert out["expected_strong_side"] == expect


def test_ambiguous_text_still_rejected():
    """两端都点名且无评级/梯度词 → 仍判歧义（不猜）。"""
    assert _canon_side("D1 vs D10 都存在") is None


# ══════════════ ③ DSL 报错可行动 ══════════════

def test_unknown_name_gets_actionable_hint():
    _, problem = _problem_from_exception(NameError("name 'TS_FILL_NAN' is not defined"))
    assert "未定义的名称" in problem
    assert "算子目录" in problem          # 明确指引，而非只说"缺少函数"
    assert "TS_FILL_NAN" in problem


def test_arity_error_gets_copyable_fix():
    exc = TypeError("AND() takes 2 positional arguments but 3 were given")
    _, problem = _problem_from_exception(exc)
    assert "只接受 2 个参数" in problem and "3" in problem
    assert "嵌套" in problem and "PIECEWISE_STATE" in problem
    assert "正确写法示例: AND($a, $b)" in problem


def test_missing_argument_gets_signature_hint():
    """2026-10-03 实测主因：`TS_ZSCORE() missing 1 required positional argument: 'window'`。

    当次 run eval_error 占 70%，模型反复漏传 window —— 报错必须给出真实签名。
    """
    exc = TypeError("TS_ZSCORE() missing 1 required positional argument: 'window'")
    _, problem = _problem_from_exception(exc)
    assert "缺少必需的参数" in problem and "window" in problem
    assert "正确写法示例: TS_ZSCORE($列, window, ddof=1)" in problem
    assert "窗口周期" in problem


def test_missing_argument_without_known_signature_still_actionable():
    exc = TypeError("NO_SUCH_OP() missing 2 required positional arguments: 'a' and 'b'")
    _, problem = _problem_from_exception(exc)
    assert "缺少必需的参数" in problem and "'a' and 'b'" in problem


def test_unknown_field_suggests_closest_column():
    with pytest.raises(Exception) as ei:
        compile_multi_line_factor("X = $turnover_rate + $close", columns=["turnover_rate_f", "close"])
    msg = str(ei.value)
    assert "不可用字段" in msg
    assert "最相近的可用列" in msg and "turnover_rate_f" in msg
