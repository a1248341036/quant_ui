# -*- coding: utf-8 -*-
"""2026-09-30 OCR review 修复回归测试：研报判定网关隔离 + 预测方向归一。

覆盖的缺陷（都是"判定链路静默失败"这一类）：
1. ``get_run_gate()`` 原实现返回进程内全局字典**本体** → 任何调用方原地改（判定通过后
   ``gate["phase"] = "diverge"``）都会永久污染全局，后续新题的复现判定被静默跳过。
   现在返回副本。
2. ``set_run_gate`` 仍能整体替换全局内容（不要把隔离改成"写不进去"）。
3. ``normalize_prediction``：方向中性主型（monotonic / monotone / linear）+ ``expected_sign=-1``
   原先恒归一为 ``monotonic_increasing`` → 形态对账拿 decreasing 实际形态去比 increasing，
   正确预测被误判未 confirmed。现在按 sign 落方向。
4. 显式方向不被过度纠正（sign=1 + 显式 decreasing 必须保持 decreasing）。
5. ``_DispatchMixin._auto_val_verify`` 必须有 docstring（原 try/except 插在 docstring 之前，
   使文档字符串沦为死语句、__doc__ 为空）。
"""
from __future__ import annotations

from alphaagent.factor.mining import report_channels
from alphaagent.factor.mining.eval.prediction import normalize_prediction
from alphaagent.factor.mining.tools._dispatch import _DispatchMixin


def test_get_run_gate_returns_copy_not_global_object():
    report_channels.set_run_gate({"phase": "reproduce", "qid": "RQ_1"})
    got = report_channels.get_run_gate()
    got["phase"] = "diverge"  # 模拟判定侧通过后的原地改
    assert report_channels.get_run_gate()["phase"] == "reproduce"
    assert report_channels.get_run_gate() is not got


def test_set_run_gate_still_replaces_contents():
    report_channels.set_run_gate({"phase": "reproduce", "qid": "RQ_1"})
    report_channels.set_run_gate({"phase": "diverge", "qid": "RQ_2"})
    assert report_channels.get_run_gate() == {"phase": "diverge", "qid": "RQ_2"}
    report_channels.set_run_gate(None)
    assert report_channels.get_run_gate() == {}


def _pred(shape, sign, side: str = "high_factor"):
    return {"expected_shape": shape, "expected_strong_side": side, "expected_sign": sign}


def test_directionless_monotone_follows_expected_sign():
    assert normalize_prediction(_pred("monotonic", -1))["expected_shape"] == "monotonic_decreasing"
    assert normalize_prediction(_pred("monotone", -1))["expected_shape"] == "monotonic_decreasing"
    assert normalize_prediction(_pred("linear", -1))["expected_shape"] == "monotonic_decreasing"
    assert normalize_prediction(_pred("monotonic", 1))["expected_shape"] == "monotonic_increasing"
    assert normalize_prediction(_pred("monotonic", "+1"))["expected_shape"] == "monotonic_increasing"


def test_explicit_direction_is_not_overridden_by_sign():
    assert normalize_prediction(_pred("monotonic_decreasing", 1))["expected_shape"] == "monotonic_decreasing"
    assert normalize_prediction(_pred("monotonic_increasing", -1))["expected_shape"] == "monotonic_increasing"


def test_non_monotone_shapes_untouched():
    assert normalize_prediction(_pred("u_shape", -1))["expected_shape"] == "u_shape"
    assert normalize_prediction(_pred("inverted_u", 1))["expected_shape"] == "inverted_u"


def test_auto_val_verify_keeps_docstring():
    doc = _DispatchMixin._auto_val_verify.__doc__ or ""
    assert "样本外验证" in doc, "try/except 插到 docstring 之前会把 docstring 变成死语句"
