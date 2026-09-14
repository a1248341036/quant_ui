"""测试数值处理工具。"""
from __future__ import annotations

import math
from core.numutil import to_float


def test_to_float_valid():
    assert to_float(1) == 1.0
    assert to_float(3.14) == 3.14
    assert to_float("123.45") == 123.45
    assert to_float("-0.5") == -0.5


def test_to_float_invalid_and_nan():
    assert to_float(None) is None
    assert to_float("invalid") is None
    assert to_float(float("nan")) is None
    assert to_float(float("inf")) is None
    assert to_float(float("-inf")) is None
    assert to_float("nan", default=0.0) == 0.0
    assert to_float(None, default=-1.0) == -1.0
