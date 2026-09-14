"""测试股票代码归一化函数。"""
from __future__ import annotations

from core.instruments import normalize_code


def test_normalize_code_standard():
    assert normalize_code("1") == "000001"
    assert normalize_code(1) == "000001"
    assert normalize_code("600000") == "600000"
    assert normalize_code("000001.SZ") == "000001.SZ"
    assert normalize_code("") == ""
    assert normalize_code(None) == ""


def test_normalize_code_extract():
    assert normalize_code("SH.600000", extract_digits=True) == "600000"
    assert normalize_code("000001.SZ", extract_digits=True) == "000001"
    assert normalize_code("sz000002", extract_digits=True) == "000002"
