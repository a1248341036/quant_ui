"""证券代码处理公共工具模块。"""
from __future__ import annotations

import re
from typing import Any

_DIGIT_6_RE = re.compile(r"(\d{6})")


def normalize_code(code: Any, *, extract_digits: bool = False) -> str:
    """股票/证券代码统一归一化。

    - 默认模式：字符串化并补齐 6 位（zfill(6)），如 '1' -> '000001'，'600000.SH' -> '600000.SH'；
    - extract_digits=True：优先提取首个 6 位连续数字，剥离 'SH.000001' 或 '000001.SZ' 前后缀。
    """
    if code is None:
        return ""
    s = str(code).strip()
    if not s:
        return ""
    if extract_digits:
        m = _DIGIT_6_RE.search(s)
        if m:
            return m.group(1)
    # 纯数字时补零，非纯数字（如带后缀的 000001.SZ）直接返回
    if s.isdigit():
        return s.zfill(6)
    return s
