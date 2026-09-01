# -*- coding: utf-8 -*-
"""共享: 从 jq_repro 脚本中提取 CODE = '''...''' 策略代码块。"""


def extract_jq_code(src: str, marker: str = "CODE = '''") -> str:
    m = src.find(marker)
    if m < 0:
        raise ValueError(f'marker not found: {marker!r}')
    s = m + len(marker)
    e = src.find("'''", s)
    if e < 0:
        raise ValueError('closing delimiter not found')
    return src[s:e]
