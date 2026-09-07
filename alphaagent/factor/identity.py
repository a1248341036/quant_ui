# -*- coding: utf-8 -*-
"""因子统一身份（中台 ID）。

factor_uid 由 factor_name 确定性派生：任何子系统（registry JSON、研究记忆库、
factor_index.db、迁移脚本）都能独立算出同一 uid，无需协调签发、天然幂等。
代价：factor_name 即身份根，改名 = 新身份（与现状 registry 以名为键一致）。

注意：这里内联 canonical_expression（与 evaluation/candidate.py 同语义），
不直接导入——evaluation 包 __init__ 会拉起整个引擎导入链，存在循环导入风险。
"""

from __future__ import annotations

import hashlib
import re

_CANON_WS = re.compile(r"\s+")


def canonical_expression(expr: str) -> str:
    return _CANON_WS.sub("", expr or "")


def factor_uid(factor_name: str) -> str:
    """因子中台 ID：f + sha256("factor:" + name)[:16]，64bit、确定性、跨库一致。"""
    base = str(factor_name or "").strip()
    return "f" + hashlib.sha256(f"factor:{base}".encode("utf-8")).hexdigest()[:16]


def expr_hash(expr: str) -> str:
    """表达式内容指纹（规范化空白后 sha256 前 16 位）。"""
    return hashlib.sha256(canonical_expression(expr).encode("utf-8")).hexdigest()[:16]
