"""预审：拦截"两个裸信号简单加减"的低级因子。"""
from __future__ import annotations

import re

# 检测顶层 ADD/SUBTRACT(RANK(x), RANK(y)) 且 x/y 均为简单 TS_/$field 变换（无结构化交互算子）。
# 例外：如果某个操作数本身含 CS_RESIDUALIZE / CS_NEUTRALIZE / GATED_SIGNAL / CS_GROUP_RANK /
# DIVERGENCE_RANK / PIECEWISE_STATE / IF_THEN_ELSE / TS_CORR / TS_RANKCORR 等结构化算子，则放行。

_STRUCTURED_OPS = frozenset({
    "CS_RESIDUALIZE", "CS_NEUTRALIZE", "GATED_SIGNAL", "CS_GROUP_RANK",
    "DIVERGENCE_RANK", "PIECEWISE_STATE", "IF_THEN_ELSE",
    "TS_CORR", "TS_RANKCORR", "TS_COV", "MUTUAL_INFO_LAG",
})

_SIMPLE_TS_RE = re.compile(r"\bTS_[A-Z]+\s*\(")
_RANK_RE = re.compile(r"\bRANK\s*\(")
_CS_RANK_RE = re.compile(r"\bCS_RANK\s*\(")
_CS_ZSCORE_RE = re.compile(r"\bCS_ZSCORE\s*\(")
_ZSCORE_RE = re.compile(r"\bZSCORE\s*\(")


def _is_naive_signal_addition(expr: str) -> bool:
    """检测表达式是否为"两个裸信号简单加减"的低级因子。

    判定逻辑：
    1. 顶层（最后一行）的算子是 ADD 或 SUBTRACT
    2. 整个表达式中至少 2 个截面标准化调用（RANK/CS_RANK/CS_ZSCORE/ZSCORE）
    3. 整个表达式中不含结构化交互算子（GATED_SIGNAL / CS_RESIDUALIZE 等）
    → 返回 True 表示应拦截

    对带赋值的多行表达式，检查顶层是否 ADD/SUBTRACT，但信号计数覆盖所有行。
    """
    if not expr or not expr.strip():
        return False

    full_text = expr.strip()
    full_upper = full_text.upper()

    # 取最后一行（因子值行），跳过注释行
    lines = full_text.split("\n")
    last_line = lines[-1].strip()
    while last_line.startswith("#") and lines:
        lines = lines[:-1]
        if not lines:
            return False
        last_line = lines[-1].strip()

    # 去掉前导赋值 "var = ..."
    if "=" in last_line and not last_line.upper().startswith(("ADD(", "SUBTRACT(")):
        parts = last_line.split("=", 1)
        if len(parts) == 2:
            last_line = parts[1].strip()

    # 检查顶层是否 ADD( 或 SUBTRACT(
    last_upper = last_line.upper()
    if not (last_upper.startswith("ADD(") or last_upper.startswith("SUBTRACT(")):
        return False

    # 在整个表达式中检查截面标准化调用数量
    # RANK / CS_RANK / CS_ZSCORE / ZSCORE 都算"裸信号"标记
    signal_count = (
        len(_RANK_RE.findall(full_upper))
        + len(_CS_RANK_RE.findall(full_upper))
        + len(_CS_ZSCORE_RE.findall(full_upper))
        + len(_ZSCORE_RE.findall(full_upper))
    )
    if signal_count < 2:
        return False

    # 检查是否含结构化交互算子——含则放行
    for op in _STRUCTURED_OPS:
        if op in full_upper:
            return False

    return True
