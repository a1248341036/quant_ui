"""预审：拦截"两个裸信号简单加减"的低级因子 + 同质化平滑变体动态熔断。"""
from __future__ import annotations

import re
from typing import Any

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


# ── 同质化平滑变体动态熔断 ──
# 目的：不针对具体信号（如隔夜/vwap），动态识别"同一信号根结构 + 套平滑"的连续重复，
# 达阈值即拦截评估，强迫 LLM 换信号根结构。阈值由 research_spec.homogenization_policy 控制。
#
# 信号根指纹 = 表达式中所有算子去掉平滑/截面归一/基础运算后剩下的核心信号算子
#              + 引用的 $ 字段集合（去掉平滑/归一内部包裹的字段）。
# 例如 vwap 反转 `dev = DIVIDE(SUBTRACT($adj_close, $adj_vwap), $adj_vwap); sm = WMA(dev, 6); CS_ZSCORE(sm)`
#   → 信号根指纹 = (frozenset(), frozenset({"$adj_close", "$adj_vwap"}))
#      （DIVIDE/SUBTRACT 是基础运算剔除，WMA 是平滑剔除，CS_ZSCORE 是归一剔除；
#       核心算子为空但字段集合区分了 vwap 反转 vs 隔夜因子）
# 隔夜因子 `overnight = DIVIDE(SUBTRACT($adj_open, DELAY($adj_close,1)), DELAY($adj_close,1)); sm = WMA(overnight, 6); CS_ZSCORE(sm)`
#   → 信号根指纹 = (frozenset(), frozenset({"$adj_open", "$adj_close"}))
# 两者字段集合不同 → 不同信号根，不会互相误判；但同族参数变体（换窗口）字段相同 → 熔断。

# 平滑算子（与 dsl/core/ast.py SMOOTHING_OPS 同源）
_SMOOTHING_OPS = frozenset({
    "EMA", "WMA", "TS_MEAN", "TS_MEDIAN", "TS_DECAY_LINEAR", "SMA",
})
# 截面归一/中性化（不算信号算子）
_NORMALIZE_OPS = frozenset({
    "RANK", "CS_RANK", "CS_ZSCORE", "CS_WINSORIZE", "CS_DEMEAN",
    "CS_QUANTILE", "CS_NEUTRALIZE", "ZSCORE", "WINSORIZE",
    "DEMEAN", "QUANTILE", "NORMALIZE", "CS_RESIDUALIZE", "CS_GROUP_RANK",
})
# 基础运算（不算信号算子）
_BASIC_OPS = frozenset({
    "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "NEG", "ABS", "LOG", "EXP",
    "MAX", "MIN", "MAXIMUM", "MINIMUM", "DELAY", "FILLNA",
})

_OP_NAME_RE = re.compile(r"\b([A-Z_][A-Z0-9_]+)\s*\(")
_FIELD_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")

# 辅助字段（中性化/残差化/分组用，不是信号源）——从信号根指纹剔除，避免因加不同中性化键
# 而误判为不同信号根。如 vwap 反转套 CS_RESIDUALIZE(..., LOG($float_cap)) 不应因 $float_cap
# 而与不带中性化的版本被判不同族。
_AUX_FIELDS = frozenset({
    "$float_cap", "$amount", "$industry_sw_l1", "$industry_sw_l2",
    "$industry_sw_l3", "$industry_em",
})


def _signal_fingerprint(expr: str) -> tuple[frozenset[str], frozenset[str]]:
    """提取表达式的信号根指纹（核心信号算子集合, 信号字段集合）。

    用于同质化检测：两个表达式信号根指纹相同 + 都含平滑 → 同质化变体。
    返回 (signal_ops, fields) 二元组，便于精确比较。
    - signal_ops: 去掉平滑/归一/基础运算后的核心算子（如 CHIP_MASS_ASYM/DIVERGENCE_RANK）
    - fields: 表达式引用的 $ 字段集合，剔除辅助字段（$float_cap/$amount/$industry_* 等中性化键）
      保留信号源字段（如 $adj_vwap/$adj_open/$adj_close/$volume/$ret）
    两者都为空 → 纯平滑/归一组合（无信号根），不参与熔断。
    """
    if not expr:
        return (frozenset(), frozenset())
    upper = expr.upper()
    ops = set(_OP_NAME_RE.findall(upper))
    signal_ops = ops - _SMOOTHING_OPS - _NORMALIZE_OPS - _BASIC_OPS
    fields = frozenset(f"${m.lower()}" for m in _FIELD_RE.findall(expr)) - _AUX_FIELDS
    return (frozenset(signal_ops), fields)


# 兼容旧调用名（_record_eval_signature 已改用 _signal_fingerprint）
def _signal_ops_fingerprint(expr: str) -> frozenset[str]:
    """旧接口：只返回信号算子集合（不含字段）。已弃用，保留兼容。"""
    return _signal_fingerprint(expr)[0]


def _has_smoothing(expr: str) -> bool:
    """表达式是否含平滑算子。"""
    if not expr:
        return False
    ops = set(_OP_NAME_RE.findall(expr.upper()))
    return bool(ops & _SMOOTHING_OPS)


def _homogenization_block(
    expr: str,
    recent_evals: list[dict[str, Any]] | None,
    *,
    max_consecutive: int = 3,
    enabled: bool = True,
) -> dict[str, Any] | None:
    """同质化平滑变体预检：连续 ≥max_consecutive 次"同一信号根+含平滑"→ 拦截评估。

    判定逻辑（动态，不针对具体信号）：
    1. 当前表达式含平滑算子；
    2. 提取当前表达式的信号根指纹（核心算子集合 + 引用字段集合）；
    3. 从最近评估倒序遍历，统计连续"信号根指纹相同 + 含平滑"的次数；
    4. 连续次数 ≥ max_consecutive → 返回拦截 result（ok=False）。

    信号根指纹 = (核心信号算子, 引用字段)。vwap 反转和隔夜因子字段不同 → 不会互相误判；
    同族参数变体（换 WMA 窗口）字段相同 → 熔断。

    与 AntiHomogenizationDiagnostic 的区别：
    - 诊断层熔断器要求换手>0.50（只拦高换手同质化），且只 block_submit（仍允许评估）；
    - 本预检在 dispatch 层评估前拦截，不依赖换手，直接拒绝评估同质化变体，
      强迫 LLM 换信号根结构（不是换平滑窗口）。

    参数：
        expr: 当前表达式
        recent_evals: 最近评估签名列表（_record_eval_signature 记录的 dict）
        max_consecutive: 连续同质化次数阈值（默认 3）
        enabled: 开关（research_spec.homogenization_policy.enabled）
    """
    if not enabled or not expr or not recent_evals:
        return None
    if not _has_smoothing(expr):
        return None
    current_sig = _signal_fingerprint(expr)
    # 信号根为空（无核心算子且无字段）→ 不参与熔断，避免误伤
    if not current_sig[0] and not current_sig[1]:
        return None

    consecutive = 0
    for item in reversed(recent_evals):
        if not isinstance(item, dict):
            break
        if not item.get("has_smoothing"):
            break
        prev_sig = item.get("signal_fingerprint")
        if isinstance(prev_sig, tuple) and prev_sig == current_sig:
            consecutive += 1
        else:
            break

    if consecutive >= max_consecutive:
        ops_str = ", ".join(sorted(current_sig[0])) or "(无核心算子)"
        fields_str = ", ".join(sorted(current_sig[1])) or "(无字段)"
        return {
            "ok": False,
            "error": (
                f"homogenization_smoothing_block: 已连续 {consecutive} 次在相同信号根"
                f"（算子: {ops_str}; 字段: {fields_str}）上套平滑算子做变体。"
                f"平滑只降换手不改信号本质，继续微调窗口纯属浪费算力。"
                f"请更换信号根结构（换数据源/换核心算子/换机制），不要再用同一信号根+平滑。"
                f"不要再用同一信号族+平滑。"
            ),
            "error_type": "HomogenizationSmoothingBlock",
        }
    return None
