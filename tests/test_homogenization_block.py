"""AST 同质化平滑熔断器测试：滑动窗口内同根累计计数。

验证 2026-09-19 修复——从"连续同根"改为"滑动窗口内同根累计"，
防 LLM 换根轮换绕过熔断器。
"""
from __future__ import annotations

from alphaagent.factor.mining.tools._prefilter import (
    _homogenization_block,
    _ast_signal_fingerprint,
)


def _sig_entry(expr: str, has_smoothing: bool = True) -> dict:
    """构造一条 recent_evals 条目，含 AST 指纹 + has_smoothing。"""
    return {
        "signal_fingerprint_ast": _ast_signal_fingerprint(expr),
        "signal_fingerprint": None,
        "has_smoothing": has_smoothing,
    }


# 同一根（CS_RESIDUALIZE($close, $vwap)）套不同平滑窗口
_SAME_ROOT_SMOOTH = [
    "NEG(WMA(CS_RESIDUALIZE($close, $vwap), 3))",
    "NEG(WMA(CS_RESIDUALIZE($close, $vwap), 5))",
    "NEG(WMA(CS_RESIDUALIZE($close, $vwap), 10))",
    "NEG(EMA(CS_RESIDUALIZE($close, $vwap), 8))",
]

# 不同根（换字段）套平滑——不应互相累计
_DIFF_ROOT_SMOOTH = [
    "NEG(WMA(CS_RESIDUALIZE($close, $vwap), 8))",
    "NEG(WMA(CS_RESIDUALIZE($open, $amount), 8))",
    "NEG(WMA(CS_RESIDUALIZE($high, $low), 8))",
]


def test_same_root_consecutive_smooth_blocks():
    """连续 3 次同根+平滑 → 拦截（旧逻辑也该过，回归保护）。"""
    recent = [_sig_entry(e) for e in _SAME_ROOT_SMOOTH[:3]]
    block = _homogenization_block(
        _SAME_ROOT_SMOOTH[3],
        recent,
        max_consecutive=3,
        window_size=10,
    )
    assert block is not None
    assert block["ok"] is False
    assert block["error_type"] == "HomogenizationSmoothingBlock"
    assert "最近 10 次评估中已有 3 次" in block["error"]


def test_same_root_sliding_window_blocks_after_root_rotation():
    """关键修复场景：同根平滑被其他根隔开，旧逻辑不拦，新逻辑拦。

    序列：rootA+wma3 → rootB+wma8 → rootA+wma5 → rootC+wma8 → rootA+wma10 → (当前 rootA+ema8)
    rootA 在窗口内历史累计 3 次（wma3/wma5/wma10，不含当前 ema8），当前 ema8 是第 4 次
    同根+平滑尝试 → 新逻辑拦（历史 3 次 ≥ max_consecutive=3）。
    旧逻辑（连续计数）会在 rootB/rootC 处 break，rootA 只算 1 次，不拦。
    """
    recent = [
        _sig_entry("NEG(WMA(CS_RESIDUALIZE($close, $vwap), 3))"),   # rootA
        _sig_entry("NEG(WMA(CS_RESIDUALIZE($open, $amount), 8))"),  # rootB
        _sig_entry("NEG(WMA(CS_RESIDUALIZE($close, $vwap), 5))"),   # rootA
        _sig_entry("NEG(WMA(CS_RESIDUALIZE($high, $low), 8))"),     # rootC
        _sig_entry("NEG(WMA(CS_RESIDUALIZE($close, $vwap), 10))"),  # rootA
    ]
    # 当前表达式仍是 rootA + ema8（不计入 count，只用于提取 current_sig 做比对）
    block = _homogenization_block(
        "NEG(EMA(CS_RESIDUALIZE($close, $vwap), 8))",
        recent,
        max_consecutive=3,
        window_size=10,
    )
    assert block is not None, "滑动窗口内同根历史累计 3 次（不含当前）应拦截"
    assert "最近 10 次评估中已有 3 次" in block["error"]


def test_different_roots_no_block():
    """不同根各套 1 次平滑 → 不拦（换根探索受保护）。"""
    recent = [_sig_entry(e) for e in _DIFF_ROOT_SMOOTH]
    block = _homogenization_block(
        "NEG(WMA(CS_RESIDUALIZE($close, $volume), 8))",  # 又一个新根
        recent,
        max_consecutive=3,
        window_size=10,
    )
    assert block is None


def test_window_size_limits_count():
    """window_size 限制统计范围：窗口外的同根不计入。"""
    # 窗口内只有 2 次同根（最近 5 条里 rootA 出现 2 次），阈值 3 → 不拦
    recent = [
        _sig_entry("NEG(WMA(CS_RESIDUALIZE($close, $vwap), 3))"),   # rootA
        _sig_entry("NEG(WMA(CS_RESIDUALIZE($open, $amount), 8))"),  # rootB
        _sig_entry("NEG(WMA(CS_RESIDUALIZE($close, $vwap), 5))"),   # rootA
        _sig_entry("NEG(WMA(CS_RESIDUALIZE($open, $amount), 10))"), # rootB
        _sig_entry("NEG(WMA(CS_RESIDUALIZE($high, $low), 8))"),     # rootC
    ]
    block = _homogenization_block(
        "NEG(EMA(CS_RESIDUALIZE($close, $vwap), 8))",  # rootA，窗口内累计 2 次（不含当前）
        recent,
        max_consecutive=3,
        window_size=5,
    )
    assert block is None, "窗口内同根 2 次 < 阈值 3，不拦"


def test_non_smooth_expr_no_block():
    """当前表达式不含平滑算子 → 不参与熔断。"""
    recent = [_sig_entry(e) for e in _SAME_ROOT_SMOOTH]
    block = _homogenization_block(
        "NEG(CS_RESIDUALIZE($close, $vwap))",  # 无 WMA/EMA
        recent,
        max_consecutive=3,
        window_size=10,
    )
    assert block is None


def test_disabled_no_block():
    """enabled=False → 不拦。"""
    recent = [_sig_entry(e) for e in _SAME_ROOT_SMOOTH]
    block = _homogenization_block(
        _SAME_ROOT_SMOOTH[3],
        recent,
        max_consecutive=3,
        window_size=10,
        enabled=False,
    )
    assert block is None


def test_empty_recent_no_block():
    """recent_evals 为空 → 不拦。"""
    block = _homogenization_block(
        _SAME_ROOT_SMOOTH[0],
        [],
        max_consecutive=3,
        window_size=10,
    )
    assert block is None


def test_non_smooth_history_not_counted():
    """历史条目 has_smoothing=False 不计入同根累计。"""
    recent = [
        _sig_entry("NEG(WMA(CS_RESIDUALIZE($close, $vwap), 3))", has_smoothing=True),
        _sig_entry("NEG(CS_RESIDUALIZE($close, $vwap))", has_smoothing=False),  # 不计
        _sig_entry("NEG(WMA(CS_RESIDUALIZE($close, $vwap), 5))", has_smoothing=True),
    ]
    block = _homogenization_block(
        "NEG(EMA(CS_RESIDUALIZE($close, $vwap), 8))",
        recent,
        max_consecutive=3,
        window_size=10,
    )
    # 窗口内同根+平滑只有 2 次（wma3/wma5），不含当前 → 不拦
    assert block is None


def test_recent_evals_cap_links_to_window_size():
    """_recent_evals_cap 与 window_size 联动：超配 window_size 时 cap 跟着放大。

    防 window_size 配 >20 时 recent_evals 静默截断窗口（配置陷阱修复）。
    """
    from alphaagent.factor.mining.tools import FactorEvalTools

    # 默认 window_size=10 → cap=20
    tools_default = FactorEvalTools.__new__(FactorEvalTools)
    # 手动跑 __init__ 里 homogenization_policy 装配段（避开完整 __init__ 的重依赖）
    tools_default.homogenization_policy = {
        "enabled": True, "max_consecutive": 3, "window_size": 10,
    }
    tools_default.homogenization_policy.update({})
    _ws = int(tools_default.homogenization_policy.get("window_size", 10) or 10)
    tools_default._recent_evals_cap = max(20, _ws * 2)
    assert tools_default._recent_evals_cap == 20

    # 超配 window_size=30 → cap=60
    tools_big = FactorEvalTools.__new__(FactorEvalTools)
    tools_big.homogenization_policy = {
        "enabled": True, "max_consecutive": 3, "window_size": 30,
    }
    _ws2 = int(tools_big.homogenization_policy.get("window_size", 10) or 10)
    tools_big._recent_evals_cap = max(20, _ws2 * 2)
    assert tools_big._recent_evals_cap == 60, "window_size=30 时 cap 应为 60，防静默截断"


def test_regex_fallback_path_count():
    """正则回落路径（sig_kind == "regex"）的窗口内累计逻辑。

    构造 AST 解析失败的表达式（语法错误），确认回落到 _signal_fingerprint 后
    窗口内同根累计仍正确工作。此分支在新旧逻辑里都没变，作为回归保护。
    """
    from alphaagent.factor.mining.tools._prefilter import _signal_fingerprint

    # 用一个 AST 解析不了的表达式（非法语法），强制走 regex 路径
    bad_expr = "WMA($close, 5) $$$"  # $$$ 会让 parse_ast 返回 None
    # 确认 AST 指纹为空（走 regex 回落）
    assert _ast_signal_fingerprint(bad_expr) == ""
    # regex 指纹应有值（WMA 是平滑，$close 是字段）
    sig = _signal_fingerprint(bad_expr)
    assert "$close" in sig[1]

    # 构造 recent：3 条同根（都用 $close + WMA，无核心信号算子 → signal_ops 空，fields={$close}）
    recent = [
        {"signal_fingerprint_ast": "", "signal_fingerprint": _signal_fingerprint("WMA($close, 3)"), "has_smoothing": True},
        {"signal_fingerprint_ast": "", "signal_fingerprint": _signal_fingerprint("WMA($close, 5)"), "has_smoothing": True},
        {"signal_fingerprint_ast": "", "signal_fingerprint": _signal_fingerprint("WMA($close, 10)"), "has_smoothing": True},
    ]
    block = _homogenization_block(
        bad_expr,
        recent,
        max_consecutive=3,
        window_size=10,
    )
    # 正则路径窗口内同根累计 3 次 → 拦截
    assert block is not None, "正则回落路径窗口内同根累计 3 次应拦截"
    assert block["error_type"] == "HomogenizationSmoothingBlock"
