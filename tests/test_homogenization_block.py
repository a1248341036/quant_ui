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

    序列：rootA+wma3 → rootB+wma8 → rootA+wma5 → rootC+wma8 → rootA+wma10 → rootA+ema8
    rootA 在窗口内累计 4 次（wma3/wma5/wma10/ema8），虽不连续 → 新逻辑拦。
    旧逻辑（连续计数）会在 rootB/rootC 处 break，rootA 只算 1 次，不拦。
    """
    recent = [
        _sig_entry("NEG(WMA(CS_RESIDUALIZE($close, $vwap), 3))"),   # rootA
        _sig_entry("NEG(WMA(CS_RESIDUALIZE($open, $amount), 8))"),  # rootB
        _sig_entry("NEG(WMA(CS_RESIDUALIZE($close, $vwap), 5))"),   # rootA
        _sig_entry("NEG(WMA(CS_RESIDUALIZE($high, $low), 8))"),     # rootC
        _sig_entry("NEG(WMA(CS_RESIDUALIZE($close, $vwap), 10))"),  # rootA
    ]
    # 当前表达式仍是 rootA + ema8
    block = _homogenization_block(
        "NEG(EMA(CS_RESIDUALIZE($close, $vwap), 8))",
        recent,
        max_consecutive=3,
        window_size=10,
    )
    assert block is not None, "滑动窗口内同根累计 4 次（含当前）应拦截"
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
