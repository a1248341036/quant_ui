# -*- coding: utf-8 -*-
"""验证同质化平滑变体动态熔断逻辑。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphaagent.factor.mining.tools._prefilter import (
    _signal_fingerprint,
    _signal_ops_fingerprint,
    _has_smoothing,
    _homogenization_block,
)


def test_signal_fingerprint():
    """信号根指纹提取：核心算子 + 引用字段。"""
    # 隔夜因子 + WMA 平滑
    expr1 = """
overnight = DIVIDE(SUBTRACT($adj_open, DELAY($adj_close, 1)), DELAY($adj_close, 1))
sm = WMA(overnight, 6)
CS_ZSCORE(sm)
"""
    ops1, fields1 = _signal_fingerprint(expr1)
    print(f"隔夜+WMA: ops={sorted(ops1)}, fields={sorted(fields1)}")
    assert "$adj_open" in fields1
    assert "$adj_close" in fields1
    assert "WMA" not in ops1
    assert "CS_ZSCORE" not in ops1

    # vwap 反转 + WMA
    expr2 = """
vwap_ma = TS_MEAN($adj_vwap, 5)
dev = DIVIDE(SUBTRACT($adj_close, vwap_ma), vwap_ma)
sm = WMA(dev, 6)
CS_ZSCORE(sm)
"""
    ops2, fields2 = _signal_fingerprint(expr2)
    print(f"vwap+TS_MEAN+WMA: ops={sorted(ops2)}, fields={sorted(fields2)}")
    assert "$adj_vwap" in fields2
    assert "$adj_close" in fields2
    # 隔夜 vs vwap 字段不同
    assert fields1 != fields2, "隔夜和 vwap 字段集合应不同"

    # 筹码因子
    expr3 = """
chip = CHIP_MASS_ASYM($adj_close, $adj_low, $adj_high, $volume, 30, $float_cap)
sm = WMA(chip, 5)
CS_RESIDUALIZE(sm, LOG($float_cap))
"""
    ops3, fields3 = _signal_fingerprint(expr3)
    print(f"筹码+WMA: ops={sorted(ops3)}, fields={sorted(fields3)}")
    assert "CHIP_MASS_ASYM" in ops3
    assert "WMA" not in ops3
    assert "CS_RESIDUALIZE" not in ops3

    # DIVERGENCE_RANK 交互结构
    expr4 = """
overnight = CS_ZSCORE(DIVIDE(SUBTRACT($adj_open, DELAY($adj_close, 1)), DELAY($adj_close, 1)))
intraday = CS_ZSCORE(DIVIDE(SUBTRACT($adj_close, $adj_open), $adj_open))
div = DIVERGENCE_RANK(overnight, intraday)
div_smooth = WMA(div, 6)
CS_RESIDUALIZE(div_smooth, LOG($float_cap))
"""
    ops4, fields4 = _signal_fingerprint(expr4)
    print(f"隔夜+日内+DIVERGENCE+WMA: ops={sorted(ops4)}, fields={sorted(fields4)}")
    assert "DIVERGENCE_RANK" in ops4
    assert "WMA" not in ops4

    print("✓ test_signal_fingerprint passed")


def test_has_smoothing():
    assert _has_smoothing("WMA(x, 5)")
    assert _has_smoothing("TS_MEAN($vwap, 10)")
    assert _has_smoothing("EMA(x, 12)")
    assert not _has_smoothing("RANK(x)")
    assert not _has_smoothing("CS_ZSCORE(x)")
    assert not _has_smoothing("DIVERGENCE_RANK(a, b)")
    print("✓ test_has_smoothing passed")


def test_homogenization_block_triggers_vwap():
    """vwap 反转连续 3 次（换 WMA 窗口）→ 拦截。"""
    exprs = [
        """
vwap_ma = TS_MEAN($adj_vwap, 5)
dev = DIVIDE(SUBTRACT($adj_close, vwap_ma), vwap_ma)
sm = WMA(dev, 5)
CS_ZSCORE(sm)
""",
        """
vwap_ma = TS_MEAN($adj_vwap, 5)
dev = DIVIDE(SUBTRACT($adj_close, vwap_ma), vwap_ma)
sm = WMA(dev, 8)
CS_ZSCORE(sm)
""",
        """
vwap_ma = TS_MEAN($adj_vwap, 5)
dev = DIVIDE(SUBTRACT($adj_close, vwap_ma), vwap_ma)
sm = WMA(dev, 10)
CS_ZSCORE(sm)
""",
    ]
    # 构造 recent_evals：前 3 次同族（换窗口）
    sig = _signal_fingerprint(exprs[0])
    recent = [
        {"fingerprint": f"fp{i}", "turnover": 0.3, "has_smoothing": True, "signal_fingerprint": sig}
        for i in range(3)
    ]
    # 第 4 次同族 → 连续 3 次在 recent 里 → 触发
    block = _homogenization_block(exprs[2], recent, max_consecutive=3, enabled=True)
    assert block is not None, "vwap 反转连续 3 次应拦截"
    assert block["ok"] is False
    assert block["error_type"] == "HomogenizationSmoothingBlock"
    print(f"  拦截消息: {block['error'][:150]}...")
    print("✓ test_homogenization_block_triggers_vwap passed")


def test_homogenization_block_triggers_overnight():
    """隔夜因子连续 3 次（换 WMA 窗口）→ 拦截。"""
    expr = """
overnight = DIVIDE(SUBTRACT($adj_open, DELAY($adj_close, 1)), DELAY($adj_close, 1))
sm = WMA(overnight, 6)
CS_ZSCORE(sm)
"""
    sig = _signal_fingerprint(expr)
    recent = [
        {"fingerprint": f"fp{i}", "turnover": 0.3, "has_smoothing": True, "signal_fingerprint": sig}
        for i in range(3)
    ]
    block = _homogenization_block(expr, recent, max_consecutive=3, enabled=True)
    assert block is not None, "隔夜因子连续 3 次应拦截"
    print(f"  拦截消息: {block['error'][:150]}...")
    print("✓ test_homogenization_block_triggers_overnight passed")


def test_homogenization_block_no_smoothing():
    """不含平滑 → 不拦截。"""
    expr = "RANK(CHIP_MASS_ASYM($adj_close, $adj_low, $adj_high, $volume, 30, $float_cap))"
    recent = [
        {"fingerprint": "fp1", "turnover": 0.3, "has_smoothing": True, "signal_fingerprint": _signal_fingerprint("WMA(CHIP_MASS_ASYM($adj_close,1),5)")},
    ]
    block = _homogenization_block(expr, recent, max_consecutive=3, enabled=True)
    assert block is None, "当前表达式不含平滑，不应拦截"
    print("✓ test_homogenization_block_no_smoothing passed")


def test_homogenization_block_different_signal():
    """不同信号根（vwap vs 隔夜）→ 不拦截。"""
    expr_vwap = """
dev = DIVIDE(SUBTRACT($adj_close, $adj_vwap), $adj_vwap)
sm = WMA(dev, 6)
CS_ZSCORE(sm)
"""
    sig_overnight = _signal_fingerprint("""
overnight = DIVIDE(SUBTRACT($adj_open, DELAY($adj_close, 1)), DELAY($adj_close, 1))
sm = WMA(overnight, 6)
CS_ZSCORE(sm)
""")
    recent = [
        {"fingerprint": f"fp{i}", "turnover": 0.3, "has_smoothing": True, "signal_fingerprint": sig_overnight}
        for i in range(3)
    ]
    block = _homogenization_block(expr_vwap, recent, max_consecutive=3, enabled=True)
    assert block is None, "vwap vs 隔夜信号根不同，不应拦截"
    print("✓ test_homogenization_block_different_signal passed")


def test_homogenization_block_disabled():
    """enabled=False → 不拦截。"""
    expr = """
chip = CHIP_MASS_ASYM($adj_close, $adj_low, $adj_high, $volume, 30, $float_cap)
sm = WMA(chip, 5)
CS_ZSCORE(sm)
"""
    sig = _signal_fingerprint(expr)
    recent = [
        {"fingerprint": f"fp{i}", "turnover": 0.3, "has_smoothing": True, "signal_fingerprint": sig}
        for i in range(3)
    ]
    block = _homogenization_block(expr, recent, max_consecutive=3, enabled=False)
    assert block is None, "disabled 时不应拦截"
    print("✓ test_homogenization_block_disabled passed")


def test_homogenization_block_below_threshold():
    """连续 2 次 < 阈值 3 → 不拦截。"""
    expr = """
chip = CHIP_MASS_ASYM($adj_close, $adj_low, $adj_high, $volume, 30, $float_cap)
sm = WMA(chip, 5)
CS_ZSCORE(sm)
"""
    sig = _signal_fingerprint(expr)
    recent = [
        {"fingerprint": "fp1", "turnover": 0.3, "has_smoothing": True, "signal_fingerprint": sig},
        {"fingerprint": "fp2", "turnover": 0.4, "has_smoothing": True, "signal_fingerprint": sig},
    ]
    block = _homogenization_block(expr, recent, max_consecutive=3, enabled=True)
    assert block is None, "连续 2 次 < 3，不应拦截"
    print("✓ test_homogenization_block_below_threshold passed")


def test_homogenization_block_broken_by_non_smoothing():
    """中间插入不含平滑的评估 → 连续中断，不拦截。"""
    expr = """
chip = CHIP_MASS_ASYM($adj_close, $adj_low, $adj_high, $volume, 30, $float_cap)
sm = WMA(chip, 5)
CS_ZSCORE(sm)
"""
    sig = _signal_fingerprint(expr)
    recent = [
        {"fingerprint": "fp1", "turnover": 0.3, "has_smoothing": True, "signal_fingerprint": sig},
        {"fingerprint": "fp2", "turnover": 0.4, "has_smoothing": False, "signal_fingerprint": sig},  # 中断
        {"fingerprint": "fp3", "turnover": 0.5, "has_smoothing": True, "signal_fingerprint": sig},
    ]
    block = _homogenization_block(expr, recent, max_consecutive=3, enabled=True)
    assert block is None, "中间被不含平滑的评估中断，不应拦截"
    print("✓ test_homogenization_block_broken_by_non_smoothing passed")


if __name__ == "__main__":
    test_signal_fingerprint()
    test_has_smoothing()
    test_homogenization_block_triggers_vwap()
    test_homogenization_block_triggers_overnight()
    test_homogenization_block_no_smoothing()
    test_homogenization_block_different_signal()
    test_homogenization_block_disabled()
    test_homogenization_block_below_threshold()
    test_homogenization_block_broken_by_non_smoothing()
    print("\n✅ 全部测试通过")
