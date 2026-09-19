# -*- coding: utf-8 -*-
"""验证 AST 版信号根指纹的精确性。"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphaagent.factor.mining.tools._prefilter import (
    _ast_signal_fingerprint,
    _signal_fingerprint,
    _has_smoothing,
    _homogenization_block,
)


def test_ast_fingerprint_basic():
    """AST 指纹基础：数字归一、字段保留、平滑统一。"""
    # vwap 反转 + WMA
    e1 = """
dev = SUBTRACT($adj_close, $adj_vwap)
sm = WMA(dev, 5)
CS_ZSCORE(sm)
"""
    # 同结构换窗口 + 换平滑算子（WMA→EMA）
    e2 = """
dev = SUBTRACT($adj_close, $adj_vwap)
sm = EMA(dev, 10)
CS_ZSCORE(sm)
"""
    fp1 = _ast_signal_fingerprint(e1)
    fp2 = _ast_signal_fingerprint(e2)
    print(f"vwap+WMA(5):  {fp1}")
    print(f"vwap+EMA(10): {fp2}")
    assert fp1 == fp2, "同结构换窗口+换平滑算子，AST 指纹应相同"
    print("✓ 同结构换窗口+换平滑算子 → 同指纹")

    # 隔夜因子（字段不同）
    e3 = """
overnight = SUBTRACT($adj_open, DELAY($adj_close, 1))
sm = WMA(overnight, 6)
CS_ZSCORE(sm)
"""
    fp3 = _ast_signal_fingerprint(e3)
    print(f"隔夜+WMA(6):  {fp3}")
    assert fp3 != fp1, "vwap vs 隔夜字段不同，AST 指纹应不同"
    print("✓ vwap vs 隔夜 → 不同指纹")


def test_ast_fingerprint_nesting():
    """AST 精确嵌套：算子集合相同但嵌套不同 → 指纹不同。"""
    # WMA 包 DIVERGENCE
    e1 = """
a = CS_ZSCORE($adj_open)
b = CS_ZSCORE($adj_close)
div = DIVERGENCE_RANK(a, b)
sm = WMA(div, 6)
CS_ZSCORE(sm)
"""
    # DIVERGENCE 包 WMA（正则版会误判同族，AST 能区分）
    e2 = """
a = WMA(CS_ZSCORE($adj_open), 6)
b = CS_ZSCORE($adj_close)
div = DIVERGENCE_RANK(a, b)
CS_ZSCORE(div)
"""
    fp1 = _ast_signal_fingerprint(e1)
    fp2 = _ast_signal_fingerprint(e2)
    print(f"WMA(DIVERG): {fp1}")
    print(f"DIVERG(WMA): {fp2}")
    assert fp1 != fp2, "嵌套结构不同，AST 指纹应不同"
    print("✓ 嵌套不同 → 不同指纹（AST 精确性）")

    # 正则版对比：算子集合相同，会误判
    sig1 = _signal_fingerprint(e1)
    sig2 = _signal_fingerprint(e2)
    print(f"正则版 e1: ops={sorted(sig1[0])}, fields={sorted(sig1[1])}")
    print(f"正则版 e2: ops={sorted(sig2[0])}, fields={sorted(sig2[1])}")
    if sig1 == sig2:
        print("  正则版误判为同族（算子集合相同）→ AST 版能区分")


def test_ast_fingerprint_aux_fields():
    """辅助字段归一：加不同中性化键不影响指纹。"""
    e1 = """
dev = SUBTRACT($adj_close, $adj_vwap)
sm = WMA(dev, 5)
CS_ZSCORE(sm)
"""
    e2 = """
dev = SUBTRACT($adj_close, $adj_vwap)
sm = WMA(dev, 5)
res = CS_RESIDUALIZE(sm, LOG($float_cap))
CS_ZSCORE(res)
"""
    fp1 = _ast_signal_fingerprint(e1)
    fp2 = _ast_signal_fingerprint(e2)
    print(f"无中性化:    {fp1}")
    print(f"加float_cap: {fp2}")
    # 注意：e2 多了 CS_RESIDUALIZE 和 LOG，结构不同，指纹会不同
    # 但 $float_cap 被归一成 AUX，不会因 $float_cap vs $amount 而不同
    assert "AUX" in fp2 or "$float_cap" not in fp2, "辅助字段应被归一"
    print("✓ 辅助字段归一成 AUX")

    # 同样加 CS_RESIDUALIZE 但换辅助字段
    e3 = """
dev = SUBTRACT($adj_close, $adj_vwap)
sm = WMA(dev, 5)
res = CS_RESIDUALIZE(sm, LOG($amount))
CS_ZSCORE(res)
"""
    fp3 = _ast_signal_fingerprint(e3)
    print(f"加amount:    {fp3}")
    assert fp2 == fp3, "换辅助字段（float_cap→amount）指纹应相同"
    print("✓ 换辅助字段 → 同指纹")


def test_homogenization_block_ast():
    """AST 指纹驱动熔断：连续 3 次同结构换窗口 → 拦截。"""
    exprs = [
        """
dev = SUBTRACT($adj_close, $adj_vwap)
sm = WMA(dev, 5)
CS_ZSCORE(sm)
""",
        """
dev = SUBTRACT($adj_close, $adj_vwap)
sm = WMA(dev, 8)
CS_ZSCORE(sm)
""",
        """
dev = SUBTRACT($adj_close, $adj_vwap)
sm = EMA(dev, 10)
CS_ZSCORE(sm)
""",
    ]
    # 验证三个表达式 AST 指纹相同
    fps = [_ast_signal_fingerprint(e) for e in exprs]
    print(f"三个变体 AST 指纹: {fps[0][:120]}")
    assert len(set(fps)) == 1, "三个换窗口+换平滑变体 AST 指纹应相同"

    # 构造 recent：前 3 次同指纹
    recent = [
        {"fingerprint": f"fp{i}", "turnover": 0.3, "has_smoothing": True,
         "signal_fingerprint_ast": fps[0], "signal_fingerprint": _signal_fingerprint(exprs[0])}
        for i in range(3)
    ]
    # 第 4 次同族 → 触发
    block = _homogenization_block(exprs[2], recent, max_consecutive=3, enabled=True)
    assert block is not None, "连续 3 次同 AST 指纹应拦截"
    assert block["error_type"] == "HomogenizationSmoothingBlock"
    print(f"  拦截消息: {block['error'][:200]}...")
    print("✓ AST 指纹驱动熔断生效")


def test_homogenization_block_ast_different_nesting():
    """嵌套不同 → 不熔断（AST 精确性）。"""
    e1 = """
a = CS_ZSCORE($adj_open)
b = CS_ZSCORE($adj_close)
div = DIVERGENCE_RANK(a, b)
sm = WMA(div, 6)
CS_ZSCORE(sm)
"""
    e2 = """
a = WMA(CS_ZSCORE($adj_open), 6)
b = CS_ZSCORE($adj_close)
div = DIVERGENCE_RANK(a, b)
CS_ZSCORE(div)
"""
    fp1 = _ast_signal_fingerprint(e1)
    fp2 = _ast_signal_fingerprint(e2)
    assert fp1 != fp2

    recent = [
        {"fingerprint": "fp1", "turnover": 0.3, "has_smoothing": True,
         "signal_fingerprint_ast": fp1, "signal_fingerprint": _signal_fingerprint(e1)},
        {"fingerprint": "fp2", "turnover": 0.4, "has_smoothing": True,
         "signal_fingerprint_ast": fp1, "signal_fingerprint": _signal_fingerprint(e1)},
        {"fingerprint": "fp3", "turnover": 0.5, "has_smoothing": True,
         "signal_fingerprint_ast": fp1, "signal_fingerprint": _signal_fingerprint(e1)},
    ]
    # e2 嵌套不同 → 不应拦截
    block = _homogenization_block(e2, recent, max_consecutive=3, enabled=True)
    assert block is None, "嵌套不同不应拦截"
    print("✓ 嵌套不同 → 不熔断（AST 精确性）")


if __name__ == "__main__":
    test_ast_fingerprint_basic()
    test_ast_fingerprint_nesting()
    test_ast_fingerprint_aux_fields()
    test_homogenization_block_ast()
    test_homogenization_block_ast_different_nesting()
    print("\n✅ 全部 AST 指纹测试通过")
