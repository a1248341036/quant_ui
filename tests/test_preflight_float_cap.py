# 预审 lint：$float_cap 信号用法 vs 合法参数用法（CHIP_*/CROWD_* 参数、分组、比值分母放行）。
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphaagent.factor.mining.agent.agentscope_tools import (  # noqa: E402
    _float_cap_signal_use,
    _preflight_check,
)


def test_chip_cap_param_allowed():
    """CHIP_* 算子的市值参数是合法用法，不再被预审拦截（用户高频误拦场景）。"""
    expr = (
        "peak_loc = CHIP_PEAK_LOC($adj_close, $adj_low, $adj_high, $volume, 60, $float_cap)\n"
        "sm = TS_MEAN(peak_loc, 5)\n"
        "CS_ZSCORE(CS_WINSORIZE(sm, 0.02, 0.98))"
    )
    assert _preflight_check(expr, "chip_peak_loc_sm") is None


def test_crowd_cap_param_allowed():
    expr = "CS_ZSCORE(CROWD_MEAN_RATIO($adj_close, $float_cap, 20, 'high', 0.8))"
    assert _preflight_check(expr, "crowd_mean_ratio_20") is None


def test_log_and_grouping_allowed():
    assert _preflight_check("CS_ZSCORE(LOG($float_cap))", "log_cap") is None
    assert _preflight_check(
        "CS_NEUTRALIZE(TS_MEAN($ret, 20), CS_BUCKET(LOG($float_cap), 10))",
        "mom_size_neutral") is None
    # 分组位置直接用原始 cap（按秩分桶，单调变换不影响分组）也放行
    assert _preflight_check(
        "CS_NEUTRALIZE(TS_MEAN($ret, 20), CS_BUCKET($float_cap, 10))",
        "mom_size_neutral") is None


def test_ratio_denominator_allowed():
    """比值（如 amount/float_cap≈换手）不是市值信号。"""
    assert _preflight_check("CS_ZSCORE(DIVIDE($amount, $float_cap))", "turnover_proxy") is None


def test_signal_use_still_blocked():
    # 裸用
    ok, detail = _float_cap_signal_use("x = $float_cap\nRANK(x)")
    assert ok and "裸用" in detail
    # 四则运算把市值量纲带进信号
    assert _float_cap_signal_use("x = MULTIPLY($amount, $float_cap)\nCS_ZSCORE(x)")[0]
    assert _float_cap_signal_use("x = SUBTRACT($close, $float_cap)\nCS_ZSCORE(x)")[0]
    # 值算子第一参数 = 市值本体（纯市值因子）
    assert _float_cap_signal_use("RANK($float_cap)")[0]
    assert _float_cap_signal_use("CS_ZSCORE(TS_MEAN($float_cap, 20))")[0]
    # 分组/中性化算子的信号位（第一参数）直接用 cap 也拦
    assert _float_cap_signal_use("CS_BUCKET($float_cap, 10)")[0]
    # 有 LOG 在场但另一处裸用 → 仍拦（旧规则全局放行的漏洞）
    assert _float_cap_signal_use(
        "a = LOG($float_cap)\nx = MULTIPLY(a, $float_cap)\nCS_ZSCORE(x)")[0]

    for expr in ("RANK($float_cap)", "x = MULTIPLY($amount, $float_cap)\nCS_ZSCORE(x)"):
        pre = _preflight_check(expr, "pure_size")
        assert pre is not None and pre.get("blocked")
        assert "纯市值因子" in pre["warning"]


def test_no_float_cap_untouched():
    assert _float_cap_signal_use("CS_ZSCORE(TS_MEAN($adj_close, 20))") == (False, "")
    assert _preflight_check("CS_ZSCORE(TS_MEAN($adj_close, 20))", "ma20") is None
