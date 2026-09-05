"""IC 衰减曲线 + IC 直方图（因子实验室图表数据）回归测试。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from alphaagent.factor.metrics.decay import (
    decay_horizons_for,
    forward_close_to_close_label,
    ic_histogram,
    panel_forward_label,
    rank_ic_decay_summary,
)


def _panel_ref(adj_close: pd.Series, hold_days: int) -> pd.Series:
    from alphaagent.data.panel import _calc_label_nd_close_to_close

    return _calc_label_nd_close_to_close(adj_close, hold_days)


def test_forward_label_matches_panel_convention() -> None:
    """衰减前瞻收益与 panel 构建期 label_{N}d_close_to_close 公式逐位一致。"""
    rng = np.random.default_rng(7)
    adj = pd.Series(rng.uniform(5.0, 50.0, size=64))
    for h in (1, 2, 3, 5, 10, 20, 40):
        mine = forward_close_to_close_label(adj, h)
        ref = _panel_ref(adj, h)
        mine_arr = mine.to_numpy(dtype=float)
        ref_arr = ref.to_numpy(dtype=float)
        np.testing.assert_allclose(mine_arr, ref_arr, equal_nan=True)


def test_panel_forward_label_per_instrument_no_leakage() -> None:
    """MultiIndex panel 上按 instrument 分组 shift，不得跨标的泄漏。"""
    dates = pd.date_range("2024-01-01", periods=6, freq="B")
    idx = pd.MultiIndex.from_product([dates, ["A", "B"]], names=["datetime", "instrument"])
    close = pd.Series(np.arange(12, dtype=float), index=idx)
    fwd = panel_forward_label(close.to_frame("adj_close"), 1)

    a = fwd.xs("A", level="instrument").to_numpy(dtype=float)
    # from_product 交错排列：A 的 close = 0,2,4,6,8,10 → entry/exit = shift(-1)/shift(-2)
    expected_a = np.array([(4.0 - 2.0) / 2.0, (6.0 - 4.0) / 4.0, (8.0 - 6.0) / 6.0,
                           (10.0 - 8.0) / 8.0, np.nan, np.nan])
    np.testing.assert_allclose(a, expected_a, equal_nan=True)

    b = fwd.xs("B", level="instrument").to_numpy(dtype=float)
    # B 的 close = 1,3,5,7,9,11
    expected_b = np.array([(5.0 - 3.0) / 3.0, (7.0 - 5.0) / 5.0, (9.0 - 7.0) / 7.0,
                           (11.0 - 9.0) / 9.0, np.nan, np.nan])
    np.testing.assert_allclose(b, expected_b, equal_nan=True)

    # 行序与 panel 一致（datetime 优先排序的行序）
    assert fwd.index.equals(idx)


def test_decay_horizons_for() -> None:
    assert decay_horizons_for(1) == (1, 2, 3, 5, 10, 20)
    assert decay_horizons_for(0) == (1, 2, 3, 5, 10, 20)
    assert decay_horizons_for(10) == (1, 5, 10, 20, 40)
    assert decay_horizons_for(20) == (5, 10, 20, 40, 60)


def test_rank_ic_decay_summary_matches_cs_ic_summary_semantics() -> None:
    idx = pd.date_range("2024-01-01", periods=10, freq="B")
    s = pd.Series(np.linspace(0.01, 0.10, 10), index=idx)

    full = rank_ic_decay_summary(s, holding_days=1)
    assert full["n_days"] == 10
    assert abs(full["mean_ic"] - float(s.mean())) < 1e-12
    expected_icir = float(s.mean()) / float(s.std(ddof=1))
    assert abs(full["ic_ir"] - expected_icir) < 1e-12

    # 持有期 5 → 每 5 天取一点去重叠：iloc[::5] → 2 点
    sub = rank_ic_decay_summary(s, holding_days=5)
    assert sub["n_days"] == 2
    assert abs(sub["mean_ic"] - float(np.mean([0.01, 0.06]))) < 1e-12

    empty = rank_ic_decay_summary(pd.Series([np.nan] * 5, index=idx[:5]), holding_days=1)
    assert empty["n_days"] == 0
    assert np.isnan(empty["mean_ic"]) and np.isnan(empty["ic_ir"])


def test_ic_histogram_basics() -> None:
    s = pd.Series(np.linspace(-0.1, 0.1, 101))
    hist = ic_histogram(s, bins=20)
    assert sum(hist["counts"]) == 101
    assert len(hist["edges"]) == 21 and len(hist["counts"]) == 20
    assert hist["edges"][0] <= -0.1 and hist["edges"][-1] >= 0.1

    empty = ic_histogram(pd.Series([np.nan, np.nan]))
    assert empty == {"edges": [], "counts": []}
    assert ic_histogram(None) == {"edges": [], "counts": []}


def test_engine_chart_data_contains_decay_and_histogram() -> None:
    """引擎 include_charts=True 输出 ic_decay/ic_histogram，且前瞻收益挂 session 缓存。"""
    from pathlib import Path

    from alphaagent.factor.evaluation.engine import EvaluationEngine
    from alphaagent.factor.evaluation.profile import default_evaluation_profiles
    from alphaagent.factor.mining.context import StockEvalContext
    from alphaagent.factor.mining.session import StockEvalSession

    dates = pd.date_range("2024-01-01", periods=16, freq="B")
    instruments = [f"S{i:03d}" for i in range(40)]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    cross_section = np.tile(np.linspace(-1, 1, len(instruments)), len(dates))
    drift = np.repeat(np.linspace(0, 0.05, len(dates)), len(instruments))
    close = 10 + cross_section + drift
    panel = pd.DataFrame(
        {
            "adj_close": close,
            "open": close * (1 + 0.001 * cross_section),
            "high": close * 1.01,
            "low": close * 0.99,
            "amount": np.tile(np.linspace(2e7, 8e7, len(instruments)), len(dates)),
            "turnover_rate": np.tile(np.linspace(0.5, 3.0, len(instruments)), len(dates)),
            "float_cap": np.tile(np.linspace(1e8, 5e10, len(instruments)), len(dates)),
            "label_1d_open_to_open": 0.002 * cross_section + 0.0001 * drift,
        },
        index=index,
    )
    session = StockEvalSession(
        session_id="test_decay",
        ctx=StockEvalContext(
            panel_path=Path("panel.parquet"),
            train_start="2024-01-01",
            train_end="2024-01-10",
            val_start="2024-01-11",
            val_end="2024-01-31",
        ),
        panel=panel,
    )
    engine = EvaluationEngine(default_evaluation_profiles())

    result = engine.evaluate(
        session, profile_id="train_screen", multi_line_expr="$adj_close",
        include_charts=True,
    )
    assert result["ok"]
    cd = result["chart_data"]
    assert cd["ic_decay"] is not None
    decay = cd["ic_decay"]
    assert decay["label_horizon"] == 1
    assert decay["convention"] == "close_to_close"
    horizons = [p["horizon"] for p in decay["points"]]
    assert horizons == [1, 2, 3, 5, 10, 20]
    # 短 horizon 有有效 IC；序列尾部（h 超出面板天数）n_days=0 且值为 None（JSON 安全）
    assert decay["points"][0]["n_days"] > 0
    assert decay["points"][0]["mean_ic"] is not None
    tail = [p for p in decay["points"] if p["n_days"] == 0]
    assert all(p["mean_ic"] is None and p["ic_ir"] is None for p in tail)

    hist = cd["ic_histogram"]["rank_ic"]
    assert sum(hist["counts"]) > 0
    assert len(hist["edges"]) == len(hist["counts"]) + 1

    # 前瞻收益缓存：panel 无 label_{N}d_close_to_close 列 → 全部 horizon 现算并缓存
    cache = getattr(session, "_ic_decay_label_cache", {})
    assert set(cache) == {1, 2, 3, 5, 10, 20}
    cache_snapshot = {h: cache[h] for h in cache}

    result2 = engine.evaluate(
        session, profile_id="train_screen", multi_line_expr="$adj_close",
        include_charts=True,
    )
    assert result2["ok"]
    # 第二次评估命中缓存（同对象复用，无重算）
    for h, series in cache_snapshot.items():
        assert session._ic_decay_label_cache[h] is series
    assert result2["chart_data"]["ic_decay"]["points"][0]["mean_ic"] == decay["points"][0]["mean_ic"]


def test_engine_decay_reuses_precomputed_label_columns() -> None:
    """panel 预置 label_{N}d_close_to_close 列时直接复用（不进缓存、不重算）。"""
    from pathlib import Path

    from alphaagent.factor.evaluation.engine import EvaluationEngine
    from alphaagent.factor.evaluation.profile import default_evaluation_profiles
    from alphaagent.factor.mining.context import StockEvalContext
    from alphaagent.factor.mining.session import StockEvalSession

    dates = pd.date_range("2024-01-01", periods=16, freq="B")
    instruments = [f"S{i:03d}" for i in range(40)]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    cross_section = np.tile(np.linspace(-1, 1, len(instruments)), len(dates))
    drift = np.repeat(np.linspace(0, 0.05, len(dates)), len(instruments))
    close = 10 + cross_section + drift
    g_close = pd.Series(close, index=index).groupby(level="instrument", sort=False)
    panel = pd.DataFrame(
        {
            "adj_close": close,
            "open": close * (1 + 0.001 * cross_section),
            "high": close * 1.01,
            "low": close * 0.99,
            "amount": np.tile(np.linspace(2e7, 8e7, len(instruments)), len(dates)),
            "turnover_rate": np.tile(np.linspace(0.5, 3.0, len(instruments)), len(dates)),
            "float_cap": np.tile(np.linspace(1e8, 5e10, len(instruments)), len(dates)),
            "label_1d_open_to_open": 0.002 * cross_section + 0.0001 * drift,
            "label_1d_close_to_close": g_close.transform(
                lambda s: forward_close_to_close_label(s, 1)
            ),
            "label_10d_close_to_close": g_close.transform(
                lambda s: forward_close_to_close_label(s, 10)
            ),
            "label_20d_close_to_close": g_close.transform(
                lambda s: forward_close_to_close_label(s, 20)
            ),
        },
        index=index,
    )
    session = StockEvalSession(
        session_id="test_decay_precomputed",
        ctx=StockEvalContext(
            panel_path=Path("panel.parquet"),
            train_start="2024-01-01",
            train_end="2024-01-10",
            val_start="2024-01-11",
            val_end="2024-01-31",
        ),
        panel=panel,
    )
    engine = EvaluationEngine(default_evaluation_profiles())
    result = engine.evaluate(
        session, profile_id="train_screen", multi_line_expr="$adj_close",
        include_charts=True,
    )
    assert result["ok"]
    cache = getattr(session, "_ic_decay_label_cache", {})
    # 1/10/20 走预置列；仅 2/3/5 现算缓存
    assert set(cache) == {2, 3, 5}


def test_engine_decay_failure_degrades_to_none() -> None:
    """decay 内部异常只降级该图表（ic_decay=None），不影响评估结果本身。"""
    import importlib

    from alphaagent.factor.evaluation.engine import EvaluationEngine
    from alphaagent.factor.evaluation.profile import default_evaluation_profiles
    from alphaagent.factor.mining.context import StockEvalContext
    from alphaagent.factor.mining.session import StockEvalSession

    dates = pd.date_range("2024-01-01", periods=16, freq="B")
    instruments = [f"S{i:03d}" for i in range(40)]
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    cross_section = np.tile(np.linspace(-1, 1, len(instruments)), len(dates))
    drift = np.repeat(np.linspace(0, 0.05, len(dates)), len(instruments))
    close = 10 + cross_section + drift
    panel = pd.DataFrame(
        {
            "adj_close": close,
            "open": close * (1 + 0.001 * cross_section),
            "high": close * 1.01,
            "low": close * 0.99,
            "amount": np.tile(np.linspace(2e7, 8e7, len(instruments)), len(dates)),
            "turnover_rate": np.tile(np.linspace(0.5, 3.0, len(instruments)), len(dates)),
            "float_cap": np.tile(np.linspace(1e8, 5e10, len(instruments)), len(dates)),
            "label_1d_open_to_open": 0.002 * cross_section + 0.0001 * drift,
        },
        index=index,
    )
    session = StockEvalSession(
        session_id="test_decay_fail",
        ctx=StockEvalContext(
            panel_path=Path("panel.parquet"),
            train_start="2024-01-01",
            train_end="2024-01-10",
            val_start="2024-01-11",
            val_end="2024-01-31",
        ),
        panel=panel,
    )
    engine = EvaluationEngine(default_evaluation_profiles())

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    engine_mod = importlib.import_module("alphaagent.factor.evaluation.engine")
    original = engine_mod.panel_forward_label
    engine_mod.panel_forward_label = _boom
    try:
        result = engine.evaluate(
            session, profile_id="train_screen", multi_line_expr="$adj_close",
            include_charts=True,
        )
    finally:
        engine_mod.panel_forward_label = original
    assert result["ok"]
    assert result["chart_data"]["ic_decay"] is None
    assert result["chart_data"]["daily_ic"]  # 其余图表不受影响
