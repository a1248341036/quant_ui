"""L1 深度曲线 + 可成交域透镜门禁。

硬保证：
- depth_ks=None（默认）时所有既有键零变化，depth_curve/depth_curve_note 为 None；
- depth_curve 末行 k=Q{n_groups} 与主口径 top_group_* 逐位一致（同一序列）；
- 构造「alpha 集中在头部 5 只」的因子：k=5 > Q10 > k=50 的超额排序可被识别
  （深度曲线的存在意义：把引擎 top_pct 小组合口径的缺口在挖掘期可视化）；
- eligibility=None 与全 True 掩码输出完全一致；收窄掩码会改变深度行；
- depth_only 模式的深度行与全量调用的对应行一致（低成本透镜路径不漂移）；
- tradable_mask 三规则（可负担 / 次日涨停 / 流动性）按构造样例生效。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaagent.factor import metrics as M
from alphaagent.factor.metrics.tradable import clear_cache, tradable_mask

LEGACY_KEYS = (
    "top_group_annualized_return",
    "top_group_annualized_excess_return",
    "top_group_gross_excess_return",
    "top_group_sharpe",
    "top_group_excess_sharpe",
    "top_group_max_drawdown",
    "avg_daily_side_turnover",
    "avg_rebalance_side_turnover",
    "group_means",
    "monotonicity",
    "spread_annualized",
)

ALPHA_STOCKS = ("S000", "S001", "S002", "S003", "S004")


def _idx(n_days=60, n_inst=200):
    return pd.MultiIndex.from_product(
        [pd.bdate_range("2021-01-04", periods=n_days), [f"S{i:03d}" for i in range(n_inst)]],
        names=["datetime", "instrument"],
    )


def _concentrated(n_days=60, n_inst=200, seed=7):
    """alpha 集中在固定 5 只：因子 +10、日收益 +1%，其余因子 N(0,1)、收益 0。"""
    rng = np.random.default_rng(seed)
    idx = _idx(n_days, n_inst)
    n = len(idx)
    inst = np.asarray(idx.get_level_values("instrument"))
    f = np.clip(rng.normal(0, 1, n), -5, 5)
    lab = np.zeros(n)
    alpha = np.isin(inst, ALPHA_STOCKS)
    f[alpha] += 10.0
    lab[alpha] = 0.01
    drop = rng.random(n) < 0.02
    return pd.Series(f, index=idx, name="f")[~drop], pd.Series(lab, index=idx)[~drop]


def _noisy(n_days=120, n_inst=40, seed=5):
    """无集中 alpha 的普通面板（fastpaths 同款构造）。"""
    rng = np.random.default_rng(seed)
    idx = _idx(n_days, n_inst)
    n = len(idx)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_days, n_inst)), axis=0)).ravel(order="F")
    c2 = close.reshape(n_days, n_inst)
    ret = np.zeros((n_days, n_inst))
    ret[1:] = c2[1:] / c2[:-1] - 1
    f = rng.normal(0, 1, n) + 0.2 * ret.ravel(order="F")
    label = pd.Series(ret.ravel(order="F") + rng.normal(0, 0.005, n), index=idx)
    factor = pd.Series(f, index=idx, name="f")
    drop = rng.random(n) < 0.03
    return factor[~drop], label[~drop]


def test_depth_keys_absent_by_default():
    factor, label = _noisy()
    out_default = M.quantile_portfolio_metrics(factor, label, cost_bps=15.0)
    assert out_default["depth_curve"] is None
    assert out_default["depth_curve_note"] is None
    out_depth = M.quantile_portfolio_metrics(factor, label, cost_bps=15.0, depth_ks=(5, 10))
    for key in LEGACY_KEYS:
        assert out_default[key] == out_depth[key], key


def test_q_reference_row_matches_primary():
    factor, label = _noisy()
    out = M.quantile_portfolio_metrics(factor, label, cost_bps=15.0, depth_ks=(5, 10))
    curve = out["depth_curve"]
    assert [r["k"] for r in curve] == [5, 10, "Q10"]
    q = curve[-1]
    assert q["gross_excess_ann"] == pytest.approx(out["top_group_gross_excess_return"], rel=1e-12)
    assert q["net_excess_ann"] == pytest.approx(out["top_group_annualized_excess_return"], rel=1e-12)
    assert q["sharpe_net"] == pytest.approx(out["top_group_excess_sharpe"], rel=1e-12)
    assert q["mdd_net"] == pytest.approx(out["top_group_max_drawdown"], rel=1e-12)
    assert q["avg_rebalance_turnover"] == out["avg_rebalance_side_turnover"]
    # 毛值 ≥ 净值（成本非负），成本年化 = 两者差 ×100
    for row in curve:
        if row["cost_annual_pp"] is not None:
            assert row["cost_annual_pp"] >= -1e-9


def test_depth_curve_detects_concentrated_alpha():
    factor, label = _concentrated()
    out = M.quantile_portfolio_metrics(factor, label, cost_bps=0.0, depth_ks=(5, 50))
    rows = {r["k"]: r for r in out["depth_curve"]}
    # k=5 恰好罩住 alpha 5 只 > Q10（前 20 只含 5 只 alpha）> k=50（稀释）
    assert rows[5]["net_excess_ann"] > rows["Q10"]["net_excess_ann"] > rows[50]["net_excess_ann"]
    assert rows[5]["avg_names"] == 5.0


def test_eligibility_filters_pool():
    factor, label = _concentrated()
    # None 与全 True 等价
    a = M.quantile_portfolio_metrics(factor, label, cost_bps=0.0, depth_ks=(5,))
    b = M.quantile_portfolio_metrics(
        factor, label, cost_bps=0.0, depth_ks=(5,), eligibility=np.ones(len(factor), dtype=bool)
    )
    for key in ("depth_curve", "top_group_annualized_excess_return", "avg_daily_side_turnover"):
        assert a[key] == b[key], key
    # 剔除 alpha 股 → k=5 深度行超额塌到 0 附近
    inst = np.asarray(factor.index.get_level_values("instrument"))
    el = ~np.isin(inst, ALPHA_STOCKS)
    c = M.quantile_portfolio_metrics(
        factor, label, cost_bps=0.0, depth_ks=(5,), eligibility=el
    )
    row5 = c["depth_curve"][0]
    assert row5["k"] == 5
    assert abs(row5["net_excess_ann"]) < 0.05
    # 长度不匹配必须报错
    with pytest.raises(ValueError):
        M.quantile_portfolio_metrics(factor, label, eligibility=np.ones(3, dtype=bool))


def test_depth_only_matches_full_curve():
    factor, label = _concentrated()
    full = M.quantile_portfolio_metrics(factor, label, cost_bps=0.0, depth_ks=(5, 10))
    only = M.quantile_portfolio_metrics(factor, label, cost_bps=0.0, depth_ks=(5, 10), depth_only=True)
    assert only["available"] is True and only["depth_only"] is True
    assert "top_group_annualized_return" not in only
    assert only["depth_curve"] == [r for r in full["depth_curve"] if r["k"] != "Q10"]
    # 无深度行时 available=False
    empty = M.quantile_portfolio_metrics(
        factor, label, cost_bps=0.0, depth_ks=(5,), depth_only=True, min_stocks=10**9
    )
    assert empty["available"] is False and empty["error"] == "insufficient_data"


def _tradable_panel_prices():
    dates = pd.bdate_range("2021-01-04", periods=4)
    insts = ["CHEAP", "RICH", "LIMITED"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["datetime", "instrument"])
    date_pos = {d: i for i, d in enumerate(dates)}
    limited_open = [100.0, 110.0, 100.0, 100.0]
    opens, closes = [], []
    for d, inst in idx:
        if inst == "LIMITED":
            opens.append(limited_open[date_pos[d]])
            closes.append(100.0)
        else:
            price = 10.0 if inst == "CHEAP" else 500.0
            opens.append(price)
            closes.append(price)
    panel = pd.DataFrame(
        {"open": opens, "close": closes, "turnover_rate": [1.0] * len(idx)},
        index=idx,
    )
    return panel, dates


def test_tradable_mask_affordable_and_limit():
    clear_cache()
    panel, dates = _tradable_panel_prices()
    mask, diag = tradable_mask(
        panel, capital=100_000, target_count=5, lot_size=100,
        min_am20_yuan=0.0, include_liquidity=False,
    )
    assert diag["available"] and diag["affordable_max_price"] == 200.0
    # 可负担：500 元 × 100 股 = 5 万 > 2 万预算 → RICH 全部不可入选
    assert not mask.loc[(dates[0], "RICH")] and not mask.loc[(dates[1], "RICH")]
    # 次日涨停：LIMITED d0→d1 开盘 +10%（≥ 10%-0.5% 容差）→ d0 不可入选；d1→d2 正常
    assert not mask.loc[(dates[0], "LIMITED")]
    assert mask.loc[(dates[1], "LIMITED")]
    # CHEAP 正常；末日无次日 → 不可执行
    assert mask.loc[(dates[0], "CHEAP")] and mask.loc[(dates[1], "CHEAP")]
    assert not mask.loc[(dates[3], "CHEAP")]
    assert 0.0 < diag["coverage"] < 1.0


def test_tradable_mask_liquidity():
    clear_cache()
    dates = pd.bdate_range("2021-02-01", periods=6)
    insts = ["A", "D"]
    idx = pd.MultiIndex.from_product([dates, insts], names=["datetime", "instrument"])
    panel = pd.DataFrame(
        {
            "open": [10.0] * len(idx),
            "close": [10.0] * len(idx),
            "amount": [1_000_000.0 if i == "A" else 100.0 for _, i in idx],
            "turnover_rate": [1.0] * len(idx),
        },
        index=idx,
    )
    mask, diag = tradable_mask(
        panel, capital=100_000, target_count=5,
        min_am20_yuan=5_000_000, include_affordable=False, include_limit=False,
    )
    # A：am20 = 1e9 元 ≥ 5e6（day4 起满 5 个观测）→ True；前几日 am20 NaN → False
    assert not mask.loc[(dates[0], "A")]
    assert mask.loc[(dates[4], "A")] and mask.loc[(dates[5], "A")]
    # D：am20 = 1e5 元 < 5e6 → 永远 False
    assert not mask.loc[(dates[4], "D")]
    assert diag["coverage"] == pytest.approx(2.0 / 12.0, abs=0.01)


def test_tradable_mask_missing_columns_is_passthrough():
    clear_cache()
    idx = _idx(n_days=6, n_inst=5)
    rng = np.random.default_rng(0)
    panel = pd.DataFrame({"close": rng.normal(10, 1, len(idx))}, index=idx)
    mask, diag = tradable_mask(panel, capital=100_000, target_count=5)
    assert diag["available"] is False
    assert mask.all()


def test_quantile_portfolio_plugin_depth_lens():
    clear_cache()
    from alphaagent.factor.evaluation.context import EvaluationContext
    from alphaagent.factor.evaluation.plugins import quantile_portfolio

    idx = _idx(n_days=40, n_inst=60)
    rng = np.random.default_rng(11)
    n = len(idx)
    f = np.clip(rng.normal(0, 1, n), -5, 5)
    inst = np.asarray(idx.get_level_values("instrument"))
    alpha = np.isin(inst, ALPHA_STOCKS)
    f[alpha] += 10.0
    factor = pd.Series(f, index=idx, name="f")
    label = pd.Series(np.where(alpha, 0.01, 0.0), index=idx)
    panel = pd.DataFrame(
        {
            "open": [10.0] * n,
            "close": [10.0] * n,
            "amount": [1_000_000.0] * n,
            "turnover_rate": [1.0] * n,
        },
        index=idx,
    )
    ctx = EvaluationContext(
        panel=panel, factor=factor, label=label, profile=None,  # type: ignore[arg-type]
        factor_name="t", label_holding_days=1,
    )
    out = quantile_portfolio(ctx, {"cost_bps": 0.0})
    curve = out["depth_curve"]
    assert curve and curve[-1]["k"] == "Q10"
    assert out["depth_curve_tradable"] and out["depth_curve_tradable"][0]["k"] == 5
    td = out["tradable_domain"]
    assert td["available"] and 0 < td["mask_coverage"] <= 1.0
    # 透镜是追加键：主口径键不受第二路调用影响
    for key in LEGACY_KEYS:
        assert key in out
    # tradable_domain=False 时无透镜键
    out2 = quantile_portfolio(ctx, {"cost_bps": 0.0, "tradable_domain": False})
    assert "depth_curve_tradable" not in out2 and "tradable_domain" not in out2


def test_eval_response_depth_passthrough():
    from alphaagent.factor.mining.eval.response import format_eval_response

    raw = {
        "ok": True, "split": "val",
        "date_range": {"start": "2023-01-01", "end": "2024-12-31"},
        "summary": {}, "timing_ms": {},
        "quantile_portfolio": {
            "depth_curve": [
                {"k": 5, "avg_names": 5.0, "gross_excess_ann": -0.11, "net_excess_ann": -0.12,
                 "sharpe_net": -0.2, "mdd_net": 0.44, "avg_rebalance_turnover": 1.7,
                 "cost_annual_pp": 1.6},
                {"k": "Q10", "avg_names": None, "gross_excess_ann": 0.07, "net_excess_ann": 0.07,
                 "sharpe_net": 1.2, "mdd_net": 0.24, "avg_rebalance_turnover": 1.3,
                 "cost_annual_pp": 0.0},
            ],
            "depth_curve_tradable": [
                {"k": 5, "gross_excess_ann": float("nan"), "net_excess_ann": -0.21},
            ],
            "tradable_domain": {"available": True, "mask_coverage": 0.91,
                                "budget_per_name": 20000.0, "affordable_max_price": 200.0,
                                "note": "内部说明不透传"},
        },
    }
    out = format_eval_response(raw)
    assert [r["k"] for r in out["depth_curve"]] == [5, "Q10"]
    # NaN → None（Starlette allow_nan=False）
    assert out["depth_curve_tradable"][0]["gross_excess_ann"] is None
    assert out["depth_curve_tradable"][0]["net_excess_ann"] == -0.21
    assert out["tradable_domain"]["affordable_max_price"] == 200.0
    assert "note" not in out["tradable_domain"]
    assert "depth_note" in out
    # 无深度键时不产出任何 L1 字段
    out2 = format_eval_response(
        {"ok": True, "split": "train", "date_range": {}, "summary": {}, "timing_ms": {}}
    )
    assert "depth_curve" not in out2 and "depth_note" not in out2


def test_engine_result_to_legacy_carries_depth():
    from alphaagent.factor.mining.eval.service import _engine_result_to_legacy

    raw = {
        "ok": True,
        "metrics": {"quantile_portfolio": {
            "depth_curve": [{"k": 5}],
            "depth_curve_tradable": None,
            "tradable_domain": {"available": True},
        }},
    }
    legacy = _engine_result_to_legacy(raw)
    assert legacy["quantile_portfolio"]["depth_curve"] == [{"k": 5}]
    assert legacy["quantile_portfolio"]["tradable_domain"] == {"available": True}
    # 无 quantile_portfolio 时不产出键
    assert "quantile_portfolio" not in _engine_result_to_legacy({"ok": True, "metrics": {}})
