import numpy as np
import pandas as pd

from core.execution import ETFExecutionAdapter, StockExecutionAdapter
from core.assets import ETF_PROFILE, validate_ohlcv_panel
from core.engine import build_factor_frames, run_backtest
from conftest import CODES


def test_etf_adapter_has_separate_type_but_shared_behavior():
    adapter = ETFExecutionAdapter(
        codes=["510300"],
        open_mat=np.array([[4.0], [4.0]]),
        valid_open=np.ones((2, 1), dtype=bool),
        am20_mat=np.full((2, 1), 1_000_000.0),
        turnover_mat=np.ones((2, 1)),
        limit_up=None,
        limit_down=None,
        dates=pd.date_range("2024-01-01", periods=2),
        buy_cost=0.0003,
        sell_cost=0.0003,
        lot_size=100,
        slippage_bps=0.0,
        max_participation=0.0,
    )
    assert isinstance(adapter, StockExecutionAdapter)
    assert adapter.asset_type == "etf"
    result = adapter.execute_targets(
        cash=1_000.0, positions={}, targets={0: 1.0}, chosen_list=[0],
        portfolio_value=1_000.0, amount_threshold=0.0,
        signal_idx=0, exec_idx=1,
    )
    assert result.bought_codes == ["510300"]
    assert result.positions == {0: 200.0}


def test_engine_selects_etf_profile(panel):
    result = run_backtest(
        panel=panel, codes=CODES, factor="mom20", ascending=False,
        start="2024-06-03", end="2024-09-09", capital=100_000,
        top_n=2, freq="monthly", limit_flags=False,
        execution_profile=ETF_PROFILE,
    )
    assert result["asset_type"] == "etf"
    assert result["execution_profile"] is ETF_PROFILE
    fees = [x["fee"] / x["amount"] for x in result["trades_detail"]
            if x["amount"]]
    assert fees and all(abs(v - ETF_PROFILE.buy_cost) < 1e-12 for v in fees)


def test_etf_panel_validation(panel):
    report = validate_ohlcv_panel(panel, ETF_PROFILE)
    assert report["asset_type"] == "etf"
    assert report["rows"] == len(panel)
    assert report["codes"] == panel["code"].nunique()
    assert report["invalid_price_rows"] == 0


def test_factor_scenarios_keep_etf_in_price_volume_family(panel):
    close = panel.pivot(index="date", columns="code", values="close")
    am20 = panel.pivot(index="date", columns="code", values="am20")
    turn20 = panel.pivot(index="date", columns="code", values="turn20")

    etf_factors = build_factor_frames(close, am20, turn20, asset_type="etf")
    fund_factors = build_factor_frames(close, am20, turn20, asset_type="fund_nav")

    assert {"am20", "turn20", "composite"} <= set(etf_factors)
    assert {"mdd20", "sharpe20", "nav_stability"} <= set(fund_factors)
    assert "mdd20" not in etf_factors
    assert "am20" not in fund_factors
