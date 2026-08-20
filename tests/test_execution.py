import numpy as np
import pandas as pd

from core.execution import StockExecutionAdapter


def _adapter(limit_up=None, limit_down=None, lot_size=100):
    return StockExecutionAdapter(
        codes=["a", "b"],
        open_mat=np.array([[10.0, 20.0], [10.0, 20.0]]),
        valid_open=np.ones((2, 2), dtype=bool),
        am20_mat=np.full((2, 2), 1_000_000.0),
        turnover_mat=np.ones((2, 2)),
        limit_up=limit_up, limit_down=limit_down,
        dates=pd.date_range("2024-01-01", periods=2),
        buy_cost=0.001, sell_cost=0.002, lot_size=lot_size,
        slippage_bps=0.0, max_participation=0.0,
    )


def test_execution_sells_before_buying_and_respects_lot():
    adapter = _adapter()
    result = adapter.execute_targets(
        cash=0.0, positions={0: 300.0}, targets={1: 1.0}, chosen_list=[1],
        portfolio_value=6000.0, amount_threshold=0.0, signal_idx=0, exec_idx=1,
    )
    assert result.sold_codes == ["a"]
    assert result.bought_codes == ["b"]
    assert result.positions == {1: 100.0}
    assert result.cash > 0
    assert [x["side"] for x in result.trades_detail] == ["sell", "buy"]


def test_execution_rejects_limit_up_buy():
    adapter = _adapter(limit_up=np.array([[False, True], [False, True]]))
    result = adapter.execute_targets(
        cash=10000.0, positions={}, targets={1: 1.0}, chosen_list=[1],
        portfolio_value=10000.0, amount_threshold=0.0, signal_idx=0, exec_idx=1,
    )
    assert not result.bought_codes
    assert result.rejections[0]["reason"] == "涨停买不进"
