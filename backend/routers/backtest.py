from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from backend import services
from core.engine import latest_signals, run_backtest
from strategies.registry import STRATEGIES


router = APIRouter()


class BacktestRequest(BaseModel):
    universe: str = "科技行业"
    strategy: str = "低换手冷门"
    top_n: int = 3
    capital: float = 5000.0
    freq: str = "monthly"
    start: str
    end: str
    exclude_kechuang: bool = True
    affordable: bool = True


@router.get("/strategies")
def strategies():
    return [{"name": k, **v} for k, v in STRATEGIES.items()]


@router.post("/backtest")
def backtest(req: BacktestRequest):
    if req.strategy not in STRATEGIES:
        return {"error": f"未知策略: {req.strategy}"}
    strat = STRATEGIES[req.strategy]
    codes = services.build_codes(req.universe, req.exclude_kechuang)
    res = run_backtest(
        panel=services.load_data()["panel"],
        codes=codes,
        factor=strat["factor"],
        ascending=strat["ascending"],
        start=req.start,
        end=req.end,
        capital=req.capital,
        top_n=req.top_n,
        freq=req.freq,
        affordable=req.affordable,
    )
    return {
        "metrics": {k: services._to_float(v) for k, v in res["metrics"].items()},
        "bench_metrics": {k: services._to_float(v) for k, v in res["bench_metrics"].items()},
        "nav": services.series_to_points(res["nav"]),
        "bench": services.series_to_points(res["bench"]),
        "drawdown": services.series_to_points(res["drawdown"]),
        "holdings": res["holdings"].to_dict(orient="records"),
        "trades": res["trades"].to_dict(orient="records"),
        "last_signal_date": str(res["last_signal_date"].date()) if res["last_signal_date"] else None,
    }


class SignalsRequest(BaseModel):
    universe: str = "科技行业"
    strategy: str = "低换手冷门"
    top_n: int = 10


@router.get("/signals")
def signals(universe: str = "科技行业", strategy: str = "低换手冷门", top_n: int = 10):
    if strategy not in STRATEGIES:
        return {"error": f"未知策略: {strategy}"}
    strat = STRATEGIES[strategy]
    codes = services.build_codes(universe, True)
    sig, sig_date = latest_signals(
        services.load_data()["panel"], codes, strat["factor"],
        strat["ascending"], top_n=top_n,
    )
    return {
        "signal_date": str(sig_date.date()),
        "factor": strat["factor"],
        "ascending": strat["ascending"],
        "items": sig.to_dict(orient="records"),
    }
