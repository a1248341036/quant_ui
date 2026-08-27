from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from core import trading_config


def load_module(path: str):
    spec = importlib.util.spec_from_file_location("lab_strategy", str(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"无法加载代码模块: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def _to_float(x) -> float | None:
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def points(s: pd.Series) -> list[dict]:
    out = []
    for idx, v in s.items():
        v = _to_float(v)
        if v is None:
            continue
        out.append({"date": str(pd.Timestamp(idx).date()), "value": v})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    payload: dict = {}
    try:
        # 用户代码模块只在临时目录编译，不落盘字节码缓存
        sys.dont_write_bytecode = True
        root = str(cfg["root"])
        if root not in sys.path:
            sys.path.insert(0, root)
        mod = load_module(cfg["module"])
        STRATEGIES = getattr(mod, "STRATEGIES", {}) or {}
        builder = getattr(mod, "build_factor_frames", None)
        EVENT_STRATEGIES = getattr(mod, "EVENT_STRATEGIES", None) or {}

        if cfg.get("parse_only"):
            names = list(STRATEGIES)
            for name in EVENT_STRATEGIES:
                if name not in names:
                    names.append(name)
            payload = {"ok": True, "strategies": names}
        else:
            strategy = cfg["strategy"] or (list(STRATEGIES)[0] if STRATEGIES else "")
            from backend import services
            from backend.routers.backtest import _calc_start

            calc_start = _calc_start(cfg["start"], cfg.get("warmup_days"))
            is_fund = cfg["universe"] == "场外基金"
            codes = services.build_codes(cfg["universe"], cfg["exclude_kechuang"])
            data = services.load_data(start=calc_start, end=cfg["end"], codes=codes,
                                      need_panel=not is_fund,
                                      need_heavy=is_fund or cfg["universe"] == "ETF")
            panel = data.get("fund_panel") if is_fund else data["panel"]

            if strategy in EVENT_STRATEGIES:
                # 事件驱动策略：signal -> order -> portfolio 由策略类控制
                from core.event_engine import run_event_backtest
                res = run_event_backtest(
                    panel=panel,
                    codes=codes,
                    strategy_class=EVENT_STRATEGIES[strategy],
                    start=cfg["start"],
                    end=cfg["end"],
                    capital=cfg["capital"],
                    warmup_days=cfg["warmup_days"],
                    amount_q=cfg["amount_q"],
                    slippage_bps=float(cfg.get("slippage_bps", 0.0) or 0.0),
                    max_participation=float(cfg.get("max_participation", 0.0) or 0.0),
                    buy_cost=float(cfg.get("buy_cost", trading_config.BUY_COST) or trading_config.BUY_COST),
                    sell_cost=float(cfg.get("sell_cost", trading_config.SELL_COST) or trading_config.SELL_COST),
                )
            else:
                if strategy not in STRATEGIES:
                    raise ValueError(
                        f"策略不存在: {strategy}，当前代码里可用: "
                        + (", ".join(list(STRATEGIES) + list(EVENT_STRATEGIES)) or "(空)")
                    )
                strat = STRATEGIES[strategy]
                from core.engine import run_backtest
                res = run_backtest(
                    panel=panel,
                    codes=codes,
                    factor=strat["factor"],
                    ascending=strat["ascending"],
                    start=cfg["start"],
                    end=cfg["end"],
                    capital=cfg["capital"],
                    top_n=cfg["top_n"],
                    freq=cfg["freq"],
                    affordable=cfg["affordable"],
                    amount_q=cfg["amount_q"],
                    warmup_days=cfg["warmup_days"],
                    industry_map=services.get_industry_map()
                    if cfg.get("industry_cap") else None,
                    industry_cap=cfg.get("industry_cap"),
                    factor_builder=builder,
                )
            nm = services.get_name_map()
            holdings = res["holdings"].copy()
            holdings["name"] = [nm.get(str(c), "") for c in holdings["code"]]
            payload = {
                "ok": True,
                "metrics": {k: _to_float(v) for k, v in res["metrics"].items()},
                "bench_metrics": {k: _to_float(v)
                                  for k, v in res["bench_metrics"].items()},
                "nav": points(res["nav"]),
                "bench": points(res["bench"]),
                "drawdown": points(res["drawdown"]),
                "holdings": holdings.to_dict(orient="records"),
                "trades": res["trades"].to_dict(orient="records"),
                "last_signal_date": (
                    str(res["last_signal_date"].date())
                    if res["last_signal_date"] else None
                ),
                "strategies": list(STRATEGIES) + [n for n in EVENT_STRATEGIES
                                                  if n not in STRATEGIES],
            }
    except Exception as exc:
        payload = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=20),
        }

    Path(args.out).write_text(
        json.dumps(payload, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
