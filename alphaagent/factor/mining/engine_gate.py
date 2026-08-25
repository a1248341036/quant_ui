"""旧引擎认证门禁：submit 进正式库前，在验证集窗口跑完整约束回测。

数据全部在内存中变换：AlphaAgent panel (datetime, instrument) → 旧引擎长表
(date, code, open/high/low/close, turnover, am20, turn20)。不产生任何磁盘中间表。
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def panel_to_engine_frame(panel: pd.DataFrame) -> pd.DataFrame:
    """把 AlphaAgent panel 转成 core.engine.run_backtest 需要的长表（纯内存）。

    价格优先取 adj_* 列（与 label 口径一致），缺失时回退原始列；
    am20 / turn20 按 code 滚动 20 日均值现算。
    """
    df = panel.reset_index().rename(columns={"datetime": "date", "instrument": "code"})

    def _col(name: str) -> pd.Series:
        adjusted = df.get(f"adj_{name}")
        return adjusted if adjusted is not None else df[name]

    amount = df.get("amount")
    if amount is None:
        raise ValueError("engine_gate_requires_amount_column")
    turnover = df.get("turnover_rate")
    if turnover is None:
        turnover = pd.Series(np.nan, index=df.index)

    out = pd.DataFrame({
        "date": df["date"],
        "code": df["code"],
        "open": _col("open").to_numpy(),
        "high": _col("high").to_numpy() if "high" in df.columns else np.nan,
        "low": _col("low").to_numpy() if "low" in df.columns else np.nan,
        "close": _col("close").to_numpy(),
        "turnover": turnover.to_numpy(),
        "amount": amount.to_numpy(),
    })
    grouped = out.groupby("code", sort=False)
    out["am20"] = grouped["amount"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    out["turn20"] = grouped["turnover"].transform(lambda s: s.rolling(20, min_periods=5).mean())
    return out


def run_engine_gate(
    panel: pd.DataFrame,
    factor_values: np.ndarray,
    *,
    val_start: str,
    val_end: str,
    direction: int = 1,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """验证集窗口的旧引擎日频 TopN 回测门禁。

    policy 来自 ResearchSpec delivery_policy.production.engine_gate：
    {enabled, top_n, min_annual_return, min_excess_annual, max_drawdown}
    """
    policy = policy or {}
    from core.engine import run_backtest

    scores = pd.Series(np.asarray(factor_values, dtype=np.float64), index=panel.index) * float(direction)
    wide = scores.unstack("instrument")
    codes = [str(c) for c in wide.columns]
    engine_panel = panel_to_engine_frame(panel)

    try:
        result = run_backtest(
            engine_panel,
            codes=codes,
            factor="pred",
            ascending=False,
            start=str(val_start),
            end=str(val_end),
            capital=float(policy.get("capital", 1_000_000)),
            top_n=int(policy.get("top_n", 30)),
            freq="daily",
            warmup_days=25,
            external_scores=wide,
            slippage_bps=float(policy.get("slippage_bps", 10.0)),
            max_participation=float(policy.get("max_participation", 0.02)),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "enabled": True,
            "passed": False,
            "fail_reasons": ["engine_backtest_error"],
            "error": f"{type(exc).__name__}: {exc}",
            "metrics": {},
        }

    m = result.get("metrics") or {}
    min_annual = float(policy.get("min_annual_return", 0.0))
    min_excess = float(policy.get("min_excess_annual", 0.03))
    max_dd = float(policy.get("max_drawdown", 0.35))
    reasons: list[str] = []
    annual = m.get("年化收益")
    if annual is None or not np.isfinite(float(annual)) or float(annual) < min_annual:
        reasons.append("annual_return")
    excess = m.get("超额年化")
    if excess is None or not np.isfinite(float(excess)) or float(excess) < min_excess:
        reasons.append("excess_annual")
    drawdown = m.get("最大回撤")
    if drawdown is None or not np.isfinite(float(drawdown)) or abs(float(drawdown)) > max_dd:
        reasons.append("max_drawdown")

    return {
        "enabled": True,
        "passed": len(reasons) == 0,
        "fail_reasons": reasons,
        "thresholds": {
            "min_annual_return": min_annual,
            "min_excess_annual": min_excess,
            "max_drawdown": max_dd,
        },
        "metrics": {
            "annual_return": annual,
            "excess_annual": excess,
            "sharpe": m.get("夏普"),
            "max_drawdown": drawdown,
            "calmar": m.get("卡玛"),
            "win_rate": m.get("胜率"),
            "total_return": m.get("总收益"),
        },
        "top_n": int(policy.get("top_n", 30)),
        "window": {"start": str(val_start), "end": str(val_end)},
    }
