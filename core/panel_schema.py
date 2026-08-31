"""面板数据契约：列 / 单位 / 索引定义、校验与跨格式转换（单一事实来源）。

项目里存在两套 panel 形态：

1. AlphaAgent panel（CNE 数据源直出）
   - 索引：MultiIndex(datetime, instrument)
   - amount：CNE stock_daily_wide 原始口径，单位千元
   - turnover_rate：tushare 口径，百分数（0.4289 = 0.4289%）
   列集见 alphaagent/core/types.py 的 OUTPUT_COLUMNS。

2. 回测引擎面板（core.engine.run_backtest 的输入）
   - 扁平长表，列含 date/code
   - amount：元
   - turnover：小数比例（0.0043）
   - 必需列：date, code, open, close, turnover, amount, turn20, am20
   - 可选列：high, low, volume

单位换算魔法数字此前散落在 core/data.py（_finalize_stock_df 的 turnover/100）
与 alphaagent/factor/mining/engine_gate.py（amount×1000）；本模块统一收口，
引擎面板的所有生产方都从这里取常量 / 做转换 / 做校验。

加新数据源时先在这里声明单位与列契约，不要在转换代码里再写一遍换算。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ── 单位换算常量（唯一事实来源）───────────────────────────────────────
AMOUNT_CNE_TO_ENGINE = 1000.0      # CNE/Tushare amount 千元 → 引擎元
AMOUNT_TEN_THOUSAND_TO_ENGINE = 10000.0  # 腾讯行情 amount 万元 → 引擎元
AMOUNT_ETF_TO_ENGINE = 1.0         # ETF panel amount 已是元（腾讯 qfq 口径）→ 引擎元 ×1
TURNOVER_PERCENT_TO_RATIO = 100.0  # turnover_rate 百分数 → 小数比例


@dataclass(frozen=True)
class PanelSpec:
    """一种 panel 形态的列 / 单位契约。"""

    name: str
    required_columns: tuple[str, ...]
    optional_columns: tuple[str, ...] = ()
    index: tuple[str, ...] = ()
    units: dict[str, str] = field(default_factory=dict)


ENGINE_PANEL_SPEC = PanelSpec(
    name="engine",
    required_columns=("date", "code", "open", "close", "turnover", "amount",
                      "turn20", "am20"),
    optional_columns=("high", "low", "volume"),
    index=("date", "code"),
    units={"open": "元", "close": "元", "high": "元", "low": "元",
           "amount": "元", "am20": "元",
           "turnover": "比例", "turn20": "比例", "volume": "手"},
)

ALPHA_CNE_PANEL_SPEC = PanelSpec(
    name="alpha_cne",
    required_columns=("open", "high", "low", "close", "amount",
                      "turnover_rate", "volume"),
    index=("datetime", "instrument"),
    units={"open": "元", "close": "元", "high": "元", "low": "元",
           "amount": "千元", "turnover_rate": "%"},
)


def validate_engine_panel(panel: pd.DataFrame) -> None:
    """结构校验引擎面板；不满足契约直接抛 ValueError。"""
    if panel is None:
        raise ValueError("engine_panel_is_none")
    missing = [c for c in ENGINE_PANEL_SPEC.required_columns
               if c not in panel.columns]
    if missing:
        raise ValueError(f"engine_panel_missing_columns:{missing}")
    if "date" in panel.columns:
        dates = pd.to_datetime(panel["date"], errors="coerce")
        if dates.isna().any():
            raise ValueError("engine_panel_invalid_date")


def validate_alpha_panel(panel: pd.DataFrame) -> None:
    """结构校验 AlphaAgent panel；不满足契约直接抛 ValueError。"""
    if panel is None:
        raise ValueError("alpha_panel_is_none")
    if not isinstance(panel.index, pd.MultiIndex):
        raise ValueError("alpha_panel_requires_multiindex")
    if list(panel.index.names) != ["datetime", "instrument"]:
        raise ValueError(
            f"alpha_panel_index_names:{list(panel.index.names)}"
        )
    missing = [c for c in ALPHA_CNE_PANEL_SPEC.required_columns
               if c not in panel.columns]
    if missing:
        raise ValueError(f"alpha_panel_missing_columns:{missing}")


def alpha_panel_to_engine_frame(
    panel: pd.DataFrame,
    *,
    asset_type: str = "stock",
) -> pd.DataFrame:
    """AlphaAgent panel → core.engine.run_backtest 长表（纯内存）。

    口径：
    - stock：amount 千元 → 元（×1000）。引擎参与率预算按元计算，不复权原始
      千元口径会让预算缩水 1000 倍 → 大面积“现金不足”拒单。
    - etf：amount 已是元（腾讯 qfq 口径，见 etf_panel.parquet），×1 不换算；
      切勿按 stock 千元口径乘 1000，否则 1.2 亿元会被误放大 1000 倍。
    - turnover_rate：百分数 → 比例（/100），与 core/data.py 引擎面板口径一致。
    - am20/turn20 按 code 滚动 20 日均值现算（与 _finalize_stock_df 同 min_periods）。

    该函数是 engine_gate / topn_portfolio 等“因子挖掘域回测”的唯一转换入口；
    不要在别处再写一遍单位换算。
    """
    validate_alpha_panel(panel)
    df = panel.reset_index().rename(
        columns={"datetime": "date", "instrument": "code"}
    ).copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("alpha_panel_invalid_datetime")

    amount_mult = (
        AMOUNT_ETF_TO_ENGINE
        if asset_type == "etf"
        else AMOUNT_CNE_TO_ENGINE
    )
    amount = pd.to_numeric(df["amount"], errors="coerce") * amount_mult
    if "turnover_rate" in df.columns:
        turnover = (pd.to_numeric(df["turnover_rate"], errors="coerce")
                    / TURNOVER_PERCENT_TO_RATIO)
    else:
        turnover = pd.Series(np.nan, index=df.index)

    out = pd.DataFrame({
        "date": df["date"],
        "code": df["code"],
        "open": pd.to_numeric(df["open"], errors="coerce").to_numpy(),
        "high": (pd.to_numeric(df["high"], errors="coerce").to_numpy()
                 if "high" in df.columns else np.nan),
        "low": (pd.to_numeric(df["low"], errors="coerce").to_numpy()
                if "low" in df.columns else np.nan),
        "close": pd.to_numeric(df["close"], errors="coerce").to_numpy(),
        "turnover": turnover.to_numpy(),
        "amount": amount.to_numpy(),
    })
    grouped = out.groupby("code", sort=False)
    out["am20"] = grouped["amount"].transform(
        lambda s: s.rolling(20, min_periods=5).mean())
    out["turn20"] = grouped["turnover"].transform(
        lambda s: s.rolling(20, min_periods=5).mean())
    return out