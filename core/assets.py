"""资产类型与执行规则。

本模块只定义资产适配边界，不负责撮合。真正的订单执行仍由
``core.engine`` / ``core.event_engine`` 完成，避免第一阶段重写交易逻辑。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from . import trading_config

AssetType = Literal["stock", "etf", "fund_nav"]


@dataclass(frozen=True)
class AssetExecutionProfile:
    asset_type: AssetType
    name: str
    lot_size: int
    buy_cost: float
    sell_cost: float
    limit_flags: bool
    uses_intraday_execution: bool
    notes: str = ""
    spread_bps: float = 0.0
    min_commission: float = 0.0


STOCK_PROFILE = AssetExecutionProfile(
    "stock", "A股股票", trading_config.LOT_SIZE,
    trading_config.BUY_COST, trading_config.SELL_COST, True, True,
    "T+1、整手、涨跌停和停牌规则由交易执行层处理；散户口径佣金万2.5/万12.5（统一配置 core/trading_config.py）。",
    min_commission=5.0,
)
ETF_PROFILE = AssetExecutionProfile(
    "etf", "场内ETF", 100, 0.0003, 0.0003, False, True,
    "日线使用前复权OHLCV；分红已反映在前复权价格，暂不重复现金分红。",
    spread_bps=2.0, min_commission=5.0,
)
FUND_NAV_PROFILE = AssetExecutionProfile(
    "fund_nav", "场外基金", 1, 0.0015, 0.0050, False, False,
    "当前是NAV确认近似模型；申购截止时间、确认日和现金到账需后续扩展。",
)

PROFILES: dict[str, AssetExecutionProfile] = {
    "stock": STOCK_PROFILE, "股票": STOCK_PROFILE,
    "etf": ETF_PROFILE, "ETF": ETF_PROFILE,
    "fund": FUND_NAV_PROFILE, "fund_nav": FUND_NAV_PROFILE,
    "场外基金": FUND_NAV_PROFILE,
}


def get_execution_profile(asset_type: str | None) -> AssetExecutionProfile:
    """按资产类型取得规则；未指定时保持历史股票默认行为。"""
    if not asset_type:
        return STOCK_PROFILE
    try:
        return PROFILES[asset_type]
    except KeyError as exc:
        raise ValueError(f"未知资产类型: {asset_type}") from exc


def panel_kind(profile: AssetExecutionProfile) -> str:
    return "nav" if profile.asset_type == "fund_nav" else "ohlcv"


def validate_ohlcv_panel(data: pd.DataFrame, profile: AssetExecutionProfile) -> dict:
    """校验行情资产面板，返回可记录到数据状态/回测归档的报告。"""
    required = {"date", "code", "open", "close", "turnover", "amount",
                "turn20", "am20", "volume"}
    missing = sorted(required - set(data.columns)) if data is not None else sorted(required)
    if missing:
        raise ValueError(f"{profile.name}面板缺少字段: {missing}")
    if data.empty:
        return {"asset_type": profile.asset_type, "rows": 0, "codes": 0,
                "start": None, "end": None, "invalid_price_rows": 0}
    dates = pd.to_datetime(data["date"], errors="coerce")
    invalid_price = ((pd.to_numeric(data["open"], errors="coerce") <= 0)
                     | (pd.to_numeric(data["close"], errors="coerce") <= 0))
    return {
        "asset_type": profile.asset_type,
        "rows": int(len(data)),
        "codes": int(data["code"].nunique()),
        "start": dates.min().date().isoformat() if dates.notna().any() else None,
        "end": dates.max().date().isoformat() if dates.notna().any() else None,
        "invalid_price_rows": int(invalid_price.fillna(True).sum()),
    }


class AssetAdapter:
    """把资产原始数据转换为统一回测输入。

    统一输入仍是 ``date/code/open/close/turnover/amount/turn20/am20``。
    适配器只负责数据口径，不负责选股、组合或订单撮合。
    """

    profile: AssetExecutionProfile

    def build_panel(self, data: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class OHLCVAdapter(AssetAdapter):
    def __init__(self, profile: AssetExecutionProfile):
        self.profile = profile

    def build_panel(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return data.copy() if data is not None else pd.DataFrame()
        required = {"date", "code", "open", "close"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"{self.profile.name}面板缺少字段: {sorted(missing)}")
        return data.copy()


class FundNavAdapter(AssetAdapter):
    """场外基金 NAV → 统一面板的兼容适配器。

    交易时点和费用仍由后续 NAV execution adapter 完善；这里保留当前
    可复现的近似模型，避免改变既有回测结果。
    """

    profile = FUND_NAV_PROFILE

    def build_panel(self, data: pd.DataFrame) -> pd.DataFrame:
        if data is None or data.empty:
            return pd.DataFrame(columns=["date", "open", "close", "turnover",
                                         "amount", "code", "turn20", "am20",
                                         "volume"])
        required = {"date", "code", "nav"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"场外基金NAV缺少字段: {sorted(missing)}")
        df = data[["date", "code", "nav"]].dropna().copy()
        df["date"] = pd.to_datetime(df["date"])
        cal = pd.DatetimeIndex(sorted(df["date"].unique()))
        mat = df.pivot_table(index="date", columns="code", values="nav",
                             aggfunc="last", observed=True).reindex(cal).ffill()
        mat.index.name = "date"
        long = mat.stack(future_stack=True).rename("nav").reset_index()
        long = long.dropna(subset=["nav"]).sort_values(["code", "date"])
        panel = long.rename(columns={"nav": "close"}).copy()
        panel["open"] = panel["close"]
        panel["turnover"] = 1.0
        panel["amount"] = 1.0
        panel["volume"] = 1.0
        panel["turn20"] = 1.0
        panel["am20"] = 1.0
        panel["code"] = panel["code"].astype("category")
        return panel


STOCK_ADAPTER = OHLCVAdapter(STOCK_PROFILE)
ETF_ADAPTER = OHLCVAdapter(ETF_PROFILE)
FUND_NAV_ADAPTER = FundNavAdapter()
