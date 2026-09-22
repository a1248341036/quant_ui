"""核心数据源插件：CNE stock_daily_wide（Tushare 完整日线宽表）。

这是 priority=0 的核心插件，提供 OHLCV + adjfactor + 估值 + 标记列，
决定 Panel 的 (datetime, instrument) 行索引。

**单位归一化（2026-09-19 修复）**：Tushare 宽表原始口径为
``amount`` 千元、``vol`` 手、``circ_mv``/``total_mv`` 万元。本插件在
``load()`` 出口统一换算为 Panel 契约口径（``amount`` 元、``volume`` 股、
``float_cap``/``tot_cap`` 元），与 ETF 插件（etf_bars，已是元/股）对齐，
使 ``build_panel_from_hq`` 的 ``vwap = amount / volume`` 得到正确的元/股。

修复前 ``vwap`` 恒为正确值的 1/10（千元/手），``vwap/close`` 恒等于 0.1；
比值型因子（``$adj_close/$adj_vwap``）因量纲相消不受影响，但直接使用
``$vwap`` 绝对值的算子（CROWD_MEAN_RATIO / CHIP_COM_W_GAP 等）输入整体
偏小 10 倍。

加新数据源时无需修改本文件——只需在 plugins/ 下新建另一个 .py 即可。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from alphaagent.data.adapters.registry import DataSourcePlugin

logger = logging.getLogger(__name__)

# CNE 配置路径：alphaagent/data/adapters/plugins/stock_daily_wide.py → 往上 4 层到 quant_ui
_CNE_ROOT = Path(__file__).resolve().parents[4] / "CNEquity"
_CNE_CONFIG = _CNE_ROOT / "configs" / "cnequity.quant_dataset.toml"

# ── 列映射：CNE 原始列 → Panel 列名 ──────────────────────────────────

_COLUMN_MAP: dict[str, str] = {
    # 行情
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "vol": "volume",
    "amount": "amount",
    "adj_factor": "adjfactor",
    # 估值（daily_basic）
    "turnover_rate": "turnover_rate",
    "turnover_rate_f": "turnover_rate_f",
    "volume_ratio": "volume_ratio",
    "pe": "pe",
    "pe_ttm": "pe_ttm",
    "pb": "pb",
    "ps": "ps",
    "ps_ttm": "ps_ttm",
    "dv_ratio": "dv_ratio",
    "dv_ttm": "dv_ttm",
    "total_share": "total_share",
    "float_share": "float_share",
    "free_share": "free_share",
    # 市值
    "circ_mv": "float_cap",
    "total_mv": "tot_cap",
    # 标记
    "is_st": "is_st",
}

# ── 插件声明 ──────────────────────────────────────────────────────────

PLUGIN = DataSourcePlugin(
    name="stock_daily_wide",
    dataset="stock_daily_wide",
    join_keys=("trade_date", "ts_code"),
    datetime_key="trade_date",
    instrument_key="ts_code",
    column_map=_COLUMN_MAP,
    priority=0,
)


# ── 加载函数 ──────────────────────────────────────────────────────────

def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    cne_root: str | None = None,
    cne_config: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """从 CNE 数据湖加载 stock_daily_wide 并转为 pandas DataFrame。

    返回的 DataFrame 包含原始列名（未映射），注册中心负责列名转换。
    """
    from cnequity.query.reader import load as cne_load

    from alphaagent.data.adapters.plugins._pitlib import load_cne_config

    cfg = load_cne_config(
        cne_root, cne_config, default_root=_CNE_ROOT, default_config=_CNE_CONFIG
    )

    logger.info("CNE load: dataset=%s start=%s end=%s", dataset, start, end)
    df = cne_load(dataset, start=start, end=end, config=cfg)
    if df is None or df.is_empty():
        raise ValueError(f"CNE {dataset} 无数据 (start={start}, end={end})")

    pdf = df.to_pandas()
    logger.info("CNE load: %s rows=%d cols=%d", dataset, len(pdf), pdf.shape[1])

    # 单位归一化：Tushare 宽表 amount=千元、vol=手、circ_mv/total_mv=万元
    # → Panel 契约 amount=元、volume=股、float_cap/tot_cap=元。
    # 换算后 vwap = amount/volume 才是正确的元/股（修复前恒为 1/10）。
    if "amount" in pdf.columns:
        pdf["amount"] = pdf["amount"] * 1000.0
    if "vol" in pdf.columns:
        pdf["vol"] = pdf["vol"] * 100.0
    if "circ_mv" in pdf.columns:
        pdf["circ_mv"] = pdf["circ_mv"] * 10000.0
    if "total_mv" in pdf.columns:
        pdf["total_mv"] = pdf["total_mv"] * 10000.0
    return pdf
