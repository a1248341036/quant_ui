"""核心数据源插件：CNE stock_daily_wide（Tushare 完整日线宽表）。

这是 priority=0 的核心插件，提供 OHLCV + adjfactor + 估值 + 标记列，
决定 Panel 的 (datetime, instrument) 行索引。

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
    from cnequity.config import load_config
    from cnequity.query.reader import load as cne_load

    root = Path(cne_root) if cne_root else _CNE_ROOT
    cfg_path = Path(cne_config) if cne_config else _CNE_CONFIG

    old = Path.cwd()
    try:
        os.chdir(root)
        cfg = load_config(cfg_path)
    finally:
        os.chdir(old)

    logger.info("CNE load: dataset=%s start=%s end=%s", dataset, start, end)
    df = cne_load(dataset, start=start, end=end, config=cfg)
    if df is None or df.is_empty():
        raise ValueError(f"CNE {dataset} 无数据 (start={start}, end={end})")

    pdf = df.to_pandas()
    logger.info("CNE load: %s rows=%d cols=%d", dataset, len(pdf), pdf.shape[1])
    return pdf
