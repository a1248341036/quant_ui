"""核心数据源插件：CNE etf_bars（本地 ETF 日线，腾讯 qfq 前复权口径）。

这是 priority=0 的核心插件，提供 ETF 的 OHLCV + 换手率，
决定 ETF Panel 的 (datetime, instrument) 行索引。

数据口径（与股票 stock_daily_wide 不同，见 data/etf/etf_panel.parquet）：
- 价格：**qfq 前复权价**（price_basis='qfq'），复权因子恒为 1.0；
  panel 侧无需 adjfactor 列（build_panel 兜底补 1.0）。
- amount：**元**（引擎回测同口径，无需 ×1000/×10000 换算）。
- turnover：**百分数**（0.5456 = 0.5456%，与 Tushare turnover_rate 同口径）。
- 无 adjfactor / float_cap / tot_cap / 估值 / ST 列——市值与基本面类
  因子在 ETF 域不存在，评估 profile 会跳过 size 相关指标。

asset_type='etf' 时加载本插件并跳过股票插件：
  load_panel_from_cne(asset_type="etf") → registry.build_panel(asset_type="etf")
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from alphaagent.data.adapters.registry import DataSourcePlugin

logger = logging.getLogger(__name__)

# ── 列映射：CNE 原始列 → Panel 列名 ──────────────────────────────────

_COLUMN_MAP: dict[str, str] = {
    "open": "open",
    "high": "high",
    "low": "low",
    "close": "close",
    "volume": "volume",
    "amount": "amount",
    # ETF turnover 已是百分数（0.5456 = 0.5456%），与 Tushare turnover_rate 同口径
    "turnover": "turnover_rate",
}

# ── 插件声明 ──────────────────────────────────────────────────────────

PLUGIN = DataSourcePlugin(
    name="etf_bars",
    dataset="etf_bars",
    join_keys=("date", "code"),
    datetime_key="date",
    instrument_key="code",
    column_map=_COLUMN_MAP,
    priority=0,
)

# CNE 配置路径：alphaagent/data/adapters/plugins/etf_bars.py → 往上 4 层到 quant_ui
_CNE_ROOT = Path(__file__).resolve().parents[4] / "CNEquity"
_CNE_CONFIG = _CNE_ROOT / "configs" / "cnequity.quant_dataset.toml"


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    cne_root: str | None = None,
    cne_config: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """从 CNE 数据湖加载 etf_bars 并转为 pandas DataFrame。

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