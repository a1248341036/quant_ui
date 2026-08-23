"""辅助数据源插件：CNE fund_flow（资金流向，东财来源）。

priority=50 的辅助插件，提供主力/超大单/大单/中单/小单净流入，
左 join 到核心行情 Panel 上。

经济直觉：超大单 vs 小单的净流入分歧反映机构与散户博弈方向，
是独立于 OHLCV 的"谁在买卖"维度信号。
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd

from alphaagent.data.adapters.registry import DataSourcePlugin

logger = logging.getLogger(__name__)

_CNE_ROOT = Path(__file__).resolve().parents[4] / "CNEquity"
_CNE_CONFIG = _CNE_ROOT / "configs" / "cnequity.quant_dataset.toml"

# ── 列映射：CNE 原始列 → Panel 列名 ──────────────────────────────────

_COLUMN_MAP: dict[str, str] = {
    "main_net_inflow": "ff_main_net",          # 主力净流入
    "super_large_net_inflow": "ff_super_net",   # 超大单净流入
    "large_net_inflow": "ff_large_net",         # 大单净流入
    "medium_net_inflow": "ff_medium_net",       # 中单净流入
    "small_net_inflow": "ff_small_net",         # 小单净流入
}

# ── 插件声明 ──────────────────────────────────────────────────────────

PLUGIN = DataSourcePlugin(
    name="fund_flow",
    dataset="fund_flow",
    join_keys=("trade_date", "symbol"),
    datetime_key="trade_date",
    instrument_key="symbol",
    column_map=_COLUMN_MAP,
    priority=50,
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
    """从 CNE 数据湖加载 fund_flow 并转为 pandas DataFrame。"""
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
