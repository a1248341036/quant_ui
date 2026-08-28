"""辅助数据源插件：CNE 季频基本面（PIT 对齐）。

直接读取 CNE curated 层的 balancesheet / income / cashflow / fina_indicator
parquet 文件，以 ann_date 为 point-in-time 锚点，展开为日频后 join 到核心行情 Panel。

设计要点：
- PIT：每条记录以 ann_date（公告日）为准，而非 end_date（报告期末），
  确保因子在 t 日只使用 t 日及之前已公开的数据。
- 去重：同一 (symbol, end_date) 可能存在多版报告，保留 ann_date 最晚的那条。
- 展开：polars join_asof(backward) 把季频记录广播到每个交易日，
  直到下一份财报披露才切换值。
"""

from __future__ import annotations

import datetime
import logging
import os
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from alphaagent.data.adapters.registry import DataSourcePlugin

logger = logging.getLogger(__name__)

# CNE curated 根目录：优先从 CNE config data.root 解析，避免硬编码路径
# 在换环境（docker/服务器）时失效；config 解析失败时回退到仓库内默认路径。
_CURATED_ROOT_FALLBACK = (
    Path(__file__).resolve().parents[4]
    / "CNEquity" / "data" / "quant_dataset" / "_cnequity" / "curated"
)
_CNE_ROOT = Path(__file__).resolve().parents[4] / "CNEquity"
_CNE_CONFIG = _CNE_ROOT / "configs" / "cnequity.quant_dataset.toml"
_curated_root_cache: Path | None = None


def _curated_root() -> Path:
    """从 CNE config 解析 curated 根目录（进程内缓存；失败回退默认路径）。"""
    global _curated_root_cache
    if _curated_root_cache is not None:
        return _curated_root_cache
    try:
        from cnequity.config import load_config

        old = Path.cwd()
        try:
            os.chdir(_CNE_ROOT)
            cfg = load_config(_CNE_CONFIG)
        finally:
            os.chdir(old)
        root = Path(cfg.curated_root)
        if root.is_dir():
            _curated_root_cache = root
            logger.info("fundamental: curated 根 = %s（来自 CNE config）", root)
            return root
    except Exception as exc:  # noqa: BLE001
        logger.warning("fundamental: 解析 CNE config curated 根失败，回退默认路径: %s", exc)
    _curated_root_cache = _CURATED_ROOT_FALLBACK
    return _curated_root_cache

# ── 各数据集选取的列 → funda_* 映射 ────────────────────────────────────

_INCOME_COLS = {
    "total_revenue": "funda_total_revenue",
    "n_income_attr_p": "funda_net_profit",
    "operate_profit": "funda_operate_profit",
    "ebit": "funda_ebit",
    "sell_exp": "funda_selling_expense",
    "admin_exp": "funda_admin_expense",
    "fin_exp": "funda_finance_expense",
    "rd_exp": "funda_rd_expense",
}

_BALANCE_COLS = {
    "total_assets": "funda_total_assets",
    "total_liab": "funda_total_liabilities",
    "total_cur_assets": "funda_current_assets",
    "total_cur_liab": "funda_current_liabilities",
    "total_hldr_eqy_exc_min_int": "funda_total_equity",
    "inventories": "funda_inventory",
    "accounts_receiv": "funda_accounts_receivable",
    "fix_assets": "funda_fixed_assets",
    "goodwill": "funda_goodwill",
    "money_cap": "funda_cash",
}

_CASHFLOW_COLS = {
    "n_cashflow_act": "funda_ocf",
    "n_cashflow_inv_act": "funda_icf",
    "n_cash_flows_fnc_act": "funda_fcf",
    "free_cashflow": "funda_free_cashflow",
}

_FINA_INDICATOR_COLS = {
    "roe": "funda_roe",
    "roa": "funda_roa",
    "roic": "funda_roic",
    "grossprofit_margin": "funda_gross_margin",
    "netprofit_margin": "funda_net_margin",
    "debt_to_assets": "funda_debt_to_assets",
    "current_ratio": "funda_current_ratio",
    "quick_ratio": "funda_quick_ratio",
    "dt_eps": "funda_eps_diluted",
    "eps": "funda_eps",
    "bps": "funda_bps",
    "ocfps": "funda_ocfps",
    "profit_dedt": "funda_profit_dedt",
    # 同比增长（%）：财报期同比，PIT 阶跃序列
    "netprofit_yoy": "funda_netprofit_yoy",
    "or_yoy": "funda_or_yoy",
    "tr_yoy": "funda_tr_yoy",
    "ocf_yoy": "funda_ocf_yoy",
    "roe_yoy": "funda_roe_yoy",
}

_ALL_FUNDA_COLS = [
    *(_INCOME_COLS | _BALANCE_COLS | _CASHFLOW_COLS | _FINA_INDICATOR_COLS).values(),
    "funda_end_date",
    "funda_ann_date",
]

# ── 插件声明 ──────────────────────────────────────────────────────────

PLUGIN = DataSourcePlugin(
    name="fundamental",
    dataset="fundamental",  # 虚拟名；实际读 curated 多个 parquet
    join_keys=("trade_date", "ts_code"),
    datetime_key="trade_date",
    instrument_key="ts_code",
    # load() 内部已完成重命名，这里 identity 映射告诉 registry 保留这些列
    column_map={c: c for c in _ALL_FUNDA_COLS},
    priority=30,
)

# ── 内部工具 ──────────────────────────────────────────────────────────


def _read_curated(dataset: str) -> pl.DataFrame:
    """扫描 curated/{dataset}/ 下所有 parquet 文件并合并。"""
    root = _curated_root() / dataset
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"CNE curated {dataset} 无 parquet 文件")
    return pl.read_parquet(root, hive_partitioning=False)


def _select_and_rename(df: pl.DataFrame, col_map: dict[str, str]) -> pl.DataFrame:
    keep = list(col_map.keys()) + ["symbol", "end_date", "ann_date"]
    available = [c for c in keep if c in df.columns]
    missing = sorted(k for k in col_map if k not in df.columns)
    if missing:
        # 声明了但 curated 层没有的列：显式告警，避免字段静默消失导致下游"不可用字段"。
        logger.warning("fundamental: curated 缺少声明列(跳过): %s", missing)
    out = df.select(available)
    rename = {k: v for k, v in col_map.items() if k in available}
    return out.rename(rename).with_columns(pl.col("ann_date").cast(pl.Date))


# ── 加载函数 ──────────────────────────────────────────────────────────


def load(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """从 CNE curated 加载季频基本面并 PIT 展开为日频。"""
    logger.info("fundamental adapter: reading curated from %s", _curated_root())

    specs = [
        ("balancesheet", _BALANCE_COLS),
        ("income", _INCOME_COLS),
        ("cashflow", _CASHFLOW_COLS),
        ("fina_indicator", _FINA_INDICATOR_COLS),
    ]
    parts: list[pl.DataFrame] = []
    for ds_name, ds_cols in specs:
        try:
            raw = _read_curated(ds_name)
            sub = _select_and_rename(raw, ds_cols)
            parts.append(sub)
            logger.info("fundamental: %s loaded (%d rows)", ds_name, sub.height)
        except Exception as exc:
            logger.warning("fundamental: skip %s: %s", ds_name, exc)

    if not parts:
        raise ValueError("fundamental: 所有 CNE 基本面数据集均无数据")

    # 1. 每个数据集聚合到 (symbol, end_date) 粒度，重命名 PIT 列避免 join 冲突
    pit_names: list[str] = []
    for i, p in enumerate(parts):
        funda_cols_p = [c for c in p.columns if c.startswith("funda_")]
        pit_name = f"_pit_{i}"
        pit_names.append(pit_name)
        parts[i] = (
            p.sort(["symbol", "end_date", "ann_date"])
            .group_by(["symbol", "end_date"])
            .agg(
                pl.col("ann_date").min().alias(pit_name),
                *[pl.col(c).first() for c in funda_cols_p],
            )
        )

    # 顺序 full outer join（只有 symbol/end_date 是共享键）
    merged = parts[0]
    for i in range(1, len(parts)):
        merged = merged.join(parts[i], on=["symbol", "end_date"], how="full", coalesce=True)

    # 合并各数据集的 _pit_* 为统一 ann_date（取最早的披露日）
    avail_pit = [pl.col(c) for c in pit_names if c in merged.columns]
    if len(avail_pit) == 1:
        merged = merged.rename({pit_names[0]: "ann_date"})
    else:
        merged = merged.with_columns(
            pl.min_horizontal(avail_pit).alias("ann_date")
        ).drop([c for c in pit_names if c in merged.columns])

    # 过滤 + 排序
    quarterly = merged.filter(pl.col("ann_date").is_not_null())
    if end:
        end_d = datetime.date.fromisoformat(end)
        quarterly = quarterly.filter(pl.col("ann_date") <= end_d)
    if start:
        start_d = datetime.date.fromisoformat(start)
        # 保留 start 前最后一份报告（作为初始值），只过滤太老的
        cutoff = start_d - datetime.timedelta(days=400)
        quarterly = quarterly.filter(pl.col("ann_date") >= cutoff)

    if quarterly.height == 0:
        raise ValueError(f"fundamental: 过滤后无数据 (start={start}, end={end})")

    logger.info("fundamental: %d unique (symbol, period) records after merge", quarterly.height)

    # 2. PIT 展开为日频
    symbols = quarterly["symbol"].unique().to_list()

    s = datetime.date.fromisoformat(start) if start else datetime.date(2015, 1, 1)
    e = datetime.date.fromisoformat(end) if end else datetime.date.today()

    # 构建工作日序列（Mon-Fri）
    all_days = []
    d = s
    while d <= e:
        if d.weekday() < 5:
            all_days.append(d)
        d += datetime.timedelta(days=1)

    grid = pl.DataFrame({
        "date": pl.Series(all_days, dtype=pl.Date)
    }).join(
        pl.DataFrame({"symbol": pl.Series(symbols, dtype=pl.Utf8)}), how="cross"
    )

    quarterly_sorted = quarterly.sort(["symbol", "ann_date"])
    grid_sorted = grid.sort(["symbol", "date"])

    expanded = grid_sorted.join_asof(
        quarterly_sorted,
        left_on="date",
        right_on="ann_date",
        by="symbol",
        strategy="backward",
    )

    # 过滤掉完全没有基本面数据的行（用 total_assets 是否非空判断）
    expanded = expanded.filter(pl.col("funda_total_assets").is_not_null())

    # 3. 输出格式适配 registry（原始列名，由 column_map 重命名）
    result = expanded.rename({"date": "trade_date", "symbol": "ts_code"})
    pdf = result.to_pandas()
    logger.info("fundamental adapter: expanded to %d rows × %d cols", *pdf.shape)
    return pdf
