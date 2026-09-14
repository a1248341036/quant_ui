"""QWeave 因子计算与面板转换核心模块。"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

ALPHA_SETS = {
    "alpha158": ("qlib_alpha158", ["close", "high", "low", "open", "volume", "vwap"]),
    "alpha101": ("worldquant_alpha101", ["close", "high", "low", "open", "volume", "vwap"]),
    "alpha191": ("gtja_alpha191", ["close", "high", "low", "open", "volume", "vwap"]),
}


def load_codes(
    codes_arg: str | None,
    max_codes: int | None,
    asset_type: str = "stock",
) -> list[str] | None:
    """--codes 逗号列表；缺省用 universe.csv；文件也没有就 None（全市场）。"""
    if codes_arg:
        return [str(c).strip().zfill(6) for c in codes_arg.split(",") if c.strip()]
    return None


def load_panel(
    start: str | None,
    end: str | None,
    codes: list[str] | None,
    asset_type: str = "stock",
) -> pd.DataFrame:
    """从本地宽表加载并标准化字段为 qweave 所需格式。"""
    from core.fetcher.stocks import load_stock_daily_wide
    from core.panel_schema import standardize_panel_for_engine

    df = load_stock_daily_wide(start=start, end=end, codes=codes)
    if df.empty:
        return df
    return standardize_panel_for_engine(df, asset_type=asset_type)


def to_qweave_df(df: pd.DataFrame) -> Any:
    """将 pandas DataFrame 转换为 polars DataFrame 以供 qweave 高性能计算。"""
    import polars as pl

    return pl.from_pandas(df)


def build_alphas(
    df: Any,
    alpha_set: str = "alpha158",
    alpha_limit: int | None = None,
) -> pd.DataFrame:
    """计算指定集合的 Alpha 因子矩阵。"""
    try:
        import qweave
    except ImportError as exc:
        raise RuntimeError("请先安装 qweave: pip install qweave") from exc

    spec = ALPHA_SETS.get(alpha_set)
    if not spec:
        raise ValueError(f"未知 alpha_set: {alpha_set}，可选: {list(ALPHA_SETS.keys())}")

    mod_name, req_cols = spec
    calculator = getattr(qweave, mod_name)
    res_pl = calculator(df)
    res_df = res_pl.to_pandas()

    if alpha_limit and alpha_limit > 0:
        meta_cols = [c for c in ["date", "code", "datetime", "instrument"] if c in res_df.columns]
        alpha_cols = [c for c in res_df.columns if c not in meta_cols][:alpha_limit]
        res_df = res_df[meta_cols + alpha_cols]

    return res_df
