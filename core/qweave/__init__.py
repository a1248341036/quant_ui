"""QWeave 因子计算与面板转换核心模块。"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
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
    project_root: Path | None = None,
) -> list[str] | None:
    """--codes 逗号列表；缺省用 universe.csv / etf.csv；文件也没有就 None（全市场）。"""
    if codes_arg:
        return [str(c).strip().zfill(6) for c in codes_arg.split(",") if c.strip()]

    root = project_root or Path(__file__).resolve().parents[2]
    uni = (root / "data" / "etf" / "etf.csv" if asset_type == "etf"
           else root / "data" / "stock" / "universe.csv")
    if uni.exists():
        codes = pd.read_csv(uni, dtype={"code": str})["code"].astype(str).str.zfill(6).tolist()
        codes = sorted(codes)
        if max_codes:
            codes = codes[:max_codes]
        return codes
    return None


def load_panel(
    start: str | None,
    end: str | None,
    codes: list[str] | None,
    asset_type: str = "stock",
) -> pd.DataFrame:
    """读取与回测面板同口径的股票/ETF前复权日线。"""
    if asset_type == "etf":
        from core.data import load_etf_panel
        panel = load_etf_panel(start=start, end=end)
        if codes:
            panel = panel[panel["code"].astype(str).isin(codes)].copy()
        return panel
    from core.data import _load_panel_pg_parquet
    return _load_panel_pg_parquet(start=start, end=end, codes=codes)


def to_qweave_df(panel: pd.DataFrame) -> Any:
    """将 pandas DataFrame 转换为 polars DataFrame 并派生标准 vwap 列。"""
    import polars as pl
    panel = panel.copy()
    panel["volume"] = panel["volume"].astype(float)
    panel["amount"] = panel["amount"].astype(float)
    # 口径：vwap = amount(元) / (volume(手) * 100)
    panel["vwap"] = np.where(
        panel["volume"] > 0,
        panel["amount"] / (panel["volume"] * 100.0),
        np.nan,
    )
    cols = [c for c in ["date", "code", "open", "high", "low", "close", "volume", "amount", "turnover", "vwap"] if c in panel.columns]
    return pl.from_pandas(panel).select(cols)


def get_alpha_expressions(
    alpha_set: str = "alpha158",
    alpha_limit: int | None = None,
) -> tuple[list, list[str]]:
    """获取 qweave 因子表达式列表（不计算，仅返回表达式对象与名称）。

    供 qweave_runner / qweave_research 调用 qweave.with_alphas() 前使用。
    """
    try:
        import qweave
    except ImportError as exc:
        raise RuntimeError("请先安装 qweave: pip install qweave") from exc

    spec = ALPHA_SETS.get(alpha_set)
    if not spec:
        raise ValueError(f"未知 alpha_set: {alpha_set}，可选: {list(ALPHA_SETS.keys())}")

    mod_name, _req_cols = spec
    alphas = getattr(qweave, mod_name)({})
    if alpha_limit:
        alphas = alphas[:alpha_limit]
    names = [a.output_name() for a in alphas]
    return alphas, names


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
