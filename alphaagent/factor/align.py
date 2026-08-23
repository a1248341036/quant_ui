"""因子值对齐到 canonical row_id 顺序。"""

from __future__ import annotations

import numpy as np
import pandas as pd

from alphaagent.factor.zoo.index import RowIndex


def series_to_long(values: pd.Series) -> pd.DataFrame:
    """MultiIndex Series → datetime, instrument, value。"""
    if not isinstance(values.index, pd.MultiIndex):
        raise ValueError("因子值须为 (datetime, instrument) MultiIndex Series")
    out = values.reset_index()
    out.columns = ["datetime", "instrument", "value"]
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["instrument"] = out["instrument"].astype(str)
    return out.dropna(subset=["datetime"])


def canonical_align(
    values: np.ndarray | pd.Series,
    *,
    factor_dt: pd.Series | None = None,
    factor_inst: pd.Series | None = None,
    row_index: RowIndex,
    n_rows: int,
) -> np.ndarray:
    """通过 merge 将因子值对齐到 canonical row_id 顺序（长度 n_rows）。"""
    if isinstance(values, pd.Series):
        long = series_to_long(values)
        factor_dt = long["datetime"]
        factor_inst = long["instrument"]
        arr = long["value"].to_numpy(dtype=np.float32, copy=False)
    else:
        arr = np.asarray(values, dtype=np.float32)
        if factor_dt is None or factor_inst is None:
            raise ValueError("ndarray 输入须提供 factor_dt 与 factor_inst")

    if len(arr) == 0:
        return np.full(n_rows, np.nan, dtype=np.float32)

    dt_series = pd.to_datetime(factor_dt, errors="coerce")
    ref = row_index.rows[["row_id", "datetime", "instrument"]].copy()
    ref["datetime"] = pd.to_datetime(ref["datetime"], errors="coerce")
    ref["instrument"] = ref["instrument"].astype(str)

    tmp = pd.DataFrame(
        {
            "_dt": dt_series,
            "_inst": factor_inst.astype(str),
            "_val": np.asarray(arr, dtype=np.float32),
        }
    )
    merged = tmp.merge(
        ref,
        left_on=["_dt", "_inst"],
        right_on=["datetime", "instrument"],
        how="inner",
    )
    out = np.full(n_rows, np.nan, dtype=np.float32)
    if merged.empty:
        return out
    rid = merged["row_id"].to_numpy(dtype=np.int64, copy=False)
    out[rid] = merged["_val"].to_numpy(dtype=np.float32, copy=False)
    return out


def align_series_to_panel(values: pd.Series, panel: pd.DataFrame) -> np.ndarray:
    """panel index 顺序对齐（panel 须已与 row_index 同序）。"""
    panel = panel.sort_index()
    aligned = values.reindex(panel.index)
    return aligned.to_numpy(dtype=np.float32, copy=False)
