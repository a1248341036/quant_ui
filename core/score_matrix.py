"""因子分数 → 回测引擎 external_scores 矩阵（date×code）的公共转换。

引擎 ``core.engine.run_backtest`` 支持通过 ``external_scores`` 注入任意
date×code 分数矩阵（配合 ``factor="pred"``）。项目里多个研究路径产出的
都是（date, instrument/code, score）形状的数据，但各自实现了自己的矩阵
对齐逻辑：

- AlphaAgent DSL 求值结果：``pd.Series``，MultiIndex(datetime, instrument)，
  instrument 带交易所后缀（如 000001.SZ）
- qweave 研究输出：``polars.DataFrame`` 长表列（date, code, <factor>），
  code 为 6 位无后缀

本模块统一收口：无论输入哪种形状，都按引擎契约输出 date×code 矩阵，
列名为 6 位代码（去后缀）；给定目标日历/代码时做 reindex 对齐。

不要在别处再写一遍 unstack/pivot + 列名清理。
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd


def _series_from_input(
    scores: pd.Series | pd.DataFrame,
    value_col: str,
) -> pd.Series:
    """把 Series / 长表 DataFrame 规整为 MultiIndex(datetime, instrument) 分数序列。"""
    if isinstance(scores, pd.DataFrame):
        if value_col not in scores.columns:
            raise ValueError(f"scores_frame_missing_value_column:{value_col}")
        needed = ("date", "code")
        missing = [c for c in needed if c not in scores.columns]
        if missing:
            raise ValueError(f"scores_frame_missing_columns:{missing}")
        s = scores.set_index(["date", "code"])[value_col]
    elif isinstance(scores, pd.Series):
        s = scores.rename("score")
    else:
        raise TypeError(
            f"scores_must_be_series_or_frame:{type(scores).__name__}"
        )

    if not isinstance(s.index, pd.MultiIndex) or s.index.nlevels != 2:
        raise ValueError("scores_requires_multiindex(date, code)")

    dt = s.index.get_level_values(0)
    inst = s.index.get_level_values(1)
    dt = pd.to_datetime(dt, errors="coerce")
    if dt.isna().any():
        raise ValueError("scores_invalid_datetime")
    return pd.Series(
        s.to_numpy(copy=False),
        index=pd.MultiIndex.from_arrays([dt, [str(c) for c in inst]],
                                        names=["datetime", "instrument"]),
        name=s.name or "score",
    )


def scores_to_engine_matrix(
    scores: pd.Series | pd.DataFrame,
    *,
    bt_dates: Sequence | pd.DatetimeIndex | None = None,
    bt_codes: Sequence[str] | None = None,
    value_col: str = "score",
) -> pd.DataFrame:
    """因子分数 → 引擎 external_scores 矩阵（date × code，6 位代码）。

    Parameters
    ----------
    scores
        ``pd.Series``（MultiIndex(datetime, instrument)，instrument 可带
        交易所后缀）或长表 ``pd.DataFrame``（date/code/value_col 三列）。
    bt_dates, bt_codes
        回测面板的交易日历与 6 位代码列表；提供时输出矩阵 reindex 对齐
        （缺失补 NaN），缺省时直接 unstack（由引擎侧自行对齐）。
    value_col
        长表 DataFrame 的分数列名。缺省 "score"。

    Returns
    -------
    pd.DataFrame
        索引为 date（DatetimeIndex）、列名为 6 位代码的分数矩阵。
        重复 (date, code) 保留最后一条（与 qweave pivot aggfunc="last" 一致）。
    """
    s = _series_from_input(scores, value_col)
    # 重复 (datetime, instrument) 取最后一条，保证 unstack 不抛错。
    s = s[~s.index.duplicated(keep="last")]

    mat = s.unstack("instrument").sort_index()
    # 去交易所后缀 -> 6 位纯代码。
    mat.columns = [str(c).split(".")[0].zfill(6) for c in mat.columns]
    # 理论上同一代码只会出现在一个交易所；防御性去重。
    if mat.columns.has_duplicates:
        mat = mat.loc[:, ~mat.columns.duplicated(keep="first")]
    mat.columns.name = "code"

    if bt_dates is not None:
        mat = mat.reindex(pd.DatetimeIndex(bt_dates))
    if bt_codes is not None:
        mat = mat.reindex(columns=[str(c).split(".")[0].zfill(6) for c in bt_codes])
    return mat