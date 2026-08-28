"""申万一级(SW2021)行业分类：拉取成员(带 in/out 日期) → 严格 PIT 映射为 panel 行业码列。

产出列 ``industry_sw_l1``：整数编码(1..N，按行业 index_code 排序，可复现)，落为 float32，
未归类样本为 NaN。DSL 里可直接用 ``$industry_sw_l1`` 做行业中性：
``CS_NEUTRALIZE(factor, $industry_sw_l1)``（行业码是离散组号，勿再套 CS_BUCKET）。

数据源 Tushare：``index_classify``(行业目录) + ``index_member``(个股成员，含 in/out 日期)。
PIT：按 in_date/out_date 用 merge_asof(backward) 把每个交易日映射到当日有效行业，无前视。
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from alphaagent.core.paths import INDUSTRY_SW_PATH
from alphaagent.data.tushare_client import call_with_retry, get_pro

INDUSTRY_COLUMN = "industry_sw_l1"
SW_SRC = "SW2021"
SW_LEVEL = "L1"

_MEMBERSHIP_COLUMNS = ["instrument", "industry_code", "industry_name", "in_date", "out_date"]


def fetch_sw_l1_membership(
    pro=None,
    *,
    sleep_sec: float = 0.35,
    verbose: bool = True,
) -> pd.DataFrame:
    """拉 SW2021 一级行业成员，返回 long 表。

    列: ``instrument, industry_code(int 1..N), industry_name, in_date, out_date``。
    整数码按行业 ``index_code`` 排序分配，保证跨次运行稳定。
    """
    if pro is None:
        pro = get_pro()

    classify = call_with_retry(
        pro.index_classify,
        level=SW_LEVEL,
        src=SW_SRC,
        label="index_classify_L1",
    )
    if classify is None or classify.empty:
        raise RuntimeError("index_classify 返回空：检查 Tushare 权限/积分（申万行业需相应权限）")

    name_col = "industry_name" if "industry_name" in classify.columns else "name"
    classify = classify.dropna(subset=["index_code"]).sort_values("index_code").reset_index(drop=True)
    code_map = {code: i + 1 for i, code in enumerate(classify["index_code"])}
    name_map = dict(zip(classify["index_code"], classify[name_col]))

    rows: list[pd.DataFrame] = []
    n = len(classify)
    for i, idx_code in enumerate(classify["index_code"]):
        if verbose and (i == 0 or (i + 1) % 10 == 0 or i + 1 == n):
            print(f"  index_member [{i + 1}/{n}] {idx_code} {name_map[idx_code]}")
        mem = call_with_retry(
            pro.index_member,
            index_code=idx_code,
            is_new="",
            label=f"index_member_{idx_code}",
        )
        time.sleep(sleep_sec)
        if mem is None or mem.empty:
            continue
        m = mem.copy()
        m["instrument"] = m["con_code"]
        m["industry_code"] = code_map[idx_code]
        m["industry_name"] = name_map[idx_code]
        m["in_date"] = pd.to_datetime(m["in_date"], errors="coerce")
        m["out_date"] = pd.to_datetime(m["out_date"], errors="coerce")
        rows.append(m[_MEMBERSHIP_COLUMNS])

    if not rows:
        raise RuntimeError("index_member 全部为空：无法构建行业映射")

    out = pd.concat(rows, ignore_index=True)
    out = out.dropna(subset=["instrument", "in_date"])
    return out.sort_values(["instrument", "in_date"]).reset_index(drop=True)


def save_membership(df: pd.DataFrame, path: Path | str = INDUSTRY_SW_PATH) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out)
    return out


def load_membership(path: Path | str = INDUSTRY_SW_PATH) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        return pd.DataFrame(columns=_MEMBERSHIP_COLUMNS)
    df = pd.read_parquet(p)
    df["in_date"] = pd.to_datetime(df["in_date"], errors="coerce")
    df["out_date"] = pd.to_datetime(df["out_date"], errors="coerce")
    return df


def sw_l1_code_map(membership: pd.DataFrame) -> dict[int, str]:
    """行业整数码 → 行业名称。"""
    if membership.empty:
        return {}
    pairs = membership[["industry_code", "industry_name"]].dropna().drop_duplicates()
    return {int(c): str(nm) for c, nm in zip(pairs["industry_code"], pairs["industry_name"])}


def build_industry_column(
    panel: pd.DataFrame,
    membership: pd.DataFrame,
    *,
    dtype: str = "float32",
) -> pd.Series:
    """按 (in_date, out_date) 严格 PIT 映射，返回与 panel 同索引的行业码列。"""
    if panel.index.names != ["datetime", "instrument"]:
        raise ValueError(f"panel 索引须为 (datetime, instrument)，当前: {panel.index.names}")

    empty = pd.Series(np.nan, index=panel.index, name=INDUSTRY_COLUMN, dtype=dtype)
    if membership is None or membership.empty:
        return empty

    left = panel.index.to_frame(index=False)[["datetime", "instrument"]]
    left["_row"] = np.arange(len(left))
    left = left.sort_values("datetime", kind="mergesort")

    right = membership[["instrument", "in_date", "out_date", "industry_code"]].dropna(
        subset=["in_date"]
    )
    right = right.sort_values("in_date", kind="mergesort")
    if right.empty:
        return empty

    merged = pd.merge_asof(
        left,
        right,
        left_on="datetime",
        right_on="in_date",
        by="instrument",
        direction="backward",
    )
    # 已离开该行业（datetime 晚于 out_date）的样本置空，避免延用过期分类。
    invalid = merged["out_date"].notna() & (merged["datetime"] > merged["out_date"])
    merged.loc[invalid, "industry_code"] = np.nan

    merged = merged.sort_values("_row", kind="mergesort")
    return pd.Series(
        merged["industry_code"].to_numpy(),
        index=panel.index,
        name=INDUSTRY_COLUMN,
    ).astype(dtype)


def enrich_panel_industry(
    panel: pd.DataFrame,
    *,
    membership_path: Path | str = INDUSTRY_SW_PATH,
    refresh: bool = False,
    dtype: str = "float32",
    sleep_sec: float = 0.35,
    verbose: bool = True,
) -> pd.DataFrame:
    """把申万一级行业码 left-join 到 panel（缺缓存或 refresh 时自动从 Tushare 拉取并落盘）。"""
    membership = pd.DataFrame() if refresh else load_membership(membership_path)
    if membership.empty:
        membership = fetch_sw_l1_membership(sleep_sec=sleep_sec, verbose=verbose)
        save_membership(membership, membership_path)

    out = panel.copy()
    if INDUSTRY_COLUMN in out.columns:
        out = out.drop(columns=[INDUSTRY_COLUMN])
    out[INDUSTRY_COLUMN] = build_industry_column(panel, membership, dtype=dtype)
    return out.sort_index()
