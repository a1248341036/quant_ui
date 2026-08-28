"""指数成分（universe）缓存：把指数成分股按月快照落盘，供拉取阶段复用与复现。

- 缓存 schema：long 表 ``[trade_date, instrument]``（每月一份成分快照）。
- 全量拉取时按月抓 ``index_weight`` 快照并落盘；相同区间再拉可直接读缓存，免于重复请求。
- 仅用于**拉取阶段**决定拉哪些股票；离线建 panel 不依赖本缓存。
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from alphaagent.core.paths import INDEX_DIR
from alphaagent.data.tushare_client import get_pro
from alphaagent.data.universe import fetch_index_members, resolve_index_code

MEMBER_COLUMNS = ["trade_date", "instrument"]


def index_members_path(index: str, *, base_dir: Path | str = INDEX_DIR) -> Path:
    """成分缓存路径，如 artifacts/index/000852_SH_members.parquet。"""
    code = resolve_index_code(index).replace(".", "_")
    return Path(base_dir) / f"{code}_members.parquet"


def load_index_members(index: str | None = None, *, path: Path | str | None = None) -> pd.DataFrame:
    """读取成分快照缓存；不存在时返回空表。"""
    p = Path(path) if path is not None else index_members_path(index or "")
    if not p.is_file():
        return pd.DataFrame(columns=MEMBER_COLUMNS)
    df = pd.read_parquet(p)
    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["instrument"] = df["instrument"].astype(str)
    return df.sort_values(MEMBER_COLUMNS).reset_index(drop=True)


def save_index_members(
    df: pd.DataFrame, index: str | None = None, *, path: Path | str | None = None
) -> Path:
    p = Path(path) if path is not None else index_members_path(index or "")
    p.parent.mkdir(parents=True, exist_ok=True)
    df.sort_values(MEMBER_COLUMNS).reset_index(drop=True).to_parquet(p)
    return p


def merge_members(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """合并成分快照，按 (trade_date, instrument) 去重。"""
    if old is None or old.empty:
        return new.sort_values(MEMBER_COLUMNS).reset_index(drop=True)
    if new is None or new.empty:
        return old.sort_values(MEMBER_COLUMNS).reset_index(drop=True)
    both = pd.concat([old, new]).drop_duplicates(subset=MEMBER_COLUMNS)
    return both.sort_values(MEMBER_COLUMNS).reset_index(drop=True)


def members_union(cache: pd.DataFrame, start: str, end: str) -> list[str]:
    """缓存中 [start, end] 内所有快照的成分并集。"""
    if cache is None or cache.empty:
        return []
    td = cache["trade_date"]
    mask = (td >= pd.Timestamp(start)) & (td <= pd.Timestamp(end))
    return sorted(cache.loc[mask, "instrument"].astype(str).unique())


def _cache_covers(cache: pd.DataFrame, start: str, end: str) -> bool:
    """缓存是否覆盖 [start, end]（月粒度：已含 start 所在月至 end 所在月的快照）。

    月快照总落在月末，故按 period[M] 比较：缓存最早月 <= start 月且最晚月 >= end 月。
    """
    if cache is None or cache.empty:
        return False
    months = cache["trade_date"].dt.to_period("M")
    start_m = pd.Timestamp(start).to_period("M")
    end_m = pd.Timestamp(end).to_period("M")
    return months.min() <= start_m and months.max() >= end_m


def fetch_monthly_snapshots(
    pro,
    index: str,
    start: str,
    end: str,
    *,
    sleep_sec: float = 0.35,
    verbose: bool = True,
) -> pd.DataFrame:
    """按月抓 index_weight 快照，返回 long 表 [trade_date, instrument]。"""
    index_code = resolve_index_code(index)
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    month_starts = pd.date_range(start_ts.replace(day=1), end_ts.replace(day=1), freq="MS")

    rows: list[dict] = []
    for i, m in enumerate(month_starts):
        snap = min(m + pd.offsets.MonthEnd(0), end_ts)
        if snap < start_ts:
            continue
        d = snap.strftime("%Y%m%d")
        df = pro.index_weight(index_code=index_code, start_date=d, end_date=d)
        if df is not None and not df.empty:
            snap_norm = pd.Timestamp(snap).normalize()
            for code in df["con_code"].dropna().astype(str).unique():
                rows.append({"trade_date": snap_norm, "instrument": code})
        if verbose and (i + 1) % 12 == 0:
            print(f"    index_weight 月快照 {i + 1}/{len(month_starts)} 月, 累计 {len(rows)} 行")
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    return pd.DataFrame(rows, columns=MEMBER_COLUMNS)


def resolve_index_members_cached(
    index: str,
    start: str,
    end: str,
    *,
    pro=None,
    path: Path | str | None = None,
    refresh: bool = False,
    sleep_sec: float = 0.35,
    verbose: bool = True,
) -> list[str]:
    """解析 [start, end] 指数成分并集，优先读缓存；缺失/未覆盖时拉取并落盘。

    返回成分股 ts_code 列表。缓存以月快照形式持久化于 artifacts/index/。
    """
    cache_path = Path(path) if path is not None else index_members_path(index)
    cache = load_index_members(path=cache_path)

    if not refresh and _cache_covers(cache, start, end):
        members = members_union(cache, start, end)
        if members:
            if verbose:
                print(f"  成分来源: 缓存 {cache_path.name}, 共 {len(members)} 只")
            return members

    if pro is None:
        pro = get_pro()

    snaps = fetch_monthly_snapshots(pro, index, start, end, sleep_sec=sleep_sec, verbose=verbose)
    if snaps.empty:
        # 月快照为空：回退到 universe 的多路解析（index_member 等），并以单份快照持久化
        members = fetch_index_members(pro, index, start, end, sleep_sec=sleep_sec, verbose=verbose)
        snaps = pd.DataFrame(
            {"trade_date": pd.Timestamp(end).normalize(), "instrument": list(members)},
            columns=MEMBER_COLUMNS,
        )

    cache = merge_members(cache, snaps)
    save_index_members(cache, path=cache_path)
    members = members_union(cache, start, end)
    if verbose:
        print(f"  成分来源: index_weight(monthly) 已缓存 → {cache_path.name}, 共 {len(members)} 只")
    return members


def append_snapshot(
    index: str,
    trade_date: str,
    instruments: list[str] | set[str],
    *,
    path: Path | str | None = None,
) -> Path:
    """把某一交易日的成分快照追加进缓存（增量更新时用）。"""
    cache_path = Path(path) if path is not None else index_members_path(index)
    snap = pd.DataFrame(
        {"trade_date": pd.Timestamp(trade_date).normalize(), "instrument": sorted(set(instruments))},
        columns=MEMBER_COLUMNS,
    )
    cache = merge_members(load_index_members(path=cache_path), snap)
    return save_index_members(cache, path=cache_path)
