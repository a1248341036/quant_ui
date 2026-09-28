# -*- coding: utf-8 -*-
"""第一批数据插件共享工具：curated 读取 + 工作日网格 + PIT 展开基元。

供 plugins/ 下新增的季频/事件类插件复用（margin 等日频插件不需要网格）。
约定与 fundamental / forecast / shareholder_counts 插件一致：
- 事件表按 (symbol, PIT 锚点日) 聚合为 step 事件
- 用 symbol × 工作日 网格做 join_asof(backward) 广播为日频阶跃序列
- 首次事件之前的行整体丢弃（输出行即"已有披露"的股票与日期）
"""

from __future__ import annotations

import datetime
import logging
import os
import re
from pathlib import Path
from typing import Any, Iterable

import polars as pl

from alphaagent.data.adapters.plugins.fundamental import _curated_root

logger = logging.getLogger(__name__)

DEFAULT_START = datetime.date(2015, 1, 1)

# PIT 状态回看窗口：窗口起点前的该天数内的事件会保留，作为网格首日的初始状态。
PIT_LOOKBACK_DAYS = 800


def load_cne_config(
    cne_root: str | None = None,
    cne_config: str | None = None,
    *,
    default_root: Path,
    default_config: Path,
) -> Any:
    """加载 CNE config（chdir 到 CNE 根目录以兼容相对路径解析，随后恢复）。

    供 stock_daily_wide / etf_bars / fund_flow 等插件复用，消除逐字复制的
    chdir + load_config 样板。
    """
    from cnequity.config import load_config

    root = Path(cne_root) if cne_root else default_root
    cfg_path = Path(cne_config) if cne_config else default_config
    old = Path.cwd()
    try:
        os.chdir(root)
        return load_config(cfg_path)
    finally:
        os.chdir(old)


def parse_window(
    start: str | None,
    end: str | None,
) -> tuple[datetime.date, datetime.date]:
    """把字符串日期窗口解析为 (start, end)，缺省补默认值。"""
    s = datetime.date.fromisoformat(start) if start else DEFAULT_START
    e = datetime.date.fromisoformat(end) if end else datetime.date.today()
    if s > e:
        raise ValueError(f"非法窗口: start={s} 晚于 end={e}")
    return s, e


# 单次 eager 读取的文件数上限：超过则改走 scan 流式读取。CNE curated 的日频数据集
# 按 trade_date=YYYY-MM-DD 每天一个文件（融资融券 2607 个、估值 2608 个），旧实现
# 一律拒绝 → 这些数据集整条链路不可用（面板缺 mgn_* 列且哨兵校验拒绝落盘）。
_MAX_EAGER_FILES = 512

# hive 日期分区目录名：trade_date=2024-01-02 / announce_date=2024-01-02 / ...
_PARTITION_DIR_RE = re.compile(r"^[a-z_]+=(\d{4}-\d{2}-\d{2})$")


def _partition_date(path: Path) -> datetime.date | None:
    """取路径中最近一层 hive 日期分区目录的日期；非日期分区返回 None。"""
    for part in reversed(path.parts[:-1]):
        m = _PARTITION_DIR_RE.match(part)
        if m:
            return datetime.date.fromisoformat(m.group(1))
    return None


def read_curated(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
) -> pl.DataFrame:
    """读取整个 curated 数据集；给出窗口时按日期分区裁剪（与 fundamental 同口径）。

    显式列出 parquet 文件逐个读取：避免 ``pl.read_parquet(root)`` 对目录的隐式
    全量扫描（curated 目录异常残留碎片文件时可能把无关文件一并读入内存）。

    日分区数据集（``trade_date=YYYY-MM-DD/part-merged.parquet``）在给出窗口时只保留
    命中的分区——5 年窗口约 1200 档，不裁剪会被文件数守卫整包拒绝。
    """
    root = _curated_root() / dataset
    files = sorted(root.rglob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"CNE curated {dataset} 无 parquet 文件")

    windowed = False
    if start or end:
        s, e = parse_window(start, end)
        dated = [(f, _partition_date(f)) for f in files]
        if any(d is not None for _, d in dated):
            kept = [f for f, d in dated if d is None or s <= d <= e]
            if not kept:
                known = sorted(d for _, d in dated if d is not None)
                raise ValueError(
                    f"CNE curated {dataset} 在窗口 {s}~{e} 内无分区数据"
                    f"（分区范围 {known[0]} ~ {known[-1]}）"
                )
            files = sorted(kept)
            windowed = True

    if len(files) > _MAX_EAGER_FILES:
        if not windowed:
            raise ValueError(
                f"CNE curated {dataset} parquet 文件数异常（{len(files)}），拒绝全量读取"
            )
        # 已按窗口裁剪到合规规模：scan 流式读取，避免 concat 上千个小文件
        return pl.scan_parquet([str(f) for f in files]).collect()
    return pl.concat([pl.read_parquet(f) for f in files], how="vertical")


def read_curated_pit(
    dataset: str,
    *,
    start: str | None = None,
    end: str | None = None,
    lookback_days: int = PIT_LOOKBACK_DAYS,
) -> pl.DataFrame:
    """读取 PIT 类数据集：起点回退 ``lookback_days``，保住窗口前的最近状态。

    ``read_curated`` 按 ``xxx_date=YYYY-MM-DD`` 分区裁剪时会把**窗口起点之前**的最后
    一份快照/事件一起裁掉，而 PIT 类插件正需要它当初始状态——否则窗口头部若干天缺值
    （行业/财报/指数）或滚动计数系统性少计（公告 5d/20d）。展开/计数仍在用户窗口内完成
    （``expand_pit_daily`` 按 start/end 产出并按首事件裁行）。
    """
    if not start and not end:
        return read_curated(dataset)
    s, _ = parse_window(start, end)
    lo = s - datetime.timedelta(days=max(0, int(lookback_days)))
    return read_curated(dataset, start=lo.isoformat(), end=end)


def weekdays_between(s: datetime.date, e: datetime.date) -> list[datetime.date]:
    days: list[datetime.date] = []
    d = s
    while d <= e:
        if d.weekday() < 5:
            days.append(d)
        d += datetime.timedelta(days=1)
    return days


def weekday_grid(
    symbols: Iterable[str],
    s: datetime.date,
    e: datetime.date,
) -> pl.DataFrame:
    """symbol × 工作日 笛卡尔网格（按 symbol, date 排序）。"""
    days = weekdays_between(s, e)
    syms = sorted(set(str(x) for x in symbols))
    return (
        pl.DataFrame({"date": pl.Series(days, dtype=pl.Date)})
        .join(pl.DataFrame({"symbol": pl.Series(syms, dtype=pl.Utf8)}), how="cross")
        .sort(["symbol", "date"])
    )


def expand_pit_daily(
    events: pl.DataFrame,
    *,
    start: str | None,
    end: str | None,
    anchor: str,
    since_col: str,
    value_cols: list[str],
    out_date: str = "trade_date",
    out_symbol: str = "ts_code",
) -> pl.DataFrame:
    """事件表 → 日频阶跃输出（网格 join_asof backward + days_since）。

    events 需已聚合到 (symbol, anchor) 粒度，含 ``value_cols``；
    输出日期范围 = [start, end]，首事件前的行丢弃。
    """
    s, e = parse_window(start, end)
    lo = s - datetime.timedelta(days=PIT_LOOKBACK_DAYS)
    events = events.filter(
        pl.col(anchor).is_not_null()
        & (pl.col(anchor) >= lo)
        & (pl.col(anchor) <= e)
    )
    if events.is_empty():
        raise ValueError("无有效事件记录")

    symbols = events["symbol"].unique().to_list()
    grid = weekday_grid(symbols, s, e)
    expanded = (
        grid.join_asof(
            events.sort(["symbol", anchor]),
            left_on="date",
            right_on=anchor,
            by="symbol",
            strategy="backward",
        )
        .with_columns(
            (pl.col("date") - pl.col(anchor))
            .dt.total_days()
            .cast(pl.Float64)
            .alias(since_col)
        )
        .filter(pl.col(since_col).is_not_null())
    )
    keep = list(value_cols)
    if since_col not in keep:
        keep.append(since_col)
    return expanded.select(
        [pl.col("date").alias(out_date), pl.col("symbol").alias(out_symbol), *keep]
    )
