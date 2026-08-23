"""增量周线聚合 + 无前视 broadcast（与 batch resample 对拍）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import numpy as np
import pandas as pd

from alphaagent.dsl.stock.intervals import bar_interval_to_timedelta
from alphaagent.dsl.stock.resample import (
    _aggregation_rule_for,
    _bucket_datetime,
    _safe_divide,
    _select_numeric_columns,
    broadcast_timeframe_to_main_freq,
    build_timeframe_panel,
)

# 派生列：partial 阶段不直接累加，finalize 时重算
_DERIVED_COLS = frozenset({"vwap", "ret"})


def _bucket_start_ts(dt: pd.Timestamp, interval: str = "1w") -> pd.Timestamp:
    s = pd.Series([pd.Timestamp(dt)])
    return pd.Timestamp(_bucket_datetime(s, interval).iloc[0])


def _finalize_partial(
    partial: Mapping[str, float],
    *,
    cols: Sequence[str],
    prev_adj_close: float | None,
) -> dict[str, float]:
    """把桶内累加状态转为周线一行（含 vwap / ret）。"""
    out = dict(partial)
    if "vwap" in cols and "volume" in out and "amount" in out:
        out["vwap"] = float(
            _safe_divide(
                pd.Series([out["amount"]]),
                pd.Series([out["volume"]]),
            ).iloc[0]
        )
    if "ret" in cols and "adj_close" in out:
        ac = out["adj_close"]
        if prev_adj_close is not None and np.isfinite(prev_adj_close) and prev_adj_close != 0:
            out["ret"] = (ac - prev_adj_close) / prev_adj_close
        else:
            out["ret"] = np.nan
    return out


def _update_running(
    acc: MutableMapping[str, float],
    bar: Mapping[str, float],
    cols: Sequence[str],
    *,
    first_bar: bool,
) -> None:
    for c in cols:
        if c in _DERIVED_COLS:
            continue
        v = float(bar[c]) if pd.notna(bar[c]) else np.nan
        rule = _aggregation_rule_for(c)
        if first_bar:
            acc[c] = v
            continue
        cur = acc.get(c, np.nan)
        if rule == "first":
            continue
        if rule == "max":
            acc[c] = v if pd.isna(cur) else max(cur, v) if pd.notna(v) else cur
        elif rule == "min":
            acc[c] = v if pd.isna(cur) else min(cur, v) if pd.notna(v) else cur
        elif rule == "sum":
            if pd.isna(cur):
                acc[c] = v
            elif pd.notna(v):
                acc[c] = cur + v
        else:  # last
            acc[c] = v


@dataclass
class _InstrumentWeekState:
    """单票周线增量状态。"""

    cols: tuple[str, ...]
    partial_bucket: pd.Timestamp | None = None
    partial_acc: dict[str, float] = field(default_factory=dict)
    # bucket_start → 定稿周线（与 batch build_timeframe_panel 一致）
    weekly_bars: dict[pd.Timestamp, dict[str, float]] = field(default_factory=dict)
    # (complete_time, values) 已排序，供单点 backward broadcast
    broadcast_ready: list[tuple[pd.Timestamp, dict[str, float]]] = field(default_factory=list)
    last_finalized_adj_close: float | None = None

    def _finalize_current_bucket(self) -> None:
        if self.partial_bucket is None or not self.partial_acc:
            return
        row = _finalize_partial(
            self.partial_acc,
            cols=self.cols,
            prev_adj_close=self.last_finalized_adj_close,
        )
        self.weekly_bars[self.partial_bucket] = row
        complete = self.partial_bucket + bar_interval_to_timedelta("1w")
        self.broadcast_ready.append((complete, dict(row)))
        if "adj_close" in row and pd.notna(row["adj_close"]):
            self.last_finalized_adj_close = float(row["adj_close"])
        self.partial_bucket = None
        self.partial_acc = {}

    def append_bar(self, dt: pd.Timestamp, bar: Mapping[str, float]) -> None:
        bucket = _bucket_start_ts(dt, "1w")
        if self.partial_bucket is not None and bucket != self.partial_bucket:
            self._finalize_current_bucket()

        first = self.partial_bucket != bucket or not self.partial_acc
        if first:
            self.partial_bucket = bucket
            self.partial_acc = {}

        _update_running(self.partial_acc, bar, self.cols, first_bar=first)

    def weekly_rows(self) -> list[tuple[pd.Timestamp, dict[str, float]]]:
        """定稿桶 + 当前未结束桶（与 batch 含 partial week 一致）。"""
        rows = sorted(self.weekly_bars.items(), key=lambda x: x[0])
        if self.partial_bucket is not None and self.partial_acc:
            partial_row = _finalize_partial(
                self.partial_acc,
                cols=self.cols,
                prev_adj_close=self.last_finalized_adj_close,
            )
            rows.append((self.partial_bucket, partial_row))
        return rows

    def broadcast_at(self, dt: pd.Timestamp, col: str) -> float:
        """单点 backward broadcast（完成时刻 ≤ dt 的最近一根已完成周 bar）。"""
        val = np.nan
        for complete, data in self.broadcast_ready:
            if complete <= dt:
                val = data.get(col, np.nan)
            else:
                break
        return float(val) if pd.notna(val) else np.nan


class IncrementalWeekEngine:
    """日频逐 bar 增量更新 @1w 辅表，语义对齐 batch resample + merge_asof backward。"""

    def __init__(self, columns: Optional[Sequence[str]] = None) -> None:
        self._columns: Optional[tuple[str, ...]] = (
            tuple(columns) if columns is not None else None
        )
        self._states: dict[str, _InstrumentWeekState] = {}

    def _cols_for(self, bar: Mapping[str, Any]) -> tuple[str, ...]:
        if self._columns is not None:
            return self._columns
        return tuple(_select_numeric_columns(pd.DataFrame([bar])))

    def _state(self, instrument: str, cols: tuple[str, ...]) -> _InstrumentWeekState:
        st = self._states.get(instrument)
        if st is None:
            st = _InstrumentWeekState(cols=cols)
            self._states[instrument] = st
        elif st.cols != cols:
            raise ValueError(f"列集合不一致: {instrument} {st.cols} vs {cols}")
        return st

    def append_bar(
        self,
        dt: pd.Timestamp,
        instrument: str,
        bar: Mapping[str, Any],
    ) -> None:
        """追加一根日 K（单票）。"""
        cols = self._cols_for(bar)
        st = self._state(instrument, cols)
        st.append_bar(pd.Timestamp(dt), bar)

    def append_panel(self, panel: pd.DataFrame) -> None:
        """按 instrument × datetime 顺序 replay 面板（可多次 append 续接）。"""
        if not isinstance(panel.index, pd.MultiIndex):
            raise ValueError("panel 必须是 (datetime, instrument) MultiIndex")
        df = panel.reset_index().sort_values(["instrument", "datetime"], kind="mergesort")
        cols = self._columns or tuple(_select_numeric_columns(panel))
        for row in df.itertuples(index=False):
            inst = row.instrument
            dt = pd.Timestamp(row.datetime)
            bar = {c: getattr(row, c) for c in cols if hasattr(row, c)}
            self._state(inst, cols).append_bar(dt, bar)

    def weekly_panel(self) -> pd.DataFrame:
        """当前周线辅表（index = 桶起点 datetime, instrument）。"""
        if not self._states:
            return pd.DataFrame(
                index=pd.MultiIndex.from_arrays(
                    [[], []], names=["datetime", "instrument"]
                )
            )
        cols = next(iter(self._states.values())).cols
        records: list[dict[str, Any]] = []
        for inst, st in sorted(self._states.items()):
            for bucket, data in st.weekly_rows():
                rec = {"datetime": bucket, "instrument": inst, **data}
                records.append(rec)
        if not records:
            return pd.DataFrame(
                index=pd.MultiIndex.from_arrays(
                    [[], []], names=["datetime", "instrument"]
                )
            )
        df = pd.DataFrame(records)
        df = df.set_index(["datetime", "instrument"]).sort_index()
        # 列顺序与 batch 一致
        return df[[c for c in cols if c in df.columns]]

    def broadcast(
        self,
        target_index: pd.MultiIndex,
        columns: Optional[Sequence[str]] = None,
    ) -> pd.DataFrame:
        """全量 broadcast（复用 batch merge_asof，weekly 来自增量状态）。"""
        weekly = self.weekly_panel()
        if weekly.empty:
            cols = list(columns or self._columns or [])
            return pd.DataFrame(np.nan, index=target_index, columns=cols)
        use = weekly if columns is None else weekly[list(columns)]
        return broadcast_timeframe_to_main_freq(use, target_index, "1w")

    def broadcast_at(
        self,
        dt: pd.Timestamp,
        instrument: str,
        col: str,
    ) -> float:
        """单点 broadcast（不跑 merge_asof，O(#已完成周)）。"""
        st = self._states.get(instrument)
        if st is None:
            return np.nan
        return st.broadcast_at(pd.Timestamp(dt), col)

    def replay_panel(self, panel: pd.DataFrame) -> pd.DataFrame:
        """清空后全量 replay，返回 weekly_panel（便于测试）。"""
        self._states.clear()
        self.append_panel(panel)
        return self.weekly_panel()


def assert_incremental_matches_batch(
    panel: pd.DataFrame,
    *,
    rtol: float = 1e-6,
    atol: float = 1e-6,
) -> None:
    """增量 replay 与 batch build + broadcast 全字段对拍。"""
    batch_weekly = build_timeframe_panel(panel, target_interval="1w", base_interval="1d")
    engine = IncrementalWeekEngine()
    inc_weekly = engine.replay_panel(panel)

    pd.testing.assert_frame_equal(
        inc_weekly.sort_index(),
        batch_weekly.sort_index(),
        check_exact=False,
        rtol=rtol,
        atol=atol,
    )

    if batch_weekly.empty:
        return

    cols = list(batch_weekly.columns)
    batch_bc = broadcast_timeframe_to_main_freq(batch_weekly[cols], panel.index, "1w")
    inc_bc = engine.broadcast(panel.index, columns=cols)
    pd.testing.assert_frame_equal(
        inc_bc.sort_index(),
        batch_bc.sort_index(),
        check_exact=False,
        rtol=rtol,
        atol=atol,
    )

    # 单点 broadcast 与 merge 结果一致（抽样）
    for i in (0, len(panel) // 2, len(panel) - 1):
        key = panel.index[i]
        dt, inst = key
        for c in cols[: min(5, len(cols))]:
            point = engine.broadcast_at(dt, inst, c)
            merged = float(inc_bc.loc[key, c])
            assert np.isclose(point, merged, rtol=rtol, atol=atol, equal_nan=True), (
                key,
                c,
                point,
                merged,
            )
