"""可成交域掩码（L1 透镜）：把引擎的「买得到吗」约束提前到统计评估层。

用途：诊断「统计口径好、引擎口径差」里属于**可成交性**的那部分缺口。本模块
只定义**候选域**，不复制引擎的盈亏口径（整手撮合/先卖后买/拒单不补仓/逐笔
费用仍由 core.engine 负责）——所以它是筛选透镜，不是净值替代品。

三处判据均对齐 core.engine 的同一真源（不重写语义）：

1. **可负担**（本轮实测的主要拒单原因：101/103 笔「现金不足/预算过小」）
   引擎侧（core/execution.py `execute_targets`）：
   ``budget = portfolio_value * pct``，``px = open[t]*(1+slippage+spread)*(1+impact)``，
   ``want_lots = int(budget // (px*(1+buy_cost)) // lot_size)``，``lots <= 0`` 即拒单。
   即买得起一手要求 ``open[t]*lot_size <= budget``。
   - 价格取**执行日（信号日下一交易日）open**，不是信号日 close；
   - 价格是面板**原始 open**（不复权），不是 adj_open——复权价与真实报价可
     差数十倍，用错会把可负担性算得完全失真；
   - budget 按等权近似 ``capital / target_count``（引擎实际用组合市值/权重）。

2. **次日买不进**
   引擎在信号日 sig 打分、t=sig 的下一交易日开盘执行，要求
   ``valid_open[t]`` 且 ``~limit_up[t]``（涨停买不进）。涨停判据复用
   ``core.limit.build_limit_flags``（单一真源，按板块 10/20/30% 与开盘涨幅）。

3. **流动性下限**
   engine_gate 把 ``am20 < min_am20_yuan`` 的打分置 NaN；执行层另要求
   ``turnover > 0``。am20 口径同 ``core.panel_schema.alpha_panel_to_engine_frame``
   （amount 千元 ×1000 → 元，按票 ``rolling(20, min_periods=5)`` 均值）。

成本：掩码按 (面板索引身份, 参数) 缓存（weakref + is 校验，防 id 复用误命中）。
同一 session 内每次评估共用同一份掩码，首次构建付一次成本，之后摊销为 0。
"""
from __future__ import annotations

import threading
import weakref
from collections import OrderedDict

import numpy as np
import pandas as pd

# 掩码缓存：键 = (id(index), 参数元组)；值 = (index 弱引用, bool ndarray)
_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_CACHE_LOCK = threading.Lock()
_CACHE_MAX = 8

_AMOUNT_CNE_TO_ENGINE = 1000.0  # 与 core.panel_schema 同源常量（千元 → 元）


def _am20_yuan_wide(panel: pd.DataFrame) -> pd.DataFrame | None:
    """按票 20 日均成交额（元）宽表，口径同 core.panel_schema（千元 ×1000）。

    用宽表向量化 rolling（等价于引擎按票 groupby rolling：NaN 日不计入均值、
    窗口按市场交易日推进；停牌缺失日的窗口语义差异对 5e6 元量级的流动性下限
    判定无影响），避免 5000 组 × Python lambda transform 的秒级开销。
    """
    if "amount" not in panel.columns:
        return None
    amount_w = _wide(panel, "amount").astype(np.float64, copy=False) * _AMOUNT_CNE_TO_ENGINE
    return amount_w.rolling(20, min_periods=5).mean()


def _wide(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    """长表列 → 宽表（行=交易日，列=代码）。重复 (date, code) 时取均值。

    面板按 (datetime, instrument) 唯一时走 unstack 快路径；重复时回落
    pivot_table（避免 unstack 抛错直接中断评估）。
    """
    s = pd.to_numeric(panel[col], errors="coerce")
    try:
        return s.unstack("instrument")
    except (ValueError, KeyError):
        return s.reset_index().pivot_table(
            index="datetime", columns="instrument", values=col, aggfunc="mean"
        )


def _empty_mask(panel: pd.DataFrame, reason: str) -> tuple[pd.Series, dict]:
    mask = pd.Series(True, index=panel.index, name="tradable", dtype=bool)
    return mask, {"available": False, "reason": reason, "coverage": 1.0}


def _cache_key(panel: pd.DataFrame, params: tuple) -> tuple:
    return (id(panel.index), params)


def tradable_mask(
    panel: pd.DataFrame,
    *,
    capital: float,
    target_count: int,
    lot_size: int = 100,
    min_am20_yuan: float = 0.0,
    cost_padding: float = 0.001,
    include_limit: bool = True,
    include_affordable: bool = True,
    include_liquidity: bool = True,
) -> tuple[pd.Series, dict]:
    """返回 ``(掩码 Series[bool], 诊断 dict)``；True = 该 (date, code) 在信号日可入选。

    掩码语义：**信号日**该票既在可成交域内（次日能买到、买得起、够流动），
    统计层据此排名即可近似引擎的候选池。掩码为 False 的票仍参与 IC/分组
    计算——只有显式把掩码传给 ``quantile_portfolio_metrics(eligibility=...)``
    时才收窄组合域（保持主口径零变化）。
    """
    need = ("open", "close")
    missing = [c for c in need if c not in panel.columns]
    if missing:
        return _empty_mask(panel, f"panel_missing_columns:{missing}")
    if panel.empty or not isinstance(panel.index, pd.MultiIndex):
        return _empty_mask(panel, "panel_empty_or_not_multiindex")

    params = (
        round(float(capital), 6), int(target_count), int(lot_size),
        round(float(min_am20_yuan), 6), round(float(cost_padding), 8),
        bool(include_limit), bool(include_affordable), bool(include_liquidity),
    )
    key = _cache_key(panel, params)
    with _CACHE_LOCK:
        hit = _CACHE.get(key)
        if hit is not None:
            ref, cached_mask, cached_diag = hit
            if ref() is panel.index:
                _CACHE.move_to_end(key)
                return pd.Series(cached_mask, index=panel.index, name="tradable"), dict(cached_diag)
            _CACHE.pop(key, None)

    open_w = _wide(panel, "open")
    close_w = _wide(panel, "close")
    dates = open_w.index
    codes = open_w.columns
    diag: dict = {"available": True, "params": {
        "capital": float(capital), "target_count": int(target_count),
        "lot_size": int(lot_size), "min_am20_yuan": float(min_am20_yuan),
    }}

    ok_wide = open_w.notna() & close_w.notna()

    # ── 1. 可负担：执行日（下一交易日）开盘一手成本 <= 等权预算 ──
    if include_affordable:
        budget = float(capital) / max(int(target_count), 1)
        open_next = open_w.shift(-1)
        per_lot_cost = open_next.astype(np.float64) * float(lot_size) * (1.0 + float(cost_padding))
        ok_wide &= per_lot_cost.le(budget) & open_next.notna()
        diag["budget_per_name"] = round(budget, 2)
        diag["affordable_max_price"] = round(budget / max(int(lot_size), 1), 2)

    # ── 2. 次日买不进：开盘涨停 / 停牌 ──
    if include_limit and len(open_w):
        from core.limit import build_limit_flags

        try:
            limit_up, _limit_down, _one_up, _one_down = build_limit_flags(close_w, open_w)
            # 下一交易日涨停对齐到当前行：row i ← limit_up[i+1]，末日无次日 = False
            limit_up_next = np.vstack([limit_up[1:], np.zeros((1, limit_up.shape[1]), dtype=bool)])
            ok_wide &= ~limit_up_next
        except Exception as exc:  # noqa: BLE001 — 涨停判定失败不应阻断评估
            diag["limit_flags_error"] = f"{type(exc).__name__}: {exc}"

    # ── 3. 流动性下限（信号日 am20）与成交量 ──
    if include_liquidity:
        if min_am20_yuan > 0:
            am20_w = _am20_yuan_wide(panel)
            if am20_w is not None:
                # am20 NaN（不足 5 日观测）→ 不可入选：执行层对 NaN am20 一律拒单
                ok_wide &= am20_w.reindex(index=dates, columns=codes).ge(float(min_am20_yuan))
        if "turnover_rate" in panel.columns:
            turn_w = _wide(panel, "turnover_rate")
            ok_wide &= turn_w.reindex(index=dates, columns=codes).gt(0)

    # ── 宽表 → 与 panel.index 对齐 ──
    arr = ok_wide.to_numpy(dtype=bool)
    d_pos = pd.Index(dates).get_indexer(panel.index.get_level_values("datetime"))
    c_pos = pd.Index(codes).get_indexer(panel.index.get_level_values("instrument"))
    found = (d_pos >= 0) & (c_pos >= 0)
    mask = np.zeros(len(panel), dtype=bool)
    if found.any():
        mask[found] = arr[d_pos[found], c_pos[found]]

    diag["coverage"] = round(float(mask.mean()), 4)
    diag["n_rows"] = int(len(mask))
    try:
        day_any = pd.Series(mask, index=panel.index).groupby(level="datetime").mean()
        diag["avg_daily_coverage"] = round(float(day_any.mean()), 4)
        diag["min_daily_coverage"] = round(float(day_any.min()), 4)
    except Exception:  # noqa: BLE001
        pass

    with _CACHE_LOCK:
        _CACHE[key] = (weakref.ref(panel.index), mask, dict(diag))
        _CACHE.move_to_end(key)
        while len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)

    return pd.Series(mask, index=panel.index, name="tradable"), diag


def clear_cache() -> None:
    """清空掩码缓存（测试用）。"""
    with _CACHE_LOCK:
        _CACHE.clear()
