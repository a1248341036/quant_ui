# -*- coding: utf-8 -*-
"""alphaagent DSL 因子表达式 -> JQ 运行时桥接(CNE 数据接入版)。

get_factor(expr, date=None) 的底层实现:
1. 首次调用时把 JQContext 的回测窗口(日期 x 代码域)映射到 AlphaAgent 的 CNE
   面板(alphaagent.data.adapters.cnequity.load_panel_from_cne, 与因子实验室
   完全同源同列):
   - 行情 open/high/low/close/amount/volume/turnover_rate + 复权 adj_* +
     估值 pe/pb/ps/dv_ratio + 股本市值 total_share/float_cap/tot_cap +
     财务 funda_* + 股东/资金流/两融/事件等全量 AlphaAgent 列(约 120+ 列)
   - artifacts/panel/cache 磁盘缓存命中时秒级加载(arrow mmap 零拷贝);
     回测区间未覆盖时自动重建一次(30-70s)并落盘, 之后复用
   - 行索引限定在 JQ 域(回测窗口日期 x 域内代码), 不加载全市场 8M+ 行
2. 用 alphaagent.dsl.eval.eval_factor 求值 -> 全历史 MultiIndex Series。
3. JQRuntime.get_factor 取信号日截面返回 Series(index=code)。

说明:
- 面板按整个回测窗口一次性构建(float32, 缓存在 ctx._alpha_panel), 构建约
  数秒~数十秒(含 CNE 面板 attach/裁剪; 缓存未覆盖时含重建); 首个表达式
  import alphaagent DSL(numba/numpy 算子编译)可能再慢 10-60s。
- 同一回测内相同表达式结果按条缓存(ctx._factor_cache, 上限 8): 逐日调度
  策略首日付一次整窗求值, 后续信号日直接复用。
- 表达式语法与 AlphaAgent 因子实验室完全一致: $close/$open/$pe_ttm/
  $funda_net_profit... 引用 CNE 面板列, TS_*/CS_* 算子, 多行表达式末行为输出
  (label_* 未来收益列由 DSL guard 拒绝, 与因子实验室一致)。
- 环境变量 JQ_FACTOR_PANEL_SOURCE=ctx 可切回旧的 JQ 引擎本地 8 列口径;
  CNE 面板加载失败时自动回退该口径, 不阻断回测。
"""
from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_PANEL_SOURCE_ENV = "JQ_FACTOR_PANEL_SOURCE"

# CNE 面板加载锁: 并发 JQ 回测线程串行化 attach/重建(缓存未覆盖时避免
# 并发触发多次 30-70s 整面板重建与内存尖峰)
_CNE_LOAD_LOCK = threading.RLock()


def _load_cne_panel_for_ctx(ctx) -> pd.DataFrame:
    """JQContext 回测窗口 -> CNE AlphaAgent 面板子集(MultiIndex, float32)。

    请求区间 = ctx.tables.dates 覆盖范围(已含回测 lookback 预热); 返回面板只保留
    该区间内、ctx 域内代码的行, 列保持 CNE 面板全列(与因子实验室同口径)。
    """
    from alphaagent.data.adapters.cnequity import load_panel_from_cne

    dates = pd.DatetimeIndex(ctx.tables.dates)
    if not len(dates):
        raise ValueError("JQ 上下文无交易日, 无法构建因子面板")
    start = dates.min().strftime("%Y-%m-%d")
    end = dates.max().strftime("%Y-%m-%d")
    domain = {str(c) for c in ctx.codes}

    t0 = time.time()
    raw = load_panel_from_cne(
        start=start, end=end,
        include_fundamentals=True,
        asset_type="stock",
        universe_mask=False,
    )
    if raw is None or not len(raw):
        raise ValueError(f"CNE 面板为空(start={start}, end={end})")
    if not isinstance(raw.index, pd.MultiIndex):
        if {"datetime", "instrument"}.issubset(raw.columns):
            raw = raw.set_index(["datetime", "instrument"])
        else:
            raise ValueError("CNE 面板缺少 datetime/instrument 索引列")
    dts = raw.index.get_level_values(0)
    inst = raw.index.get_level_values(1).astype(str)
    code6 = inst.str.split(".").str[0].str.zfill(6)
    mask = (
        dts.isin(dates)
        & code6.isin(list(domain))
    )
    panel = raw.loc[mask]
    if not len(panel):
        raise ValueError(
            f"CNE 面板裁剪后为空: 窗口 {start}~{end}, 域 {len(domain)} 只"
        )
    logger.info(
        "JQ factor: CNE 面板就绪 rows=%d cols=%d (%.1fs)",
        len(panel), panel.shape[1], time.time() - t0,
    )
    return panel


def build_ctx_core_panel(ctx) -> pd.DataFrame:
    """JQ 引擎本地 8 列口径(旧实现保留为回退路径)。

    JQContext -> MultiIndex(datetime, instrument) 面板, 前复权 OHLC + amount(千元)
    + volume(手, 近似) + turnover_rate(%) + mv(万元)。行 = ctx 域内有收盘的格子。
    """
    t = ctx.tables
    dates, codes = t.dates, t.codes
    T, K = len(dates), len(codes)

    def _mat(attr):
        m = getattr(t, attr, None)
        return m if m is not None else np.full((T, K), np.nan)

    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = t.close_qfq / t.close_raw        # 复权比 -> OHLC 统一前复权
        ratio = np.where(np.isfinite(ratio) & (ratio > 0), ratio, 1.0)
        close = t.close_qfq
        open_ = t.open_raw * ratio
        high = _mat("high_raw") * ratio
        low = _mat("low_raw") * ratio
        amount = _mat("amount")                   # 千元
        volume = amount * 1e3 / np.where(close > 0, close, np.nan) / 100.0
    tr = (ctx.panel.pivot_table(index="date", columns="code",
                                values="turnover", aggfunc="last")
          .reindex(index=dates, columns=codes).to_numpy())

    idx = pd.MultiIndex.from_product([dates, codes],
                                     names=["datetime", "instrument"])
    df = pd.DataFrame({
        "open": open_.astype(np.float32).ravel(),
        "high": high.astype(np.float32).ravel(),
        "low": low.astype(np.float32).ravel(),
        "close": close.astype(np.float32).ravel(),
        "amount": amount.astype(np.float32).ravel(),
        "volume": volume.astype(np.float32).ravel(),
        "turnover_rate": tr.astype(np.float32).ravel(),
        "mv": _mat("mv").astype(np.float32).ravel(),   # 万元(额外列)
    }, index=idx)
    return df[np.isfinite(df["close"].to_numpy())]


def build_alpha_panel(ctx) -> pd.DataFrame:
    """JQContext -> AlphaAgent 因子面板。

    默认走 CNE 数据接入(与因子实验室同源同列); 环境变量
    JQ_FACTOR_PANEL_SOURCE=ctx 或 CNE 加载失败时回退 JQ 本地 8 列口径。
    CNE attach/重建有进程内锁防抖: 并发 JQ 回测不重复触发 30-70s 的整面板重建。
    """
    source = os.environ.get(_PANEL_SOURCE_ENV, "cne").strip().lower()
    if source == "ctx":
        logger.warning("JQ factor: %s=ctx, 使用 JQ 本地 8 列口径", _PANEL_SOURCE_ENV)
        return build_ctx_core_panel(ctx)
    t0 = time.time()
    with _CNE_LOAD_LOCK:
        try:
            panel = _load_cne_panel_for_ctx(ctx)
            if panel.empty:
                raise ValueError("CNE 面板为空")
            return panel
        except Exception as exc:  # noqa: BLE001 - CNE 不可用不阻断 JQ 回测
            logger.warning(
                "JQ factor: CNE 面板加载失败(%.0fs), 回退 JQ 本地 8 列口径: %s: %s",
                time.time() - t0, type(exc).__name__, exc,
            )
            return build_ctx_core_panel(ctx)


# 同 ctx 内按表达式缓存求值结果(整窗确定性结果): 每日调度的策略只首日付
# 一次整窗求值成本, 之后各信号日直接复用; 上限 8 条防常驻内存膨胀。
_FACTOR_CACHE_MAX = 8


def factor_series(ctx, expr) -> pd.Series:
    """在 ctx 域面板上求值 DSL 表达式, 返回 MultiIndex Series。

    同一 ctx 内重复表达式命中缓存(表达式与面板确定 => 结果确定)。
    """
    expr = str(expr)
    panel = getattr(ctx, "_alpha_panel", None)
    if panel is None:
        t0 = time.time()
        with _CNE_LOAD_LOCK:
            panel = build_alpha_panel(ctx)
        logger.info("JQ factor: 因子面板构建完成(%.1fs), 行数=%d, 列数=%d",
                    time.time() - t0, len(panel), panel.shape[1])
        try:
            ctx._alpha_panel = panel
        except AttributeError:
            pass
    cache = getattr(ctx, "_factor_cache", None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            ctx._factor_cache = cache
        except AttributeError:
            pass
    ser = cache.get(expr)
    if ser is None:
        from alphaagent.dsl.eval import eval_factor
        ser = eval_factor(expr, panel, operator_monitor=False)
        cache[expr] = ser
        if len(cache) > _FACTOR_CACHE_MAX:
            for k in list(cache)[: len(cache) - _FACTOR_CACHE_MAX]:
                cache.pop(k, None)
    return ser
