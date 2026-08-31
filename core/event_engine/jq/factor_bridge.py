# -*- coding: utf-8 -*-
"""alphaagent DSL 因子表达式 -> JQ 运行时桥接。

get_factor(expr, date=None) 的底层实现:
1. 把 JQContext 的域数据(date x code 矩阵)组装成 AlphaAgent panel 契约
   (core/panel_schema.ALPHA_CNE_PANEL_SPEC):
   MultiIndex(datetime, instrument), 列 open/high/low/close(前复权, 元),
   amount(千元), turnover_rate(%), volume(手,近似), 另附 mv(万元)。
2. 用 alphaagent.dsl.eval.eval_factor 求值 -> 全历史 MultiIndex Series。
3. JQRuntime.get_factor 取信号日截面返回 Series(index=code)。

说明:
- 面板按整个回测窗口一次性构建(float32, 缓存在 ctx._alpha_panel);
  典型 3-5 年窗口约占 200-300MB 内存, 构建约 2-5s。
- 首次调用会 import alphaagent DSL(numaba/numpy 算子编译), 首个表达式可能
  慢 10-60s, 之后走磁盘/内存缓存。
- 表达式语法与 AlphaAgent 因子实验室完全一致: $close/$open/$amount...
  引用列, TS_*/CS_* 算子, 多行表达式最后一行为输出。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def build_alpha_panel(ctx) -> pd.DataFrame:
    """JQContext -> AlphaAgent panel(全 OHLC 前复权一致口径)。"""
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


def factor_series(ctx, expr) -> pd.Series:
    """在 ctx 域面板上求值 DSL 表达式, 返回 MultiIndex Series。"""
    panel = getattr(ctx, "_alpha_panel", None)
    if panel is None:
        panel = build_alpha_panel(ctx)
        try:
            ctx._alpha_panel = panel
        except AttributeError:
            pass
    from alphaagent.dsl.eval import eval_factor
    return eval_factor(str(expr), panel, operator_monitor=False)
