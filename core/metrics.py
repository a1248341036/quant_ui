from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(nav: pd.Series, periods_per_year: int = 244) -> dict:
    nav = nav.dropna()
    if len(nav) < 2:
        return {"总收益": np.nan, "年化收益": np.nan, "年化波动": np.nan,
                "夏普": np.nan, "最大回撤": np.nan, "卡玛": np.nan,
                "胜率": np.nan}

    rets = nav.pct_change().dropna()
    total = nav.iloc[-1] / nav.iloc[0] - 1
    n = len(rets)
    ann = (1 + total) ** (periods_per_year / n) - 1 if total > -1 else -1.0
    vol = rets.std(ddof=1) * np.sqrt(periods_per_year)
    # 算术年化夏普（行业惯例，与旧 backtest_5w 口径一致）
    sharpe = (rets.mean() * periods_per_year) / vol if vol and not np.isnan(vol) else np.nan

    running_max = nav.cummax()
    drawdown = nav / running_max - 1
    max_dd = float(drawdown.min())
    calmar = ann / abs(max_dd) if max_dd < 0 else np.nan

    win = float((rets > 0).mean()) if len(rets) else np.nan
    return {
        "总收益": float(total),
        "年化收益": float(ann),
        "年化波动": float(vol),
        "夏普": float(sharpe),
        "最大回撤": float(max_dd),
        "卡玛": float(calmar),
        "胜率": win,
    }


def compute_excess_metrics(nav: pd.Series, bench: pd.Series) -> dict:
    """Compute active performance from aligned strategy and benchmark NAVs."""
    aligned = pd.concat([nav.rename("nav"), bench.rename("bench")], axis=1).dropna()
    if len(aligned) < 2:
        return {"超额年化": np.nan, "超额夏普": np.nan}

    active_nav = aligned["nav"] / aligned["bench"]
    active_metrics = compute_metrics(active_nav)
    return {
        "超额年化": active_metrics["年化收益"],
        "超额夏普": active_metrics["夏普"],
    }


def drawdown_series(nav: pd.Series) -> pd.Series:
    return nav / nav.cummax() - 1


# ============================================================
# 聚宽回测详情面板同口径指标
# 口径经聚宽官方导出样本反推校验(年化因子 250, 无风险利率 4%,
# 几何超额, 日胜率=策略日收益>基准日收益, 索提诺 MAR=日无风险利率 ddof=0,
# 超额夏普=(超额年化-4%)/算术日超额TE年化), 见 scripts/jq_repro/_calib_jq_metrics*.py
# ============================================================
_JQ_RF = 0.04
_JQ_ANN = 250


def _jq_ann(total: float, n: int) -> float:
    return (1 + total) ** (_JQ_ANN / n) - 1 if n else np.nan


def _jq_max_dd(curve: pd.Series):
    """返回 (最大回撤, 峰值日, 谷底日)。"""
    running = curve.cummax()
    dd = curve / running - 1
    trough = dd.idxmin()
    peak = curve.loc[:trough].idxmax()
    return float(dd.min()), peak, trough


def _jq_round_trips(fills: list[dict]) -> list[float]:
    """逐笔成交 -> FIFO 回合盈亏(含费, 面板价空间)。"""

    def _fee(entry: dict) -> float:
        return float(entry.get("fee") or 0.0)

    lots: dict[str, list[list[float]]] = {}   # code -> [[shares, cost/股], ...]
    pnl: list[float] = []
    for f in fills or []:
        code = str(f.get("code"))
        side = str(f.get("side"))
        px = float(f.get("price") or 0.0)
        sh = float(f.get("shares") or 0.0)
        if sh <= 0 or px <= 0:
            continue
        if side == "buy":
            fee = _fee(f)
            lots.setdefault(code, []).append([sh, px + fee / sh])
        else:
            queue = lots.get(code) or []
            sell_fee = _fee(f)
            remaining = sh
            gain = 0.0
            while remaining > 1e-9 and queue:
                q = queue[0]
                take = min(q[0], remaining)
                gain += take * (px - sell_fee / sh - q[1])
                q[0] -= take
                remaining -= take
                if q[0] <= 1e-9:
                    queue.pop(0)
            pnl.append(gain)
    return pnl


def compute_jq_panel(nav: pd.Series, bench: pd.Series | None,
                     fills: list[dict] | None = None) -> dict:
    """聚宽回测详情面板同口径指标(中文名, 与聚宽逐项对齐)。

    契约(与聚宽基点语义一致, 已用聚宽官方导出双样本逐位校验):
    - nav: 策略净值, 长度 n, 首点 = 1+首日收益(基点=初始资金, 窗口前);
      索引为交易日
    - bench: 基准净值, 与 nav 同长同索引, 值 = 指数收盘/回测前收盘
      (基点 = 窗口首日**之前**的收盘, 故首点 = 1+首日基准涨跌, 一般 != 1)
    - fills: 引擎逐笔成交(code/side/shares/price/fee), 回合盈亏统计
    """
    nav = nav.dropna()
    n = len(nav)
    if n < 2:
        return {"策略收益": np.nan}
    # 基点 = 窗口前(1.0): 策略收益 = nav[-1] - 1;
    # 日收益 = 逐日 pct_change, 首日 = nav[0] - 1(相对初始资金)
    total_s = float(nav.iloc[-1] - 1.0)
    rets = nav.pct_change()
    rets.iloc[0] = float(nav.iloc[0] - 1.0)
    rets = rets.dropna()
    ann_s = _jq_ann(total_s, n)
    vol_s = float(rets.std(ddof=1) * np.sqrt(_JQ_ANN))

    mdd, peak, trough = _jq_max_dd(nav)

    out: dict = {}
    out["策略收益"] = total_s
    out["策略年化收益"] = ann_s
    out["基准收益"] = np.nan
    out["超额收益"] = np.nan
    out["阿尔法"] = np.nan
    out["贝塔"] = np.nan
    out["夏普比率"] = (ann_s - _JQ_RF) / vol_s if vol_s else np.nan
    out["索提诺比率"] = np.nan
    out["胜率"] = np.nan
    out["盈亏比"] = np.nan
    out["盈利次数"] = np.nan
    out["亏损次数"] = np.nan
    out["最大回撤"] = mdd
    out["最大回撤区间"] = f"{pd.Timestamp(peak).date()}~{pd.Timestamp(trough).date()}"
    out["策略波动率"] = vol_s
    out["基准波动率"] = np.nan
    out["日胜率"] = np.nan
    out["信息比率"] = np.nan
    out["日均超额收益"] = np.nan
    out["超额收益最大回撤"] = np.nan
    out["超额收益夏普比率"] = np.nan

    if bench is not None and len(bench) > 1:
        aligned = pd.concat([nav.rename("s"), bench.rename("b")],
                            axis=1).dropna()
        if len(aligned) > 2:
            b = aligned["b"]
            # 基准基点 = 窗口前(1.0): 收益 = b[-1] - 1; 日收益首日 = b[0]-1
            total_b = float(b.iloc[-1] - 1.0)
            rb = b.pct_change()
            rb.iloc[0] = float(b.iloc[0] - 1.0)
            rs = aligned["s"].pct_change()
            rs.iloc[0] = float(aligned["s"].iloc[0] - 1.0)
            rb = rb.dropna()
            rs = rs.dropna()
            nb = len(rb)
            ann_b = _jq_ann(total_b, nb)
            vol_b = float(rb.std(ddof=1) * np.sqrt(_JQ_ANN))
            beta = float(rs.cov(rb) / rb.var(ddof=1)) if rb.var(ddof=1) else np.nan
            alpha = ann_s - (_JQ_RF + beta * (ann_b - _JQ_RF))
            excess = (1 + total_s) / (1 + total_b) - 1
            ex_ann = (1 + excess) ** (_JQ_ANN / nb) - 1
            te = float((rs - rb).std(ddof=1) * np.sqrt(_JQ_ANN))
            ratio = aligned["s"] / aligned["b"]
            mdd_x, _, _ = _jq_max_dd(ratio)
            out["基准收益"] = total_b
            out["超额收益"] = excess
            out["阿尔法"] = alpha
            out["贝塔"] = beta
            out["基准波动率"] = vol_b
            out["日均超额收益"] = excess / nb
            out["超额收益最大回撤"] = mdd_x
            out["超额收益夏普比率"] = ((ex_ann - _JQ_RF) / te) if te else np.nan
            out["信息比率"] = ((ann_s - ann_b) / te) if te else np.nan
            out["日胜率"] = float((rs > rb).mean())

    dn = rets[rets < _JQ_RF / _JQ_ANN]
    if len(dn) > 1:
        dn_vol = float(dn.std(ddof=0) * np.sqrt(_JQ_ANN))
        out["索提诺比率"] = (ann_s - _JQ_RF) / dn_vol if dn_vol else np.nan

    if fills:
        pnl = _jq_round_trips(fills)
        wins = [x for x in pnl if x > 0]
        losses = [x for x in pnl if x <= 0]
        out["盈利次数"] = len(wins)
        out["亏损次数"] = len(losses)
        out["胜率"] = (len(wins) / len(pnl)) if pnl else np.nan
        avg_w = float(np.mean(wins)) if wins else 0.0
        avg_l = abs(float(np.mean(losses))) if losses else np.nan
        out["盈亏比"] = (avg_w / avg_l) if avg_l else np.nan
    return out
