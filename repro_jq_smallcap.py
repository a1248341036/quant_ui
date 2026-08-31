# -*- coding: utf-8 -*-
"""复现聚宽《致敬市场(15)——小盘三正低风险中等收益策略》(实盘降噪版 v2.0)

用本平台日线事件驱动引擎 (core.event_engine) 复现。近似点（与聚宽原版的差异）：
1) 日内时序(9:05/10:00/14:00/14:50)折叠为 "T日收盘信号 -> T+1开盘成交"；
   14:00 涨停开板卖出近似为: 昨收==涨停价 且 今日开盘 < 昨日涨停价 -> 开盘卖出
2) 选股域: 399101 中小板综历史成分数据缺失 -> 沪深主板(00/60前缀) + 策略自身硬过滤
   (非ST/非停牌/上市满375天/净利>0/营收>1亿/市值3~1000亿)
3) MA 择时基准: 中小板综指日线缺失 -> 用面板自算的域等权指数,
   原 sigmoid(diff/500) 按指数点位~2万分之500 等效换算为相对幅度 diff_pct/2.5%
4) 4月空仓持币(本地无 511260 国债ETF历史净值)
5) 涨停卖出后当日补仓(check_remain_amount)省略, 由下一调仓日自然补足
6) 审计意见/分红过滤原版默认关闭, 未实现
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(r"D:\Quant\quant_ui")
sys.path.insert(0, str(ROOT))

QDATA = ROOT / "data" / "quant_dataset"
PG = ROOT / "data" / "pg_parquet"
OUT = ROOT / "repro_out"
OUT.mkdir(exist_ok=True)

LOAD_FROM = "2015-11-01"   # 预热缓冲(am20/MA10)
START = "2016-01-04"
END = "2026-08-28"
CAPITAL = 1_000_000.0

# 策略参数（与原版一致）
STOCK_NUM = 7
MAX_SINGLE_WEIGHT = 0.12
MAX_SMALLCAP_EXPOSURE = 0.70
STOPLOSS_LIMIT = 0.07
HIGHEST = 60.0            # 元
MIN_MV = 3e4              # 万元 (3亿)
MAX_MV = 1e7              # 万元 (1000亿)
LISTED_DAYS = 375
PASS_MONTHS = (4,)
WARMUP_DAYS = 45

t0 = time.time()


def log(msg: str) -> None:
    print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)


# ============================================================
# 1. 行情面板（未复权 -> 前复权）
# ============================================================
def build_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = ["ts_code", "trade_date", "open", "close", "pre_close", "amount",
            "adj_factor", "up_limit", "total_mv", "turnover_rate", "is_st",
            "listed_days"]
    years = sorted({p.name for p in QDATA.iterdir() if p.is_dir() and p.name.isdigit()})
    frames, meta_frames = [], []
    for y in years:
        f = QDATA / y / y / "day" / "stock_daily.parquet"
        if not f.exists():
            continue
        try:
            schema_names = set(pq.read_schema(f).names)
        except Exception:
            schema_names = set(cols)
        usecols = [c for c in cols if c in schema_names]
        df = pd.read_parquet(f, columns=usecols)
        df["trade_date"] = pd.to_datetime(df["trade_date"])
        df = df[(df["trade_date"] >= LOAD_FROM) & (df["trade_date"] <= END)]
        if df.empty:
            continue
        if "listed_days" not in df.columns:
            df["listed_days"] = np.nan
        meta_frames.append(df[["ts_code", "trade_date", "pre_close", "up_limit",
                               "total_mv", "is_st", "listed_days"]].copy())
        # 域过滤：主板 00/60
        code6 = df["ts_code"].str[:6]
        df = df[code6.str.startswith(("00", "60"))]
        if df.empty:
            continue
        frames.append(df[["ts_code", "trade_date", "open", "close", "amount",
                          "adj_factor", "turnover_rate"]])
        log(f"  {y}: {len(df):,} rows")
    raw = pd.concat(frames, ignore_index=True)
    meta = pd.concat(meta_frames, ignore_index=True)

    raw["code"] = raw["ts_code"].str[:6]
    raw = raw.sort_values(["code", "trade_date"], kind="stable")

    # 前复权因子：每股以其最后交易日的 adj_factor 为锚
    last_factor = raw.groupby("code")["adj_factor"].last()
    raw["factor_last"] = raw["code"].map(last_factor)
    adj = raw["adj_factor"] / raw["factor_last"]
    raw["open"] = raw["open"] * adj
    raw["close"] = raw["close"] * adj

    # am20（元）
    raw["am20"] = (raw["amount"] * 1e3).groupby(raw["code"]).transform(
        lambda s: s.rolling(20, min_periods=5).mean())

    panel = pd.DataFrame({
        "date": raw["trade_date"],
        "code": raw["code"],
        "open": raw["open"],
        "close": raw["close"],
        "turnover": raw["turnover_rate"],
        "am20": raw["am20"],
        "close_raw": raw["close"] / adj,  # 未复权收盘（60元价格过滤用）
    })
    panel = panel.dropna(subset=["close"], how="any")
    panel = panel[(panel["open"] > 0) & (panel["close"] > 0)]   # 防 o2o inf
    log(f"panel: {len(panel):,} rows, {panel['code'].nunique()} codes, "
        f"{panel['date'].min().date()} ~ {panel['date'].max().date()}")
    return panel, meta


# ============================================================
# 2. 预计算：市值/ST/上市天数/涨停/财务/候选池/MA仓位
# ============================================================
def precompute(panel: pd.DataFrame, meta: pd.DataFrame) -> dict:
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    codes = sorted(panel["code"].unique())
    dmap = {d: i for i, d in enumerate(dates)}
    cmap = {c: j for j, c in enumerate(codes)}
    T, K = len(dates), len(codes)
    log(f"precompute grid: {T} days x {K} codes")

    meta["code"] = meta["ts_code"].str[:6]
    m = meta[meta["code"].isin(cmap)].copy()
    m["di"] = m["trade_date"].map(dmap)
    m["ki"] = m["code"].map(cmap)
    m = m.dropna(subset=["di", "ki"])
    m["di"] = m["di"].astype(int)
    m["ki"] = m["ki"].astype(int)

    def scatter(col: str, dtype) -> np.ndarray:
        a = np.full((T, K), np.nan, dtype=dtype)
        a[m["di"].values, m["ki"].values] = m[col].astype(dtype).values
        return a

    mv = scatter("total_mv", np.float64)          # 万元
    up_limit = scatter("up_limit", np.float64)    # 元(未复权)
    st = scatter("is_st", np.float64) == 1

    # 未复权收盘（价格过滤/涨停判定）
    close_raw = np.full((T, K), np.nan)
    cp = panel.pivot_table(index="date", columns="code", values="close_raw",
                           aggfunc="last").reindex(index=dates, columns=codes)
    close_raw[:] = cp.values

    # 上市天数（day 文件逐日 listed_days，点时口径；缺失回退 stock_basic）
    listed = scatter("listed_days", np.float64)
    basic = pd.read_parquet(PG / "stock_basic.parquet",
                            columns=["ts_code", "name", "list_date"])
    basic["code"] = basic["ts_code"].str[:6]
    basic = basic.drop_duplicates("code").set_index("code")
    list_date = pd.to_datetime(basic["list_date"], errors="coerce")
    ld = pd.to_datetime(list_date.reindex(codes)).values.astype("datetime64[D]")
    age_fallback = ((dates.values.astype("datetime64[D]")[:, None] - ld[None, :]) /
                    np.timedelta64(1, "D"))
    age_ok = np.where(np.isfinite(listed), listed >= LISTED_DAYS,
                      age_fallback >= LISTED_DAYS)
    name = basic["name"].reindex(codes).fillna("")
    delist_name = name.str.contains("退").values

    # 停牌：当日无有效收盘价即视为不可交易
    paused = ~np.isfinite(close_raw)

    # 涨停收盘（精确涨停价）
    hl = np.zeros((T, K), dtype=bool)
    ok = np.isfinite(close_raw) & np.isfinite(up_limit)
    hl[ok] = close_raw[ok] >= up_limit[ok] - 1e-4

    # 财务：最近公告报表 归母净利>0 且 净利润>0 且 营收>1亿
    inc = pd.read_parquet(PG / "income.parquet",
                          columns=["ts_code", "ann_date", "end_date",
                                   "n_income", "n_income_attr_p", "revenue",
                                   "report_type"])
    inc = inc[pd.to_numeric(inc["report_type"], errors="coerce") == 1]
    inc["code"] = inc["ts_code"].str[:6]
    inc = inc[inc["code"].isin(cmap)]
    inc["ann_date"] = pd.to_datetime(inc["ann_date"], errors="coerce")
    inc["end_date"] = pd.to_datetime(inc["end_date"], errors="coerce")
    inc = inc.dropna(subset=["ann_date"])
    inc = inc.sort_values(["code", "end_date", "ann_date"], kind="stable")
    inc = inc.drop_duplicates(["code", "end_date"], keep="last")
    inc = inc.sort_values(["code", "ann_date"], kind="stable")

    cal_d = dates.values.astype("datetime64[D]")
    fin_ok = np.zeros((T, K), dtype=bool)
    for code, g in inc.groupby("code", sort=False):
        j = cmap[code]
        ann = g["ann_date"].values.astype("datetime64[D]")
        flag = ((g["n_income_attr_p"].values > 0) &
                (g["n_income"].values > 0) &
                (g["revenue"].values > 1e8))
        pos = np.searchsorted(ann, cal_d, side="right") - 1
        valid = pos >= 0
        fin_ok[valid, j] = flag[pos[valid]]

    # 候选池：市值硬过滤 + 财务 + ST/停牌/次新/退市名
    eligible = ((mv >= MIN_MV) & (mv <= MAX_MV) & fin_ok & ~st &
                ~paused & age_ok & ~delist_name[None, :])
    mv_e = np.where(eligible, mv, np.inf)
    cand: dict[pd.Timestamp, list] = {}
    top_k = 50
    for i, d in enumerate(dates):
        row = mv_e[i]
        if not np.isfinite(row).any():
            cand[d] = []
            continue
        idx = np.argpartition(row, min(top_k, K - 1))[:top_k]
        idx = idx[np.isfinite(row[idx])]
        idx = idx[np.argsort(row[idx], kind="stable")]
        cand[d] = [(codes[j], bool(close_raw[i, j] <= HIGHEST), bool(hl[i, j]))
                   for j in idx]

    # 域等权指数 + MA10 仓位（复刻原 sigmoid，相对幅度等效 500点/2万点）
    # 日收益裁剪 ±20%：剔除前复权微价股/长期停牌复牌造成的伪影收益
    cl = panel.pivot_table(index="date", columns="code", values="close",
                           aggfunc="last").reindex(index=dates, columns=codes)
    ret = cl.pct_change(fill_method=None).clip(-0.2, 0.2)
    ew_ret = ret.mean(axis=1, skipna=True).fillna(0.0)
    level = (1.0 + ew_ret).cumprod()
    ma10 = level.rolling(10, min_periods=1).mean()
    diff_rel = (level - ma10) / ma10
    frac = 1.0 / (1.0 + np.exp(-diff_rel / 0.025))
    num_raw = (STOCK_NUM - 3 * frac).round()          # 与原版 int(round(x)) 一致
    ma_num = {d: int(min(STOCK_NUM, max(4, v))) for d, v in num_raw.items()}

    # 涨停股的涨停价（供开板近似）
    hl_price: dict[pd.Timestamp, dict[str, float]] = {}
    for i, d in enumerate(dates):
        js = np.where(hl[i])[0]
        if len(js):
            hl_price[d] = {codes[j]: float(up_limit[i, j]) for j in js}

    n_elig = np.mean([len(v) for v in cand.values()])
    log(f"precompute done: 平均每日合格候选 {n_elig:.0f} 只, "
        f"hl日均值 {np.mean(hl):.4f}, st占比 {np.mean(st):.4f}")
    return {"dates": dates, "codes": codes, "dmap": dmap, "cand": cand,
            "ma_num": ma_num, "hl_price": hl_price, "bench_level": level}


# ============================================================
# 3. 策略类
# ============================================================
def make_strategy(PREP: dict, buy_cost: float):
    from core.event_engine import EventStrategy

    class SmallCapJQ(EventStrategy):
        """小市值三正策略 —— 日线近似复现。"""

        def init(self, ctx) -> None:
            self.cost: dict[str, float] = {}

        # ---- 工具 ----
        def _pv(self, ctx) -> float:
            return ctx.portfolio_value or CAPITAL

        def _sell(self, ctx, code: str) -> None:
            ctx.order_target_pct(code, 0.0)
            self.cost.pop(code, None)

        # ---- 每日 ----
        def on_bar(self, ctx, bar) -> None:
            sig = bar.date
            # 清理已不存在的成本记录
            for c in list(self.cost):
                if c not in ctx.positions:
                    del self.cost[c]

            # 4月空仓（原版 close_account + 空仓月持币）
            if bar.exec_date.month in PASS_MONTHS:
                for c in list(ctx.positions):
                    self._sell(ctx, c)
                return

            self._daily_risk(ctx, bar, sig)

            # 周二调仓（原版 run_weekly(..., 2, '10:00')）
            if bar.exec_date.weekday() == 1:
                self._weekly(ctx, bar, sig)

        # ---- 止损/止盈/大盘惨跌/涨停开板 ----
        def _daily_risk(self, ctx, bar, sig) -> None:
            hl_price = PREP["hl_price"].get(sig, {})
            for code in list(ctx.positions):
                ac = self.cost.get(code)
                px = ctx.last_close(code)
                if ac and px:
                    if px >= ac * 2:
                        self._sell(ctx, code)          # 100%止盈
                        continue
                    if px < ac * (1 - STOPLOSS_LIMIT):  # 个股止损
                        self._sell(ctx, code)
                        continue
                # 涨停开板近似：昨收涨停 + 今日开盘未封死 -> 开盘卖出
                lim = hl_price.get(code)
                if lim is not None and code in bar.open and bar.open[code] < lim:
                    self._sell(ctx, code)

            # 大盘惨跌：域内平均(1-收/开) >= 5%（原版 399101 平均降幅）
            o, c = bar.open, bar.close
            both = [1.0 - c[s] / o[s] for s in c if s in o and o[s] > 0]
            if both and float(np.mean(both)) >= 0.05:
                for code in list(ctx.positions):
                    self._sell(ctx, code)

        # ---- 周度调仓 ----
        def _weekly(self, ctx, bar, sig) -> None:
            num = PREP["ma_num"].get(sig, STOCK_NUM)
            target: list[str] = []
            for code, price_ok, is_hl in PREP["cand"].get(sig, []):
                if len(target) >= num:
                    break
                if code in ctx.positions:
                    target.append(code)            # 持仓豁免价格/涨停过滤
                elif price_ok and not is_hl:
                    target.append(code)

            # 卖出：不在目标 且 昨日未涨停（原版豁免昨日涨停, 含全部持仓）
            hl_set = set(PREP["hl_price"].get(sig, {}))
            for code in list(ctx.positions):
                if code not in target and code not in hl_set:
                    self._sell(ctx, code)

            # 买入：新进目标，受单票12%与小票总暴露70%约束
            buy_list = [s for s in target if s not in ctx.positions]
            if not buy_list:
                return
            pv = self._pv(ctx)
            exposure = sum(ctx.position_value(s) for s in ctx.positions) / pv
            sold = sum(ctx.position_value(s) for s in ctx.positions
                       if s not in target) / pv
            avail = MAX_SMALLCAP_EXPOSURE - (exposure - sold)
            per = min(MAX_SINGLE_WEIGHT, max(avail, 0.0) / len(buy_list))
            if per <= 0:
                return
            for code in buy_list:
                ctx.order_target_pct(code, per)
                op = bar.open.get(code)
                if op:
                    self.cost[code] = op * (1 + buy_cost)

    return SmallCapJQ


# ============================================================
# 4. 主流程
# ============================================================
def main() -> None:
    log("构建行情面板...")
    panel, meta = build_panel()
    log("预计算选股表/MA仓位...")
    prep = precompute(panel, meta)

    codes = prep["codes"]
    panel = panel[panel["code"].isin(codes)]

    buy_cost, sell_cost, slippage_bps = 0.0001, 0.0011, 0.0   # 聚宽原版费率
    from core.event_engine import run_event_backtest
    strat = make_strategy(prep, buy_cost)
    log(f"开始回测 {START} ~ {END}, 资金 {CAPITAL:,.0f}")
    res = run_event_backtest(
        panel=panel, codes=codes, strategy_class=strat,
        start=START, end=END, capital=CAPITAL,
        buy_cost=buy_cost, sell_cost=sell_cost,
        slippage_bps=slippage_bps, max_participation=0.0,
        limit_flags=True, warmup_days=WARMUP_DAYS, amount_q=0.2,
    )

    metrics = {k: (float(v) if np.isscalar(v) or isinstance(v, (int, float))
                   else v) for k, v in res["metrics"].items()}

    # 干净基准：裁剪后的域等权指数（对齐回测区间），重算超额指标
    from core.metrics import compute_metrics
    level = prep["bench_level"]
    bench_s = pd.Series(level, index=prep["dates"], name="bench")
    bench_s = bench_s.reindex(res["nav"].index)
    bench_s = bench_s / bench_s.iloc[0]
    bench_m = {k: float(v) for k, v in compute_metrics(bench_s).items()}
    excess_daily = res["nav"].pct_change() - bench_s.pct_change()
    excess_annual = float((res["nav"].iloc[-1] / bench_s.iloc[-1])
                          ** (252 / len(res["nav"])) - 1)
    excess_sharpe = float(excess_daily.mean() / excess_daily.std()
                          * np.sqrt(252)) if excess_daily.std() > 0 else 0.0
    metrics["超额年化"] = excess_annual
    metrics["超额夏普"] = excess_sharpe
    bench = bench_m
    log("策略指标: " + json.dumps(metrics, ensure_ascii=False, default=str))
    log("基准指标: " + json.dumps(bench, ensure_ascii=False, default=str))

    res["nav"].to_frame("nav").assign(bench=bench_s).to_csv(OUT / "nav.csv")
    res["drawdown"].to_frame("drawdown").to_csv(OUT / "drawdown.csv")
    res["trades"].to_csv(OUT / "trades.csv", index=False)
    res["holdings"].to_csv(OUT / "holdings.csv", index=False)
    (OUT / "metrics.json").write_text(
        json.dumps({"strategy": metrics, "bench": bench,
                    "start": START, "end": END, "capital": CAPITAL},
                   ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    log(f"完成。输出目录: {OUT}")


if __name__ == "__main__":
    main()
