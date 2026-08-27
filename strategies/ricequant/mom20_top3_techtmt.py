# ==========================================================================
# 米筐(RiceQuant) 在线策略 · 多因子对比版 · 科技TMT池
# （兼容 rqalpha 系平台的防御式 API 层）
#
# 用途：与本地 quant_ui 引擎(core/engine.py)的同名回测做引擎结果对齐。
# 支持四个因子 x 三种调仓频率，改下面 CONFIG 两行即可切换：
#
#   FACTOR   : "mom20"(20日动量) | "mom60"(60日动量)
#              | "vol20"(20日低波动,配 ASCENDING=True)
#              | "ma_cross20_30"(MA20/MA30乖离)
#              | "brk20"(趋势突破: 收盘/前20日最高收盘-1)
#   ASCENDING: False=买因子最大(动量/乖离/突破)  True=买最小(反转/低波动)
#   FREQ     : "monthly"(每月首个交易日) | "weekly"(每周首个交易日)
#              | "daily"(每交易日)
#
# 因子定义与本地 build_factor_frames 逐条对齐（前复权、截至上一交易日）：
#   mom20         = C_t / C_(t-20) - 1
#   mom60         = C_t / C_(t-60) - 1
#   vol20         = 最近20个日收益率的样本标准差(ddof=1)
#   ma_cross20_30 = mean(C,20) / mean(C,30) - 1
#   brk20         = C_t / max(C_{t-1..t-20}) - 1（>0 即创20日新高）
#
# 本地基准导出（同窗口先跑一遍，产物在 strategies/ricequant/baseline_*）：
#   .venv\Scripts\python.exe scripts\ricequant_picks_export.py ^
#       --start 2024-01-01 --end 2025-12-31 ^
#       --factor ma_cross20_30 --out strategies\ricequant\baseline_ma2030
#
# 口径要点：
#   - 初始资金必须 5000；佣金倍率 1；滑点 0；撮合方式平台只有收盘就选它。
#   - 流动性过滤 am20 >= 池内当日 20% 分位 且近20日有成交（等价 turn20>0）。
#   - 先卖后买、整手、现金封顶由平台处理。
# ==========================================================================

# ---------------- CONFIG（对比哪个组合就改这里） ----------------
FACTOR = "mom20"       # mom20 / mom60 / vol20 / ma_cross20_30 / brk20
ASCENDING = False      # False=买最大  True=买最小
FREQ = "monthly"       # monthly / weekly / daily
TOP_N = 3
AMOUNT_Q = 0.2
# -----------------------------------------------------------------

import numpy as np
import pandas as pd

# 各因子需要的收盘行数（截至上一交易日）
_LOOKBACK = {
    "mom20": 21,
    "mom60": 61,
    "vol20": 21,
    "ma_cross20_30": 30,
    "brk20": 21,
}

# 科技TMT池 ∩ 本地面板 ∩ 剔除创业板/科创板，共 90 只（快照见 pytmp/techtmt_pool.json）
POOL = [
    "000021", "000034", "000050", "000062", "000063", "000100", "000725",
    "000938", "000977", "000997", "001309", "001389", "002027", "002049",
    "002065", "002130", "002138", "002152", "002153", "002185", "002195",
    "002230", "002236", "002241", "002261", "002273", "002281", "002371",
    "002384", "002402", "002409", "002410", "002415", "002436", "002463",
    "002475", "002517", "002558", "002583", "002600", "002602", "002624",
    "002739", "002841", "002916", "002920", "002938", "003031", "600050",
    "600100", "600105", "600131", "600171", "600183", "600363", "600460",
    "600498", "600522", "600536", "600563", "600570", "600584", "600588",
    "600601", "600602", "600637", "600707", "600845", "600941", "600977",
    "601019", "601098", "601138", "601360", "601728", "601869", "601928",
    "603000", "603019", "603160", "603175", "603290", "603296", "603341",
    "603444", "603501", "603893", "603920", "603986", "605358",
]


def _ob(code):
    return code + (".XSHG" if code.startswith("6") else ".XSHE")


CODES = [_ob(c) for c in POOL]


# ---------- 平台适配层 ----------

def _g(name):
    return globals().get(name)


def _log(msg):
    lg = _g("logger")
    if lg is not None and hasattr(lg, "info"):
        lg.info(msg)
    else:
        print(msg)


def _hist(field, count, codes):
    """截至今日、不含今日的日频数据 -> DataFrame(columns=标的)。"""
    hist = _g("history")
    if hist is not None:
        try:
            return hist(count, unit="1d", field=field,
                        security_list=codes, df=True)
        except TypeError:
            return hist(count, frequency="1d", field=field,
                        security_list=codes, df=True)

    hb = _g("history_bars")
    if hb is not None:
        cols = {}
        for s in codes:
            try:
                cols[s] = _to_float_1d(hb(s, count, "1d", field))
            except Exception:
                cols[s] = np.empty(0)
        width = max((len(v) for v in cols.values()), default=0)
        data = {}
        for s, v in cols.items():
            pad = np.full(width, np.nan)
            if len(v):
                pad[width - len(v):] = v
            data[s] = pad
        return pd.DataFrame(data)

    raise RuntimeError("平台既无 history() 也无 history_bars()")


def _to_float_1d(arr):
    if arr is None:
        return np.empty(0)
    dt = getattr(arr, "dtype", None)
    if dt is not None and getattr(dt, "names", None):
        arr = arr[dt.names[0]]
    return np.asarray(arr, dtype=float).ravel()


def _pos_qty(pos):
    q = getattr(pos, "quantity", None)
    if q is None:
        q = getattr(pos, "total_amount", 0)
    return q or 0


def _target_value(ob_id, value):
    fn = _g("order_target_value")
    if fn is None:
        raise RuntimeError("平台未提供 order_target_value()")
    return fn(ob_id, value)


# ---------- 策略主体 ----------

def compute_score(closes):
    """单只票的收盘序列 -> 因子得分（与 build_factor_frames 同口径）。"""
    c = closes[np.isfinite(closes)]
    n = _LOOKBACK[FACTOR]
    if len(c) < n:
        return np.nan
    c = c[-n:]
    if FACTOR in ("mom20", "mom60"):
        return c[-1] / c[0] - 1.0
    if FACTOR == "ma_cross20_30":
        return c[-20:].mean() / c.mean() - 1.0
    if FACTOR == "vol20":
        rets = c[1:] / c[:-1] - 1.0
        return float(np.std(rets, ddof=1))
    if FACTOR == "brk20":
        # 收盘 / 前20日最高收盘 - 1（c 最后一行是信号日收盘）
        return c[-1] / c[:-1].max() - 1.0
    raise ValueError(FACTOR)


def init(context):
    context.codes = CODES
    context.cur_key = None

    bm = _g("set_benchmark")
    if bm is not None:
        bm("000300.XSHG")
    else:
        _log("[init] 无 set_benchmark，跳过（仅影响展示）")

    cm = _g("set_commission")
    if cm is not None:
        try:
            cm(buy_cost=0.0003, sell_cost=0.0013, min_cost=5.0)
            _log("[init] 费率 买万3/卖万13+税/最低5元")
        except Exception:
            _log("[init] set_commission 签名不符，请在设置页保持佣金倍率=1")
    _log("[init] FACTOR={} ASCENDING={} FREQ={}".format(
        FACTOR, ASCENDING, FREQ))


def rebalance(context, bar_dict=None):
    lb = _LOOKBACK[FACTOR]
    closes = _hist("close", lb, context.codes)
    amounts = _hist("total_turnover", 20, context.codes)

    scores = {}
    for s in context.codes:
        v = compute_score(closes[s].values.astype(float))
        if v == v:
            scores[s] = v
    am20 = amounts.mean()

    valid = [s for s in scores
             if s in am20.index and am20[s] == am20[s] and am20[s] > 0]
    if not valid:
        _log("[{}] 无有效候选".format(_today(context)))
        _sell_all(context, [])
        return

    vals = sorted(am20[s] for s in valid)
    thr = vals[int(AMOUNT_Q * (len(vals) - 1))]
    cand = [s for s in valid if am20[s] >= thr]

    ranked = sorted(cand, key=lambda s: scores[s], reverse=not ASCENDING)
    targets = ranked[:TOP_N]

    pv = context.portfolio.portfolio_value
    weight = 1.0 / len(targets) if targets else 0.0
    _log("[{}] {} 目标: {} | 市值 {:.0f}".format(
        _today(context), FACTOR,
        ",".join(t.split(".")[0] for t in targets), pv))

    _sell_all(context, targets)
    for s in targets:
        _target_value(s, pv * weight)


def _sell_all(context, keep):
    pos_map = context.portfolio.positions
    items = pos_map.items() if hasattr(pos_map, "items") else pos_map
    for ob_id, pos in list(items):
        if ob_id in keep or _pos_qty(pos) <= 0:
            continue
        _target_value(ob_id, 0)


def _today(context):
    now = getattr(context, "now", None)
    return str(now.date()) if now is not None and hasattr(now, "date") else "?"


def _period_key(context):
    d = getattr(context, "now", None)
    if d is None:
        return ""
    if FREQ == "monthly":
        return str(d)[:7]
    if FREQ == "weekly":
        iso = d.isocalendar()
        return "{}-W{:02d}".format(iso[0], iso[1])
    return str(d)[:10]


def _on_bar(context, bar_dict):
    key = _period_key(context)
    if key and key != context.cur_key:
        context.cur_key = key
        rebalance(context, bar_dict)


def handle_bar(context, bar_dict):
    _on_bar(context, bar_dict)


def handle_data(context, data):
    _on_bar(context, None)
