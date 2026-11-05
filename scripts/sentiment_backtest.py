#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""舆情/新闻情绪策略回测（舆情双源 + 行情面板）。

调研要点落地方案：
  1. 舆情动量     SENTI/SENATT 思路：买入近期新闻情绪最正面的一批（ascending=False）
  2. 舆情反转     A股反转 + 情绪：买入近期情绪最负面的一批（ascending=True）
  3. 舆情关注度   注意力因子：买入近期新闻条数最多的一批（ascending=False）
  4. 舆情+反转增强 媒体增强效应(MRE)：买入「低动量 + 低情绪」的组合（ascending=True）

数据源（2026-11 起）：
- 舆情：``core.sentiment_store`` 合并旧 ``articles.db``（2026-02~08）与 CNE curated
  ``sentiment_articles``（2026-09 起），口径与页面 /api/sentiment 完全一致。
- 行情：默认 ``--panel-source cne``（CNE 数据湖面板，覆盖率与舆情同步推进）；
  ``--panel-source legacy`` 保留旧 ``data/stock/panel.parquet`` 复现历史结果。

注意：舆情库覆盖为「按成交额挑选的部分股票」，窗口短、股票少，
结果只能作为方法论验证，不能外推。
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.engine import run_backtest  # noqa: E402
from strategies.registry import get_strategy  # noqa: E402

PANEL = ROOT / "data/stock/panel.parquet"

# CNE 面板取列：回测契约要求 close/open/turnover/am20/turn20（high/low/volume 可选）
_CNE_COLUMNS = ("close", "open", "high", "low", "volume", "amount", "turnover_rate")
_ROLL_WINDOW = 20


def load_sentiment() -> pd.DataFrame:
    """合并双源舆情文章（与页面同源，口径零漂移）。"""
    from core import sentiment_store

    df, _ = sentiment_store.load_articles()
    if df.empty:
        raise SystemExit("舆情数据为空：确认 data/sentiment-mvp/articles.db 或 CNE curated sentiment_articles")
    out = df.loc[:, ["code", "publish_time", "source", "title", "label", "score"]].copy()
    out["code"] = out["code"].astype(str).str.zfill(6)
    out["pt"] = pd.to_datetime(out["publish_time"], errors="coerce")
    # 无时间的热榜/快照行不参与情绪打分因子，仅参与条数统计
    out["score"] = pd.to_numeric(out["score"], errors="coerce").fillna(0.0)
    return out


def _normalize_codes(code: pd.Series) -> pd.Series:
    return code.astype(str).str.split(".").str[0].str.zfill(6)


def _add_rolling_columns(df: pd.DataFrame) -> pd.DataFrame:
    """按 code 现算 turn20 / am20（与旧面板同口径：20 日均值）。"""
    df = df.sort_values(["code", "date"], kind="stable")
    for src, dst in (("turnover", "turn20"), ("amount", "am20")):
        df[dst] = (
            df.groupby("code", sort=False)[src]
            .transform(lambda s: s.rolling(_ROLL_WINDOW, min_periods=1).mean())
        )
    return df


def load_panel(source: str = "cne", *, start: str | None = None, end: str | None = None) -> pd.DataFrame:
    """加载行情长表面板（date/code/close/open/turnover/am20/turn20/...）。"""
    if source == "legacy":
        panel = pd.read_parquet(PANEL)
        panel["code"] = _normalize_codes(panel["code"])
        panel["date"] = pd.to_datetime(panel["date"])
        return panel

    from alphaagent.data.adapters.cnequity import load_panel_from_cne

    wide = load_panel_from_cne(
        start=start or "2026-02-01",
        end=end or None,
        include_fundamentals=False,
        asset_type="stock",
        keep_columns=_CNE_COLUMNS,
    )
    df = wide.reset_index().rename(
        columns={"datetime": "date", "instrument": "code", "turnover_rate": "turnover"}
    )
    df["code"] = _normalize_codes(df["code"])
    df["date"] = pd.to_datetime(df["date"])
    # CNE turnover_rate 是百分数（中位 ~1.76），旧面板 turnover 是小数比例（中位 ~0.029），
    # 统一到小数比例，避免 amount_q / 低换手排序口径漂移。
    df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce") / 100.0
    for col in ("close", "open", "high", "low", "volume", "amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df[df["code"].str.len() == 6]
    return _add_rolling_columns(df)


def _daily_sentiment(art: pd.DataFrame) -> pd.DataFrame:
    """文章按 (code, 自然日) 聚合日均情绪分与条数（一次聚合，多次复用）。"""
    art = art.assign(d=art["pt"].dt.normalize(), s=art["score"], c=1)
    return (
        art.dropna(subset=["d"])
        .groupby(["code", "d"])
        .agg(s=("s", "mean"), c=("c", "sum"))
        .reset_index()
    )


def _build_frames(daily: pd.DataFrame, close: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """按引擎交易日历对齐，构造情绪因子矩阵（无前视：信号日当天可用、次日开盘执行）。"""
    cal = close.index
    by_code = {code: sub.set_index("d") for code, sub in daily.groupby("code")}
    sent = pd.DataFrame(index=cal, columns=close.columns, dtype=float)
    cnt = pd.DataFrame(index=cal, columns=close.columns, dtype=float)
    for code in close.columns:
        sub = by_code.get(code)
        if sub is None:
            continue
        s = sub["s"].reindex(cal)
        c = sub["c"].reindex(cal).fillna(0.0)
        sent[code] = s.rolling(_ROLL_WINDOW, min_periods=1).mean()
        cnt[code] = c.rolling(_ROLL_WINDOW, min_periods=1).sum()
    return {"sent20": sent, "cnt20": cnt}


def zscore_frame(df: pd.DataFrame) -> pd.DataFrame:
    z = df.sub(df.mean(axis=1), axis=0).div(df.std(axis=1), axis=0)
    return z.replace([np.inf, -np.inf], np.nan)


def make_factor_builder(daily: pd.DataFrame):
    """factor_builder 接口：close/am20/turn20 是引擎切好的子区间矩阵。"""

    def builder(close, am20, turn20):
        frames = _build_frames(daily, close)
        mom20 = close.pct_change(20, fill_method=None)
        frames["mre"] = zscore_frame(frames["sent20"]) + zscore_frame(mom20)
        return frames

    return builder


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="舆情情绪策略回测（双源舆情 + 行情面板）")
    ap.add_argument(
        "--panel-source",
        choices=("cne", "legacy"),
        default="cne",
        help="行情来源：cne=CNE 数据湖面板（默认，与舆情数据同步推进）；legacy=旧 data/stock/panel.parquet",
    )
    ap.add_argument("--start", default="2026-03-01", help="回测起点（默认 2026-03-01）")
    ap.add_argument("--end", default=None, help="回测终点（默认面板最后交易日）")
    ap.add_argument("--panel-start", default=None, help="面板加载起点（CNE 默认 2026-02-01）")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    art = load_sentiment()
    panel = load_panel(
        args.panel_source,
        start=args.panel_start or "2026-02-01",
        end=args.end,
    )
    daily = _daily_sentiment(art)
    builder = make_factor_builder(daily)

    sent_codes = sorted(set(art["code"]) & set(panel["code"].unique()))
    if not sent_codes:
        raise SystemExit("舆情覆盖股票与面板股票无交集：检查数据源区间是否重叠")
    print(f"舆情覆盖 {art['code'].nunique()} 只：{art['pt'].min().date()} ~ {art['pt'].max().date()}")
    print(f"面板区间 {panel['date'].min().date()} ~ {panel['date'].max().date()}（source={args.panel_source}）")

    start = args.start
    end = args.end or str(panel["date"].max().date())
    capital, top_n = 50000.0, 3
    freq = "monthly"
    print(f"回测 {start} ~ {end} · {len(sent_codes)} 只 · Top{top_n} · {freq}")

    rows = []
    ic_rows = []
    sent_strategies = [
        ("舆情正面(动量)", "sent20", False, "买入近期新闻情绪最正面"),
        ("舆情负面(反转)", "sent20", True, "买入近期情绪最负面（博反转）"),
        ("舆情关注度", "cnt20", False, "买入近期新闻条数最多"),
        ("舆情+反转增强(MRE)", "mre", True, "买入低动量+低情绪组合"),
    ]
    for name, factor, asc, desc in sent_strategies:
        res = run_backtest(
            panel=panel, codes=sent_codes, factor=factor, ascending=asc,
            start=start, end=end, capital=capital, top_n=top_n, freq=freq,
            affordable=True, amount_q=0.2, warmup_days=45,
            factor_builder=builder,
            analyze=True,
        )
        m = res["metrics"]
        q = res.get("factor_quality")
        rows.append({"策略": name, "类型": "舆情", "说明": desc,
                     "总收益%": round(m["总收益"] * 100, 2),
                     "年化%": round(m["年化收益"] * 100, 2),
                     "夏普": round(m["夏普"], 2),
                     "最大回撤%": round(m["最大回撤"] * 100, 2),
                     "信号日": str(res["last_signal_date"].date()) if res["last_signal_date"] else "-"})
        if q:
            sign = -1.0 if asc else 1.0
            ic_rows.append({"策略": name, "类型": "舆情",
                            "IC均值": q["ic"]["mean_ic"], "ICIR": q["ic"]["icir"],
                            "t值": q["ic"]["t_stat"], "IC>0占比": q["ic"]["win_rate"],
                            "多空价差%": q["group"]["spread"] * 100 if q["group"]["spread"] is not None else None,
                            "多空价差年化%": q["group"]["spread_pa"] * 100 if q["group"]["spread_pa"] is not None else None,
                            "方向调整IC": q["ic"]["mean_ic"] * sign if q["ic"]["mean_ic"] is not None else None,
                            "方向调整价差%": q["group"]["spread"] * sign * 100 if q["group"]["spread"] is not None else None})

    # 同股票池下对照：现有价格/量因子策略
    for name in ("动量 20 日", "反转 20 日", "低波动", "低换手冷门"):
        s = get_strategy(name)
        res = run_backtest(
            panel=panel, codes=sent_codes, factor=s["factor"], ascending=s["ascending"],
            start=start, end=end, capital=capital, top_n=top_n, freq=freq,
            affordable=True, amount_q=0.2, warmup_days=45,
            analyze=True,
        )
        m = res["metrics"]
        q = res.get("factor_quality")
        rows.append({"策略": name, "类型": "对照", "说明": s["desc"],
                     "总收益%": round(m["总收益"] * 100, 2),
                     "年化%": round(m["年化收益"] * 100, 2),
                     "夏普": round(m["夏普"], 2),
                     "最大回撤%": round(m["最大回撤"] * 100, 2),
                     "信号日": str(res["last_signal_date"].date()) if res["last_signal_date"] else "-"})
        if q:
            sign = -1.0 if s["ascending"] else 1.0
            ic_rows.append({"策略": name, "类型": "对照",
                            "IC均值": q["ic"]["mean_ic"], "ICIR": q["ic"]["icir"],
                            "t值": q["ic"]["t_stat"], "IC>0占比": q["ic"]["win_rate"],
                            "多空价差%": q["group"]["spread"] * 100 if q["group"]["spread"] is not None else None,
                            "多空价差年化%": q["group"]["spread_pa"] * 100 if q["group"]["spread_pa"] is not None else None,
                            "方向调整IC": q["ic"]["mean_ic"] * sign if q["ic"]["mean_ic"] is not None else None,
                            "方向调整价差%": q["group"]["spread"] * sign * 100 if q["group"]["spread"] is not None else None})

    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    out = ROOT / "results" / "sentiment_backtest.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"saved -> {out}")
    if ic_rows:
        ic_df = pd.DataFrame(ic_rows)
        ic_out = ROOT / "results" / "sentiment_ic_group.csv"
        ic_df.to_csv(ic_out, index=False, encoding="utf-8-sig")
        print(f"saved -> {ic_out}")
        # 回测区间/面板来源写 sidecar，页面据此标注而不是让读者猜
        meta_out = ROOT / "results" / "sentiment_ic_meta.json"
        meta_out.write_text(
            json.dumps(
                {
                    "window": f"{start} ~ {end}",
                    "panel_source": args.panel_source,
                    "n_codes": len(sent_codes),
                    "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"saved -> {meta_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
