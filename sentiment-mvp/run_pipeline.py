#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""舆情情绪 MVP 端到端管线。

用法（在 sentiment-mvp 目录下）：
    python run_pipeline.py all               # 一键：选股→抓新闻(EM+CLS)→打情绪→事件研究
    python run_pipeline.py fetch             # 抓东方财富个股新闻
    python run_pipeline.py fetch-cls         # 抓财联社当日电报并匹配标的
    python run_pipeline.py score [--llm]     # 打情绪
    python run_pipeline.py study             # 事件研究（收盘/次日开盘双口径）
    python run_pipeline.py audit             # LLM vs 词典标签一致性审计
    python run_pipeline.py snapshot          # 今日舆情快照（财联社当日）
    python run_pipeline.py daily             # 每日入库：增量合并到 SQLite + 事件研究
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import requests
import yaml
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from lib import audit as au
from lib import event_study as es
from lib import fetch_cls as fc
from lib import fetch_news as fn
from lib import ml_sentiment as mlm
from lib import prices as px
from lib import sentiment as st
from lib import sources_extra as sx
from lib import store as stk
from lib.universe import build_universe


def load_cfg():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _name_map(cfg):
    uni = pd.read_csv(cfg["universe"]["universe_csv"], dtype={"code": str})
    tech = pd.read_csv(cfg["universe"]["tech_csv"], dtype={"code": str})
    uni["code"] = uni["code"].astype(str).str.zfill(6)
    tech["code"] = tech["code"].astype(str).str.zfill(6)
    m = dict(zip(uni["code"], uni["name"]))
    m.update(dict(zip(tech["code"], tech["name"])))
    return m


def cmd_universe(cfg):
    df = build_universe(cfg)
    print(f"universe: {len(df)} stocks -> {cfg['universe']['output']}")
    print(df.head(10).to_string(index=False))


def cmd_fetch(cfg, codes=None):
    uni = build_universe(cfg)
    if codes:
        uni = uni[uni["code"].isin(codes)]
    arts = _fetch_em_parallel(uni, cfg, max_pages=int(cfg["news"]["max_pages"]))
    fn.save_raw(arts, ROOT / cfg["paths"]["news_raw"])
    print(f"saved {len(arts)} articles -> {cfg['paths']['news_raw']}")


def _fetch_em_parallel(uni, cfg, max_pages):
    """并发抓东方财富个股新闻（每线程独立 session），返回 source=em 的行。"""
    workers = int(cfg["news"].get("workers", 8))
    delay = float(cfg["news"].get("request_delay", 0.25))
    timeout = float(cfg["news"].get("timeout", 15))
    arts = []

    def worker(code):
        with requests.Session() as session:
            batch = fn.fetch_stock_news(session, code, max_pages=max_pages,
                                        delay=delay, timeout=timeout)
            batch = fn.dedup(batch)
            for a in batch:
                a["source"] = "em"
            return batch

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(worker, str(code)) for code in uni["code"]]
        done = 0
        for fut in as_completed(futs):
            try:
                arts.extend(fut.result())
            except Exception as e:
                print(f"[em_parallel] failed: {e}", file=sys.stderr)
            done += 1
            if done % 300 == 0:
                print(f"[em_parallel] {done}/{len(futs)}", flush=True)
    return arts


def cmd_fetch_cls(cfg, codes=None):
    uni = build_universe(cfg)
    if codes:
        uni = uni[uni["code"].isin(codes)]
    code_list = set(uni["code"])
    name_map = _name_map(cfg)
    with requests.Session() as session:
        items = fc.fetch_cls(session, max_pages=int(cfg.get("cls", {}).get("max_pages", 10)))
    rows = fc.map_mentions(items, name_map, code_list)
    fc.save_jsonl(rows, ROOT / cfg["paths"]["news_cls"])
    print(f"cls items={len(items)} matched={len(rows)} -> {cfg['paths']['news_cls']}")
    if rows:
        print(pd.DataFrame(rows)[["code", "publish_time", "title"]].head(10).to_string(index=False))


def cmd_fetch_extra(cfg, codes=None):
    uni = build_universe(cfg)
    if codes:
        uni = uni[uni["code"].isin(codes)]
    rows = sx.fetch_all(cfg, list(uni["code"]))
    sx.save_jsonl(rows, ROOT / cfg["paths"]["news_extra"])
    print(f"saved {len(rows)} extra articles -> {cfg['paths']['news_extra']}")
    if rows:
        print(pd.DataFrame(rows)[["code", "source", "publish_time", "title"]].head(10).to_string(index=False))


def _load_articles(cfg, include_cls=True):
    arts = fn.load_raw(ROOT / cfg["paths"]["news_raw"])
    if include_cls and (ROOT / cfg["paths"]["news_cls"]).exists():
        arts += fc.load_jsonl(ROOT / cfg["paths"]["news_cls"])
    if include_cls and (ROOT / cfg["paths"]["news_extra"]).exists():
        arts += sx.load_jsonl(ROOT / cfg["paths"]["news_extra"])
    return arts


def cmd_score(cfg, use_llm=False, limit=None):
    arts = _load_articles(cfg)
    if not arts:
        raise SystemExit("no articles, run fetch / fetch-cls first")
    if limit:
        import random
        random.seed(42)
        arts = random.sample(arts, min(limit, len(arts)))
        print(f"limited to {len(arts)} articles (seed=42)")
    method = "llm" if use_llm else cfg["sentiment"]["method"]
    llm_cfg = dict(cfg["sentiment"]["llm"])
    if use_llm:
        llm_cfg["api_key"] = os.environ.get("NEWAPI_KEY", "")
    workers = int(cfg["sentiment"]["llm"].get("concurrency", 4))
    if method == "llm":
        scored = []
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(st.score_one, a, llm_cfg, "llm") for a in arts]
            for fut in as_completed(futs):
                scored.append(fut.result())
                done += 1
                if done % 100 == 0 or done == len(arts):
                    print(f"llm progress {done}/{len(arts)} {pd.Timestamp.now():%H:%M:%S}", flush=True)
                if done % 300 == 0:
                    partial = ROOT / (cfg["paths"]["news_sentiment_llm"] + ".partial")
                    pd.DataFrame(scored).to_csv(partial, index=False, encoding="utf-8-sig")
    else:
        scored = st.score_articles(arts, method=method)
    df = pd.DataFrame(scored)
    if "source" in df:
        df["source"] = df["source"].fillna("em")
    out = ROOT / (cfg["paths"]["news_sentiment_llm"] if use_llm else cfg["paths"]["news_sentiment"])
    df.to_csv(out, index=False, encoding="utf-8-sig")
    partial = ROOT / (cfg["paths"]["news_sentiment_llm"] + ".partial")
    if partial.exists():
        partial.unlink()
    print(f"scored {len(df)} articles -> {out}")
    if "source" in df:
        print(df.groupby("source")["label"].value_counts().to_string())
    else:
        print(df["label"].value_counts().to_string())


def cmd_train_ml(cfg, force=False):
    metrics = mlm.train(force=force)
    print(f"ml model ready: acc={metrics['accuracy']:.4f} f1(pos)={metrics['f1']:.4f}")


def _prepare_sentiment(cfg, label_source="lexicon", sentiment_file=None):
    if sentiment_file:
        path = ROOT / sentiment_file
    else:
        key = "news_sentiment_llm" if label_source == "llm" else "news_sentiment"
        path = ROOT / cfg["paths"][key]
    if not path.exists():
        raise SystemExit(f"missing {path}, run score first")
    df = pd.read_csv(path)
    df["code"] = df["code"].astype(str).str.zfill(6)
    if "source" not in df:
        df["source"] = "em"
    else:
        df["source"] = df["source"].fillna("em")
    df["publish_time"] = df["publish_time"].astype(str)
    df["dt"] = df["publish_time"].map(fn.parse_time)
    df = df.dropna(subset=["dt"]).copy()
    index_df = px.fetch_index(ROOT / cfg["paths"]["index_file"])
    index_frame = es.frame_from_df(index_df)
    cal = es.build_calendar(index_frame)
    df["event_date"] = df["dt"].map(lambda x: es.event_trade_date(x, cal))
    df = df.dropna(subset=["event_date"]).copy()
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.normalize()
    df["event_key"] = df["code"] + "_" + df["event_date"].dt.strftime("%Y-%m-%d")
    return df, index_frame


def cmd_study(cfg, refresh_prices=False, label_source="lexicon", sentiment_file=None, tag=None):
    sent_df, index_frame = _prepare_sentiment(cfg, label_source=label_source, sentiment_file=sentiment_file)
    codes = sorted(sent_df["code"].unique())
    prices = {}
    for code in codes:
        try:
            df = px.fetch_stock_price(code, cache_dir=ROOT / cfg["paths"]["prices_dir"], refresh=refresh_prices)
            prices[code] = es.frame_from_df(df)
        except Exception as e:
            print(f"price fetch failed for {code}: {e}", file=sys.stderr)
    prices = {c: f for c, f in prices.items() if len(f) > 10}

    horizons = [int(h) for h in cfg["event_study"]["horizons"]]
    threshold = float(cfg["sentiment"]["pos_threshold"])
    es_cfg = cfg.get("event_study", {})
    source_weights = es_cfg.get("source_weights", {})
    decay_half_life = es_cfg.get("decay_half_life_hours")
    autocorr = bool(es_cfg.get("autocorr", {}).get("enabled", True))

    day_df = es.aggregate_stock_days(sent_df, pos_threshold=threshold,
                                     source_weights=source_weights,
                                     decay_half_life=decay_half_life)
    day_df["event_date"] = pd.to_datetime(day_df["event_date"]).dt.normalize()
    events = es.compute_events(day_df, prices, index_frame, horizons)

    vol_filter_note = ""
    vf = es_cfg.get("volume_filter", {})
    if vf.get("enabled"):
        before = len(events)
        events = es.apply_volume_filter(events, prices,
                                        window=int(vf.get("window", 20)),
                                        min_vol_ratio=float(vf.get("min_vol_ratio", 1.0)))
        vol_filter_note = f"事件日成交量 < 过去{int(vf.get('window', 20))}日中位数 × {float(vf.get('min_vol_ratio', 1.0))} 的剔除（{before} → {len(events)}）"

    industry_map = {}
    if es_cfg.get("industry_breakdown"):
        try:
            ind_df = pd.read_csv(cfg["universe"]["tech_csv"], dtype={"code": str})
            industry_map = dict(zip(ind_df["code"].astype(str).str.zfill(6), ind_df["industry"]))
        except Exception:
            try:
                ind_df = pd.read_csv(ROOT.parent / "quant_ui" / "data" / "tech.csv", dtype={"code": str})
                industry_map = dict(zip(ind_df["code"].astype(str).str.zfill(6), ind_df["industry"]))
            except Exception:
                pass
        if industry_map:
            events = es.add_industry(events, industry_map)

    ll_df, ll_corr = es.compute_leadlag(day_df, prices, horizons)
    event_days = set(zip(events["code"], events["event_date"])) if len(events) else set()
    baseline = es.compute_baseline(prices, index_frame, event_days, horizons, lead_days=int(cfg["event_study"]["baseline_lead_days"]))
    summary_close = es.summarize_by_label(events, baseline, horizons, mode="close", autocorr=autocorr)
    summary_open = es.summarize_by_label(events, baseline, horizons, mode="open", autocorr=autocorr)
    ind_close = es.summarize_by_industry(events, horizons, mode="close", autocorr=autocorr) if industry_map else pd.DataFrame()
    ind_open = es.summarize_by_industry(events, horizons, mode="open", autocorr=autocorr) if industry_map else pd.DataFrame()

    suffix = f"_{tag}" if tag else ("_llm" if label_source == "llm" else "")
    out_csv = ROOT / cfg["paths"]["study_csv"].replace(".csv", f"{suffix}.csv")
    events.to_csv(out_csv, index=False, encoding="utf-8-sig")
    if len(baseline):
        baseline.to_csv(ROOT / cfg["paths"]["study_csv"].replace(".csv", f"{suffix}_baseline.csv"), index=False, encoding="utf-8-sig")
    if len(ll_df):
        ll_df.to_csv(ROOT / cfg["paths"]["study_csv"].replace(".csv", f"{suffix}_leadlag.csv"), index=False, encoding="utf-8-sig")
    if len(ll_corr):
        ll_corr.to_csv(ROOT / cfg["paths"]["study_csv"].replace(".csv", f"{suffix}_leadlag_corr.csv"), index=False, encoding="utf-8-sig")
    if len(ind_close):
        ind_close.to_csv(ROOT / cfg["paths"]["study_csv"].replace(".csv", f"{suffix}_by_industry.csv"), index=False, encoding="utf-8-sig")

    report = es.build_report(events, baseline, summary_close, summary_open, sent_df, cfg,
                             leadlag_corr=ll_corr,
                             industry_close=ind_close, industry_open=ind_open,
                             vol_filter_note=vol_filter_note)
    report_path = ROOT / cfg["paths"]["report"].replace(".md", f"{suffix}.md")
    report_path.write_text(report, encoding="utf-8")
    print(f"events={len(events)} baseline_days={len(baseline)} leadlag={len(ll_df)} industry_groups={ind_close['industry'].nunique() if len(ind_close) else 0}")
    print(f"report -> {report_path}")
    if len(ll_corr):
        print("== 领先/滞后相关性 ==")
        print(ll_corr.to_string(index=False))
    if len(summary_open):
        cols = ["horizon", "name", "n", "mean", "hit", "t", "diff_t"]
        show = summary_open[cols].copy()
        for c in ("mean", "hit", "t", "diff_t"):
            show[c] = show[c].round(4)
        print("== 次日开盘买入口径 ==")
        print(show.to_string(index=False))


def cmd_audit(cfg, sample_size=300):
    arts = _load_articles(cfg)
    if not arts:
        raise SystemExit("no articles, run fetch / fetch-cls first")
    llm_cfg = dict(cfg["sentiment"]["llm"])
    llm_cfg["api_key"] = os.environ.get("NEWAPI_KEY", "")
    df = au.audit_sample(
        arts,
        sample_size=sample_size,
        llm_cfg=llm_cfg,
        workers=int(cfg["sentiment"]["llm"].get("concurrency", 4)),
    )
    out = ROOT / cfg["paths"]["audit_report"]
    extra = f"- 审计样本量：{sample_size}（随机抽样，seed=42）\n- LLM：{llm_cfg['model']}，本地 new-api"
    au.write_report(df, out, extra=extra)
    agree = int((df["lex_label"] == df["llm_label"]).sum())
    print(f"一致 {agree}/{len(df)} = {agree / len(df):.1%}")
    print(f"audit report -> {out}")


def cmd_snapshot(cfg):
    cls_path = ROOT / cfg["paths"]["news_cls"]
    cached = fc.load_jsonl(cls_path) if cls_path.exists() else []
    uni = build_universe(cfg)
    name_map = _name_map(cfg)
    with requests.Session() as session:
        items = fc.fetch_cls(session, max_pages=int(cfg.get("cls", {}).get("max_pages", 10)))
    rows = fc.map_mentions(items, name_map, set(uni["code"]))
    if rows:
        fc.save_jsonl(rows, cls_path)
        archive_dir = ROOT / cfg["cls"]["archive_dir"]
        archive_dir.mkdir(parents=True, exist_ok=True)
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        archive = archive_dir / f"{today}.jsonl"
        old = fc.load_jsonl(archive) if archive.exists() else []
        fc.save_jsonl(fn.dedup(rows + old), archive)
        arts = rows
    else:
        arts = cached
        print("CLS fetch empty, using cached file")
    if not arts:
        print("today: no cls items matched universe")
        return
    out = _write_snapshot(cfg, arts)
    print(f"snapshot -> {out}")


def _write_snapshot(cfg, arts):
    uni = build_universe(cfg)
    name_map = dict(zip(uni["code"], uni["name"]))
    scored = st.score_articles(arts, llm_cfg=None, method="lexicon")
    df = pd.DataFrame(scored)
    df["name"] = df["code"].map(name_map)
    day = df.groupby(["code", "name"]).agg(
        n=("score", "size"),
        mean_score=("score", "mean"),
        positive=("label", lambda s: int((s == "positive").sum())),
        negative=("label", lambda s: int((s == "negative").sum())),
    ).sort_values("mean_score", ascending=False)
    out = ROOT / cfg["paths"]["snapshot"]
    lines = [f"# 今日舆情快照（财联社）  {pd.Timestamp.now():%Y-%m-%d %H:%M}", ""]
    lines.append(f"匹配股票 {len(day)} 只，电报 {len(df)} 条")
    lines.append("")
    lines.append(day.reset_index().to_markdown(index=False))
    out.write_text("\n".join(lines), encoding="utf-8")
    print(day.reset_index().to_string(index=False))
    return out


def seed_store(cfg):
    """首次运行时把已有 news_sentiment.csv 灌入 SQLite，避免重复抓取。"""
    db = ROOT / cfg["paths"]["articles_db"]
    if Path(db).exists() and stk.stats(db)["total"] > 0:
        return 0
    sent = ROOT / cfg["paths"]["news_sentiment"]
    if not sent.exists():
        return 0
    df = pd.read_csv(sent)
    df["code"] = df["code"].astype(str).str.zfill(6)
    if "source" not in df:
        df["source"] = "em"
    else:
        df["source"] = df["source"].fillna("em")
    rows = [
        {
            "code": r["code"], "publish_time": str(r["publish_time"]), "media": r.get("media", ""),
            "title": r["title"], "content": r.get("content", ""), "url": r.get("url", ""),
            "source": r.get("source", "em"), "label": r["label"], "score": r["score"],
            "pos_hits": r.get("pos_hits", 0), "neg_hits": r.get("neg_hits", 0),
        }
        for _, r in df.iterrows()
    ]
    ins, _ = stk.upsert_rows(db, rows)
    print(f"seeded store with {ins} rows")
    return ins


def cmd_daily(cfg, refresh_prices=False, full_history=False, no_study=False):
    """每日入库：抓 EM + CLS → 只给新条目标注 → 合并 SQLite → 导出 → 跑事件研究 + 快照。

    full_history=True 按 news.max_pages 全量补历史（首次扩容用）；
    no_study=True 跳过事件研究（大批量补库时避免拉全量价格）。
    """
    seed_store(cfg)
    db = ROOT / cfg["paths"]["articles_db"]
    uni = build_universe(cfg)

    max_pages = int(cfg["news"]["max_pages"] if full_history
                    else cfg["news"].get("daily_max_pages", 5))
    arts_em = _fetch_em_parallel(
        uni, cfg,
        max_pages=max_pages,
    )
    print(f"daily: em fetched {len(arts_em)} (max_pages={max_pages})")

    name_map = _name_map(cfg)
    with requests.Session() as session:
        items = fc.fetch_cls(session, max_pages=int(cfg.get("cls", {}).get("max_pages", 10)))
    rows_cls = fc.map_mentions(items, name_map, set(uni["code"]))
    if rows_cls:
        fc.save_jsonl(rows_cls, ROOT / cfg["paths"]["news_cls"])
        archive_dir = ROOT / cfg["cls"]["archive_dir"]
        archive_dir.mkdir(parents=True, exist_ok=True)
        today = pd.Timestamp.now().strftime("%Y-%m-%d")
        archive = archive_dir / f"{today}.jsonl"
        old = fc.load_jsonl(archive) if archive.exists() else []
        fc.save_jsonl(fn.dedup(rows_cls + old), archive)
    print(f"daily: cls fetched {len(rows_cls)}")

    rows_extra = sx.fetch_all(cfg, list(uni["code"]))
    if rows_extra:
        sx.save_jsonl(rows_extra, ROOT / cfg["paths"]["news_extra"])
    print(f"daily: extra fetched {len(rows_extra)}")

    combined = arts_em + rows_cls + rows_extra
    existing = stk.existing_keys(db)
    new_rows = [a for a in combined if stk.key_of(a) not in existing]
    if new_rows:
        scored = st.score_articles(new_rows, llm_cfg=None, method="lexicon")
        ins, ex = stk.upsert_rows(db, scored)
    else:
        ins, ex = 0, len(combined)
    print(f"daily: inserted {ins}, existing {ex}")

    df = stk.load_articles(db)
    df.to_csv(ROOT / cfg["paths"]["news_sentiment_daily"], index=False, encoding="utf-8-sig")
    print(f"daily: exported {len(df)} -> {cfg['paths']['news_sentiment_daily']}")
    print(stk.stats(db))

    if no_study:
        print("daily: skip event study (--no-study)")
    else:
        cmd_study(
            cfg,
            refresh_prices=refresh_prices,
            label_source="lexicon",
            sentiment_file=cfg["paths"]["news_sentiment_daily"],
            tag="daily",
        )
    today_cls = stk.load_articles(db, source="cls")
    if len(today_cls):
        today_cls["publish_time"] = today_cls["publish_time"].astype(str)
        today_str = pd.Timestamp.now().strftime("%Y-%m-%d")
        todays = today_cls[today_cls["publish_time"].str.startswith(today_str)].to_dict("records")
        if todays:
            out = _write_snapshot(cfg, todays)
            print(f"snapshot -> {out}")


def main():
    ap = argparse.ArgumentParser(description="舆情情绪 MVP 管线")
    ap.add_argument("cmd", nargs="?", default="all", choices=["universe", "fetch", "fetch-cls", "fetch-extra", "score", "train-ml", "study", "audit", "snapshot", "daily", "all"])
    ap.add_argument("--codes", help="逗号分隔的股票代码，覆盖 universe")
    ap.add_argument("--llm", action="store_true", help="用本地 LLM 打情绪")
    ap.add_argument("--refresh-prices", action="store_true", help="忽略价格缓存重新抓取")
    ap.add_argument("--full", action="store_true", help="daily 按 news.max_pages 全量补历史")
    ap.add_argument("--no-study", action="store_true", help="daily 跳过事件研究（大批量补库时用）")
    ap.add_argument("--sample", type=int, default=300, help="audit 抽样数量")
    ap.add_argument("--label-source", choices=["lexicon", "llm"], default="lexicon", help="study 使用的情绪标签来源")
    ap.add_argument("--limit", type=int, default=0, help="LLM 打分只取随机 N 条（seed=42），0=全部")
    ap.add_argument("--sentiment-file", default="", help="study 使用指定情绪 CSV（相对项目根目录）")
    ap.add_argument("--tag", default="", help="study 输出文件后缀")
    ap.add_argument("--force", action="store_true", help="train-ml 强制重训")
    args = ap.parse_args()

    cfg = load_cfg()
    codes = [c.strip() for c in args.codes.split(",")] if args.codes else None

    if args.cmd in ("fetch", "all"):
        cmd_fetch(cfg, codes=codes)
    if args.cmd in ("fetch-cls", "all"):
        cmd_fetch_cls(cfg, codes=codes)
    if args.cmd in ("fetch-extra", "all"):
        cmd_fetch_extra(cfg, codes=codes)
    if args.cmd in ("score", "all"):
        cmd_score(cfg, use_llm=args.llm, limit=args.limit)
    if args.cmd == "train-ml":
        cmd_train_ml(cfg, force=args.force)
    if args.cmd in ("study", "all"):
        cmd_study(
            cfg,
            refresh_prices=args.refresh_prices,
            label_source=args.label_source,
            sentiment_file=args.sentiment_file or None,
            tag=args.tag or None,
        )
    if args.cmd == "audit":
        cmd_audit(cfg, sample_size=args.sample)
    if args.cmd == "snapshot":
        cmd_snapshot(cfg)
    if args.cmd == "daily":
        cmd_daily(cfg, refresh_prices=args.refresh_prices,
                  full_history=args.full, no_study=args.no_study)
    if args.cmd == "universe":
        cmd_universe(cfg)


if __name__ == "__main__":
    main()
