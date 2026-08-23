from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
from fastapi import APIRouter

from core.store import SENTIMENT_DIR


router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])

SENT_ROOT = SENTIMENT_DIR
SENT_DB = SENT_ROOT / "articles.db"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SENT_IC_CSV = PROJECT_ROOT / "results" / "sentiment_ic_group.csv"


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except Exception:
        return None


def _query(sql: str, params: tuple = ()) -> pd.DataFrame:
    conn = sqlite3.connect(str(SENT_DB))
    try:
        return pd.read_sql_query(sql, conn, params=params)
    finally:
        conn.close()


@router.get("/status")
def sentiment_status():
    return {
        "data_dir": str(SENT_ROOT),
        "has_news": SENT_DB.exists(),
        "has_ic": SENT_IC_CSV.exists(),
        "has_articles_db": SENT_DB.exists(),
        "source": "articles.db",
    }


@router.get("/ic")
def sentiment_ic():
    """舆情分桶回测的 IC/分组摘要（scripts/sentiment_backtest.py 输出）。"""
    df = _read_csv(SENT_IC_CSV)
    if df is None or df.empty:
        return {"items": [], "error": "暂无舆情 IC 结果，先运行 scripts/sentiment_backtest.py"}
    return {"items": df.to_dict(orient="records")}


@router.get("/stats")
def sentiment_stats():
    """舆情统计，读取全量 articles.db（词典打分全量入库）。"""
    if not SENT_DB.exists():
        return {"ok": False, "error": "暂无舆情数据，先运行 sentiment-mvp 流水线"}
    total = _query("SELECT COUNT(*) AS n FROM articles").iloc[0]["n"]
    n_codes = _query("SELECT COUNT(DISTINCT code) AS n FROM articles").iloc[0]["n"]
    label_df = _query("SELECT label, COUNT(*) AS n FROM articles GROUP BY label")
    daily_df = _query(
        "SELECT substr(publish_time,1,10) AS d, COUNT(*) AS n "
        "FROM articles WHERE publish_time != '' "
        "GROUP BY d ORDER BY d DESC LIMIT 30")
    row = _query(
        "SELECT MAX(publish_time) AS last, AVG(score) AS avg_score "
        "FROM articles WHERE publish_time != ''").iloc[0]
    label_dist = {str(k): int(v) for k, v in zip(label_df["label"], label_df["n"])}
    return {
        "ok": True,
        "n_articles": int(total),
        "n_codes": int(n_codes),
        "last_date": str(row["last"])[:10],
        "label_dist": label_dist,
        "mean_score": float(row["avg_score"]) if row["avg_score"] is not None else None,
        "daily": [{"date": str(d), "n": int(n)}
                  for d, n in zip(daily_df["d"], daily_df["n"])],
    }


@router.get("/news")
def sentiment_news(top: int = 10, sort: str = "high", days: int = 7):
    """情绪最强（sort=high）或最弱（sort=low）新闻，默认只取最近 days 天。"""
    if not SENT_DB.exists():
        return {"items": []}
    df = _query(
        "SELECT code,publish_time,media,title,url,source,label,score "
        "FROM articles WHERE publish_time != ''")
    if df.empty:
        return {"items": []}
    df["dt"] = pd.to_datetime(df["publish_time"], errors="coerce")
    df = df.dropna(subset=["dt"])
    last = df["dt"].max()
    cutoff = last - pd.Timedelta(days=max(1, int(days)))
    sub = df[df["dt"] >= cutoff]
    if sub.empty:
        sub = df
    sub = sub.sort_values("score", ascending=(sort != "high"))
    out = []
    for _, r in sub.head(max(1, min(int(top), 100))).iterrows():
        out.append({
            "code": str(r["code"]).zfill(6),
            "publish_time": str(r.get("publish_time", ""))[:19],
            "media": str(r.get("media", "")),
            "title": str(r.get("title", "")),
            "url": str(r.get("url", "")),
            "label": str(r.get("label", "")),
            "score": float(r["score"]),
            "source": str(r.get("source", "")),
        })
    return {"items": out}
