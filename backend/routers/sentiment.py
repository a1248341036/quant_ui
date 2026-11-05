from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from core import sentiment_store

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SENT_IC_CSV = PROJECT_ROOT / "results" / "sentiment_ic_group.csv"
SENT_IC_META = PROJECT_ROOT / "results" / "sentiment_ic_meta.json"


def _read_ic_csv():
    if not SENT_IC_CSV.exists():
        return None
    try:
        import pandas as pd

        return pd.read_csv(SENT_IC_CSV, encoding="utf-8-sig")
    except Exception:
        return None


def _read_ic_meta() -> dict:
    if not SENT_IC_META.exists():
        return {}
    try:
        return json.loads(SENT_IC_META.read_text(encoding="utf-8"))
    except Exception:
        return {}


@router.get("/status")
def sentiment_status():
    """数据源健康：旧 articles.db + CNE curated sentiment_articles 两段各自状态。"""
    df, sources = sentiment_store.load_articles()
    return {
        "data_dir": str(sentiment_store.LEGACY_DB.parent),
        "has_news": len(df) > 0,
        "has_ic": SENT_IC_CSV.exists(),
        "has_articles_db": sentiment_store.LEGACY_DB.exists(),
        "source": "legacy_articles_db+cne_curated",
        "n_articles": int(len(df)),
        "sources": sources,
    }


@router.get("/ic")
def sentiment_ic():
    """舆情分桶回测的 IC/分组摘要（scripts/sentiment_backtest.py 输出）。"""
    df = _read_ic_csv()
    meta = _read_ic_meta()
    if df is None or df.empty:
        return {
            "items": [],
            "window": meta.get("window"),
            "error": "暂无舆情 IC 结果，先运行 scripts/sentiment_backtest.py",
        }
    return {
        "items": df.to_dict(orient="records"),
        "window": meta.get("window"),
        "panel_source": meta.get("panel_source"),
        "generated_at": meta.get("generated_at"),
    }


@router.get("/stats")
def sentiment_stats():
    """舆情统计：旧库（2026-02~08）+ CNE curated（2026-09 起）合并。"""
    return sentiment_store.stats()


@router.get("/news")
def sentiment_news(top: int = 10, sort: str = "high", days: int = 7):
    """情绪最强（sort=high）或最弱（sort=low）新闻，默认只取最近 days 天。"""
    return {"items": sentiment_store.news(top=top, sort=sort, days=days)}
