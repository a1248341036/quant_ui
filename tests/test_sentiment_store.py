# -*- coding: utf-8 -*-
"""core.sentiment_store 双源合并行为回归（不依赖真实数据文件）。"""

from __future__ import annotations

import pandas as pd

from core import sentiment_store as S


def _frame(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in S.COLUMNS:
        if col not in df:
            df[col] = None
    return df.loc[:, list(S.COLUMNS)]


def test_norm_code_strips_exchange_suffix_and_pads() -> None:
    assert S._norm_code("688519.SH") == "688519"
    assert S._norm_code("000001") == "000001"
    assert S._norm_code("1") == "000001"
    assert S._norm_code("") == ""
    assert S._norm_code(None) == ""


def test_finalize_dedupes_by_article_id_keeping_last() -> None:
    df = _frame(
        [
            {"article_id": "a1", "code": "000001", "publish_time": "2026-08-01 10:00:00",
             "media": "旧", "title": "旧标题", "url": "u1", "source": "cls",
             "label": "neutral", "score": 0.0},
            {"article_id": "a1", "code": "000001", "publish_time": "2026-08-01 10:00:00",
             "media": "新", "title": "新标题", "url": "u1", "source": "em",
             "label": "positive", "score": 0.8},
        ]
    )
    out = S._finalize(df)
    assert len(out) == 1
    assert out.iloc[0]["title"] == "新标题"
    assert out.iloc[0]["label"] == "positive"


def test_finalize_derives_publish_date_from_time() -> None:
    df = _frame(
        [{"article_id": "a1", "code": "000001", "publish_time": "2026-09-10 08:30:00",
          "media": "m", "title": "t", "url": "u", "source": "em",
          "label": "neutral", "score": 0.1}]
    )
    out = S._finalize(df)
    assert str(out.iloc[0]["publish_date"].date()) == "2026-09-10"


def test_finalize_synthesizes_id_for_empty_article_id() -> None:
    df = _frame(
        [
            {"article_id": "", "code": "000001", "publish_time": "2026-08-01 10:00:00",
             "media": "m", "title": "t1", "url": "u", "source": "hot",
             "label": "neutral", "score": 0.0},
            {"article_id": "", "code": "000001", "publish_time": "2026-08-01 10:00:00",
             "media": "m", "title": "t2", "url": "u", "source": "hot",
             "label": "neutral", "score": 0.0},
        ]
    )
    out = S._finalize(df)
    # 标题不同 → 合成键不同 → 两行都保留
    assert len(out) == 2
    assert out["article_id"].str.startswith("synth:").all()


def test_stats_and_news_use_merged_cache(monkeypatch) -> None:
    legacy = _frame(
        [{"article_id": "old1", "code": "000001", "publish_time": "2026-03-02 09:00:00",
          "media": "旧媒", "title": "三月利空", "url": "u1", "source": "cls",
          "label": "negative", "score": -0.9}]
    )
    cne = _frame(
        [{"article_id": "new1", "code": "600000", "publish_time": "2026-10-09 10:00:00",
          "media": "证券时报", "title": "十月利好", "url": "u2", "source": "em",
          "label": "positive", "score": 0.9},
         {"article_id": "new2", "code": "600000", "publish_time": "2026-10-10 10:00:00",
          "media": "证券时报", "title": "十月利空", "url": "u3", "source": "em",
          "label": "negative", "score": -0.8}]
    )
    monkeypatch.setattr(S, "_legacy_frame", lambda: (legacy, {"name": "legacy_articles_db", "rows": 1}))
    monkeypatch.setattr(S, "_cne_frame", lambda: (cne, {"name": "cne_curated", "rows": 2}))
    monkeypatch.setattr(S, "_signature", lambda: ("test-sig",))
    S.invalidate_cache()

    st = S.stats()
    assert st["ok"] is True
    assert st["n_articles"] == 3
    assert st["n_codes"] == 2
    assert st["first_date"] == "2026-03-02"
    assert st["last_date"] == "2026-10-10"
    assert st["label_dist"] == {"negative": 2, "positive": 1}
    assert len(st["sources"]) == 2

    high = S.news(top=2, sort="high", days=7)
    assert high[0]["title"] == "十月利好"
    assert high[0]["code"] == "600000"

    low = S.news(top=2, sort="low", days=7)
    assert low[0]["title"] == "十月利空"

    S.invalidate_cache()


def test_stats_reports_unavailable_sources(monkeypatch) -> None:
    empty = pd.DataFrame(columns=list(S.COLUMNS))
    monkeypatch.setattr(S, "_legacy_frame", lambda: (empty, {"name": "legacy_articles_db", "rows": 0, "error": "x"}))
    monkeypatch.setattr(S, "_cne_frame", lambda: (empty, {"name": "cne_curated", "rows": 0, "error": "y"}))
    monkeypatch.setattr(S, "_signature", lambda: ("empty-sig",))
    S.invalidate_cache()

    st = S.stats()
    assert st["ok"] is False
    assert len(st["sources"]) == 2
    assert S.news() == []
    S.invalidate_cache()
