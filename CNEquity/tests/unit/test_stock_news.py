from datetime import date

import pytest

from cnequity.adapters.eastmoney.stock_news import fetch_stock_news
from cnequity.config import Config
from cnequity.domain.sentiment import aggregate_scores, keyword_score, score_text
from cnequity.query.on_demand import OnDemandService


class FakeNewsClient:
    def __init__(self, items: list[dict]):
        self.items = items

    def get(self, url, **kwargs):
        class Resp:
            def __init__(self, data):
                self._data = data

            def raise_for_status(self):
                return None

            def json(self):
                return {"data": {"list": self._data}}

        return Resp(self.items)

    def close(self):
        return None


def test_keyword_score_positive():
    score, method = score_text("业绩超预期增长", use_snownlp=False)
    assert score > 0
    assert method == "keyword"


def test_keyword_score_negative():
    assert keyword_score("收到行政处罚决定书") < 0


def test_aggregate_scores():
    assert aggregate_scores([1.0, -1.0]) == 0.0


def test_fetch_stock_news_parses_and_scores():
    client = FakeNewsClient(
        [
            {
                "title": "公司签订重大合同利好",
                "showtime": "2024-06-28 15:00:00",
                "art_code": "n1",
                "url": "https://example.com/1",
            },
            {
                "title": "日常经营简报",
                "showtime": "2024-06-27 10:00:00",
                "art_code": "n2",
            },
        ]
    )
    payload = fetch_stock_news(
        "600519.SH",
        on_date=date(2024, 6, 28),
        use_snownlp=False,
        client=client,  # type: ignore[arg-type]
    )
    assert payload["headline_count"] == 1
    assert payload["items"][0]["sentiment_method"] == "keyword"
    assert payload["aggregate_sentiment"] > 0


def test_fetch_stock_news_skips_non_object_rows():
    client = FakeNewsClient(
        [
            None,
            {
                "title": "有效新闻",
                "showtime": "2024-06-28 15:00:00",
                "art_code": "n1",
            },
        ]
    )
    payload = fetch_stock_news(
        "600519.SH",
        on_date=date(2024, 6, 28),
        use_snownlp=False,
        client=client,  # type: ignore[arg-type]
    )
    assert payload["headline_count"] == 1
    assert payload["items"][0]["news_id"] == "n1"


def test_fetch_stock_news_date_filter_drops_undated_items():
    client = FakeNewsClient(
        [
            {"title": "无日期新闻", "art_code": "undated"},
            {
                "title": "有效新闻",
                "showtime": "2024-06-28 15:00:00",
                "art_code": "dated",
            },
        ]
    )

    payload = fetch_stock_news(
        "600519.SH",
        on_date=date(2024, 6, 28),
        use_snownlp=False,
        client=client,  # type: ignore[arg-type]
    )

    assert payload["headline_count"] == 1
    assert payload["items"][0]["news_id"] == "dated"


def test_fetch_stock_news_date_filter_walks_older_pages():
    pages = [
        [
            {"title": "前一日新闻 1", "showtime": "2024-06-29 15:00:00", "art_code": "old-1"},
            {"title": "前一日新闻 2", "showtime": "2024-06-29 14:00:00", "art_code": "old-2"},
        ],
        [{"title": "目标日新闻", "showtime": "2024-06-28 15:00:00", "art_code": "target"}],
    ]

    class _PagedClient:
        def __init__(self):
            self.page_indexes: list[str] = []

        def get(self, url, *, params):
            self.page_indexes.append(params["page_index"])

            class Resp:
                def __init__(self, page):
                    self.page = page

                def raise_for_status(self):
                    return None

                def json(self):
                    return {"data": {"list": self.page}}

            return Resp(pages[len(self.page_indexes) - 1])

        def close(self):
            return None

    client = _PagedClient()
    payload = fetch_stock_news(
        "600519.SH",
        on_date=date(2024, 6, 28),
        limit=2,
        use_snownlp=False,
        client=client,  # type: ignore[arg-type]
    )

    assert client.page_indexes == ["1", "2"]
    assert payload["headline_count"] == 1
    assert payload["items"][0]["news_id"] == "target"


def test_fetch_stock_news_rejects_non_positive_limit():
    with pytest.raises(ValueError, match="limit must be positive"):
        fetch_stock_news("600519.SH", limit=0)


def test_fetch_stock_news_dedupes_stable_article_ids():
    client = FakeNewsClient(
        [
            {
                "title": "签约利好",
                "showtime": "2024-06-28 15:00:00",
                "art_code": "n1",
            },
            {
                "title": "签约利好修订",
                "showtime": "2024-06-28 15:01:00",
                "art_code": "n1",
            },
        ]
    )
    payload = fetch_stock_news(
        "600519.SH", on_date=date(2024, 6, 28), use_snownlp=False, client=client
    )
    assert payload["headline_count"] == 1
    assert payload["items"][0]["title"] == "签约利好修订"


def test_on_demand_stock_news_caches(tmp_path):
    cfg = Config(
        data_root=tmp_path / "data",
        on_demand_datasets=["stock_news"],
        sources={"eastmoney": True},
    )

    class StubService(OnDemandService):
        def _fetch_remote(self, dataset, symbol, **kwargs):
            return {
                "symbol": symbol,
                "source": "eastmoney",
                "items": [{"news_id": "1", "title": "分红方案公布", "sentiment_score": 0.5}],
                "headline_count": 1,
                "aggregate_sentiment": 0.5,
                "data_version": "v1",
                "fetched_at": "2024-06-28T00:00:00+00:00",
            }

    svc = StubService(cfg)
    first = svc.fetch("stock_news", "600519.SH")
    second = svc.fetch("stock_news", "600519.SH")
    assert first == second
    assert (cfg.meta_root / "on_demand" / "stock_news" / "600519_SH.json").exists()
