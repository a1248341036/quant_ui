"""Unit tests for EastMoney rotation adapters."""

from datetime import date
from unittest.mock import MagicMock

import pytest

from cnequity.adapters.eastmoney.rotation import (
    _fetch_board_rows,
    _hot_symbol,
    _news_symbols,
    fetch_hot_rank,
    fetch_news_headlines,
)


def test_hot_symbol_parsing():
    assert _hot_symbol("SZ002185") == "002185.SZ"
    assert _hot_symbol("SH600519") == "600519.SH"


def test_hot_symbol_rejects_malformed_or_non_a_codes():
    assert _hot_symbol("SZ00abc1") is None
    assert _hot_symbol("SZ810001") is None


def test_news_symbols_routes_beijing_codes():
    assert _news_symbols(["2.920001", "2.830001"]) == "830001.BJ,920001.BJ"


def test_news_symbols_skips_malformed_codes():
    assert _news_symbols(["2.abc", "2.920001"]) == "920001.BJ"


def test_fetch_board_rows_rejects_rows_without_codes(monkeypatch):
    monkeypatch.setattr(
        "cnequity.adapters.eastmoney.rotation.fetch_clist_pages",
        lambda *args, **kwargs: [{"f12": "BK0001"}, {"f14": "missing-code"}],
    )
    with pytest.raises(RuntimeError, match=r"concept board clist returned 1 row\(s\) without f12"):
        _fetch_board_rows(object(), "m:90+t:3", "concept")


def test_fetch_hot_rank_normalizes(monkeypatch):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"sc": "SZ002185", "rk": 1, "rc": 2, "hisRc": 3}],
    }
    mock_client.post.return_value = mock_resp

    class CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("cnequity.adapters.eastmoney.rotation.EastMoneyClient", lambda: CM())
    df = fetch_hot_rank(date(2026, 7, 14), top_n=10)
    assert df.height == 1
    assert df["symbol"][0] == "002185.SZ"
    assert df["rank"][0] == 1


def test_fetch_hot_rank_preserves_missing_rank_fields(monkeypatch):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"sc": "SZ002185", "rk": "", "rc": None, "hisRc": "bad"}]
    }
    mock_client.post.return_value = mock_resp

    class CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("cnequity.adapters.eastmoney.rotation.EastMoneyClient", lambda: CM())
    df = fetch_hot_rank(date(2026, 7, 14), top_n=10)
    row = df.row(0, named=True)
    assert row["rank"] is None
    assert row["rank_change"] is None
    assert row["hist_rank"] is None


def test_fetch_hot_rank_preserves_invalid_integer_fields(monkeypatch):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [{"sc": "SZ002185", "rk": "1.5", "rc": 1e300, "hisRc": -1e300}]
    }
    mock_client.post.return_value = mock_resp

    class CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("cnequity.adapters.eastmoney.rotation.EastMoneyClient", lambda: CM())
    row = fetch_hot_rank(date(2026, 7, 14), top_n=10).row(0, named=True)
    assert row["rank"] is None
    assert row["rank_change"] is None
    assert row["hist_rank"] is None


def test_fetch_hot_rank_skips_non_object_rows(monkeypatch):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [None, {"sc": "SZ002185", "rk": 1, "rc": 0, "hisRc": 2}],
    }
    mock_client.post.return_value = mock_resp

    class CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("cnequity.adapters.eastmoney.rotation.EastMoneyClient", lambda: CM())
    df = fetch_hot_rank(date(2026, 7, 14), top_n=10)
    assert df.height == 1
    assert df["symbol"][0] == "002185.SZ"


def test_fetch_hot_rank_dedupes_symbols(monkeypatch):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": [
            {"sc": "SZ002185", "rk": 1, "rc": 0, "hisRc": 2},
            {"sc": "SZ002185", "rk": 2, "rc": 1, "hisRc": 3},
        ],
    }
    mock_client.post.return_value = mock_resp

    class CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("cnequity.adapters.eastmoney.rotation.EastMoneyClient", lambda: CM())
    df = fetch_hot_rank(date(2026, 7, 14), top_n=10)
    assert df.height == 1
    assert df["rank"][0] == 2


def test_fetch_hot_rank_continues_when_a_full_page_repeats_symbols(monkeypatch):
    mock_client = MagicMock()
    first = MagicMock()
    first.json.return_value = {
        "data": [{"sc": "SZ002185", "rk": i, "rc": 0, "hisRc": i} for i in range(100)]
    }
    second = MagicMock()
    second.json.return_value = {
        "data": [
            {"sc": "SZ000001", "rk": 101, "rc": 0, "hisRc": 101},
            {"sc": "SH600519", "rk": 102, "rc": 0, "hisRc": 102},
        ]
    }
    mock_client.post.side_effect = [first, second]

    class CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("cnequity.adapters.eastmoney.rotation.EastMoneyClient", lambda: CM())
    df = fetch_hot_rank(date(2026, 7, 14), top_n=3)

    assert set(df["symbol"].to_list()) == {"002185.SZ", "000001.SZ", "600519.SH"}
    assert mock_client.post.call_count == 2


def test_fetch_hot_rank_rejects_a_repeated_page(monkeypatch):
    mock_client = MagicMock()
    response = MagicMock()
    response.json.return_value = {
        "data": [{"sc": "SZ002185", "rk": i, "rc": 0, "hisRc": i} for i in range(100)]
    }
    mock_client.post.return_value = response

    class CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("cnequity.adapters.eastmoney.rotation.EastMoneyClient", lambda: CM())
    with pytest.raises(RuntimeError, match="repeated page"):
        fetch_hot_rank(date(2026, 7, 14), top_n=2)
    assert mock_client.post.call_count == 2


def test_fetch_hot_rank_can_reject_an_incomplete_top_n(monkeypatch):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": [{"sc": "SZ002185", "rk": 1, "rc": 0, "hisRc": 2}]}
    mock_client.post.return_value = mock_resp

    class CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("cnequity.adapters.eastmoney.rotation.EastMoneyClient", lambda: CM())
    with pytest.raises(RuntimeError, match="only 1 unique A-share symbols; expected 10"):
        fetch_hot_rank(date(2026, 7, 14), top_n=10, require_top_n=True)


def test_fetch_hot_rank_rejects_negative_top_n():
    with pytest.raises(ValueError, match="top_n must be non-negative"):
        fetch_hot_rank(date(2026, 7, 14), top_n=-1)


def test_fetch_news_headlines_filters_date(monkeypatch):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": {
            "fastNewsList": [
                None,
                {
                    "code": "n1",
                    "showTime": "2026-07-14 16:00:00",
                    "title": "测试新闻",
                    "summary": "摘要",
                    "stockList": ["0.600519"],
                },
                {
                    "code": "n2",
                    "showTime": "2026-07-13 16:00:00",
                    "title": "旧新闻",
                },
                {
                    "showTime": "2026-07-14 17:00:00",
                    "title": "无法定位的新闻",
                },
                {
                    "code": "bad-time",
                    "showTime": "2026-07-14 not-a-time",
                    "title": "坏时间新闻",
                },
                {
                    "code": "empty-title",
                    "showTime": "2026-07-14 18:00:00",
                    "title": " ",
                },
            ]
        }
    }
    mock_client.get.return_value = mock_resp

    class CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("cnequity.adapters.eastmoney.rotation.EastMoneyClient", lambda: CM())
    df = fetch_news_headlines(date(2026, 7, 14))
    assert df.height == 1
    assert df["news_id"][0] == "n1"
    assert "600519.SH" in df["related_symbols"][0]


def test_fetch_news_headlines_follows_cursor_until_target_date_ends(monkeypatch):
    mock_client = MagicMock()
    first = MagicMock()
    first.json.return_value = {
        "data": {
            "sortEnd": "cursor-1",
            "fastNewsList": [
                {
                    "code": "n1",
                    "showTime": "2026-07-14 16:00:00",
                    "title": "较新的新闻",
                }
            ],
        }
    }
    second = MagicMock()
    second.json.return_value = {
        "data": {
            "sortEnd": "cursor-2",
            "fastNewsList": [
                {
                    "code": "n2",
                    "showTime": "2026-07-14 09:00:00",
                    "title": "较早的新闻",
                },
                {
                    "code": "old",
                    "showTime": "2026-07-13 18:00:00",
                    "title": "更早的新闻",
                },
            ],
        }
    }
    mock_client.get.side_effect = [first, second]

    class CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("cnequity.adapters.eastmoney.rotation.EastMoneyClient", lambda: CM())
    df = fetch_news_headlines(date(2026, 7, 14), page_size=1)

    assert set(df["news_id"].to_list()) == {"n1", "n2"}
    assert mock_client.get.call_count == 2
    assert "sortEnd=cursor-1" in mock_client.get.call_args_list[1].args[0]


def test_fetch_news_headlines_rejects_a_stalled_cursor(monkeypatch):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": {
            "sortEnd": "same-cursor",
            "fastNewsList": [{"code": "n1", "showTime": "2026-07-14 16:00:00", "title": "新闻"}],
        }
    }
    mock_client.get.return_value = mock_resp

    class CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("cnequity.adapters.eastmoney.rotation.EastMoneyClient", lambda: CM())
    with pytest.raises(RuntimeError, match="cursor did not advance"):
        fetch_news_headlines(date(2026, 7, 14), page_size=1)


def test_fetch_news_headlines_rejects_non_positive_page_size():
    with pytest.raises(ValueError, match="page_size must be positive"):
        fetch_news_headlines(date(2026, 7, 14), page_size=0)


def test_fetch_news_headlines_rejects_non_list_rows(monkeypatch):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": {"fastNewsList": {"code": "n1"}}}
    mock_client.get.return_value = mock_resp

    class CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("cnequity.adapters.eastmoney.rotation.EastMoneyClient", lambda: CM())
    with pytest.raises(RuntimeError, match="news response rows are not a list"):
        fetch_news_headlines(date(2026, 7, 14))


def test_fetch_news_headlines_dedupes_news_ids(monkeypatch):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.json.return_value = {
        "data": {
            "fastNewsList": [
                {
                    "code": "n1",
                    "showTime": "2026-07-14 16:00:00",
                    "title": "旧版本",
                },
                {
                    "code": "n1",
                    "showTime": "2026-07-14 16:01:00",
                    "title": "修订版本",
                },
            ]
        }
    }
    mock_client.get.return_value = mock_resp

    class CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, *a):
            pass

    monkeypatch.setattr("cnequity.adapters.eastmoney.rotation.EastMoneyClient", lambda: CM())
    df = fetch_news_headlines(date(2026, 7, 14))
    assert df.height == 1
    assert df["title"][0] == "修订版本"


def _cm(mock_client):
    class CM:
        def __enter__(self):
            return mock_client

        def __exit__(self, *a):
            pass

    return CM


def test_fetch_sector_fund_flow_falls_back_to_tushare_on_clist_failure(monkeypatch):
    import polars as pl

    from cnequity.adapters.eastmoney import rotation

    def _boom(*_a, **_kw):
        raise RuntimeError("EastMoney clist page 1 failed on all hosts")

    monkeypatch.setattr(rotation, "_fetch_board_rows", _boom)

    ts_raw = pl.DataFrame(
        {
            "trade_date": ["20260923"] * 3,
            "content_type": ["概念", "行业", "地域"],
            "ts_code": ["BK0447.DC", "BK0420.DC", "BK0145.DC"],
            "name": ["互联网服务", "航空机场", "上海板块"],
            "pct_change": [-1.64, 0.03, -0.09],
            "net_amount": [-3.64e9, 1.06e7, -5.74e9],
        }
    )
    monkeypatch.setattr(
        rotation, "_get_pro", lambda config: object()
    )
    monkeypatch.setattr(
        rotation,
        "_fetch_with_retry",
        lambda pro, api, **kwargs: ts_raw if api == "moneyflow_ind_dc" else pl.DataFrame(),
    )

    class FakeConfig:
        external_tushare_wide_interval = 0.0

    df = rotation.fetch_sector_fund_flow(date(2026, 9, 23), config=FakeConfig())
    assert df.height == 2  # 地域 dropped
    assert set(df["board_type"].to_list()) == {"concept", "industry"}
    assert set(df["source"].unique().to_list()) == {"tushare"}
    row = df.filter(pl.col("sector_code") == "BK0447").row(0, named=True)
    assert row["sector_name"] == "互联网服务"
    assert row["main_net_inflow"] == -3.64e9
    assert row["change_pct"] == -1.64
    assert row["turnover_pct"] is None


def test_fetch_sector_fund_flow_tushare_raises_without_config(monkeypatch):
    from cnequity.adapters.eastmoney import rotation

    def _boom(*_a, **_kw):
        raise RuntimeError("EastMoney clist page 1 failed on all hosts")

    monkeypatch.setattr(rotation, "_fetch_board_rows", _boom)
    with pytest.raises(RuntimeError, match="clist page 1 failed"):
        rotation.fetch_sector_fund_flow(date(2026, 9, 23))
