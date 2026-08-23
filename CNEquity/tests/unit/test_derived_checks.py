from datetime import date

import polars as pl

from cnequity.config import Config
from cnequity.quality.derived_checks import industry_index_findings, market_breadth_findings
from cnequity.quality.pit_checks import pit_announce_date_findings


def _write_breadth(cfg: Config, rows: list[dict]) -> None:
    root = cfg.curated_root / "market_breadth" / "trade_date=2026"
    root.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(root / "part-000.parquet")


def _valid_rows(day: date) -> list[dict]:
    values = {
        "advance_count": 4.0,
        "decline_count": 3.0,
        "flat_count": 3.0,
        "limit_up_count": 1.0,
        "limit_down_count": 1.0,
        "advance_ratio": 0.4,
        "total_count": 10.0,
    }
    return [
        {"trade_date": day, "metric_id": metric, "value": value} for metric, value in values.items()
    ]


def test_market_breadth_reports_partial_observation(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    rows = _valid_rows(date(2026, 8, 7))
    _write_breadth(cfg, rows[:2])

    findings = market_breadth_findings(cfg, date(2026, 8, 7))

    incomplete = [f for f in findings if f["check"] == "market_breadth_incomplete_day"]
    assert len(incomplete) == 1
    assert incomplete[0]["severity"] == "error"


def test_market_breadth_reports_inconsistent_metrics(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    rows = _valid_rows(date(2026, 8, 7))
    rows[-1]["value"] = 99.0
    _write_breadth(cfg, rows)

    findings = market_breadth_findings(cfg, date(2026, 8, 7))

    assert any(f["check"] == "market_breadth_inconsistent_metrics" for f in findings)


def test_market_breadth_accepts_complete_observation(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    _write_breadth(cfg, _valid_rows(date(2026, 8, 7)))

    assert market_breadth_findings(cfg, date(2026, 8, 7)) == []


def _write_industry(cfg: Config, rows: list[dict]) -> None:
    root = cfg.derived_root / "industry_index" / "trade_date=2026"
    root.mkdir(parents=True)
    pl.DataFrame(rows).write_parquet(root / "part-000.parquet")


def _write_sw_membership(cfg: Config) -> None:
    root = cfg.curated_root / "industry_members" / "as_of_date=2026-08"
    root.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["600000.SH", "600001.SH"],
            "classification_system": ["sw", "sw"],
            "industry_code": ["240301", "240401"],
            "industry_name": ["铝", "钢铁"],
            "as_of_date": [date(2026, 8, 1), date(2026, 8, 1)],
            "source": ["sw", "sw"],
            "data_version": ["v1", "v1"],
            "fetched_at": ["2026-08-01T00:00:00+00:00"] * 2,
        }
    ).write_parquet(root / "part-000.parquet")


def _industry_rows(day: date) -> list[dict]:
    rows = []
    for weighting in ("equal", "amount"):
        rows.append(
            {
                "trade_date": day,
                "industry_code": "2403",
                "level": "L2",
                "weighting": weighting,
                "ret": 0.01,
                "n_members": 10,
                "n_priced": 9,
                "n_excluded": 1,
            }
        )
    return rows


def test_industry_index_reports_missing_weighting(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    _write_industry(cfg, _industry_rows(date(2026, 8, 7))[:1])

    findings = industry_index_findings(cfg, date(2026, 8, 7))

    assert any(f["check"] == "industry_index_incomplete_weightings" for f in findings)


def test_industry_index_reports_invalid_accounting(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    rows = _industry_rows(date(2026, 8, 7))
    rows[0]["n_excluded"] = 3
    _write_industry(cfg, rows)

    findings = industry_index_findings(cfg, date(2026, 8, 7))

    assert any(f["check"] == "industry_index_invalid_accounting" for f in findings)


def test_industry_index_accepts_complete_group(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    _write_industry(cfg, _industry_rows(date(2026, 8, 7)))

    assert industry_index_findings(cfg, date(2026, 8, 7)) == []


def test_industry_index_reports_missing_membership_group(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    _write_sw_membership(cfg)
    _write_industry(cfg, _industry_rows(date(2026, 8, 7)))

    findings = industry_index_findings(cfg, date(2026, 8, 7))

    missing = [f for f in findings if f["check"] == "industry_index_missing_groups"]
    assert len(missing) == 1
    assert missing[0]["severity"] == "error"
    assert missing[0]["missing_groups"] == 4


def test_financial_statement_items_reports_sentinel_announce_dates(tmp_path):
    cfg = Config(data_root=tmp_path / "data")
    root = cfg.curated_root / "financial_statement_items" / "report_period=2004Q2"
    root.mkdir(parents=True)
    pl.DataFrame(
        {
            "symbol": ["600000.SH"],
            "report_period": ["2004Q2"],
            "statement_type": ["balance"],
            "item_code": ["total_assets"],
            "item_value": [1.0],
            "announce_date": [date(1900, 1, 1)],
        }
    ).write_parquet(root / "part-000.parquet")

    findings = pit_announce_date_findings(cfg)

    assert findings[0]["check"] == "pit_invalid_announce_date"
    assert findings[0]["invalid_rows"] == 1
