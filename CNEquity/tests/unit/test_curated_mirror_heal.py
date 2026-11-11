"""curated 镜像缺口自愈（heal_curated_mirror）的回归测试。

2026-11-06 事故形态：external 归档有当天 5573 行，curated 镜像却因为 compact 被
源超时打断而少了 ``trade_date=2026-11-06`` 分区；水位读的是 external，正常走到
11-09，于是日常调度永远看不见这个洞，只能人工 backfill。
"""

from datetime import date

import polars as pl
import pytest

from cnequity.steps import finalize

DAYS = [
    date(2026, 11, 3),
    date(2026, 11, 4),
    date(2026, 11, 5),
    date(2026, 11, 6),
    date(2026, 11, 9),
]
MISSING = date(2026, 11, 6)


def _write_partition(config, day: date, rows: int = 2) -> None:
    part = config.curated_root / "daily_bars" / f"trade_date={day.isoformat()}"
    part.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        {"symbol": [f"00000{i}.SZ" for i in range(rows)], "trade_date": [day] * rows}
    ).write_parquet(part / "part-sync.parquet")


def _patch_sources(monkeypatch, first: date = DAYS[0], last: date = DAYS[-1]) -> None:
    """把 external 覆盖范围与交易日历固定成可预测的小窗口。"""
    monkeypatch.setattr(
        "cnequity.external.registry.external_coverage_bounds",
        lambda cfg, ds: (first, last) if ds == "daily_bars" else (None, None),
    )
    monkeypatch.setattr(
        "cnequity.steps.common.list_trading_dates",
        lambda cfg, start, end: [d for d in DAYS if start <= d <= end],
    )


def test_heal_fills_missing_trading_day(config, monkeypatch):
    for day in DAYS:
        if day != MISSING:
            _write_partition(config, day)
    _patch_sources(monkeypatch)

    calls: list[tuple[str, date]] = []

    def _fake_sync(cfg, dataset, day):
        calls.append((dataset, day))
        return 3

    monkeypatch.setattr(finalize, "_sync_external_to_curated", _fake_sync)

    result = finalize.heal_curated_mirror(config)

    assert calls == [("daily_bars", MISSING)]
    assert result["gaps"] == {"daily_bars": [MISSING.isoformat()]}
    assert result["healed_days"] == 1
    assert result["rows"] == 3
    assert result["datasets_scanned"] == ["daily_bars"]


def test_dry_run_reports_without_writing(config, monkeypatch):
    for day in DAYS:
        if day != MISSING:
            _write_partition(config, day)
    _patch_sources(monkeypatch)

    calls: list[tuple] = []
    monkeypatch.setattr(
        finalize, "_sync_external_to_curated", lambda *a, **k: calls.append(a) or 1
    )

    result = finalize.heal_curated_mirror(config, apply=False)

    assert calls == []
    assert result["applied"] is False
    assert result["gaps"] == {"daily_bars": [MISSING.isoformat()]}
    assert result["rows"] == 0


def test_nothing_to_heal_when_mirror_complete(config, monkeypatch):
    for day in DAYS:
        _write_partition(config, day)
    _patch_sources(monkeypatch)
    monkeypatch.setattr(
        finalize,
        "_sync_external_to_curated",
        lambda *a, **k: pytest.fail("complete mirror must not be rewritten"),
    )

    result = finalize.heal_curated_mirror(config)

    assert result["gaps"] == {}
    assert result["healed_days"] == 0
    assert result["rows"] == 0


def test_heal_respects_lookback(config, monkeypatch):
    """窗口外的旧缺口不动：回看限制决定扫描范围，不是"补全历史"。"""
    for day in DAYS:
        if day != MISSING:
            _write_partition(config, day)
    _patch_sources(monkeypatch)
    monkeypatch.setattr(finalize, "_sync_external_to_curated", lambda *a, **k: 1)

    result = finalize.heal_curated_mirror(config, lookback_days=1)

    # 最近 1 个交易日是 11-09，它已经在盘上 → 无缺口。
    assert result["gaps"] == {}
