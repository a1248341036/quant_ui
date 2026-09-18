"""Tushare 风险警示板源（stock_st）：分页、回填窗口、契约注册。

背景：`trading_status` 的历史 ST 靠 baostock 逐票扫（5557 票 × 1s + 批次冷却
≈ 11 小时）；Tushare `stock_st` 是"当日全市场 ST 名单"，区间调用 + offset 分页
即可回填全历史（2016→今约 440 页 / 2~3 分钟）。本文件锁住三件事：
单页截断不再静默、`cne backfill --start/--end` 真正生效、契约（schema/PK/spec）齐全。
"""

from __future__ import annotations

from datetime import date

import polars as pl
import pytest

import cnequity.steps  # noqa: F401  — 触发 step 注册
from cnequity.config import Config
from cnequity.domain import schemas
from cnequity.domain.datasets import get_dataset
from cnequity.orchestrator.registry import get_step
from cnequity.steps import tushare_wide as tw


@pytest.fixture
def cfg(tmp_path):
    c = Config(
        data_root=tmp_path / "data",
        external_tushare_wide_token="test-token",
        external_tushare_wide_interval=0.0,
    )
    c.staging_root.mkdir(parents=True, exist_ok=True)
    return c


def _page(start_index: int, n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "ts_code": [f"{600000 + start_index + i:06d}.SH" for i in range(n)],
            "name": ["ST测试"] * n,
            "trade_date": [date(2024, 6, 28)] * n,
            "type": ["ST"] * n,
            "type_name": ["风险警示板"] * n,
        }
    )


# ─── 契约 ─────────────────────────────────────────────────────────────


def test_stock_st_contract_registered():
    spec = get_dataset("stock_st")
    assert spec.primary_source == "tushare"
    assert spec.partition_col == "trade_date"
    assert spec.partition_granularity == "month"

    pk = schemas.PRIMARY_KEYS["stock_st"]
    assert pk == ["symbol", "trade_date"]
    schema = schemas.DATASET_SCHEMAS["stock_st"]
    # 主键必须落在 schema 里，否则 compact 去重与校验会对不上
    assert set(pk) <= set(schema)


def test_stock_st_step_registered_in_daily_core_group():
    entry = get_step("stock_st")
    assert entry.group == "core"
    assert entry.depends_on == ["instruments"]


# ─── 分页 ─────────────────────────────────────────────────────────────


def test_fetch_range_paged_walks_until_short_page(monkeypatch):
    """满页必须继续翻页——旧实现只取首页，历史回填会被静默截断到最近 1000 行。"""
    calls: list[dict] = []

    def fake_fetch(pro, api, interval=0.0, **kw):
        calls.append(kw)
        if kw["offset"] == 0:
            return _page(0, tw._TUSHARE_PAGE_LIMIT)
        if kw["offset"] == tw._TUSHARE_PAGE_LIMIT:
            return _page(tw._TUSHARE_PAGE_LIMIT, 5)
        return pl.DataFrame()

    monkeypatch.setattr("cnequity.external.tushare_fetch._fetch_with_retry", fake_fetch)

    # 短窗口（≤366 天）= 单窗口，避免与按年切窗行为混在一起测
    out = tw._fetch_range_paged(
        object(), "stock_st", interval=0.0, start=date(2024, 1, 1), end=date(2024, 6, 28)
    )

    assert out.height == tw._TUSHARE_PAGE_LIMIT + 5
    assert [c["offset"] for c in calls][:2] == [0, tw._TUSHARE_PAGE_LIMIT]
    assert calls[0]["limit"] == tw._TUSHARE_PAGE_LIMIT
    assert calls[0]["start_date"] == "20240101"


def test_transient_empty_page_does_not_truncate_the_window(monkeypatch):
    """源端偶发空页不得当作"取完"：实测这样丢过 2024-01-02..02-22 共 32 个交易日。"""
    per_offset: dict[int, int] = {}

    def fake_fetch(pro, api, interval=0.0, **kw):
        off = kw["offset"]
        per_offset[off] = per_offset.get(off, 0) + 1
        if off == 0:
            return _page(0, tw._TUSHARE_PAGE_LIMIT)  # 满页
        if off == tw._TUSHARE_PAGE_LIMIT:
            # 第 1 次空（偶发），第 2 次仍有数据 → 必须继续翻页
            if per_offset[off] == 1:
                return pl.DataFrame()
            return _page(tw._TUSHARE_PAGE_LIMIT, 40)
        return pl.DataFrame()  # 更深处才是真的取完

    monkeypatch.setattr("cnequity.external.tushare_fetch._fetch_with_retry", fake_fetch)

    out = tw._fetch_range_paged(
        object(), "stock_st", interval=0.0, start=date(2024, 1, 1), end=date(2024, 6, 28)
    )

    assert out.height == tw._TUSHARE_PAGE_LIMIT + 40
    assert per_offset[tw._TUSHARE_PAGE_LIMIT] == 2  # 空页被重取过一次


def test_fetch_range_paged_splits_long_windows_by_year(monkeypatch):
    """长窗口必须按年切窗：中间件 offset 上限实测在 10 万~12 万之间，
    单窗口一路翻到底会在 44 万行的全历史上报「参数校验失败, offset」。"""
    calls: list[dict] = []

    def fake_fetch(pro, api, interval=0.0, **kw):
        calls.append(kw)
        if kw["offset"] == 0:
            return _page(0, 7)
        return pl.DataFrame()

    monkeypatch.setattr("cnequity.external.tushare_fetch._fetch_with_retry", fake_fetch)

    out = tw._fetch_range_paged(
        object(), "stock_st", interval=0.0, start=date(2016, 1, 1), end=date(2026, 9, 17)
    )

    windows = sorted({(c["start_date"], c["end_date"]) for c in calls if c["offset"] == 0})
    assert out.height == 7 * len(windows)
    assert windows[0] == ("20160101", "20161231")
    assert windows[-1] == ("20260101", "20260917")
    assert len(windows) == 11  # 2016..2026
    # 每窗跨度不超过阈值（保证窗内行数远低于 offset 上限）
    for start_s, end_s in windows:
        span = date.fromisoformat(
            f"{end_s[:4]}-{end_s[4:6]}-{end_s[6:]}"
        ) - date.fromisoformat(f"{start_s[:4]}-{start_s[4:6]}-{start_s[6:]}")
        assert span.days <= tw._PAGE_WINDOW_DAYS


def test_fetch_range_paged_single_page_costs_one_confirm_call(monkeypatch):
    """日更窗口（百余行）只多一次确认空页的调用，不引入分页开销。"""
    calls: list[dict] = []

    def fake_fetch(pro, api, interval=0.0, **kw):
        calls.append(kw)
        if kw["offset"] == 0:
            return _page(0, 173)
        return pl.DataFrame()

    monkeypatch.setattr("cnequity.external.tushare_fetch._fetch_with_retry", fake_fetch)

    out = tw._fetch_range_paged(
        object(), "stock_st", interval=0.0, start=date(2024, 6, 28), end=date(2024, 6, 28)
    )

    assert out.height == 173
    assert [c["offset"] for c in calls] == [0, tw._TUSHARE_PAGE_LIMIT, tw._TUSHARE_PAGE_LIMIT]


# ─── 回填窗口 ─────────────────────────────────────────────────────────


def test_step_honors_backfill_window(cfg, monkeypatch):
    """`cne backfill stock_st --start/--end` 必须真拉历史窗口，而不是最近 30 天。"""
    seen: dict = {}

    def fake_paged(pro, api, *, interval, start, end, page_limit=tw._TUSHARE_PAGE_LIMIT):
        seen.update(api=api, start=start, end=end)
        return _page(0, 3)

    monkeypatch.setattr(tw, "_fetch_range_paged", fake_paged)
    monkeypatch.setattr(tw, "_get_pro", lambda config: object())

    cfg._backfill = True
    cfg._backfill_start = date(2016, 1, 1)
    cfg._backfill_end = date(2026, 9, 17)

    result = get_step("stock_st").fn(cfg, date(2026, 9, 17), "run1", {})

    assert seen == {"api": "stock_st", "start": date(2016, 1, 1), "end": date(2026, 9, 17)}
    assert result["rows_written"] == 3


def test_step_daily_window_is_incremental(cfg, monkeypatch):
    """非回填模式保持增量语义：watermark+1 → trade_date。"""
    seen: dict = {}

    def fake_paged(pro, api, *, interval, start, end, page_limit=tw._TUSHARE_PAGE_LIMIT):
        seen.update(start=start, end=end)
        return _page(0, 1)

    monkeypatch.setattr(tw, "_fetch_range_paged", fake_paged)
    monkeypatch.setattr(tw, "_get_pro", lambda config: object())

    from cnequity.storage.state import StateStore

    StateStore(cfg.meta_root).set_date("stock_st", date(2024, 6, 20))

    get_step("stock_st").fn(cfg, date(2024, 6, 28), "run1", {})

    assert seen == {"start": date(2024, 6, 21), "end": date(2024, 6, 28)}


def test_step_dedupes_pages_on_primary_key(cfg, monkeypatch):
    """分页边界重叠（源端往回返回时可能重复）必须按 PK 去重后才入 staging。"""

    def fake_paged(pro, api, *, interval, start, end, page_limit=tw._TUSHARE_PAGE_LIMIT):
        return pl.concat([_page(0, 3), _page(0, 3)])

    monkeypatch.setattr(tw, "_fetch_range_paged", fake_paged)
    monkeypatch.setattr(tw, "_get_pro", lambda config: object())

    result = get_step("stock_st").fn(cfg, date(2024, 6, 28), "run1", {})

    assert result["rows_written"] == 3
