"""window_config 动态解析与窗口函数单测。

覆盖：
- 静态边界常量不变性
- resolve_test_end 优先级链（external → curated → watermark → today）
- 防御：解析结果不低于 TEST_START
- lru_cache 进程内缓存行为
- test_window / coverage_window / window_defaults 一致性
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest

import alphaagent.factor.window_config as wc


# ── 静态边界 ──────────────────────────────────────────────────────────────


class TestStaticBounds:
    def test_train_bounds(self) -> None:
        assert wc.DEFAULT_TRAIN_START == "2020-01-01"
        assert wc.DEFAULT_TRAIN_END == "2022-12-31"

    def test_val_bounds(self) -> None:
        assert wc.DEFAULT_VAL_START == "2023-01-01"
        assert wc.DEFAULT_VAL_END == "2024-12-31"

    def test_test_start(self) -> None:
        assert wc.DEFAULT_TEST_START == "2025-01-01"

    def test_test_end_default_is_none(self) -> None:
        assert wc.DEFAULT_TEST_END is None


# ── resolve_test_end 优先级链 ──────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _clear_cache():
    """每个测试清除 lru_cache，避免互相干扰。"""
    wc.resolve_test_end.cache_clear()
    yield
    wc.resolve_test_end.cache_clear()


class TestResolveTestEnd:
    def test_external_priority(self) -> None:
        """stock_daily_wide 优先于 daily_bars。"""
        fake_cfg = object()
        with (
            patch.object(wc, "_load_cne_config", return_value=fake_cfg),
            patch.object(wc, "_external_coverage_end", return_value="2026-08-28") as ext_mock,
            patch.object(wc, "_curated_coverage_end", return_value="2026-08-27") as cur_mock,
        ):
            result = wc.resolve_test_end()
        assert result == "2026-08-28"
        ext_mock.assert_called_once_with(fake_cfg, "stock_daily_wide")
        cur_mock.assert_not_called()

    def test_curated_fallback(self) -> None:
        """external 返回 None 时回落到 curated。"""
        fake_cfg = object()
        with (
            patch.object(wc, "_load_cne_config", return_value=fake_cfg),
            patch.object(wc, "_external_coverage_end", return_value=None),
            patch.object(wc, "_curated_coverage_end", return_value="2026-08-28") as cur_mock,
        ):
            result = wc.resolve_test_end()
        assert result == "2026-08-28"
        cur_mock.assert_called_once_with(fake_cfg, "daily_bars")

    def test_watermark_fallback(self) -> None:
        """external + curated 都 None → watermark。"""
        fake_cfg = object()
        with (
            patch.object(wc, "_load_cne_config", return_value=fake_cfg),
            patch.object(wc, "_external_coverage_end", return_value=None),
            patch.object(wc, "_curated_coverage_end", return_value=None),
            patch.object(wc, "_state_watermark", return_value="2026-08-25"),
        ):
            result = wc.resolve_test_end()
        assert result == "2026-08-25"

    def test_today_fallback_no_cfg(self) -> None:
        """CNE 配置加载失败 → 兜底今天。"""
        today = date.today().isoformat()
        with patch.object(wc, "_load_cne_config", return_value=None):
            result = wc.resolve_test_end()
        assert result == today

    def test_today_fallback_all_none(self) -> None:
        """配置加载成功但所有数据源都 None → 兜底今天。"""
        today = date.today().isoformat()
        fake_cfg = object()
        with (
            patch.object(wc, "_load_cne_config", return_value=fake_cfg),
            patch.object(wc, "_external_coverage_end", return_value=None),
            patch.object(wc, "_curated_coverage_end", return_value=None),
            patch.object(wc, "_state_watermark", return_value=None),
        ):
            result = wc.resolve_test_end()
        assert result == today

    def test_floor_to_test_start(self) -> None:
        """解析结果早于 TEST_START → 回退 TEST_START。"""
        fake_cfg = object()
        with (
            patch.object(wc, "_load_cne_config", return_value=fake_cfg),
            patch.object(wc, "_external_coverage_end", return_value="2024-06-01"),
        ):
            result = wc.resolve_test_end()
        assert result == wc.DEFAULT_TEST_START

    def test_exact_test_start_not_floored(self) -> None:
        """解析结果恰好等于 TEST_START → 原样返回。"""
        fake_cfg = object()
        with (
            patch.object(wc, "_load_cne_config", return_value=fake_cfg),
            patch.object(wc, "_external_coverage_end", return_value="2025-01-01"),
        ):
            result = wc.resolve_test_end()
        assert result == "2025-01-01"


# ── lru_cache 行为 ──────────────────────────────────────────────────────────


class TestCache:
    def test_cached_on_second_call(self) -> None:
        """第二次调用不重新解析（命中缓存）。"""
        fake_cfg = object()
        with (
            patch.object(wc, "_load_cne_config", return_value=fake_cfg),
            patch.object(wc, "_external_coverage_end", return_value="2026-08-28") as mock,
        ):
            wc.resolve_test_end()
            wc.resolve_test_end()
        # 外部解析只调用一次
        assert mock.call_count == 1

    def test_cache_clear_refreshes(self) -> None:
        """cache_clear 后重新解析。"""
        fake_cfg = object()
        with (
            patch.object(wc, "_load_cne_config", return_value=fake_cfg),
            patch.object(wc, "_external_coverage_end", side_effect=["2026-08-28", "2026-08-29"]),
        ):
            first = wc.resolve_test_end()
            wc.resolve_test_end.cache_clear()
            second = wc.resolve_test_end()
        assert first == "2026-08-28"
        assert second == "2026-08-29"


# ── 窗口函数 ────────────────────────────────────────────────────────────────


class TestWindowFunctions:
    def test_test_window(self) -> None:
        with patch.object(wc, "resolve_test_end", return_value="2026-08-28"):
            ts, te = wc.test_window()
        assert ts == "2025-01-01"
        assert te == "2026-08-28"

    def test_mining_window(self) -> None:
        ts, te, vs, ve = wc.mining_window()
        assert (ts, te, vs, ve) == (
            "2020-01-01",
            "2022-12-31",
            "2023-01-01",
            "2024-12-31",
        )

    def test_coverage_window(self) -> None:
        with patch.object(wc, "resolve_test_end", return_value="2026-08-28"):
            start, end = wc.coverage_window()
        assert start == "2020-01-01"
        assert end == "2026-08-28"

    def test_window_defaults_keys(self) -> None:
        with patch.object(wc, "resolve_test_end", return_value="2026-08-28"):
            d = wc.window_defaults()
        assert set(d.keys()) == {
            "train_start", "train_end", "val_start", "val_end",
            "test_start", "test_end",
        }
        assert d["test_end"] == "2026-08-28"

    def test_window_defaults_consistent_with_individual(self) -> None:
        """window_defaults 与单独窗口函数结果一致。"""
        with patch.object(wc, "resolve_test_end", return_value="2026-08-28"):
            d = wc.window_defaults()
            ts, te = wc.test_window()
            ms, me, vs, ve = wc.mining_window()
        assert d["test_start"] == ts
        assert d["test_end"] == te
        assert d["train_start"] == ms
        assert d["train_end"] == me
        assert d["val_start"] == vs
        assert d["val_end"] == ve