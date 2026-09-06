"""panel 磁盘缓存列族哨兵校验。

背景：建缓存当天某个辅助插件加载失败（如 CNE 同步占用 parquet 文件锁）会把
缺列面板固化，之后每次命中都返回残缺面板，下游 funda_*/dt_* 因子集体报
"不可用字段"。命中与写入两侧都必须拒绝缺哨兵列的面板。
"""
from __future__ import annotations

import pandas as pd
import pytest

from alphaagent.data.adapters import cnequity


def _make_panel(*, complete: bool, with_funda: bool = True) -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["000001.SZ", "000002.SZ", "000003.SZ"]],
        names=["datetime", "instrument"],
    )
    cols = {"close": 1.0, "holder_count_chg_pct": 0.1, "dt_net_buy_90d": 2.0}
    if with_funda:
        base = {
            "funda_total_assets": 1.0,
            "funda_net_profit": 1.0,
            "funda_ocf": 1.0,
            "funda_netprofit_yoy": 1.0,
        }
        if not complete:
            base.pop("funda_ocf")
            base.pop("funda_netprofit_yoy")
        cols.update(base)
    n = len(idx)
    return pd.DataFrame({c: [v] * n for c, v in cols.items()}, index=idx)


def test_missing_funda_sentinels_complete_panel() -> None:
    assert cnequity._missing_funda_sentinels(_make_panel(complete=True)) == frozenset()


def test_missing_funda_sentinels_deficient_panel() -> None:
    missing = cnequity._missing_funda_sentinels(_make_panel(complete=False))
    assert missing == frozenset({"funda_ocf", "funda_netprofit_yoy"})


def test_missing_funda_sentinels_no_funda_family() -> None:
    # include_fundamentals=False 的面板无任何 funda_ 列：不算残缺
    assert cnequity._missing_funda_sentinels(_make_panel(complete=False, with_funda=False)) == frozenset()


@pytest.fixture()
def cache_root(tmp_path, monkeypatch):
    root = tmp_path / "cache"
    root.mkdir()
    monkeypatch.setattr(cnequity, "_CACHE_ROOT", root)
    return root


def _write_cache(root, name: str, panel: pd.DataFrame) -> None:
    panel.reset_index().to_parquet(root / name, index=False)


def test_find_skips_deficient_cache(cache_root) -> None:
    _write_cache(cache_root, "panel_v3_deficient.parquet", _make_panel(complete=False))
    assert cnequity._find_cached_panel("2024-01-02", "2024-01-03", include_fundamentals=True) is None


def test_find_hits_complete_cache(cache_root) -> None:
    _write_cache(cache_root, "panel_v3_complete.parquet", _make_panel(complete=True))
    hit = cnequity._find_cached_panel("2024-01-02", "2024-01-03", include_fundamentals=True)
    assert hit is not None and len(hit) == 6


def test_find_no_funda_cache_never_serves_funda_request(cache_root) -> None:
    _write_cache(cache_root, "panel_v3_nofunda.parquet", _make_panel(complete=False, with_funda=False))
    assert cnequity._find_cached_panel("2024-01-02", "2024-01-03", include_fundamentals=True) is None
