# -*- coding: utf-8 -*-
"""pred 下发价格池口径的回归测试。

核心断言（2026-11-05 事故回归）：默认"低价池"必须**保留低价、剔除高价**。
此前 ``b_ops_filter.py`` 把比较符写反，产出高价池，模拟盘 1549 只里 10 元以下仅 15 只。
"""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.push_pred_scores import (
    DEFAULT_PRICE_MAX,
    apply_price_pool,
    build_pool,
    mainboard_mask,
    normalize_codes,
)


def _scores(closes: list[float], codes: list[str] | None = None,
            dates: list[str] | None = None) -> pd.DataFrame:
    n = len(closes)
    return pd.DataFrame({
        "date": pd.to_datetime(dates or ["2026-10-27"] * n),
        "code": codes or [f"60000{i}" for i in range(n)],
        "score": [1.0] * n,
        "close": closes,
    })


# ── 方向回归：默认低价池 ──────────────────────────────────────────────
def test_default_pool_keeps_low_price_drops_high_price():
    """默认 price_max=10 必须保留 5 元、剔除 15 元（否则就是事故复发）。"""
    df = _scores([5.0, 10.0, 10.01, 15.0])
    out, stats = apply_price_pool(df, price_max=DEFAULT_PRICE_MAX)
    kept = out.dropna(subset=["score"])["close"].tolist()
    assert kept == [5.0, 10.0], f"低价池保留了 {kept}，方向可能再次写反"
    assert stats["dropped_over_max"] == 2


def test_default_pool_drops_all_below_equivalently_when_inverted():
    """对照：若把边界写成 close<=10 剔除，则 5 元会被剔——断言不会发生。"""
    df = _scores([5.0, 20.0])
    out, _ = apply_price_pool(df, price_max=DEFAULT_PRICE_MAX)
    kept = out.dropna(subset=["score"])["close"].tolist()
    assert 5.0 in kept and 20.0 not in kept


# ── 上/下限语义 ──────────────────────────────────────────────────────
def test_price_max_inclusive_boundary():
    df = _scores([9.99, 10.0, 10.0001])
    out, _ = apply_price_pool(df, price_max=10.0)
    kept = out.dropna(subset=["score"])["close"].tolist()
    assert kept == [9.99, 10.0]


def test_price_min_drops_below():
    df = _scores([5.0, 10.0, 15.0])
    out, _ = apply_price_pool(df, price_max=None, price_min=10.0)
    kept = out.dropna(subset=["score"])["close"].tolist()
    assert kept == [10.0, 15.0]


def test_price_band_both_bounds():
    df = _scores([5.0, 10.0, 15.0, 20.0])
    out, stats = apply_price_pool(df, price_max=15.0, price_min=10.0)
    kept = out.dropna(subset=["score"])["close"].tolist()
    assert kept == [10.0, 15.0]
    assert stats["dropped_below_min"] == 1
    assert stats["dropped_over_max"] == 1


def test_no_bounds_keeps_everything_scored():
    df = _scores([1.0, 50.0])
    out, stats = apply_price_pool(df, price_max=None, price_min=None)
    assert stats["kept_scored"] == 2


def test_missing_close_is_dropped():
    df = _scores([5.0, float("nan"), 15.0])
    out, stats = apply_price_pool(df, price_max=10.0)
    kept = out.dropna(subset=["score"])["close"].tolist()
    assert kept == [5.0]
    assert stats["dropped_price_missing"] == 1


def test_requires_close_and_score_columns():
    with pytest.raises(ValueError):
        apply_price_pool(pd.DataFrame({"close": [1.0]}), price_max=10.0)


# ── 代码与板块 ──────────────────────────────────────────────────────
def test_normalize_codes_from_suffix_form():
    s = pd.Series(["000001.SZ", "601398.SH", "300750.SZ"])
    assert normalize_codes(s).tolist() == ["000001", "601398", "300750"]


def test_mainboard_mask_excludes_star_and_chinext():
    s = pd.Series(["600036", "601398", "000001", "002594", "003816",
                   "688981", "300750", "301236", "830799"])
    mask = mainboard_mask(s).tolist()
    assert mask == [True, True, True, True, True, False, False, False, False]


# ── build_pool 端到端 ───────────────────────────────────────────────
def test_build_pool_end_to_end_main_and_lowprice():
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-27"] * 5),
        "code": ["600000", "601398", "688981", "000001", "300750"],
        "score": [1.0, 1.0, 1.0, 1.0, 1.0],
    })
    close_long = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-27"] * 5),
        "code": ["600000", "601398", "688981", "000001", "300750"],
        "close": [8.0, 20.0, 5.0, 9.5, 4.0],
    })
    out, stats = build_pool(raw, close_long, board="main",
                            price_max=DEFAULT_PRICE_MAX, price_min=None)
    # 600000(8元,主板) 与 000001(9.5元,主板) 保留；
    # 601398 价格超上限；688981/300750 非主板。
    assert out["code"].tolist() == ["000001", "600000"]
    assert stats["final_rows"] == 2
    assert stats["after_board"] == 3


def test_build_pool_all_board_keeps_star_market():
    raw = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-27"]),
        "code": ["688981"],
        "score": [1.0],
    })
    close_long = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-27"]),
        "code": ["688981"],
        "close": [5.0],
    })
    out, _ = build_pool(raw, close_long, board="all", price_max=10.0)
    assert out["code"].tolist() == ["688981"]


# ── 生产文件保护 ────────────────────────────────────────────────────
def test_main_refuses_to_overwrite_production_without_flag(tmp_path, monkeypatch):
    import scripts.push_pred_scores as mod

    raw_path = tmp_path / "raw.parquet"
    pd.DataFrame({
        "date": pd.to_datetime(["2026-10-27"]),
        "code": ["600000"],
        "score": [1.0],
    }).to_parquet(raw_path, index=False)

    prod = tmp_path / "pred_demo.parquet"
    monkeypatch.setattr(mod, "PROD_PRED", prod)
    rc = mod.main([str(raw_path), "--out", str(prod)])
    assert rc == 2
    assert not prod.exists()


def test_main_writes_default_lowprice_output(tmp_path, monkeypatch):
    import scripts.push_pred_scores as mod

    raw_path = tmp_path / "raw.parquet"
    pd.DataFrame({
        "date": pd.to_datetime(["2026-10-27"]),
        "code": ["600000"],
        "score": [1.0],
    }).to_parquet(raw_path, index=False)

    out_path = tmp_path / "lowprice.parquet"
    close_long = pd.DataFrame({
        "date": pd.to_datetime(["2026-10-27"]),
        "code": ["600000"],
        "close": [8.0],
    })
    monkeypatch.setattr(mod, "load_close_long", lambda s, e: close_long)
    rc = mod.main([str(raw_path), "--out", str(out_path)])
    assert rc == 0
    assert out_path.exists()
    assert pd.read_parquet(out_path)["code"].tolist() == ["600000"]
