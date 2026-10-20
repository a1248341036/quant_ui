# -*- coding: utf-8 -*-
"""ml_weekly_ops 单测：stacking run 选取、Top-N 名单、风格漂移、台账、暴露画像（合成面板）。"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def _load_mwo():
    spec = importlib.util.spec_from_file_location("ml_weekly_ops", SCRIPTS / "ml_weekly_ops.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_scores(tmp: Path, dates, codes) -> Path:
    rng = np.random.default_rng(7)
    rows = []
    for d in dates:
        for c in codes:
            rows.append({"date": d, "code": c,
                         "score": float(rng.normal()) if rng.random() > 0.1 else np.nan})
    p = tmp / "scores.parquet"
    pd.DataFrame(rows).to_parquet(p, index=False)
    return p


def test_latest_stacking_run(tmp_path: Path):
    mwo = _load_mwo()
    assert mwo.latest_stacking_run(tmp_path) is None
    (tmp_path / "20260101_000000").mkdir()
    (tmp_path / "20260102_000000").mkdir()
    (tmp_path / "20260102_000000" / "scores.parquet").write_bytes(b"x")
    rid, scores = mwo.latest_stacking_run(tmp_path)
    assert rid == "20260102_000000" and scores.name == "scores.parquet"


def test_build_list_latest_date_topn(tmp_path: Path):
    mwo = _load_mwo()
    dates = [pd.Timestamp("2026-01-05"), pd.Timestamp("2026-01-09")]
    p = _make_scores(tmp_path, dates, ["000001", "000002", "000003"])
    top, latest = mwo.build_list(p, top_n=2)
    assert latest == "2026-01-09"          # 只取最新截日
    assert len(top) == 2 and top["score"].is_monotonic_decreasing
    assert top["score"].notna().all()


def test_write_list_files(tmp_path: Path):
    mwo = _load_mwo()
    top = pd.DataFrame({"code": ["000001", "000002"], "score": [1.2, 0.8]})
    md = mwo.write_list(top, "2026-01-09", "r1", out_dir=tmp_path)
    assert md.name == "ml_weekly_list_r1.md" and "000001" in md.read_text(encoding="utf-8")
    assert (tmp_path / "ml_weekly_list_r1.csv").is_file()


def test_compute_drift_semantics():
    mwo = _load_mwo()
    assert mwo.compute_drift({"a": 0.1}, None) == {}          # 首跑无对照
    assert mwo.compute_drift({"a": 0.1, "b": 0.0}, {"a": 0.04, "c": 0.5}) == {"a": 0.06}


def test_exposure_ledger_roundtrip(tmp_path: Path):
    mwo = _load_mwo()
    ledger = tmp_path / "exposure_ledger.jsonl"
    mwo.append_exposure_ledger({"run_id": "r1", "style_exposure": {"am20": -0.09}}, path=ledger)
    rows = mwo.load_exposure_ledger(ledger)
    assert len(rows) == 1 and rows[0]["style_exposure"]["am20"] == -0.09


def test_exposure_snapshot_synthetic_panel(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    mwo = _load_mwo()
    rng = np.random.default_rng(11)
    n_days, n_codes = 60, 60
    dates = pd.bdate_range("2026-01-05", periods=n_days)
    codes = [f"{600000 + i}" for i in range(n_codes)]
    idx = pd.MultiIndex.from_product([dates, codes], names=["datetime", "code"])
    base = 10.0 * np.cumprod(1 + rng.normal(0, 0.01, (n_days, n_codes)), axis=0).ravel()
    # amount = 每股持久水平 + 微噪声 → 20 日滚动均值 ≈ 水平本身，score 与风格列强相关
    code_amt = rng.uniform(5e7, 2e8, n_codes)
    am20 = np.tile(code_amt, n_days) * (1 + rng.normal(0, 0.02, len(idx)))
    turn = rng.uniform(0.5, 5.0, len(idx))
    close_series = pd.Series(base, index=idx)
    panel = pd.DataFrame({"close": close_series, "amount": am20,
                          "turnover_rate": turn}, index=idx)
    monkeypatch.setattr(mwo, "load_panel_from_cne", lambda **kw: panel)

    # score 与 amount（流动性代理）强相关，确保 liquidity 暴露系数显著
    score_df = pd.DataFrame({
        "date": np.repeat(dates, n_codes),
        "code": np.tile(codes, n_days),
        "score": (am20 * rng.normal(1, 0.05, len(idx))
                  + rng.normal(0, 1e-6, len(idx))),
    })
    p = tmp_path / "scores.parquet"
    score_df.to_parquet(p, index=False)

    snap = mwo.exposure_snapshot(p, window=20)
    # build_exposures 输出语义风格名（liquidity=am20 代理，turnover，volatility，momentum）
    assert set(snap["style_exposure"]) == {"liquidity", "turnover", "volatility", "momentum"}
    coefs = snap["style_exposure"]
    # score 与 amount 同向 → liquidity 暴露为正且显著主导（内部做时序标准化，不卡绝对阈值）
    assert coefs["liquidity"] > 0.2
    assert coefs["liquidity"] > 2 * max(abs(v) for k, v in coefs.items() if k != "liquidity")
    assert 0.0 <= snap["r2_style"] <= 1.0
