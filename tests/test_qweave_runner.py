from __future__ import annotations

import pytest

from backend.qweave_runner import _default_code, execute, parse_code


def _local_market_data_available() -> bool:
    """qweave 研究跑在本地 CNE parquet 上；CI 无数据时跳过而非失败。"""
    try:
        from core.data import _pg_parquet_end

        return bool(_pg_parquet_end())
    except Exception:
        return False


_requires_market_data = pytest.mark.skipif(
    not _local_market_data_available(), reason="本地无 CNE 行情 parquet（CI 环境）"
)


def test_default_qweave_code_parses():
    result = parse_code(_default_code())
    assert result["ok"] is True
    assert result["factor_count"] == 30
    assert "KMID" in result["factors"]


def test_qweave_code_requires_factor_protocol():
    result = execute({"code": "x = 1"})
    assert result["ok"] is False
    assert "build_alphas" in result["error"] or "ALPHAS" in result["error"]


@_requires_market_data
def test_qweave_small_run_returns_json_safe_tables():
    result = execute({
        "code": _default_code(),
        "universe": "科技TMT",
        "start": "2025-01-01",
        "end": "2025-03-31",
        "alpha_set": "alpha158",
        "alpha_limit": 2,
        "horizons": [1],
        "quantiles": 3,
        "min_cs_count": 5,
        "cost_bps": 8,
    })
    assert result["ok"] is True
    assert result["factor_count"] == 2
    assert result["summary"]
    for row in result["summary"]:
        assert all(value is not None or key not in {"ic_mean", "ls_ir"}
                   for key, value in row.items())


@_requires_market_data
def test_qweave_score_can_flow_into_daily_backtest():
    result = execute({
        "code": _default_code(),
        "universe": "科技TMT",
        "start": "2025-01-01",
        "end": "2025-03-31",
        "alpha_set": "alpha158",
        "alpha_limit": 2,
        "horizons": [1],
        "quantiles": 3,
        "min_cs_count": 5,
        "run_backtest": True,
        "score_factor": "KMID",
        "top_n": 5,
        "capital": 100000,
        "freq": "weekly",
    })
    assert result["ok"] is True
    assert result["backtest"]["factor"] == "KMID"
    assert result["backtest"]["nav"]
