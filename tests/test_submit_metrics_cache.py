"""submit 指标缓存门禁。

submit 流水线内 train/val/全窗三次 ``compute_ingest_metrics`` 是 20~30s 级的
重复计算。``_cached_ingest_metrics`` 按 (因子值指纹, 窗口口径指纹) 做会话级
缓存：命中返回拷贝（调用方原地修改不污染缓存）、FIFO 上限防膨胀。
"""
from __future__ import annotations

import dataclasses

from alphaagent.factor.mining.submit import (
    _cached_ingest_metrics,
    _ingest_metrics_fingerprint,
)
from alphaagent.factor.types import IngestPolicy


def test_policy_fingerprint_discriminates_windows():
    base = IngestPolicy(
        train_start="2020-01-01", val_end="2024-12-31",
        label_col="label_10d_close_to_close",
    )
    # submit 的两个窗口口径即用这两个 replace 构造
    train_window = dataclasses.replace(base, val_end=base.train_start)
    val_window = dataclasses.replace(base, train_start="2023-01-01")

    assert _ingest_metrics_fingerprint(base) == _ingest_metrics_fingerprint(
        dataclasses.replace(base)
    )
    assert _ingest_metrics_fingerprint(base) != _ingest_metrics_fingerprint(train_window)
    assert _ingest_metrics_fingerprint(base) != _ingest_metrics_fingerprint(val_window)
    assert _ingest_metrics_fingerprint(base) != _ingest_metrics_fingerprint(
        dataclasses.replace(base, label_col="label_1d_open_to_open")
    )


def test_cached_ingest_metrics_hits_and_isolates():
    cache: dict = {}
    calls = {"n": 0}

    def compute() -> dict:
        calls["n"] += 1
        return {"ic": 0.03, "icir": 0.4}

    key = ("values_fp", "policy_fp")
    first = _cached_ingest_metrics(cache, key, compute)
    second = _cached_ingest_metrics(cache, key, compute)
    assert calls["n"] == 1, "同键第二次调用必须命中缓存"
    assert first == second == {"ic": 0.03, "icir": 0.4}

    # 命中返回拷贝：调用方原地修改（如补 val_long_excess）不得污染缓存
    second["val_long_excess"] = 0.02
    third = _cached_ingest_metrics(cache, key, compute)
    assert "val_long_excess" not in third
    assert calls["n"] == 1

    # 不同键不串
    _cached_ingest_metrics(cache, ("other_fp", "policy_fp"), compute)
    assert calls["n"] == 2


def test_cached_ingest_metrics_fifo_eviction(monkeypatch):
    import alphaagent.factor.mining.submit as sm

    monkeypatch.setattr(sm, "_INGEST_METRICS_CACHE_MAX", 3)
    cache: dict = {}
    for i in range(5):
        _cached_ingest_metrics(cache, (f"k{i}", "p"), lambda i=i: {"v": i})
    assert len(cache) == 3
    assert ("k0", "p") not in cache and ("k4", "p") in cache  # FIFO：最旧的 k0 被淘汰


def test_stage_one_turnover_gate():
    """stage_one 换手硬门：日单边换手 >0.5 的候选必须被拒（回溯重筛的规则来源）。"""
    from alphaagent.factor.mining.delivery_checker import DeliveryChecker

    from alphaagent.factor.mining.delivery_criteria import DeliveryCriteria
    checker = DeliveryChecker(DeliveryCriteria.defaults())
    base = {"ic": 0.03, "icir": 0.4, "factor_coverage": 0.95, "cs_pearson_autocorr": 0.8}
    ok = checker.stage_one_stats({**base, "quantile_portfolio": {"avg_daily_side_turnover": 0.35}})
    bad = checker.stage_one_stats({**base, "quantile_portfolio": {"avg_daily_side_turnover": 1.2}})
    no_data = checker.stage_one_stats(dict(base))  # 无换手数据不误伤

    assert ok.passed
    assert not bad.passed
    assert any("avg_daily_side_turnover" in r for r in bad.fail_reasons)
    assert no_data.passed or not any("avg_daily_side_turnover" in r for r in no_data.fail_reasons)
