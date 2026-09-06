"""挖掘评估吞吐改造回归测试（2026-09-06）：

1. ``_day_slices`` 身份缓存：同一索引重复调用命中缓存、不同索引互不污染、
   弱引用不延长面板生命周期、override 注入点仍然生效。
2. ``StockEvalService.eval_train`` 两段式海选：未过线走 lite（跳过诊断件、
   规则随响应返回、decile 保留）；profiles 无 lite 规则时回退全量。
"""

from __future__ import annotations

import gc

import numpy as np
import pandas as pd
import pytest

from alphaagent.factor import metrics as M
from alphaagent.factor.evaluation.profile import (
    default_evaluation_profiles,
    resolve_profiles,
)
from alphaagent.factor.mining.eval.schemas import EvalTrainRequest
from alphaagent.factor.mining.eval.service import StockEvalService
from alphaagent.factor.mining.eval.session import StockEvalSession


# ── _day_slices 缓存 ────────────────────────────────────────────────


def _idx(n_days=120, n_inst=40):
    return pd.MultiIndex.from_product(
        [pd.bdate_range("2021-01-01", periods=n_days), [f"S{i:03d}" for i in range(n_inst)]],
        names=["datetime", "instrument"],
    )


def test_day_slices_cache_identity_hit():
    idx = _idx()
    M._slices_cache.clear()
    first = M._day_slices(idx)
    second = M._day_slices(idx)
    assert first is not None
    assert first is second  # 同一索引命中同一结果对象（零拷贝复用）


def test_day_slices_cache_distinct_indexes():
    M._slices_cache.clear()
    a = M._day_slices(_idx())
    b = M._day_slices(_idx(n_days=60))
    assert a is not None and b is not None and a is not b
    assert len(M._slices_cache) == 2


def test_day_slices_cache_weakref_no_leak():
    idx = _idx()
    M._slices_cache.clear()
    M._day_slices(idx)
    assert len(M._slices_cache) == 1
    del idx
    gc.collect()
    # 弱引用死亡不延长原索引生命周期；死条目靠 LRU 上限与惰性 ref() 检查兜底
    fresh = _idx(n_days=30)
    M._day_slices(fresh)
    assert len(M._slices_cache) == 2
    stale_key = next(iter(M._slices_cache))
    dead_ref = M._slices_cache[stale_key][0]
    assert dead_ref() is None


def test_day_slices_result_readonly():
    idx = _idx()
    M._slices_cache.clear()
    bounds, day_vals = M._day_slices(idx)
    with pytest.raises(ValueError):
        bounds[0] = 123
    with pytest.raises(ValueError):
        day_vals[0] = day_vals[0]


def test_day_slices_override_still_wins():
    idx = _idx()
    M._slices_cache.clear()
    sentinel = ([1, 2, 3], ["x"])
    M._day_slices_override = lambda index, time_level: sentinel
    try:
        assert M._day_slices(idx) is sentinel
        assert M._slices_cache == {}  # override 路径不写缓存
    finally:
        M._day_slices_override = None


def test_day_slices_unsorted_returns_none_consistently():
    idx = _idx()
    shuffled = pd.Series(
        np.random.default_rng(0).normal(size=len(idx)), index=idx
    ).sample(frac=1.0, random_state=0)
    M._slices_cache.clear()
    assert M._day_slices(shuffled.index) is None
    assert M._day_slices(shuffled.index) is None  # 未排序不缓存，每次走重算
    assert M._slices_cache == {}


# ── eval_train 两段式 ───────────────────────────────────────────────


def _synth_panel(n_days=300, n_inst=60, seed=7):
    rng = np.random.default_rng(seed)
    idx = pd.MultiIndex.from_product(
        [pd.bdate_range("2021-01-01", periods=n_days), [f"S{i:03d}" for i in range(n_inst)]],
        names=["datetime", "instrument"],
    )
    ret = rng.normal(0, 0.02, len(idx))
    label = rng.normal(0, 0.01, len(idx))
    return pd.DataFrame(
        {"ret": ret, "label_1d_close_to_close": label, "close": 100.0, "float_cap": 1e6},
        index=idx,
    )


def _make_service_and_session(panel, profiles):
    service = StockEvalService(max_parallel_eval=1, profiles=profiles)
    from alphaagent.factor.mining.context import StockEvalContext

    real_ctx = StockEvalContext(
        panel_path="unused://", label_col="label_1d_close_to_close",
    )
    session = StockEvalSession(session_id="test-session", ctx=real_ctx, panel=panel)
    service.sessions._sessions[session.session_id] = session
    return service, session


def test_eval_train_two_stage_lite_path():
    panel = _synth_panel()
    service, session = _make_service_and_session(panel, resolve_profiles({}))
    req = EvalTrainRequest(
        session_id=session.session_id,
        multi_line_expr="f = CS_ZSCORE($ret)\nNEG(f)",
        factor_name="noise_factor",
        include_detail_tables=False,
        label_quantile_n=10,
    )
    resp = service.eval_train(req)
    assert resp.get("ok"), resp.get("error")
    assert resp.get("screen_stage") == "lite"
    skipped = resp.get("skipped_diagnostics") or []
    assert "mls_fmb" in skipped and "quantile_portfolio" in skipped
    rules = resp.get("screen_rules") or []
    assert rules and not all(r["passed"] for r in rules)
    # 预测对账数据源不因 lite 缺失（decile_mean_label 在 core 内）
    assert isinstance(resp["summary"].get("decile_mean_label"), list)


def test_eval_train_fallback_when_lite_missing():
    """profiles 无 train_screen_lite 规则（如旧调用方注入 defaults）→ 回退全量。"""
    panel = _synth_panel(n_days=40, n_inst=10, seed=1)
    service, session = _make_service_and_session(panel, default_evaluation_profiles())
    req = EvalTrainRequest(
        session_id=session.session_id,
        multi_line_expr="f = CS_ZSCORE($ret)\nNEG(f)",
        factor_name="legacy_profiles",
        include_detail_tables=False,
        label_quantile_n=10,
    )
    resp = service.eval_train(req)
    assert resp.get("ok"), resp.get("error")
    assert "screen_stage" not in resp
