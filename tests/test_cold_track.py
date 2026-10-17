# -*- coding: utf-8 -*-
"""冷门赛道双轨评估单测（spec: alphaagent_mining_anti_convergence_spec.md 验收 §7.1）。

覆盖：配置三件套（默认/覆盖/越界）、客观赛道判定（含套壳反例）、信号根剥壳、
crowd 细族->粗族映射、骨架清单（top-IC/自噬剔除）、残差化 train-before-val 冻结、
fail_detail 三标签、dispatch 短路三分支、cold_dead verdict 传导、M3 父本屏蔽、
cold_dead retryable 负池判定。全部本地轻量，不触真实引擎/网络。
"""
from __future__ import annotations

import json
import types

import numpy as np
import pandas as pd
import pytest

from alphaagent.factor.mining.memory.cold_track import (
    COLD_TRACK_APPROVAL_PROVENANCE,
    build_crowd_skeletons,
    classify_track,
    crowd_coarse_of,
    is_crowd_family,
    signal_root_op,
)
from alphaagent.factor.mining.eval.residualize import (
    _slice_bounds,
    apply_residual,
    evaluate_cold_track,
    pool_residual_betas,
)
from alphaagent.factor.mining.research_spec import normalize_research_spec


# ── 共享合成面板 ──

@pytest.fixture(scope="module")
def ct_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.date_range("2021-01-01", periods=8, freq="D")
    instruments = ["S%02d" % i for i in range(12)]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    n = len(idx)
    return pd.DataFrame(
        {"label_1d_open_to_open": rng.normal(size=n)},
        index=idx,
    )


def _ctx():
    return types.SimpleNamespace(
        train_start="2021-01-01", train_end="2021-01-04",
        val_start="2021-01-05", val_end="2021-01-08",
        label_col="label_1d_open_to_open",
    )


# ── 1. 配置三件套 ──

class TestResearchSpec:
    def test_cold_track_defaults(self):
        spec = normalize_research_spec({})
        cold = spec["cold_track"]
        assert cold["enabled"] is False
        assert cold["min_val_resid_ic"] == 0.02
        assert cold["max_crowd_proxy_corr"] == 0.3
        assert cold["max_ctrl_families"] == 8
        assert cold["cold_eval_budget_ratio"] == 0.20
        assert cold["op_cool_after_dead"] == 3
        assert cold["op_cool_skips"] == 4

    def test_cold_track_override(self):
        spec = normalize_research_spec({
            "cold_track": {
                "enabled": True,
                "min_val_resid_ic": 0.01,
                "max_ctrl_families": 10,
            }
        })
        cold = spec["cold_track"]
        assert cold["enabled"] is True
        assert cold["min_val_resid_ic"] == 0.01
        assert cold["max_ctrl_families"] == 10

    @pytest.mark.parametrize("key,bad", [
        ("min_val_resid_ic", 0.1),
        ("max_crowd_proxy_corr", 2.0),
        ("max_ctrl_families", 99),
        ("cold_eval_budget_ratio", 0.99),
        ("op_cool_after_dead", 0),
    ])
    def test_cold_track_bounds_rejected(self, key, bad):
        with pytest.raises(ValueError):
            normalize_research_spec({"cold_track": {key: bad}})

    def test_cold_track_enabled_must_be_bool(self):
        with pytest.raises(ValueError):
            normalize_research_spec({"cold_track": {"enabled": "yes"}})


# ── 2. 客观赛道判定 + 骨架辅助 ──

class TestTrackClassification:
    def test_cold_true_positive(self):
        track, new_fam, cold_root = classify_track(
            "RANK(TS_PERMUTATION_ENTROPY($volume,20))", "排列熵"
        )
        assert track == "cold"
        assert new_fam is True and cold_root is True

    def test_chip_entropy_cold(self):
        track, new_fam, cold_root = classify_track(
            "CS_NEUTRALIZE(CHIP_ENTROPY($close,20), $industry)", "筹码熵"
        )
        assert track == "cold" and new_fam is True and cold_root is True

    def test_shell_crowd_share_not_cold(self):
        # 套壳反例：顶层冷门算子但机制归回拥挤族 -> crowd + cold_root=True（供 M3 advisory）
        track, new_fam, cold_root = classify_track(
            "RANK(CROWD_SHARE($volume,10))", "拥挤度"
        )
        assert track == "crowd"
        assert new_fam is False and cold_root is True

    def test_new_family_but_non_cold_root(self):
        track, new_fam, cold_root = classify_track(
            "RANK(TS_MA(TS_RANK($close,20),5))", "新族普通"
        )
        assert track == "crowd"
        assert new_fam is True and cold_root is False


class TestSignalRoot:
    def test_peels_normalize_shell(self):
        assert signal_root_op(
            "CS_NEUTRALIZE(TS_PERMUTATION_ENTROPY($volume,20), $industry)"
        ) == "TS_PERMUTATION_ENTROPY"

    def test_peels_smoothing_stack(self):
        assert signal_root_op("TS_MEAN(EMA(TS_TREND_RANK($high,20),5),5)") == "TS_TREND_RANK"

    def test_bare_column_no_root(self):
        # 信号根为裸列引用 -> 空串（不命中条件 (2)）
        assert signal_root_op("RANK($close)") == ""


class TestCrowdCoarse:
    def test_mapping(self):
        assert crowd_coarse_of("volume") == "volume_liquidity"
        assert crowd_coarse_of("volatility") == "volatility"
        assert crowd_coarse_of("reversal") == "momentum_reversal"
        assert crowd_coarse_of("chip") is None
        assert crowd_coarse_of("fundamental") is None

    def test_is_crowd_family(self):
        assert is_crowd_family("volume") is True
        assert is_crowd_family("fundamental") is False


class TestSkeletons:
    @staticmethod
    def _fake_registry():
        return {
            # volatility 族两条：取 train_ic 最高的 v2
            "v1": {"expr": "RANK(TS_STD($close,20))", "name": "低波A", "provenance": "mine",
                   "ingest_metrics": {"ic": 0.04}},
            "v2": {"expr": "RANK(TS_DEMEAN(TS_STD($close,10), $industry))", "name": "低波B",
                   "provenance": "mine", "ingest_metrics": {"ic": 0.06}},
            # reversal 族一条（拥挤）
            "r1": {"expr": "RANK(NEG(TS_MEAN($turnover,10)))", "name": "换手反转",
                   "provenance": "mine", "ingest_metrics": {"ic": 0.05}},
            # 冷门自噬：历史冷门赛道通过者，即使高 IC 也不能进骨架
            "self1": {"expr": "RANK(TS_STD($high,20))", "name": "低波自噬",
                      "provenance": COLD_TRACK_APPROVAL_PROVENANCE,
                      "ingest_metrics": {"ic": 0.09}},
        }

    def test_top_ic_and_self_heal(self, monkeypatch, tmp_path):
        import alphaagent.factor.mining.infra.registry_io as reg_io
        monkeypatch.setattr(reg_io, "load_mining_registry", lambda p: self._fake_registry())
        snap = build_crowd_skeletons(tmp_path / "prod.json", tmp_path / "cand.json",
                                     repo_root=tmp_path)
        ske = snap["skeletons"]
        fams = {f for f in ske}
        assert "volatility" in fams
        assert "momentum_reversal" in fams
        # 自噬条目被剔除
        assert "RANK(TS_STD($high,20))" not in [m["expr"] for m in ske.values()]
        # volatility 族保留 IC 最高的 v2
        assert any("TS_DEMEAN(TS_STD($close,10)" in m["expr"] for m in ske.values())
        assert not any(m["expr"] == "RANK(TS_STD($close,20))" for m in ske.values())
        assert snap["signature"] and snap["crowd_families"]

    def test_max_ctrl_families_cap(self, monkeypatch, tmp_path):
        import alphaagent.factor.mining.infra.registry_io as reg_io
        monkeypatch.setattr(reg_io, "load_mining_registry", lambda p: self._fake_registry())
        snap = build_crowd_skeletons(tmp_path / "prod.json", tmp_path / "cand.json",
                                     repo_root=tmp_path, max_ctrl_families=1)
        assert len(snap["skeletons"]) == 1


# ── 3. 残差化纪律 ──

class TestResidualize:
    def test_pool_beta_matches_train_ols(self, ct_panel):
        rng = np.random.default_rng(11)
        n = len(ct_panel)
        y = rng.normal(size=n)
        x = rng.normal(size=n)
        dt = pd.to_datetime(ct_panel.index.get_level_values("datetime"))
        bounds = _slice_bounds(dt)  # 8 天 -> 8 段（每天 12 行）
        train_bounds = bounds[:4]  # 仅 train 4 天
        beta, n_used = pool_residual_betas(
            y, x.reshape(-1, 1), train_bounds, min_rows=5, min_ctrl_cover=0.5
        )
        # spec：逐截面 OLS 后按有效日等权平均（非整段合并 OLS）
        per_day = []
        for st, en in train_bounds:
            yy = y[st:en]; xx = x[st:en]
            design = np.column_stack([np.ones(len(yy)), xx])
            b, *_ = np.linalg.lstsq(design, yy, rcond=None)
            per_day.append(b)
        beta_mean = np.mean(np.vstack(per_day), axis=0)
        assert n_used == 4  # 恰好 4 个 train 截面
        np.testing.assert_allclose(beta, beta_mean, atol=1e-8)

    def test_residual_uses_frozen_beta(self, ct_panel):
        rng = np.random.default_rng(5)
        y = rng.normal(size=len(ct_panel)).astype(float)
        x = rng.normal(size=len(ct_panel)).astype(float)
        beta = np.array([0.3, 0.7])
        resid = apply_residual(y, x.reshape(-1, 1), beta)
        good = np.isfinite(resid)
        expected = y - (0.3 + 0.7 * x)
        np.testing.assert_allclose(resid[good], expected[good], atol=1e-10)

    def test_fail_detail_data_gap(self, ct_panel):
        rng = np.random.default_rng(3)
        res = evaluate_cold_track(
            rng.normal(size=len(ct_panel)).astype(float), ct_panel, {}, _ctx(), min_pairs=3
        )
        assert res["fail_detail"] == "data_gap"

    def test_fail_detail_underpowered_sign_flip(self):
        rng = np.random.default_rng(21)
        dates = pd.date_range("2021-01-01", periods=8, freq="D")
        instruments = ["S%02d" % i for i in range(12)]
        idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        n = len(idx)
        panel = pd.DataFrame({"label_1d_open_to_open": rng.normal(size=n)}, index=idx)
        # 强确定性信号独立于骨架，val 段符号翻转 -> train/val 残差 IC 异号
        signal = 5.0 * rng.normal(size=n)
        dates = pd.to_datetime(panel.index.get_level_values("datetime"))
        train_b = dates < pd.Timestamp("2021-01-05")
        y = np.where(train_b, signal, -signal).astype(float)
        ctrl = {"volatility": rng.normal(size=n).astype(float),
                "volume_liquidity": rng.normal(size=n).astype(float)}
        res = evaluate_cold_track(
            y.astype(float), panel, ctrl, _ctx(),
            policy={"min_val_resid_ic": 0.02, "max_crowd_proxy_corr": 0.3}, min_pairs=3
        )
        assert res["fail_detail"] == "underpowered"

    def test_train_before_val_fit_days_bounded(self, ct_panel):
        rng = np.random.default_rng(9)
        n = len(ct_panel)
        common = rng.normal(size=n)
        ctrl = {"volatility": (0.7 * common + 0.3 * rng.normal(size=n)).astype(float),
                "volume_liquidity": rng.normal(size=n).astype(float)}
        cand = (common + 0.5 * rng.normal(size=n)).astype(float)
        res = evaluate_cold_track(cand, ct_panel, ctrl, _ctx(),
                                  policy={"min_val_resid_ic": 0.02, "max_crowd_proxy_corr": 0.3},
                                  min_pairs=3)
        assert res["n_fit_days"] <= 4
        assert res["train_n_days"] <= 4
        assert res["val_n_days"] <= 4

    def test_proxy_masked_on_blind_segment(self, monkeypatch):
        # train/val/test 三段面板；monkeypatch crowd_proxy_corr 记录实际收到的
        # factor/ctrl 序列日期，断言 test（盲测）段日期未进入 proxy 相似度计算。
        import alphaagent.factor.mining.eval.residualize as rmod

        rng = np.random.default_rng(11)
        dates = pd.date_range("2021-01-01", periods=12, freq="D")
        instruments = ["S%02d" % i for i in range(12)]
        idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
        n = len(idx)
        panel = pd.DataFrame({"label_1d_open_to_open": rng.normal(size=n)}, index=idx)
        seen = {}

        def spy_proxy(factor, ctrl_series, **kw):
            seen["dates"] = set(pd.to_datetime(factor.index.get_level_values("datetime")))
            return 0.05, "volatility"

        monkeypatch.setattr(rmod, "crowd_proxy_corr", spy_proxy)
        ctrl = {
            "volatility": rng.normal(size=n).astype(float),
            "volume_liquidity": rng.normal(size=n).astype(float),
        }
        cand = rng.normal(size=n).astype(float)
        res = evaluate_cold_track(cand, panel, ctrl, _ctx(), min_pairs=3)
        assert res["crowd_proxy_corr"] == 0.05
        test_dates = {
            pd.Timestamp(d)
            for d in ("2021-01-09", "2021-01-10", "2021-01-11", "2021-01-12")
        }
        assert not ((seen.get("dates") or set()) & test_dates)  # 盲测段未进入 proxy


# ── 4. dispatch 短路（gate FactorEvalTools） ──

def _make_tools(enabled: bool, skeletons=None):
    import alphaagent.dsl as dsl
    import alphaagent.factor.align as alignmod

    rng = np.random.default_rng(3)
    dates = pd.date_range("2021-01-01", periods=8, freq="D")
    instruments = ["S%02d" % i for i in range(12)]
    idx = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])
    n = len(idx)
    panel = pd.DataFrame(
        {"label_1d_open_to_open": rng.normal(size=n),
         "volume": rng.normal(size=n), "close": rng.normal(size=n)},
        index=idx,
    )
    session = types.SimpleNamespace(panel=panel, ctx=_ctx())
    service = types.SimpleNamespace(sessions={"s1": session})

    def fake_eval(expr, panel):
        return pd.Series(rng.normal(size=len(panel)), index=panel.index)

    def fake_align(out, panel):
        if hasattr(out, "to_numpy"):
            out = out.to_numpy(dtype=np.float64)
        arr = np.asarray(out, dtype=np.float64)
        return arr if len(arr) == len(panel) else np.full(len(panel), np.nan)

    dsl.eval_factor = fake_eval
    alignmod.align_series_to_panel = fake_align

    from alphaagent.factor.mining.tools import FactorEvalTools

    return FactorEvalTools(
        service, "s1",
        cold_track_policy={"enabled": enabled, "min_val_resid_ic": 0.02,
                           "max_crowd_proxy_corr": 0.3},
        crowd_skeletons=skeletons or {"skeletons": {
            "volatility": {"expr": "RANK(TS_STD($close,20))"},
            "volume_liquidity": {"expr": "RANK(TS_MEAN($volume,10))"},
        }},
    ), service


class TestDispatchShortcut:
    def test_disabled_returns_none(self):
        tools, _ = _make_tools(False)
        assert tools._cold_track_shortcut(
            "RANK(TS_PERMUTATION_ENTROPY($volume,20))", "排列熵"
        ) is None

    def test_cold_returns_result(self):
        tools, _ = _make_tools(True)
        r = tools._cold_track_shortcut(
            "RANK(TS_PERMUTATION_ENTROPY($volume,20))", "排列熵"
        )
        assert isinstance(r, dict)
        assert r.get("track") == "cold"
        assert r.get("cold_track") is True
        for k in ("resid_ic", "crowd_proxy_corr", "family_novelty"):
            assert k in r

    def test_shell_returns_advisory_not_shortcircuit(self):
        tools, _ = _make_tools(True)
        r = tools._cold_track_shortcut("RANK(CROWD_SHARE($volume,10))", "拥挤度")
        assert r is not None
        assert r.get("cold_track_advisory")
        assert r.get("track") is None  # 不短路，调用方继续走既有评估

    def test_new_family_non_cold_root_advisory(self):
        tools, _ = _make_tools(True)
        r = tools._cold_track_shortcut(
            "RANK(TS_MA(TS_RANK($close,20),5))", "新族普通"
        )
        assert r is not None and r.get("cold_track_advisory")
        assert r.get("track") is None

    def test_op_cooling_after_repeated_no_alpha(self, monkeypatch):
        tools, _ = _make_tools(True)
        import alphaagent.factor.mining.eval.residualize as rmod

        calls = []

        def fake_ect(*a, **k):
            calls.append(1)
            return {"ok": True, "passed": False, "fail_detail": "no_alpha",
                    "resid_ic": 0.001, "crowd_proxy_corr": 0.05}

        monkeypatch.setattr(rmod, "evaluate_cold_track", fake_ect)
        expr = "RANK(TS_PERMUTATION_ENTROPY($volume,20))"
        # 默认 op_cool_after_dead=3：前 3 次正常评估（cold），第 4 次进入冷却拦截
        for _ in range(3):
            r = tools._cold_track_shortcut(expr, "排列熵")
            assert r.get("track") == "cold"
        r = tools._cold_track_shortcut(expr, "排列熵")
        assert r.get("track") == "cold"  # 冷却拦截：短路返回阻断结果
        assert r.get("cold_track_blocked") is True
        assert r.get("cold_track_advisory") and "冷却" in r["cold_track_advisory"]
        assert len(calls) == 3  # 冷却拦截不再触发评估

    def test_op_cooling_reset_on_data_gap(self, monkeypatch):
        tools, _ = _make_tools(True)
        import alphaagent.factor.mining.eval.residualize as rmod

        state = {"n": 0}

        def fake_ect(*a, **k):
            state["n"] += 1
            # 前 2 次 no_alpha，随后 1 次 data_gap（归零）后再 no_alpha 需重新累计 3 次
            if state["n"] <= 2:
                fd = "no_alpha"
            elif state["n"] == 3:
                fd = "data_gap"
            else:
                fd = "no_alpha"
            return {"ok": True, "passed": False, "fail_detail": fd,
                    "resid_ic": 0.001, "crowd_proxy_corr": 0.05}

        monkeypatch.setattr(rmod, "evaluate_cold_track", fake_ect)
        expr = "RANK(TS_PERMUTATION_ENTROPY($volume,20))"
        # 2x no_alpha + 1x data_gap（归零）
        for _ in range(3):
            r = tools._cold_track_shortcut(expr, "排列熵")
            assert r.get("track") == "cold"
        # 再 3x no_alpha：n=4,5,6 累计到 3（data_gap 已归零），评估仍正常
        for _ in range(3):
            r = tools._cold_track_shortcut(expr, "排列熵")
            assert r.get("track") == "cold"
        # 第 7 次调用（累计已达 3）-> 评估前冷却拦截
        r = tools._cold_track_shortcut(expr, "排列熵")
        assert r.get("track") == "cold"
        assert r.get("cold_track_blocked") is True
        assert "冷却" in r["cold_track_advisory"]

    def test_shortcut_skeleton_empty_returns_none(self):
        tools, _ = _make_tools(True, skeletons={"skeletons": {}})
        r = tools._cold_track_shortcut(
            "RANK(TS_PERMUTATION_ENTROPY($volume,20))", "排列熵"
        )
        # 骨架缺失回落既有评估，不把冷门候选统一判 cold_dead(data_gap)
        assert r is None

    def test_cold_budget_ratio_shortcircuit(self):
        tools, _ = _make_tools(True)
        tools.cold_track_policy["cold_eval_budget_ratio"] = 0.19
        tools._total_eval_count = 10
        tools._cold_eval_count = 2  # 2/10 = 0.20 >= 0.19 -> 超限回落既有评估
        assert tools._cold_track_shortcut(
            "RANK(TS_PERMUTATION_ENTROPY($volume,20))", "排列熵"
        ) is None
        tools._cold_eval_count = 0  # 0/10 < 0.19 -> 预算内正常评估
        r = tools._cold_track_shortcut(
            "RANK(TS_PERMUTATION_ENTROPY($volume,20))", "排列熵"
        )
        assert r is not None and r.get("track") == "cold"

    def test_dispatch_cooling_shortcircuit_keeps_advisory(self, monkeypatch):
        tools, _ = _make_tools(True)
        import alphaagent.factor.mining.eval.residualize as rmod

        monkeypatch.setattr(
            rmod,
            "evaluate_cold_track",
            lambda *a, **k: {"ok": True, "passed": False, "fail_detail": "no_alpha",
                             "resid_ic": 0.001, "crowd_proxy_corr": 0.05},
        )
        tools._total_eval_count = 100
        tools._cold_op_cool_left["TS_PERMUTATION_ENTROPY"] = 3  # 直接进入冷却拦截态
        r = tools.dispatch(
            "eval_on_train_set",
            {"multi_line_expr": "RANK(TS_PERMUTATION_ENTROPY($volume,20))",
             "factor_name": "排列熵"},
        )
        assert isinstance(r, dict)
        assert r.get("track") == "cold"
        assert r.get("cold_track_blocked") is True
        # 阻断分支的 advisory 不被调用层 pop 吞掉（短路返回必须是完整自解释结果）
        assert r.get("cold_track_advisory") and "冷却" in r["cold_track_advisory"]


# ── 5. cold_dead verdict 传导（schema._classify） ──

class TestVerdict:
    def test_cold_dead_verdict(self):
        from alphaagent.factor.mining.memory.store import ResearchMemoryStore
        v, conclusion = ResearchMemoryStore._classify(
            "排列熵", {"cold_track": True, "cold_track_passed": False,
                      "fail_detail": "no_alpha"}, {}, ""
        )
        assert v == "cold_dead"
        assert "no_alpha" in conclusion

    def test_cold_approval_routes_to_cold_approved(self):
        from alphaagent.factor.mining.memory.store import ResearchMemoryStore
        v, _ = ResearchMemoryStore._classify(
            "排列熵", {"cold_track": True, "cold_track_passed": True}, {}, ""
        )
        assert v == "cold_approved"
        assert v != "train_passed"  # 未过综合门槛，不得按"已验证父本"记忆

    def test_cold_dead_retryable_predicate(self):
        from alphaagent.factor.mining.memory.retrieval import _cold_dead_retryable
        assert _cold_dead_retryable({"verdict": "cold_dead",
                                     "metrics": {"fail_detail": "data_gap"}}) is True
        assert _cold_dead_retryable({"verdict": "cold_dead",
                                     "metrics": {"fail_detail": "underpowered"}}) is True
        assert _cold_dead_retryable({"verdict": "cold_dead",
                                     "metrics": {"fail_detail": "no_alpha"}}) is False
        assert _cold_dead_retryable({"verdict": "weak", "metrics": {}}) is False


# ── 6. M3 记忆父本屏蔽（retrieval._evidence_block） ──

class TestM3ParentShield:
    @staticmethod
    def _seed(store, rows):
        now = "2026-10-17T00:00:00"
        with store._open() as conn:
            for i, row in enumerate(rows):
                conn.execute(
                    """INSERT INTO memory_entries
                       (id, factor_name, expression, conclusion, verdict, stage,
                        metrics_json, family, facets_json, attempts, created_at, updated_at)
                       VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (row["id"], row["factor_name"], row["expression"], row["conclusion"],
                     row["verdict"], row.get("stage", "candidate"),
                     json.dumps(row.get("metrics", {})), row.get("family", "other"),
                     json.dumps(row.get("facets", [])), 1, now, now),
                )

    @pytest.fixture
    def store(self, tmp_path):
        from alphaagent.factor.mining.memory.store import ResearchMemoryStore
        st = ResearchMemoryStore(tmp_path / "mem.db")
        self._seed(st, [
            {"id": "e_vol", "factor_name": "低波强者", "expression": "RANK(TS_STD($close,20))",
             "conclusion": "已入库", "verdict": "validated", "family": "volatility",
             "metrics": {"ic": 0.05}},
            # 记忆条目 family 存细族名（ingestion 侧 classify_family 输出）
            {"id": "e_liq", "factor_name": "量价强者", "expression": "RANK(TS_MEAN($amount,10))",
             "conclusion": "已入库", "verdict": "production_approved", "family": "volume",
             "metrics": {"ic": 0.04}},
            {"id": "e_chip", "factor_name": "排列熵", "expression": "RANK(TS_PERMUTATION_ENTROPY($volume,20))",
             "conclusion": "候选", "verdict": "candidate_approved", "family": "chip",
             "metrics": {"ic": 0.03}},
            {"id": "e_neg", "factor_name": "死路", "expression": "RANK(TS_STD($high,10))",
             "conclusion": "被拒", "verdict": "rejected", "family": "volatility",
             "metrics": {"ic": 0.001}},
        ])
        return st

    def test_crowd_strong_exprs_blocked(self, store):
        block = store._evidence_block("", limit=8, crowd_filter={"volatility", "volume_liquidity"})
        # 拥挤族强者表达式被屏蔽
        assert "RANK(TS_STD($close,20))" not in block
        assert "RANK(TS_MEAN($amount,10))" not in block
        # 屏蔽占位提示在场
        assert "已屏蔽" in block
        # 非拥挤族新族父本仍注入
        assert "TS_PERMUTATION_ENTROPY" in block

    def test_baseline_unblocked(self, store):
        block = store._evidence_block("", limit=8)
        # crowd_filter=None（黄金基线）：拥挤族强者表达式照常注入
        assert "RANK(TS_STD($close,20))" in block

    def test_context_for_passes_shield(self, store):
        ctx = store.context_for("", limit=8, enable_factor_retrieval=True,
                                crowd_filter={"volatility"})
        assert "RANK(TS_STD($close,20))" not in ctx
        ctx0 = store.context_for("", limit=8, enable_factor_retrieval=True)
        assert "RANK(TS_STD($close,20))" in ctx0
