# -*- coding: utf-8 -*-
"""评估引擎低离散度/重复评估修复的回归测试（2026-09-05）。

覆盖：
- A) quantile_portfolio_metrics：分箱塌缩（每天截面唯一值 < 组数）→
  available=False + insufficient_distinct_values，不输出 top_group_*；
  正常连续因子不受影响
- B) prediction_check：decile_rows 组数不足 → unverifiable（incomplete_binning）
- D) exact_duplicate_prior：同指纹+逐字相同表达式的正向历史查询
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from alphaagent.factor.metrics import quantile_portfolio_metrics
from alphaagent.factor.mining.eval.prediction import (
    build_prediction_check,
    classify_decile_shape,
)


def _panel(n_days: int = 60, n_inst: int = 40) -> pd.DataFrame:
    days = pd.bdate_range("2024-01-01", periods=n_days)
    inst = [f"S{i:03d}" for i in range(n_inst)]
    idx = pd.MultiIndex.from_product([days, inst], names=["datetime", "instrument"])
    rng = np.random.default_rng(11)
    ret = rng.normal(0, 0.02, len(idx))
    return pd.DataFrame({"adj_close": 10.0 * np.exp(np.cumsum(ret))}, index=idx)


def _label_from(panel: pd.DataFrame) -> pd.Series:
    close = panel["adj_close"]
    entry = close.groupby(level="instrument", sort=False).shift(-1)
    exit_ = close.groupby(level="instrument", sort=False).shift(-2)
    return (exit_ / entry - 1.0)


class TestQuantileCollapse:
    def test_collapsed_bins_marked_unavailable(self):
        """每天截面只有 4 个唯一值（<10 组）→ available=False + 塌缩原因。"""
        panel = _panel()
        dts = panel.index.get_level_values("datetime")
        days = pd.DatetimeIndex(pd.unique(dts))
        vals = np.full(len(panel), np.nan)
        for i, d in enumerate(days):
            mask = np.asarray(dts == d)
            # 每天 40 只股票轮转分配 4 个离散值（模拟 VPIN 类分段常数因子）
            vals[mask] = np.tile([0.1, 0.2, 0.3, 0.5], int(mask.sum()) // 4 + 1)[: int(mask.sum())]
        fac = pd.Series(vals, index=panel.index)
        lab = _label_from(panel)
        out = quantile_portfolio_metrics(fac, lab, cost_bps=0.0, n_groups=10)
        assert out["available"] is False
        assert out["error"] == "insufficient_distinct_values"
        assert out["collapse_ratio"] == 1.0
        assert "top_group_annualized_return" not in out
        assert "message" in out

    def test_continuous_factor_still_works(self):
        """正常连续因子（每天 30 个唯一值 > 10 组）不受影响，指标照常输出。"""
        panel = _panel()
        rng = np.random.default_rng(7)
        fac = pd.Series(
            rng.normal(0, 1, len(panel)) + np.arange(len(panel)) * 1e-6,
            index=panel.index,
        )
        lab = _label_from(panel)
        out = quantile_portfolio_metrics(fac, lab, cost_bps=0.0, n_groups=10)
        assert out["available"] is True
        assert "top_group_annualized_return" in out
        assert "top_group_sharpe" in out

    def test_partial_collapse_still_available(self):
        """部分天塌缩（<50%）→ 仍可用（跳过塌缩天），并记录 collapse_ratio。"""
        panel = _panel(n_days=60)
        dts = panel.index.get_level_values("datetime")
        days = pd.DatetimeIndex(pd.unique(dts))
        rng = np.random.default_rng(3)
        vals = rng.normal(0, 1, len(panel))
        ser_vals = np.asarray(vals, dtype=float).copy()
        # 只让前 10 天（<50%）塌缩成 2 个唯一值
        for d in days[:10]:
            mask = np.asarray(dts == d)
            tile = np.tile([0.1, 0.2], int(mask.sum()) // 2 + 1)[: int(mask.sum())]
            ser_vals[mask] = tile
        fac = pd.Series(ser_vals, index=panel.index)
        lab = _label_from(panel)
        out = quantile_portfolio_metrics(fac, lab, cost_bps=0.0, n_groups=10)
        assert out["available"] is True
        # 60 天 - 10 塌缩天 - 尾部 label NaN 的 2 天 = 48 天有效
        assert out["n_days"] == 48


class TestPredictionIncompleteBinning:
    def test_incomplete_deciles_unverifiable(self):
        """decile_rows 只有 7 组（≥5 但 <10，分箱塌缩）→ unverifiable + incomplete_binning。"""
        rows = [
            {"decile": 1, "mean_label": 0.0008},
            {"decile": 2, "mean_label": 0.0005},
            {"decile": 3, "mean_label": 0.0006},
            {"decile": 7, "mean_label": 0.0006},
            {"decile": 8, "mean_label": 0.0007},
            {"decile": 9, "mean_label": 0.0006},
            {"decile": 10, "mean_label": 0.0006},
        ]
        pred = {"expected_shape": "monotonic_increasing",
                "expected_strong_side": "high_factor", "expected_sign": 1}
        check = build_prediction_check(pred, ic=0.0349, decile_rows=rows)
        assert check is not None
        assert check["verdict"] == "unverifiable"
        assert check["actual"]["incomplete_binning"] is True
        assert check["actual"]["n_bins_used"] == 7

    def test_classify_marks_incomplete(self):
        rows = [{"decile": i, "mean_label": 0.001 * (i % 3 + 1)} for i in (1, 2, 3, 4, 5, 6, 7)]
        actual = classify_decile_shape(rows)
        assert actual["incomplete"] is True
        assert actual["shape"] == "incomplete_binning"
        assert actual["n_bins_used"] == 7

    def test_complete_deciles_unaffected(self):
        """完整 10 组不受影响，正常分类。"""
        rows = [{"decile": i, "mean_label": 0.001 * i} for i in range(1, 11)]
        actual = classify_decile_shape(rows)
        assert actual.get("incomplete") is None or actual.get("incomplete") is not True
        assert actual["shape"] == "monotonic_increasing"


class TestExactDuplicatePrior:
    def _seed(self, store, expr: str, factor_name: str, verdict: str):
        """直接 SQL 插入一条 memory_entries（绕过 record_tool_result 的 JSONL 形参）。"""
        from alphaagent.factor.mining.memory.expressions import _structure_fingerprint

        import json as _json
        fp = _structure_fingerprint(expr.strip())
        with store._open() as conn:
            conn.execute(
                "INSERT INTO memory_entries (id, factor_name, expression, verdict, stage, "
                "metrics_json, structure_fingerprint, created_at, updated_at, attempts) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1)",
                (
                    fp, factor_name, expr, verdict, "train",
                    _json.dumps({"ic": 0.0349, "icir": 0.643}),
                    fp, "2026-09-05T04:32:58", "2026-09-05T04:32:58",
                ),
            )
            conn.commit()

    def test_exact_duplicate_detected(self, tmp_path):
        """同指纹+逐字相同表达式 → 检出；参数变异（不同表达式）→ 不检出。"""
        from alphaagent.factor.mining.memory.store import ResearchMemoryStore

        db = tmp_path / "mem.db"
        store = ResearchMemoryStore(str(db))
        expr = "vp = VOLUME_CLOCK_VPIN($adj_close, $volume, 20, 0.1)\nTS_MEAN(RANK(vp), 3)"
        self._seed(store, expr, "vpin_smooth3", "promising")
        hit = store.exact_duplicate_prior(expr)
        assert hit is not None
        assert hit["factor_name"] == "vpin_smooth3"
        assert hit["verdict"] == "promising"
        assert hit["ic"] == pytest.approx(0.0349)
        # 变异表达式（不同参数）→ 不同指纹 → 不检出
        variant = "vp = VOLUME_CLOCK_VPIN($adj_close, $volume, 40, 0.1)\nTS_MEAN(RANK(vp), 3)"
        assert store.exact_duplicate_prior(variant) is None
        # 行首尾空白差异被规范化后视为相同（strip 归一）
        spaced = "  " + expr.replace("\n", "\n  ") + " "
        hit2 = store.exact_duplicate_prior(spaced)
        assert hit2 is not None and hit2["factor_name"] == "vpin_smooth3"

    def test_weak_entries_not_intercepted(self, tmp_path):
        """weak/rejected 历史不算 prior_result——不拦（参数扫描合理行为）。"""
        from alphaagent.factor.mining.memory.store import ResearchMemoryStore

        db = tmp_path / "mem2.db"
        store = ResearchMemoryStore(str(db))
        expr = "f = TS_MEAN($volume, 10)"
        self._seed(store, expr, "f_v1", "weak")
        assert store.exact_duplicate_prior(expr) is None
