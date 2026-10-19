# -*- coding: utf-8 -*-
"""换手透传口径回归（2026-10-19）。

**缺陷**：``eval/service.py::_engine_result_to_legacy`` 只把换手写进**嵌套**
``out["quantile_portfolio"]``，而两条主力消费路径都读**顶层**：

- ``scripts/harness_mine.py::_extract`` → ``res.get("avg_rebalance_side_turnover")``
- ``alphaagent/mcp_server/tools.py`` → ``res.get("avg_daily_side_turnover")``

于是 harness 直驱 / MCP 路径拿到的 turn 恒为 None，而 ``harness_mine._flags``
的换手门是 ``if turn is None or turn > limit: fail`` —— **换手门结构性永不通过**，
任何因子都无法判定 stage_one 过门（2026-10-19 实测 15/15 因子全部额外报
``turnover>0.8`` 而 turn=None，掩盖了真实的 IC/ICIR 结论）。

修复：service 侧同时写顶层（与 agentscope 链路 response.py 同口径），
harness 侧 ``_extract`` 顶层优先、回落嵌套，防单边回归。
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from alphaagent.factor.mining.eval.service import _engine_result_to_legacy

_ROOT = Path(__file__).resolve().parents[1]


def _load_harness_mine():
    """载入 scripts/harness_mine.py（非包模块，需按路径加载）。"""
    path = _ROOT / "scripts" / "harness_mine.py"
    spec = importlib.util.spec_from_file_location("_harness_mine_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _raw(qp: dict | None) -> dict:
    """最小 EvaluationEngine 结果，只关心 quantile_portfolio 段。"""
    metrics: dict = {
        "cross_sectional_core": {
            "ic": 0.06, "icir": 0.7, "rank_ic": 0.05,
            "n_days": 700, "n_instruments": 4000,
            "factor_coverage": 0.95, "cs_pearson_autocorr": 0.9,
        },
    }
    if qp is not None:
        metrics["quantile_portfolio"] = qp
    return {
        "ok": True, "split": "train", "date_range": ["2020-01-01", "2022-12-31"],
        "label_col": "label_20d_close_to_close", "timing_ms": {"total_ms": 1000},
        "metrics": metrics,
    }


class TestServiceWritesTopLevel:
    def test_turnover_written_both_levels(self):
        out = _engine_result_to_legacy(_raw({
            "avg_daily_side_turnover": 0.31,
            "avg_rebalance_side_turnover": 0.42,
            "depth_curve": [],
        }))
        # 顶层（harness/MCP 读这里）
        assert out["avg_daily_side_turnover"] == 0.31
        assert out["avg_rebalance_side_turnover"] == 0.42
        # 嵌套（向后兼容）
        assert out["quantile_portfolio"]["avg_daily_side_turnover"] == 0.31
        assert out["quantile_portfolio"]["avg_rebalance_side_turnover"] == 0.42

    def test_turnover_absent_when_qp_missing(self):
        out = _engine_result_to_legacy(_raw(None))
        assert "avg_daily_side_turnover" not in out
        assert "avg_rebalance_side_turnover" not in out

    def test_none_values_not_injected(self):
        """qp 存在但换手为 None 时不得写 None 键（避免下游误判为"有值"）。"""
        out = _engine_result_to_legacy(_raw({
            "avg_daily_side_turnover": None,
            "avg_rebalance_side_turnover": None,
            "depth_curve": [],
        }))
        assert "avg_daily_side_turnover" not in out
        assert "quantile_portfolio" not in out or "avg_daily_side_turnover" not in out["quantile_portfolio"]


class TestHarnessExtract:
    def test_extract_reads_top_level(self):
        hm = _load_harness_mine()
        got = hm._extract({
            "summary": {"ic": 0.06, "icir": 0.7, "factor_coverage": 0.95,
                        "cs_pearson_autocorr": 0.9, "n_days": 700},
            "avg_daily_side_turnover": 0.31,
            "avg_rebalance_side_turnover": 0.42,
        })
        assert got["turnover_daily"] == 0.31
        assert got["turnover_rebalance"] == 0.42

    def test_extract_falls_back_to_nested(self):
        """service 未修复/旧响应只有嵌套层时仍能取到（防单边回归）。"""
        hm = _load_harness_mine()
        got = hm._extract({
            "summary": {"ic": 0.06, "icir": 0.7},
            "quantile_portfolio": {
                "avg_daily_side_turnover": 0.31,
                "avg_rebalance_side_turnover": 0.42,
            },
        })
        assert got["turnover_daily"] == 0.31
        assert got["turnover_rebalance"] == 0.42

    def test_extract_zero_turnover_is_not_treated_as_missing(self):
        """0.0 是合法换手值，不能被 `or` 兜底吞掉。"""
        hm = _load_harness_mine()
        got = hm._extract({
            "summary": {},
            "avg_rebalance_side_turnover": 0.0,
            "quantile_portfolio": {"avg_rebalance_side_turnover": 0.42},
        })
        assert got["turnover_rebalance"] == 0.0


class TestFlagsUseRealTurnover:
    def test_flags_pass_turnover_when_value_present(self):
        """换手有值且在门内 → 不再报 turnover 门。"""
        hm = _load_harness_mine()
        row = {"ic": 0.08, "icir": 0.9, "coverage": 0.95,
               "cs_autocorr": 0.95, "turnover_rebalance": 0.10}
        fails = hm._flags("technical_monthly", row)
        assert not any("turnover" in f for f in fails), fails

    def test_flags_fail_turnover_above_limit(self):
        hm = _load_harness_mine()
        row = {"ic": 0.08, "icir": 0.9, "coverage": 0.95,
               "cs_autocorr": 0.95, "turnover_rebalance": 5.0}
        fails = hm._flags("technical_monthly", row)
        assert any(f.startswith("turnover>") for f in fails), fails

    def test_flags_still_fail_when_turnover_unknown(self):
        """取不到换手时仍应 fail（保守：不能把"未验证"当"通过"）。"""
        hm = _load_harness_mine()
        row = {"ic": 0.08, "icir": 0.9, "coverage": 0.95,
               "cs_autocorr": 0.95, "turnover_rebalance": None}
        fails = hm._flags("technical_monthly", row)
        assert any(f.startswith("turnover>") for f in fails), fails
