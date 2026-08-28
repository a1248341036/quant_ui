from __future__ import annotations

import json
from pathlib import Path

from alphaagent.factor.mining.research_memory import ResearchMemoryStore
from alphaagent.factor.mining.tools import FactorEvalTools


class _ProfileService:
    def __init__(self) -> None:
        self.request = None

    def eval_profile(self, request):
        self.request = request
        return {
            "ok": True,
            "split": "val",
            "profile": {"profile_id": request.profile_id},
            "profile_hash": "profilehash",
            "candidate": {"candidate_id": "cand_1", "factor_name": request.factor_name, "expression": request.multi_line_expr},
            "metrics": {
                "cross_sectional_core": {"ic": 0.02, "icir": 0.3, "rank_ic": 0.02, "factor_coverage": 1.0},
            },
            "rule_results": [],
        }


def test_generic_profile_tool_dispatches_frozen_profile() -> None:
    service = _ProfileService()
    tools = FactorEvalTools(service, "session")
    result = tools.dispatch(
        "evaluate_factor",
        {"profile_id": "size_neutral_validation", "factor_name": "candidate", "multi_line_expr": "$adj_close"},
    )
    assert result["ok"]
    assert service.request.profile_id == "size_neutral_validation"
    assert service.request.multi_line_expr == "$adj_close"


def test_generic_profile_evidence_is_persisted_in_research_memory(tmp_path: Path) -> None:
    store = ResearchMemoryStore(tmp_path / "research_memory.json")
    entry = store.record_tool_result(
        run_id="run",
        row={
            "name": "evaluate_factor",
            "arguments_raw": json.dumps({"factor_name": "candidate", "multi_line_expr": "$adj_close"}),
            "result": {
                "ok": True,
                "split": "val",
                "profile": {"profile_id": "validation"},
                "profile_hash": "profilehash",
                "candidate": {"candidate_id": "cand_1"},
                "metrics": {"cross_sectional_core": {"ic": 0.02, "icir": 0.3, "factor_coverage": 1.0}},
            },
        },
    )
    assert entry is not None
    assert entry["stage"] == "val"
    assert entry["profile_id"] == "validation"
    assert entry["candidate_id"] == "cand_1"


def _make_row(*, name: str, factor_name: str, expr: str, ic: float, icir: float, split: str = "train") -> dict:
    return {
        "name": name,
        "arguments_raw": json.dumps({"factor_name": factor_name, "multi_line_expr": expr}),
        "result": {
            "ok": True,
            "split": split,
            "profile": {"profile_id": "train_screen" if split == "train" else "validation"},
            "profile_hash": "h",
            "candidate": {"candidate_id": "c1"},
            "metrics": {"cross_sectional_core": {"ic": ic, "icir": icir, "factor_coverage": 0.95}},
        },
    }


def test_positive_entries_prioritized_and_segmented(tmp_path: Path) -> None:
    """context_for 应将肯定条目排在否定之前，并分两段输出。"""
    store = ResearchMemoryStore(tmp_path / "rm.json")

    # 两条 BM25 相似的动量因子：一条 validated（肯定），一条 weak（否定）
    store.record_tool_result(run_id="r1", row=_make_row(
        name="evaluate_factor", factor_name="momentum_20d",
        expr="TS_MEAN($ret, 20)", ic=0.03, icir=0.5, split="val",
    ))
    store.record_tool_result(run_id="r2", row=_make_row(
        name="evaluate_factor", factor_name="momentum_10d",
        expr="TS_MEAN($ret, 10)", ic=0.005, icir=0.1, split="train",
    ))

    ctx = store.context_for("动量因子 momentum", limit=10, include_expression=False, enable_factor_retrieval=True)

    # 肯定段在否定段之前
    pos_idx = ctx.find("已验证")
    neg_idx = ctx.find("已否定")
    assert pos_idx != -1, "缺少肯定段标题"
    assert neg_idx != -1, "缺少否定段标题"
    assert pos_idx < neg_idx, "肯定段应排在否定段之前"


def test_positive_entry_surfaces_above_equally_relevant_negative(tmp_path: Path) -> None:
    """BM25 分数相同时，肯定条目应在否定之前。"""
    store = ResearchMemoryStore(tmp_path / "rm.json")

    # 两条完全相同表达式的因子，一条 validated 一条 weak（第二次评估覆盖）
    # 用不同 factor_name 但相同表达式，BM25 打分相同
    store.record_tool_result(run_id="r1", row=_make_row(
        name="evaluate_factor", factor_name="vol_ratio",
        expr="DIVIDE($volume, TS_MEAN($volume, 20))", ic=0.025, icir=0.4, split="val",
    ))
    # 第二条不同表达式但 BM25 相似
    store.record_tool_result(run_id="r2", row=_make_row(
        name="evaluate_factor", factor_name="vol_ratio_weak",
        expr="DIVIDE($volume, TS_MEAN($volume, 10))", ic=0.001, icir=0.05, split="train",
    ))

    ctx = store.context_for("成交量比率 volume", limit=5, include_expression=False, enable_factor_retrieval=True)

    # validated 条目位置应在 weak 之前
    validated_pos = ctx.find("[validated]")
    weak_pos = ctx.find("[weak]")
    assert validated_pos != -1, "缺少 validated 条目"
    assert weak_pos != -1, "缺少 weak 条目"
    assert validated_pos < weak_pos, "肯定条目应排在否定之前"


def test_rejected_entries_still_appear(tmp_path: Path) -> None:
    """include_rejected=True 时否定条目仍出现，用于避免重复死路。"""
    store = ResearchMemoryStore(tmp_path / "rm.json")
    store.record_tool_result(run_id="r1", row=_make_row(
        name="evaluate_factor", factor_name="bad_factor",
        expr="TS_MEAN($ret, 5)", ic=0.001, icir=0.01, split="train",
    ))

    ctx = store.context_for("动量", limit=5, include_rejected=True, include_expression=False, enable_factor_retrieval=True)
    assert "[weak]" in ctx, "否定条目应出现在输出中"

    ctx_filtered = store.context_for("动量", limit=5, include_rejected=False, include_expression=False, enable_factor_retrieval=True)
    assert "[weak]" not in ctx_filtered, "include_rejected=False 时否定条目不应出现"
