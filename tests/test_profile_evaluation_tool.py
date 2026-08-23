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
