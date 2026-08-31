# 回归：purge_factor 按表达式签名与 factor_name 清除记忆条目（RAG/FTS 同步消失）
import json

from alphaagent.factor.mining.research_memory import ResearchMemoryStore


def _row(name: str, expr: str):
    return {
        "name": "evaluate_factor",
        "arguments_raw": json.dumps({
            "multi_line_expr": expr,
            "factor_name": name,
        }),
        "result": {"ok": True, "metrics": {"ic": 0.03, "icir": 0.4, "coverage": 0.95}},
    }


def test_purge_factor_removes_memory_and_rag(tmp_path):
    store = ResearchMemoryStore(tmp_path / "mem.db")
    expr_a = "leg_a = RANK(x)\nleg_a"
    expr_b = "leg_b = NEG(y)"
    store.record_tool_result(run_id="r1", row=_row("factor_a", expr_a))
    store.record_tool_result(run_id="r1", row=_row("factor_b", expr_b))

    # 两条都在
    assert store.purge_factor(factor_names=["factor_a"], expressions=[expr_a]) >= 1
    remaining, _ = store.recent()
    names = {e.get("factor_name") for e in remaining}
    assert "factor_a" not in names

    # 按 name 清除另一条
    assert store.purge_factor(factor_names=["factor_b"]) >= 1
    assert all(e.get("factor_name") != "factor_b" for e in store.recent()[0])

    # 再清一次应为 0（幂等）
    assert store.purge_factor(factor_names=["factor_b"], expressions=[expr_b]) == 0
