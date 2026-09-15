# -*- coding: utf-8 -*-
"""消融实验修复加固测试（P0 决策断层 / P1 异步落盘 / P2 低换手引导）。"""

from __future__ import annotations

import json
import sqlite3
import pytest
from pathlib import Path

from alphaagent.factor.mining.memory import ResearchMemoryStore
from alphaagent.factor.mining.tools._dispatch import _attach_yield_hints
from alphaagent.factor.mining.tools import FactorEvalTools


def _eval_row(name: str, expr: str, factor_name: str, *, ic: float = 0.025, icir: float = 0.35, error: str = "") -> dict:
    return {
        "name": name,
        "arguments_raw": json.dumps({
            "multi_line_expr": expr,
            "factor_name": factor_name,
        }),
        "result": {
            "ok": not bool(error),
            "split": "train",
            "summary": {
                "ic": ic,
                "icir": icir,
                "rank_ic": ic,
                "factor_coverage": 0.95,
            },
            "error": error,
        },
    }


def test_advisory_exempt_from_block(tmp_path: Path):
    """P0 测试：同结构正向条目具备豁免属性，不被 hard_block_duplicates 误拦。"""
    store = ResearchMemoryStore(tmp_path / "m.db", hard_block_duplicates=True)
    expr_a = "RANK(TS_MEAN($adj_close, 10))"
    expr_b = "RANK(TS_MEAN($adj_close, 20))"

    # 1. 记录一个正向条目
    store.record_tool_result(run_id="run_1", row=_eval_row("eval_on_train_set", expr_a, "f_pos", ic=0.03))
    store.flush_writes()

    # 2. 查询同指纹表达式的 advisory
    adv = store.advisory_for(expr_b)
    assert adv is not None
    advisories = adv.get("advisories", [])
    assert any(a.get("kind") == "duplicate_prior_result" for a in advisories)

    prior_item = next(a for a in advisories if a.get("kind") == "duplicate_prior_result")
    assert prior_item.get("exempt_from_block") is True

    # 3. 验证 _memory_gate 在 hard_block_duplicates=True 时不拦截
    tools = FactorEvalTools(service=None, session_id="test_sess", memory_store=store)
    gate_res = tools._memory_gate(expr_b, {})
    # 应当返回 advisory 字典（未拦截），而不是 {"ok": False, "error_type": "MemoryAdvisoryBlock"}
    assert gate_res is not None
    assert gate_res.get("error_type") != "MemoryAdvisoryBlock"


def test_async_write_persistence_and_integrity(tmp_path: Path):
    """P1 测试：异步写入后 flush_writes()，数据完整落盘且与独立连接一致。"""
    db_path = tmp_path / "async_mem.db"
    store = ResearchMemoryStore(db_path)

    n_entries = 12
    for i in range(n_entries):
        expr = f"RANK(TS_MEAN($adj_close, {10 + i}))"
        store.record_tool_result(run_id="run_async", row=_eval_row("eval_on_train_set", expr, f"f_{i}", ic=0.02 + i * 0.001))

    # 主线程刷新落盘
    store.flush_writes()

    # 使用全新的独立只读 SQLite 连接读取，验证完全落盘
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    count = conn.execute("SELECT COUNT(*) AS c FROM memory_entries").fetchone()["c"]
    conn.close()

    assert count == n_entries
    store.close()


def test_thread_safe_entry_snapshot(tmp_path: Path):
    """P1 测试：外部修改返回的 entry 不会污染后台落盘的数据（深拷贝快照防御）。"""
    db_path = tmp_path / "snapshot.db"
    store = ResearchMemoryStore(db_path)
    expr = "RANK(TS_MEAN($volume, 20))"

    entry = store.record_tool_result(run_id="run_snap", row=_eval_row("eval_on_train_set", expr, "f_snap", ic=0.028))
    assert entry is not None

    # 外部恶意篡改 entry 字段
    entry["conclusion"] = "MALICIOUS_TAMPERED_CONCLUSION"
    entry["metrics"]["ic"] = 999.99

    store.flush_writes()

    # 验证数据库中落盘的是原始安全快照
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT conclusion, metrics_json FROM memory_entries WHERE factor_name = 'f_snap'").fetchone()
    conn.close()

    assert row is not None
    assert "MALICIOUS_TAMPERED_CONCLUSION" not in row["conclusion"]
    m = json.loads(row["metrics_json"])
    assert m.get("ic") != 999.99
    store.close()


def test_async_write_batch_failure_resilience(tmp_path: Path):
    """P1 测试：批量落盘中若单条失败，回退机制确保其余正常条目仍成功写入。"""
    store = ResearchMemoryStore(tmp_path / "batch_fail.db")
    store._ensure_async_writer()

    def _good_task_1(conn):
        conn.execute("CREATE TABLE IF NOT EXISTS test_tbl (val TEXT)")
        conn.execute("INSERT INTO test_tbl VALUES ('good1')")

    def _bad_task(conn):
        # 故意引发异常
        raise ValueError("Simulated write error in bad task")

    def _good_task_2(conn):
        conn.execute("CREATE TABLE IF NOT EXISTS test_tbl (val TEXT)")
        conn.execute("INSERT INTO test_tbl VALUES ('good2')")

    # 批量投递
    batch = [
        (_good_task_1, (), {}),
        (_bad_task, (), {}),
        (_good_task_2, (), {}),
    ]
    store._flush_batch_to_db(batch)

    with store._open() as conn:
        rows = [r["val"] for r in conn.execute("SELECT val FROM test_tbl ORDER BY val").fetchall()]

    # good1 与 good2 均应成功落盘，不被 bad_task 拖累丢失
    assert "good1" in rows
    assert "good2" in rows
    store.close()


def test_turnover_reduction_hints_present():
    """P2 测试：日换手超标时输出 3 种具体降噪路径与 turnover_reduction_hints 字段。"""
    res = {
        "split": "train",
        "passed": True,
        "metrics": {
            "cross_sectional_core": {"ic": 0.028, "icir": 0.35, "factor_coverage": 0.95},
            "quantile_portfolio": {"avg_daily_side_turnover": 0.55},
        },
    }
    _attach_yield_hints(res, "RANK(TS_PCTCHANGE($adj_close, 1))", {})

    assert "submit_decision_required" in res
    hint = res["submit_decision_required"]
    assert "请勿提交" in hint
    assert "CS_ZSCORE" in hint
    assert "基本面 PIT" in hint

    hints_list = res.get("turnover_reduction_hints")
    assert isinstance(hints_list, list) and len(hints_list) >= 3
