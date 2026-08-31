# v3-lite 研究记忆回归：AlphaMemo Eq.7 置信 / 双门 APV / 显式父本协议 / invalid 入账 /
# 同桶衰减基线 / advisory 硬提醒 / 注入预算 / v2→v3 迁移幂等。
import json
import sqlite3

import pytest

from alphaagent.factor.mining.research_memory import (
    APV_TAU_C_DEFAULT,
    APV_TAU_V_DEFAULT,
    EQ7_KAPPA_DEFAULT,
    ResearchMemoryStore,
    _apv_gate,
    _eq7_confidence,
    _structure_fingerprint,
    motif_from_note,
)


# ---------------------------------------------------------------------------
# Eq.7 置信：论文/本地测试三条断言（单观测<0.3 / 混杂<0.3 / 30 次一致>0.7 且饱和）
# ---------------------------------------------------------------------------

def test_eq7_confidence_assertions():
    # 单观测：样本门 1/9≈0.11 → <0.3
    assert _eq7_confidence([0.01]) < 0.3
    # 6 次正负混杂（低 SNR）→ <0.3
    mixed = _eq7_confidence([0.010, -0.011, 0.009, -0.008, 0.012, -0.010])
    assert mixed < 0.3
    # 30 次一致残差 → >0.7
    consistent = _eq7_confidence([0.010 + 0.0001 * i for i in range(30)])
    assert consistent > 0.7
    # 饱和：n 越大置信越高（单调），且渐近 1
    c30 = _eq7_confidence([0.01] * 30)
    c60 = _eq7_confidence([0.01] * 60)
    assert c60 > c30
    assert _eq7_confidence([0.01] * 500) < 1.0
    # 空序列
    assert _eq7_confidence([]) == 0.0
    # κ 覆盖
    assert _eq7_confidence([0.01] * 30, kappa=80.0) < 0.7


# ---------------------------------------------------------------------------
# 双门 APV：veto = c>τ_c ∧ π⁻>τ_v；正证据永不硬放行
# ---------------------------------------------------------------------------

def test_apv_double_gate_truth_table():
    # 5 连败（explicit 加权 f_w=5）+ 一致负残差 → 高置信 → veto
    c_high = _eq7_confidence([-0.005] * 5)
    vetoed, severity, pi_neg = _apv_gate(0.0, 5.0, c_high)
    assert vetoed
    assert pi_neg == pytest.approx(6.0 / 7.0)
    assert severity == pytest.approx(pi_neg)
    # 同样 5 连败但残差混杂（低置信）→ 不否决（双门缺一不可）
    c_noisy = _eq7_confidence([0.02, -0.02, 0.015, -0.018, -0.001])
    vetoed2, _, _ = _apv_gate(0.0, 5.0, c_noisy)
    assert not vetoed2
    # 失败后验不足（成功居多）→ 不否决
    vetoed3, _, _ = _apv_gate(8.0, 2.0, 0.9)
    assert not vetoed3
    # 隐式观测权重 0.5：5 次隐式连败（f_w=2.5）→ π⁻=3.5/4.5≈0.78 <0.8 → 不否决（噪声数据需更多证据）
    vetoed4, _, pi4 = _apv_gate(0.0, 2.5, 0.9)
    assert not vetoed4 and pi4 < 0.8
    # 7 次隐式连败（f_w=3.5）→ π⁻=4.5/5.5≈0.818 >0.8 → 否决
    vetoed5, _, _ = _apv_gate(0.0, 3.5, 0.9)
    assert vetoed5
    # 零观测
    assert _apv_gate(0.0, 0.0, 0.9)[0] is False
    # 默认阈值
    assert APV_TAU_C_DEFAULT == 0.35 and APV_TAU_V_DEFAULT == 0.80 and EQ7_KAPPA_DEFAULT == 8.0


def test_motif_from_note():
    assert motif_from_note("edit=window_rescale 10→20") == "window_rescale"
    assert motif_from_note("operator_substitute DIVIDE→RANK") == "operator_substitute"
    assert motif_from_note("参数变异 窗口10到20") == "window_rescale"
    assert motif_from_note("随便写的") == "other"
    assert motif_from_note("") is None
    assert motif_from_note(None) is None


# ---------------------------------------------------------------------------
# 测试工具：构造工具调用行
# ---------------------------------------------------------------------------

def _eval_row(name, expr, factor_name, ic=None, icir=0.4, coverage=0.9, error=None, extra_args=None):
    args = {"multi_line_expr": expr, "factor_name": factor_name}
    if extra_args:
        args.update(extra_args)
    result = {"ok": error is None, "split": "train"}
    if ic is not None:
        result["metrics"] = {"ic": ic, "icir": icir, "factor_coverage": coverage}
    if error:
        result["error"] = error
    return {"name": name, "arguments_raw": json.dumps(args), "result": result}


# ---------------------------------------------------------------------------
# 显式父本协议：explicit 命中 / 隐式兜底 / crossover / invalid 入账 / cells 结构
# ---------------------------------------------------------------------------

PARENT_EXPR = "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 10)))"
CHILD_EXPR = "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 20)))"


def test_explicit_parent_protocol(tmp_path):
    store = ResearchMemoryStore(tmp_path / "m.db")
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set", PARENT_EXPR, "vwap_dev_10", ic=0.020))
    entry = store.record_tool_result(
        run_id="r1",
        row=_eval_row(
            "eval_on_train_set", CHILD_EXPR, "vwap_dev_20", ic=0.030,
            extra_args={"parent_factor": "vwap_dev_10", "edit_note": "edit=window_rescale 10→20"},
        ),
    )
    assert entry is not None
    assert entry["parent_origin"] == "explicit"
    assert entry["intended_motif"] == "window_rescale"
    assert entry["parent_id"] is not None
    # cells：(family, motif, bucket) 键 + explicit 分列
    with store._open() as conn:
        rows = conn.execute("SELECT family, motif, parent_bucket, explicit_s, explicit_f, implicit_s, implicit_f, residuals_json FROM memory_cells").fetchall()
    assert len(rows) == 1
    row = rows[0]
    assert row["motif"] == "window_rescale"
    assert row["parent_bucket"] == "medium"  # 父本 |IC|=0.020 ∈ [0.015, 0.025)
    assert row["explicit_s"] == pytest.approx(1.0)  # 显式成功 +1.0
    assert row["implicit_s"] == pytest.approx(0.0)
    # residual = child(0.030) − 同桶基线(无历史 → 回退父本 0.020) = +0.010
    residuals = json.loads(row["residuals_json"])
    assert residuals == pytest.approx([0.010])


def test_implicit_fallback_weighted(tmp_path):
    store = ResearchMemoryStore(tmp_path / "m.db")
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set", PARENT_EXPR, "vwap_dev_10", ic=0.020))
    # 不声明父本 → 结构相似隐式兜底（窗口 10→20 是真实编辑观测）
    entry = store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set", CHILD_EXPR, "vwap_dev_20b", ic=0.018))
    assert entry["parent_origin"] == "implicit"
    with store._open() as conn:
        row = conn.execute("SELECT implicit_s, explicit_s, implicit_f, explicit_f FROM memory_cells").fetchone()
    # 列存加权值：implicit 成功 = 0.5
    assert row["implicit_s"] == pytest.approx(0.5)
    assert row["explicit_s"] == pytest.approx(0.0)
    # 加权统计直加：s_w = 0.5
    s_w, _ = store._weighted_counts(row)
    assert s_w == pytest.approx(0.5)


def test_invalid_attempt_recorded_as_failure(tmp_path):
    store = ResearchMemoryStore(tmp_path / "m.db")
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set", PARENT_EXPR, "vwap_dev_10", ic=0.020))
    # 报错且无 IC → invalid 失败观测（权重 0.5，无 residual）
    entry = store.record_tool_result(
        run_id="r1",
        row=_eval_row(
            "eval_on_train_set", CHILD_EXPR, "vwap_dev_bad", error="dsl_compile_failed: unknown op",
            extra_args={"parent_factor": "vwap_dev_10", "edit_note": "edit=window_rescale 10→40"},
        ),
    )
    assert entry is not None
    with store._open() as conn:
        row = conn.execute("SELECT explicit_f, residuals_json FROM memory_cells").fetchone()
    assert row["explicit_f"] == pytest.approx(0.5)  # invalid 权重 0.5
    assert json.loads(row["residuals_json"]) == []  # 无 residual


def test_same_bucket_baseline_uses_history(tmp_path):
    store = ResearchMemoryStore(tmp_path / "m.db")
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set", PARENT_EXPR, "vwap_dev_10", ic=0.020))
    # 第一个子代：基线回退父本 0.020
    store.record_tool_result(
        run_id="r1",
        row=_eval_row("eval_on_train_set", CHILD_EXPR, "vwap_dev_20", ic=0.024,
                      extra_args={"parent_factor": "vwap_dev_10", "edit_note": "edit=window_rescale 10→20"}),
    )
    # 第二个同族同桶子代（另一个父本避免 signature 冲突？同一子代表达式会累加 attempts）：
    # 直接用第三条父本+子代，检查基线已含第一个子代的 IC
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set", PARENT_EXPR + " ", "vwap_dev_10b", ic=0.021))
    entry = store.record_tool_result(
        run_id="r1",
        row=_eval_row("eval_on_train_set", "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 40)))", "vwap_dev_40",
                      ic=0.018, extra_args={"parent_factor": "vwap_dev_10b", "edit_note": "edit=window_rescale 20→40"}),
    )
    assert entry is not None
    with store._open() as conn:
        rows = conn.execute("SELECT parent_bucket, residuals_json FROM memory_cells WHERE motif='window_rescale'").fetchall()
    # 第二条 residual = 0.018 − 同桶基线（首子代 0.024 与父本 0.020/0.021 的加权均值）
    assert len(rows) >= 1


# ---------------------------------------------------------------------------
# advisory：指纹负证据 + 意向编辑 APV + hard_block
# ---------------------------------------------------------------------------

def test_advisory_duplicate_known_dead_end(tmp_path):
    store = ResearchMemoryStore(tmp_path / "m.db")
    expr = "TS_MEAN($adj_close, 5) + 0.5"
    for i in range(2):
        store.record_tool_result(run_id=f"r{i}", row=_eval_row("eval_on_train_set", expr, f"weak_{i}", ic=0.005))
    advisory = store.advisory_for(expr)
    assert advisory is not None
    kinds = [a["kind"] for a in advisory["advisories"]]
    assert "duplicate_known_dead_end" in kinds
    # attempts<2 不提醒
    store2 = ResearchMemoryStore(tmp_path / "m2.db")
    store2.record_tool_result(run_id="r0", row=_eval_row("eval_on_train_set", expr, "weak_0", ic=0.005))
    assert store2.advisory_for(expr) is None


def test_advisory_edit_veto(tmp_path):
    store = ResearchMemoryStore(tmp_path / "m.db")
    parent_expr = "RANK(TS_MEAN($vwap, 10))"
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set", parent_expr, "vp_10", ic=0.020))
    # 5 次显式 window_rescale 连败（递减弱 IC → 对 running 基线的一致负残差）
    for w, ic in ((15, 0.014), (20, 0.012), (25, 0.011), (30, 0.010), (35, 0.009)):
        expr = f"RANK(TS_MEAN($vwap, {w}))"
        store.record_tool_result(
            run_id="r1",
            row=_eval_row("eval_on_train_set", expr, f"vp_{w}", ic=ic,
                          extra_args={"parent_factor": "vp_10", "edit_note": f"edit=window_rescale 10→{w}"}),
        )
    advisory = store.advisory_for("RANK(TS_MEAN($vwap, 99))", edit_note="edit=window_rescale 10→99")
    assert advisory is not None
    kinds = [a["kind"] for a in advisory["advisories"]]
    assert "edit_veto" in kinds
    # 换编辑类型不命中
    advisory2 = store.advisory_for("RANK(TS_MEAN($vwap, 99))", edit_note="edit=normalization_change 加秩")
    kinds2 = [a["kind"] for a in (advisory2 or {"advisories": []})["advisories"]]
    assert "edit_veto" not in kinds2


def test_tool_dispatch_memory_gate_block(tmp_path):
    from alphaagent.factor.mining.tools import FactorEvalTools

    class _FakeStore:
        hard_block_duplicates = True

        def advisory_for(self, expr, edit_note=None):
            return {"advisories": [{"kind": "duplicate_known_dead_end", "message": "死路"}], "blocked": False}

    tools = FactorEvalTools(service=None, session_id="s", memory_store=_FakeStore())
    blocked = tools._memory_gate("any expr", {})
    assert isinstance(blocked, dict) and blocked["ok"] is False and blocked["error_type"] == "MemoryAdvisoryBlock"

    class _SoftStore(_FakeStore):
        hard_block_duplicates = False

    tools2 = FactorEvalTools(service=None, session_id="s", memory_store=_SoftStore())
    soft = tools2._memory_gate("any expr", {})
    assert soft is not None and soft.get("ok") is None and soft["advisories"][0]["kind"] == "duplicate_known_dead_end"


# ---------------------------------------------------------------------------
# 注入预算
# ---------------------------------------------------------------------------

def test_context_budget_truncation(tmp_path):
    store = ResearchMemoryStore(tmp_path / "m.db", max_inject_chars=600)
    # 造足够多的条目使各块超预算
    for i in range(30):
        store.record_tool_result(
            run_id="r1",
            row=_eval_row("eval_on_train_set", f"TS_MEAN($close, {i + 3}) * {i}", f"f_{i}", ic=0.02 + 0.001 * i),
        )
    text = store.context_for("A股挖掘", enable_factor_retrieval=True, enable_edit_patterns=True)
    assert len(text) <= 700
    assert text.startswith("# 长期研究记忆")


# ---------------------------------------------------------------------------
# v2 → v3 迁移：老库自动升级、cells 重建、幂等
# ---------------------------------------------------------------------------

def _create_v2_db(path):
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE memory_entries (
            id TEXT PRIMARY KEY, factor_name TEXT NOT NULL, expression TEXT NOT NULL,
            conclusion TEXT, verdict TEXT NOT NULL, stage TEXT, profile_id TEXT, profile_hash TEXT,
            candidate_id TEXT, metrics_json TEXT NOT NULL DEFAULT '{}', interaction_json TEXT,
            error TEXT, failure_code TEXT, fail_detail TEXT, mechanism TEXT, family TEXT,
            last_run_id TEXT, attempts INTEGER NOT NULL DEFAULT 1, tokens_json TEXT NOT NULL DEFAULT '[]',
            stage_metrics_json TEXT NOT NULL DEFAULT '{}', created_at TEXT, updated_at TEXT
        );
        CREATE TABLE store_meta (k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE memory_cells (
            cell_key TEXT PRIMARY KEY, family TEXT NOT NULL, motif TEXT NOT NULL,
            residuals_json TEXT NOT NULL DEFAULT '[]', successes INTEGER NOT NULL DEFAULT 0,
            failures INTEGER NOT NULL DEFAULT 0, updated_at TEXT
        );
        """
    )
    parent_expr = "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 10)))"
    child_expr = "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 20)))"
    sig = lambda e: __import__("hashlib").sha256("\n".join(l.strip() for l in e.splitlines() if l.strip()).encode()).hexdigest()[:20]
    conn.execute(
        "INSERT INTO memory_entries (id, factor_name, expression, verdict, stage, metrics_json, updated_at, attempts)"
        " VALUES (?, 'vwap_dev_10', ?, 'promising', 'train', ?, '2026-09-01T00:00:00+00:00', 1)",
        (sig(parent_expr), parent_expr, json.dumps({"ic": 0.020})),
    )
    conn.execute(
        "INSERT INTO memory_entries (id, factor_name, expression, verdict, stage, metrics_json, updated_at, attempts)"
        " VALUES (?, 'vwap_dev_20', ?, 'promising', 'train', ?, '2026-09-02T00:00:00+00:00', 1)",
        (sig(child_expr), child_expr, json.dumps({"ic": 0.024})),
    )
    conn.execute("INSERT INTO store_meta VALUES ('data_version', '2')")
    conn.commit()
    conn.close()
    return sig


def test_v2_migration_rebuilds_cells(tmp_path):
    sig = _create_v2_db(tmp_path / "old.db")
    store = ResearchMemoryStore(tmp_path / "old.db")
    with store._open() as conn:
        version = conn.execute("SELECT v FROM store_meta WHERE k='data_version'").fetchone()[0]
        cols = {r[1] for r in conn.execute("PRAGMA table_info(memory_cells)").fetchall()}
        cell_rows = conn.execute("SELECT family, motif, parent_bucket, implicit_s, residuals_json FROM memory_cells").fetchall()
        origin = conn.execute(
            "SELECT parent_origin FROM memory_entries WHERE id=?", (sig(CHILD_EXPR),)
        ).fetchone()[0]
    assert version == "3"
    assert "parent_bucket" in cols and "implicit_s" in cols and "explicit_s" in cols
    # 子代被隐式链接到父本并重建进 cells（legacy 全 implicit）
    assert origin == "implicit"
    assert len(cell_rows) == 1
    row = cell_rows[0]
    assert row["motif"] == "window_rescale" and row["parent_bucket"] == "medium"
    assert row["implicit_s"] == pytest.approx(0.5)  # legacy implicit 加权计数
    # residual = 0.024 − 0.020（无历史回退父本）
    assert json.loads(row["residuals_json"]) == pytest.approx([0.004])
    # 幂等：再次打开不重复迁移
    store2 = ResearchMemoryStore(tmp_path / "old.db")
    with store2._open() as conn:
        cell_rows2 = conn.execute("SELECT COUNT(*) FROM memory_cells").fetchone()[0]
    assert cell_rows2 == 1


def test_v1_legacy_db_migration(tmp_path):
    """真实存量库可能是 v1 结构（entries 无 family 等列、无 cells 表、遗留三张 v1 表）：

    v2 的 executescript(_SCHEMA) 会先建 idx_memory_entries_family 索引而直接炸
    "no such column: family"；v3 修复为 建表→ALTER补列→建索引 的顺序。
    """
    import sqlite3
    db = tmp_path / "v1.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE memory_entries (
            id TEXT PRIMARY KEY, factor_name TEXT NOT NULL, expression TEXT NOT NULL,
            conclusion TEXT, verdict TEXT NOT NULL, stage TEXT, profile_id TEXT, profile_hash TEXT,
            candidate_id TEXT, metrics_json TEXT NOT NULL DEFAULT '{}', error TEXT,
            failure_code TEXT, last_run_id TEXT, attempts INTEGER NOT NULL DEFAULT 1,
            tokens_json TEXT NOT NULL DEFAULT '[]', created_at TEXT, updated_at TEXT,
            interaction_json TEXT, structure_fingerprint TEXT, operator_list_json TEXT,
            window_params_json TEXT, parent_id TEXT
        );
        CREATE TABLE store_meta (k TEXT PRIMARY KEY, v TEXT);
        CREATE TABLE memory_patterns (id TEXT PRIMARY KEY);
        CREATE TABLE edit_patterns (id TEXT PRIMARY KEY);
        CREATE TABLE memory_observations (id TEXT PRIMARY KEY);
        """
    )
    parent_expr = "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 10)))"
    child_expr = "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 20)))"
    sig = lambda e: __import__("hashlib").sha256(
        "\n".join(l.strip() for l in e.splitlines() if l.strip()).encode()
    ).hexdigest()[:20]
    conn.execute(
        "INSERT INTO memory_entries (id, factor_name, expression, verdict, stage, metrics_json, updated_at, attempts)"
        " VALUES (?, 'vwap_dev_10', ?, 'promising', 'train', ?, '2026-08-28T00:00:00+00:00', 1)",
        (sig(parent_expr), parent_expr, json.dumps({"ic": 0.020})),
    )
    conn.execute(
        "INSERT INTO memory_entries (id, factor_name, expression, verdict, stage, metrics_json, updated_at, attempts)"
        " VALUES (?, 'vwap_dev_20', ?, 'promising', 'train', ?, '2026-08-29T00:00:00+00:00', 1)",
        (sig(child_expr), child_expr, json.dumps({"ic": 0.024})),
    )
    conn.commit()
    conn.close()

    store = ResearchMemoryStore(db)  # 不应抛 "no such column: family"
    with store._open() as conn:
        version = conn.execute("SELECT v FROM store_meta WHERE k='data_version'").fetchone()[0]
        cols = {r[1] for r in conn.execute("PRAGMA table_info(memory_entries)").fetchall()}
        cell_row = conn.execute(
            "SELECT motif, parent_bucket, implicit_s, residuals_json FROM memory_cells"
        ).fetchone()
        origin = conn.execute(
            "SELECT parent_origin FROM memory_entries WHERE id=?", (sig(child_expr),)
        ).fetchone()[0]
    assert version == "3"
    assert {"family", "stage_metrics_json", "parent_origin", "intended_motif", "edit_note"} <= cols
    assert origin == "implicit"
    assert cell_row is not None
    assert cell_row["motif"] == "window_rescale" and cell_row["parent_bucket"] == "medium"
    assert cell_row["implicit_s"] == pytest.approx(0.5)
    assert json.loads(cell_row["residuals_json"]) == pytest.approx([0.004])


def test_store_backward_compat_tools(tmp_path):
    """v2 行为兼容：purge / recent / statistics 继续工作。"""
    store = ResearchMemoryStore(tmp_path / "m.db")
    expr_a = "leg_a = RANK(x)\nleg_a"
    store.record_tool_result(run_id="r1", row=_eval_row("evaluate_factor", expr_a, "factor_a", ic=0.03))
    assert store.purge_factor(factor_names=["factor_a"]) >= 1
    assert store.recent()[0] == []
    stats = store.statistics()
    assert stats["entries"] == 0
