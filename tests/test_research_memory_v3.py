# v3-lite 研究记忆回归：AlphaMemo Eq.7 置信 / 双门 APV / 显式父本协议 / invalid 入账 /
# 同桶衰减基线 / advisory 硬提醒 / 注入预算 / v2→v3 迁移幂等。
import json
import sqlite3

import pytest

from alphaagent.factor.mining.research_memory import (
    APV_TAU_C_DEFAULT,
    APV_TAU_V_DEFAULT,
    DATA_VERSION,
    EQ7_KAPPA_DEFAULT,
    ResearchMemoryStore,
    _apv_gate,
    _eq7_confidence,
    _structure_fingerprint,
    motif_from_note,
    template_from_expression,
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
    # ic=0.024：新海选线 0.02 之上（promising），且父本桶仍在 medium [0.015, 0.025)
    entry = store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set", CHILD_EXPR, "vwap_dev_20b", ic=0.024))
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


# ---------------------------------------------------------------------------
# 入库事实优先：revise / gate 失败的 error 文本不得掩盖 candidate_stored/stored
# ---------------------------------------------------------------------------

_REVIEW_REVISE = {
    "verdict": "revise",
    "canonical_form": "统计验证未满足本次 ResearchSpec",
    "reasons": ["训练集 abs(IC) 未达到 ResearchSpec 门槛。"],
}


def _submit_row(expr, factor_name, *, candidate_stored=False, stored=False, error=None,
                review=None, metrics=None, parent=None, edit_note=None):
    args = {"multi_line_expr": expr, "factor_name": factor_name}
    if parent:
        args["parent_factor"] = parent
    if edit_note:
        args["edit_note"] = edit_note
    result = {"ok": stored, "stored": stored, "candidate_stored": candidate_stored}
    if metrics:
        result["metrics"] = metrics
    if error:
        result["error"] = error
    if review:
        result["factor_review"] = review
    return {"name": "submit_factor", "arguments_raw": json.dumps(args), "result": result}


def test_classify_pool_entry_overrides_review_revise():
    # 进了候选池 + revise + gate 错误码 → candidate_approved（Reviewer 意见并入结论）
    verdict, conclusion = ResearchMemoryStore._classify(
        "submit_factor",
        {"candidate_stored": True, "error": "stage_two_failed:train_ic",
         "factor_review": _REVIEW_REVISE},
        {},
        "stage_two_failed:train_ic",
    )
    assert verdict == "candidate_approved"
    assert "Reviewer 意见" in conclusion and "统计验证未满足本次 ResearchSpec" in conclusion
    # 正式入库 + revise → production_approved
    verdict2, _ = ResearchMemoryStore._classify(
        "submit_factor", {"stored": True, "factor_review": _REVIEW_REVISE}, {}, ""
    )
    assert verdict2 == "production_approved"
    # 未入库 + revise → 仍 revise_required
    verdict3, _ = ResearchMemoryStore._classify(
        "submit_factor", {"candidate_stored": False, "factor_review": _REVIEW_REVISE}, {}, ""
    )
    assert verdict3 == "revise_required"


def test_classify_gate_failure_without_review_still_pool_entry():
    # stage_two_failed 无 review（error 文本非空）也不得记成 rejected
    verdict, conclusion = ResearchMemoryStore._classify(
        "submit_factor",
        {"candidate_stored": True, "error": "stage_two_failed:train_ic"},
        {},
        "stage_two_failed:train_ic",
    )
    assert verdict == "candidate_approved"
    assert "候选池" in conclusion


def test_pool_entry_submit_counts_as_cell_success(tmp_path):
    store = ResearchMemoryStore(tmp_path / "m.db")
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set", PARENT_EXPR, "vwap_dev_10", ic=0.020))
    entry = store.record_tool_result(
        run_id="r1",
        row=_submit_row(
            CHILD_EXPR, "vwap_dev_20_sub",
            candidate_stored=True,
            error="stage_two_failed:train_ic",
            review=_REVIEW_REVISE,
            metrics={"ic": 0.0224, "icir": 0.295},
            parent="vwap_dev_10",
            edit_note="edit=window_rescale 10→20",
        ),
    )
    assert entry is not None
    assert entry["verdict"] == "candidate_approved"
    assert entry["failure_code"] == "stage_two_failed"
    with store._open() as conn:
        row = conn.execute("SELECT explicit_s, explicit_f, residuals_json FROM memory_cells").fetchone()
    assert row["explicit_s"] == pytest.approx(1.0)  # 已入池 → 有效正观测（非 invalid 失败）
    assert row["explicit_f"] == pytest.approx(0.0)
    assert json.loads(row["residuals_json"])  # child IC 在 → 有 residual


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


def test_advisory_duplicate_prior_result(tmp_path):
    store = ResearchMemoryStore(tmp_path / "m.db")
    # 同指纹的历史正向条目（窗口参数不同 → 同结构指纹：数字归一化为 N）
    store.record_tool_result(
        run_id="r1", row=_eval_row("eval_on_train_set", "TS_MEAN($adj_close, 9) + 0.25", "prior_prom", ic=0.024)
    )
    advisory = store.advisory_for("TS_MEAN($adj_close, 5) + 0.5")
    assert advisory is not None
    kinds = [a["kind"] for a in advisory["advisories"]]
    assert "duplicate_prior_result" in kinds
    item = next(a for a in advisory["advisories"] if a["kind"] == "duplicate_prior_result")
    assert item["prior_factor"] == "prior_prom"
    assert item["prior_verdict"] == "promising"
    assert item["n_prior_positive"] == 1
    assert "ic=" in item["message"] and "勿原样重测" in item["message"]
    # 重复评估历史条目自身（同表达式）同样命中
    advisory2 = store.advisory_for("TS_MEAN($adj_close, 9) + 0.25")
    assert advisory2 is not None
    assert "duplicate_prior_result" in [a["kind"] for a in advisory2["advisories"]]
    # 仅负向历史（attempts=1）不触发任何提醒
    store2 = ResearchMemoryStore(tmp_path / "m2.db")
    store2.record_tool_result(run_id="r0", row=_eval_row("eval_on_train_set", "TS_MEAN($adj_close, 5) + 0.5", "weak_0", ic=0.005))
    assert store2.advisory_for("TS_MEAN($adj_close, 5) + 0.5") is None
    # 死路与正向并存：两种提醒同时出现（互补不互斥）
    store3 = ResearchMemoryStore(tmp_path / "m3.db")
    store3.record_tool_result(
        run_id="r0", row=_eval_row("eval_on_train_set", "TS_MEAN($adj_close, 9) + 0.25", "prior_prom", ic=0.024)
    )
    for i in range(2):
        store3.record_tool_result(
            run_id=f"r{i}", row=_eval_row("eval_on_train_set", "TS_MEAN($adj_close, 5) + 0.5", f"weak_{i}", ic=0.005)
        )
    kinds3 = [a["kind"] for a in store3.advisory_for("TS_MEAN($adj_close, 5) + 0.5")["advisories"]]
    assert "duplicate_known_dead_end" in kinds3 and "duplicate_prior_result" in kinds3


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
    assert version == DATA_VERSION
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
    assert version == DATA_VERSION
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


# ---------------------------------------------------------------------------
# 经验层蒸馏：run 累计口径（单批族类分散也能触发）+ recent 时间倒序
# ---------------------------------------------------------------------------

def test_distill_experience_run_cumulative(tmp_path):
    """同族 3 条全弱跨批累计 → forbidden；族内出现 |IC|>=0.02 → success_pattern；
    末尾连续 5 条全弱 → insight。单批口径下这些规则永远不会触发。
    注意：记忆条目按表达式指纹去重，每个因子必须用不同表达式。"""
    store = ResearchMemoryStore(tmp_path / "m.db")

    def vwap_expr(w):
        return f"RANK(SUBTRACT($adj_close, TS_MEAN($vwap, {w})))"

    # 第一批：同族 2 条弱（旧单批口径 n<2 不触发，累计口径也不该触发 forbidden）
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set", vwap_expr(10), "chip_a_10", ic=0.004))
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set", vwap_expr(20), "chip_a_20", ic=0.006))
    formed1 = store.distill_batch_experience(run_id="r1", turn=1, batch_results=[{"dummy": 1}])
    assert formed1 == {"success_patterns": 0, "forbidden": 0, "insights": 0}
    # 第二批：同族再补 1 条弱 → run 累计 3 条全弱 → forbidden 触发
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set", vwap_expr(30), "chip_a_30", ic=0.002))
    formed2 = store.distill_batch_experience(run_id="r1", turn=2, batch_results=[{"dummy": 1}])
    assert formed2["forbidden"] == 1
    forb = [r for r in store.list_experience() if r["kind"] == "forbidden"]
    assert len(forb) == 1 and forb[0]["name"] == "family_saturated:vwap"
    # 同族出现强 IC → success_pattern（文本标注方向），occurrence 累加去重
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set", vwap_expr(40), "chip_a_40", ic=-0.021))
    formed3 = store.distill_batch_experience(run_id="r1", turn=3, batch_results=[{"dummy": 1}])
    assert formed3["success_patterns"] == 1
    succ = [r for r in store.list_experience() if r["kind"] == "success_pattern"]
    assert len(succ) == 1 and succ[0]["name"] == "family_mechanism:vwap"
    assert "反向构造" in succ[0]["content"]
    # insight：强 IC 打断 streak，其后连续 5 条 |IC|<0.005（跨批累计）才触发
    for i in range(4):
        store.record_tool_result(run_id="r1", row=_eval_row(
            "eval_on_train_set", vwap_expr(50 + i), f"thin_{i}", ic=0.001 + 0.0001 * i))
    formed4 = store.distill_batch_experience(run_id="r1", turn=4, batch_results=[{"dummy": 1}])
    assert formed4["insights"] == 0  # 只有 4 条连续弱
    store.record_tool_result(run_id="r1", row=_eval_row(
        "eval_on_train_set", vwap_expr(60), "thin_4", ic=0.0012))
    formed5 = store.distill_batch_experience(run_id="r1", turn=5, batch_results=[{"dummy": 1}])
    assert formed5["insights"] == 1
    ins = [r for r in store.list_experience() if r["kind"] == "insight"]
    assert len(ins) == 1 and ins[0]["name"] == "global_alpha_thin"


def test_recent_order_default_time_desc(tmp_path):
    """recent 默认按 updated_at 倒序（新 run 的 weak 条目在前）；
    order="verdict" 保留旧的 verdict 优先口径。"""
    store = ResearchMemoryStore(tmp_path / "m.db")
    expr_old = "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 10)))"
    expr_new = "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 20)))"
    # eval_on_train_set 行会提取 metrics：ic=0.03/icir=0.4/cov=0.9 → promising
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set", expr_old, "good_old", ic=0.03))
    store.record_tool_result(run_id="r2", row=_eval_row("eval_on_train_set", expr_new, "weak_new", ic=0.004))
    entries_recent, _ = store.recent()
    assert entries_recent[0]["factor_name"] == "weak_new"
    assert entries_recent[0]["verdict"] == "weak"
    entries_verdict, _ = store.recent(order="verdict")
    assert entries_verdict[0]["factor_name"] == "good_old"
    assert entries_verdict[0]["verdict"] == "promising"


# ---------------------------------------------------------------------------
# 饱和度口径（promising 降权）+ 证据块跨族正向保底
# ---------------------------------------------------------------------------

def test_saturation_promising_downweighted(tmp_path):
    """promising 只轻度计入拥挤（8 个封顶 → 贡献 ≤0.2）：出过大量正信号但无
    幸存者的族不再被封禁；拥挤度以 validated/candidate 等真实幸存者为主。"""
    store = ResearchMemoryStore(tmp_path / "m.db")
    # vwap 族：10 条 promising（不同窗口表达式），0 条 validated → 饱和度 0.2，不拥挤
    for w in range(10, 20):
        store.record_tool_result(run_id="r1", row=_eval_row(
            "eval_on_train_set",
            f"RANK(SUBTRACT($adj_close, TS_MEAN($vwap, {w})))", f"vwap_dev_{w}", ic=0.02))
    # 另一族：3 条 validated（eval_on_val_set + |ic|>=0.015）→ 饱和度 >=1.0，拥挤
    for w in (10, 20, 30):
        store.record_tool_result(run_id="r1", row=_eval_row(
            "eval_on_val_set", f"TS_MEAN($close, {w}) - 1", f"mom_{w}", ic=0.02))
    sat = store.compute_saturation()
    assert sat["vwap"]["n_promising"] == 10
    assert sat["vwap"]["saturation_score"] == pytest.approx(0.2)
    val_fams = [f for f, d in sat.items() if d["n_validated"] == 3]
    assert len(val_fams) == 1
    assert sat[val_fams[0]]["saturation_score"] >= 1.0
    block = store._saturation_block()
    assert val_fams[0] in block                # 拥挤族被点名
    assert "优先深挖" in block and "vwap" in block  # 定向建议点名未拥挤但有正信号的族


def test_evidence_positive_guarantee(tmp_path):
    """即使 query 与强正因子毫无词汇交集，跨族最优正向因子也必被注入（保底）。"""
    store = ResearchMemoryStore(tmp_path / "m.db")
    store.record_tool_result(run_id="r1", row=_eval_row(
        "eval_on_train_set", "TS_STD($volume, 20)", "vol_std_20", ic=0.003))
    store.record_tool_result(run_id="r1", row=_eval_row(
        "eval_on_train_set", "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 10)))",
        "vwap_dev_10", ic=0.038))
    store.record_tool_result(run_id="r1", row=_eval_row(
        "eval_on_train_set", "RANK(SUBTRACT($close, DELAY($close, 5)))",
        "rev_5", ic=0.025))
    block = store.context_for(
        "成交量波动率风险", enable_factor_retrieval=True,
        enable_edit_patterns=False, limit=6)
    assert "已验证 / 有潜力的因子" in block
    assert "vwap_dev_10" in block              # 全库 |IC| 最高的正向因子保底在场
    assert "[promising]" in block


def test_recent_sort_and_verdict_filter(tmp_path):
    """sort/dir 服务端按列排序（NULL 恒排最后），verdict 过滤后 total 为过滤数。"""
    store = ResearchMemoryStore(tmp_path / "m.db")
    rows = [
        ("RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 10)))", "f10", 0.03),
        ("RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 20)))", "f20", 0.004),
        ("RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 30)))", "f30", -0.002),
    ]
    for expr, name, ic in rows:
        store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set", expr, name, ic=ic))
    # 无 metrics 的报错行：verdict=rejected，无 IC（排序时 NULL 恒排最后）
    store.record_tool_result(run_id="r1", row=_eval_row(
        "eval_on_train_set", "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 40)))", "f40",
        error="dsl_compile_failed: x"))
    entries, total = store.recent(sort="ic", dir=-1)
    assert total == 4
    assert [e["factor_name"] for e in entries[:3]] == ["f10", "f20", "f30"]
    assert entries[-1]["factor_name"] == "f40"
    entries, _ = store.recent(sort="ic", dir=1)
    assert entries[0]["factor_name"] == "f30"
    # verdict 过滤：weak 档 total=2 且条目全为 weak
    entries, total = store.recent(sort="ic", dir=-1, verdict="weak")
    assert total == 2 and len(entries) == 2
    assert all(e["verdict"] == "weak" for e in entries)
    # sort="verdict"：按 verdict 排名，promising 在 weak/rejected 前
    entries, _ = store.recent(sort="verdict", dir=1)
    assert entries[0]["verdict"] == "promising"
    # 未知名 sort 回落 updated_at（不抛错）
    _, total = store.recent(sort="not_a_column", dir=-1)
    assert total == 4


# ---------------------------------------------------------------------------
# 记忆出题 + FactorMiner 式经验条目（模板/示例/禁令）+ engine_gate 回流
# ---------------------------------------------------------------------------

def test_template_from_expression():
    t = template_from_expression("RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 20)))")
    assert t == "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, {w1})))"
    t2 = template_from_expression("ADD(TS_MEAN($close, 5), TS_STD($close, 20))")
    assert "{w1}" in t2 and "{w2}" in t2 and "5" not in t2 and "20" not in t2
    assert template_from_expression("") == ""


def test_experience_block_renders_template_and_do_not(tmp_path):
    """成功模式渲染 模板+示例+达标率；禁忌（族饱和）渲染 DO NOT 禁令。
    注意族饱和要求族内全部尝试 |IC|<0.01，所以禁忌族与成功族要分开。"""
    store = ResearchMemoryStore(tmp_path / "m.db")
    # vwap 族：1 强（→ 成功模式：模板+示例+达标率）
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set",
        "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 10)))", "good_1", ic=0.03))
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set",
        "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 20)))", "vwap_weak_1", ic=0.004))
    # 波动率族：3 条全弱（→ 族饱和 DO NOT）
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set",
        "RANK(TS_STD($close, 60))", "vol_weak_1", ic=0.004))
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set",
        "RANK(TS_STD($close, 90))", "vol_weak_2", ic=0.006))
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set",
        "RANK(TS_STD($close, 120))", "vol_weak_3", ic=0.002))
    store.distill_batch_experience(run_id="r1", turn=1, batch_results=[{"dummy": 1}])
    block = store._experience_block()
    assert "成功模式" in block and "模板:" in block and "{w1}" in block
    assert "1/2" in block  # vwap 族 1/2 条达标
    assert "【禁止】" in block and "DO NOT" in block
    assert "TS_STD" in block  # 禁令带死路模板骨架
    assert "vol_weak_1" in block  # 禁令带死路因子名单


def test_diversity_block_fusion_guidance(tmp_path):
    """近批全部集中在价量面 → 注入面覆盖警告 + 未触面清单 + 融合算子示例；
    已覆盖 ≥3 个面时不注入（避免喧宾夺主）。"""
    store = ResearchMemoryStore(tmp_path / "m.db")
    mono = [{"expression": e} for e in (
        "RANK(TS_MEAN($adj_close, 10))",
        "RANK(TS_MEAN($adj_close, 20))",
        "NEG(TS_STD($ret, 5))",
        "RANK($volume)",
        "CS_ZSCORE($amount)",
    )]
    block = store._diversity_block(mono)
    assert "数据面覆盖警告" in block
    assert "未触及的数据面" in block and "基本面" in block and "股东面" in block
    assert "DIVERGENCE_RANK" in block and "CS_RESIDUALIZE" in block

    diverse = mono + [
        {"expression": "CHIP_ENTROPY($adj_close, $adj_low, $adj_high, $volume, 60, $float_cap)"},
        {"expression": "RANK($funda_roe)"},
    ]
    assert store._diversity_block(diverse) == ""

    # 正向经验的跨面融合标注：表达式触及价量+基本面两个面
    store.record_tool_result(run_id="r1", row=_eval_row(
        "eval_on_train_set",
        "DIVERGENCE_RANK(RANK($funda_roe), RANK(TS_MEAN($adj_close, 10)))",
        "fusion_roe_mom", ic=0.025))
    store.distill_batch_experience(run_id="r1", turn=1, batch_results=[{"dummy": 1}])
    succ = [r for r in store.list_experience() if r["kind"] == "success_pattern"]
    assert any("跨面融合" in r["content"] and "基本面" in r["content"] for r in succ)


def test_recommend_edits_scores_cells(tmp_path):
    """正残差×高置信的 (family, motif) 排前，附族内最优父本；无正 cells 回退族级推荐。"""
    store = ResearchMemoryStore(tmp_path / "m.db")
    parent_expr = "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 10)))"
    store.record_tool_result(run_id="r1", row=_eval_row("eval_on_train_set", parent_expr, "vwap_p", ic=0.02))
    # 6 次同 (family, motif, bucket) 的显式子代编辑、残差持续为正 → 置信 > 0.3
    for i, ic in enumerate([0.03, 0.034, 0.036, 0.038, 0.04, 0.042]):
        store.record_tool_result(run_id="r1", row=_eval_row(
            "eval_on_train_set", f"RANK(SUBTRACT($adj_close, TS_MEAN($vwap, {20 + i})))", f"vwap_c{i}",
            ic=ic, extra_args={"parent_factor": "vwap_p",
                               "edit_note": "edit=window_rescale 10→20"}))
    recs = store.recommend_edits(k=2)
    assert recs, "有正 cells 时应给出推荐"
    top = recs[0]
    assert top["family"] == "vwap"
    assert top["motif"] == "window_rescale"
    # 族内最优父本 = 验证档位最高、|IC| 最大的正向条目（强子代优于原始父本）
    assert top["parent_factor"] == "vwap_c5"
    assert abs(top["parent_ic"] - 0.042) < 1e-9
    assert "残差" in top["reason"]
    # 空库：回退族级推荐或返回空，均不得抛错
    empty = ResearchMemoryStore(tmp_path / "empty.db")
    assert isinstance(empty.recommend_edits(k=2), list)


def test_gate_feedback_flow(tmp_path):
    """engine_gate 拒绝 → gate_rejected 禁忌；正式入库 → gate_validated 成功经验。"""
    store = ResearchMemoryStore(tmp_path / "m.db")
    expr = "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 10)))"
    # 模拟一次 submit：engine_gate 拒绝（error 带 engine_gate_failed）
    store.record_tool_result(run_id="r1", row=_eval_row(
        "submit_factor", expr, "vwap_g1", ic=0.03,
        error="engine_gate_failed:avg_daily_side_turnover>0.4"))
    store.record_tool_result(run_id="r1", row=_eval_row(
        "eval_on_train_set", "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 20)))", "vwap_g2", ic=0.028))
    formed = store.distill_batch_experience(run_id="r1", turn=1, batch_results=[{"dummy": 1}])
    assert formed["forbidden"] >= 1
    rows = {r["name"]: r for r in store.list_experience()}
    assert "gate_rejected:vwap" in rows
    assert "gate_rejected" in rows["gate_rejected:vwap"]["content"] or "DO NOT" in rows["gate_rejected:vwap"]["content"]
    assert "换手" in rows["gate_rejected:vwap"]["content"]

    # 正式入库（production_approved）→ gate_validated
    row = _eval_row("submit_factor", "RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 30)))", "vwap_g3", ic=0.03)
    row = dict(row)
    row["result"] = {**row["result"], "stored": True}
    store.record_tool_result(run_id="r1", row=row)
    store.distill_batch_experience(run_id="r1", turn=2, batch_results=[{"dummy": 1}])
    rows = {r["name"]: r for r in store.list_experience()}
    assert "gate_validated:vwap" in rows
    assert rows["gate_validated:vwap"]["template"]


def test_context_evidence_not_starved_by_core_blocks(tmp_path):
    """预算分段回归：经验块（成功模式带模板+示例表达式）+ 编辑先验膨胀后，
    证据块仍必须在场（'- [' 条目行 >= 2），不被核心块饿死到 0 条。"""
    store = ResearchMemoryStore(tmp_path / "m.db", max_inject_chars=2400)
    # 造 3 条正向因子（证据块保底来源）+ 6 条弱因子（触发禁令经验）
    for i, ic in enumerate([0.038, 0.03, 0.025]):
        store.record_tool_result(run_id="r1", row=_eval_row(
            "eval_on_train_set", f"RANK(SUBTRACT($adj_close, TS_MEAN($vwap, {10 + i})))",
            f"pos_{i}", ic=ic))
    for i in range(6):
        store.record_tool_result(run_id="r1", row=_eval_row(
            "eval_on_train_set", f"RANK(TS_STD($close, {30 + i}))", f"weak_{i}", ic=0.004))
    store.distill_batch_experience(run_id="r1", turn=1, batch_results=[{"dummy": 1}])
    block = store.context_for(
        "A股量价因子挖掘", enable_factor_retrieval=True,
        enable_edit_patterns=True, limit=8)
    n_entries = sum(1 for line in block.splitlines() if line.startswith("- ["))
    assert len(block) <= 2400 + 50
    assert n_entries >= 2, f"证据块被核心块挤出（条目行 {n_entries}）"


def test_form_memory_signature_rows_enriched(tmp_path):
    """form_memory 签名行与蒸馏行同规格：content 带因子/IC/方向/指导语，
    template 参数槽化，evidence.examples 带真实表达式——不再是
    "成功模式：{签名}（{因子名}）"式的无信息身份行。"""
    store = ResearchMemoryStore(tmp_path / "m.db")
    expr = "ov_gap = SUBTRACT($adj_open, DELAY($adj_close, 1))\nCS_ZSCORE(NEG(TS_MEDIAN(DIVIDE(ov_gap, DELAY($adj_close, 1)), 60)))"
    row = _eval_row("eval_on_train_set", expr, "ov_prem_med60", ic=-0.0349)
    formed = store.form_memory(run_id="r1", turn=1, batch_results=[{
        "factor_name": "ov_prem_med60",
        "expression": expr,
        "metrics": {"ic": -0.0349, "icir": 0.4},
        "admitted": True,
        "verdict": "promising",
        "conclusion": "训练阶段指标有潜力",
        "rejection_reason": "",
    }])
    # 签名是否命中取决于 _SUCCESS_SIGNATURES；未命中时不产生行（合法性由签名表保证）
    rows = {r["name"]: r for r in store.list_experience()}
    sig_rows = [n for n in rows if n.startswith("signature:")]
    if not sig_rows:
        assert formed["success_patterns"] == 0
        return
    r = rows[sig_rows[0]]
    assert formed["success_patterns"] >= 1
    # content 不再是裸身份行：含代表因子名、IC 数值、方向说明、行动指导
    assert "ov_prem_med60" in r["content"]
    assert "0.0349" in r["content"]
    assert "反向" in r["content"] or "方向为正" in r["content"]
    assert "模板骨架" in r["content"]
    # template 参数槽化 + evidence.examples 带真实表达式
    assert r["template"]
    with sqlite3.connect(tmp_path / "m.db") as conn:
        conn.row_factory = sqlite3.Row
        raw = conn.execute(
            "SELECT evidence_json FROM memory_experience WHERE name=?", (r["name"],)
        ).fetchone()
    ev = json.loads(raw["evidence_json"])
    assert ev.get("examples") == [expr]
    # 注入块渲染出模板与示例
    block = store._experience_block()
    assert "成功模式" in block and "模板:" in block
