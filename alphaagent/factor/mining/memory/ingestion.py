# -*- coding: utf-8 -*-
"""研究证据入库与父子关系解析。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from .calibration import _parent_bucket
from .constants import INVALID_WEIGHT, PARENT_ORIGIN_WEIGHT, POSITIVE_VERDICTS
from .diagnostics import _extract_fail_detail, _failure_code, _now, _parse_args, _rebuild_conclusion, _safe_float
from ..runlog import log_step
from .expressions import (
    _structure_fingerprint,
    _tokens,
    classify_family_ex,
    expression_features,
    expression_ops,
    expression_windows,
    extract_edit_motif,
    motif_from_note,
)


class IngestionMixin:
    """研究证据入库：显式/隐式父本解析、cells 更新和编辑观测入账。"""

    # ── 入库主流程 ──

    def record_tool_result(
        self,
        *,
        run_id: str,
        row: dict[str, Any],
        run_freq_context: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        name = str(row.get("name") or "")
        if name not in {"evaluate_factor", "eval_on_train_set", "eval_on_val_set", "submit_factor"}:
            return None
        args = _parse_args(row.get("arguments_raw"))
        result = row.get("result") if isinstance(row.get("result"), dict) else {}
        expression = str(args.get("multi_line_expr") or "").strip()
        factor_name = str(args.get("factor_name") or result.get("factor_name") or "expr").strip()
        if not expression:
            return None

        profile_metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else profile_metrics.get("cross_sectional_core", {})
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else summary
        if name == "evaluate_factor":
            metrics = summary
        # 提取嵌套回测指标到扁平 metrics，供 _compact_metrics 捕获
        metrics = self._flatten_backtest_metrics(metrics, profile_metrics)
        error = str(result.get("error") or result.get("skipped_reason") or "")
        verdict, conclusion = self._classify(name, result, metrics, error)
        # A) 预测-对账：被证伪的预测追加进 conclusion（FTS 可检索），
        #    对账摘要随 observation.metrics 持久化。
        prediction_check = result.get("prediction_check") if isinstance(result.get("prediction_check"), dict) else None
        if prediction_check and str(prediction_check.get("verdict")) == "contradicted":
            reconcile_msg = str(prediction_check.get("message") or "")[:180]
            if reconcile_msg:
                conclusion = f"{conclusion}；预测对账:{reconcile_msg}"
        canonical_expression = "\n".join(line.strip() for line in expression.splitlines() if line.strip())
        signature = self.entry_signature(canonical_expression)
        failure_code = _failure_code(name, result, error, verdict)

        # Phase 1: 解析表达式结构
        struct = expression_features(expression)
        family, entry_facets = classify_family_ex(factor_name, expression)

        # v3-lite: 显式父本协议（edit_note / parent_factor）
        parent_factor = str(args.get("parent_factor") or "").strip() or None
        edit_note = str(args.get("edit_note") or "").strip() or None
        parent_origin: str | None = None
        intended_motif: str | None = None
        parent_id: str | None = None

        with self._open() as conn:
            previous_row = conn.execute(
                "SELECT attempts, created_at FROM memory_entries WHERE id = ?",
                (signature,),
            ).fetchone()
            parent_id = self._resolve_parent_conn(
                conn, expression, struct,
                parent_factor=parent_factor,
                exclude_id=signature,
            )
            if parent_id:
                parent_origin = "explicit" if parent_factor else "implicit"
                if edit_note:
                    intended_motif = motif_from_note(edit_note) or "other"
                elif not parent_factor:
                    # 隐式兜底：用结构编辑识别（10→20 窗口扩展）
                    parent_expr_row = conn.execute(
                        "SELECT expression FROM memory_entries WHERE id = ?", (parent_id,)
                    ).fetchone()
                    parent_expr = parent_expr_row["expression"] if parent_expr_row else ""
                    intended_motif = extract_edit_motif(parent_expr, expression)
                # 记录 edit_note 原文（可追溯）
                _set_edit_note(conn, signature, edit_note)

        previous_attempts = int(previous_row["attempts"]) if previous_row else 0
        metrics_compact = self._compact_metrics(metrics)
        # 调仓频率/研究档位随评估落库（run 启动即确定的门禁口径），使研究总结
        # 的频率列不依赖入库关联；与 prediction_check 同为 metrics 携带的元数据。
        if run_freq_context:
            freq_meta = {
                key: run_freq_context.get(key)
                for key in ("rebalance_freq", "research_mode", "freq_source")
                if run_freq_context.get(key)
            }
            if freq_meta:
                metrics_compact = {**metrics_compact, **freq_meta}
        if prediction_check:
            # 对账摘要入 observations（verdict + 预期/实际核心字段），全量 check 在评估结果里
            metrics_compact = {
                **metrics_compact,
                "prediction_check": {
                    "verdict": prediction_check.get("verdict"),
                    "expected": prediction_check.get("expected") or {},
                    "actual": prediction_check.get("actual") or {},
                },
            }
        observation = {
            "run_id": run_id,
            "at": _now(),
            "stage": result.get("split") or name.removeprefix("eval_on_").removesuffix("_set"),
            "verdict": verdict,
            "failure_code": failure_code,
            "metrics": metrics_compact,
            "interaction": _parse_args(args.get("interaction")),
        }
        entry = {
            "id": signature,
            "factor_name": factor_name,
            "expression": expression,
            "tokens": _tokens(factor_name, expression, conclusion),
            "verdict": verdict,
            "conclusion": conclusion,
            "stage": result.get("split") or name.removeprefix("eval_on_").removesuffix("_set"),
            "profile_id": result.get("profile", {}).get("profile_id") if isinstance(result.get("profile"), dict) else None,
            "profile_hash": result.get("profile_hash"),
            "candidate_id": result.get("candidate", {}).get("candidate_id") if isinstance(result.get("candidate"), dict) else None,
            "metrics": metrics_compact,
            "error": error[:500],
            "failure_code": failure_code,
            "fail_detail": _extract_fail_detail(name, result, error),
            "mechanism": (args.get("interaction") or {}).get("economic_mechanism") if isinstance(args.get("interaction"), dict) else None,
            "family": family,
            "facets_json": json.dumps(sorted(entry_facets), ensure_ascii=False) if entry_facets else None,
            "stage_metrics": {},
            "last_run_id": run_id,
            "attempts": previous_attempts + 1,
            "updated_at": _now(),
            "observations": [observation],
            "structure_fingerprint": struct.get("fingerprint"),
            "operator_list_json": json.dumps(struct.get("operators", []), ensure_ascii=False),
            "window_params_json": json.dumps(struct.get("window_params", {}), ensure_ascii=False),
            "parent_id": parent_id,
            "parent_origin": parent_origin,
            "intended_motif": intended_motif,
            "edit_note": edit_note,
        }

        if previous_row:
            entry["created_at"] = previous_row["created_at"] or entry["updated_at"]
            with self._open() as conn:
                old_rows = conn.execute(
                    """
                    SELECT * FROM memory_observations
                    WHERE entry_id = ?
                    ORDER BY observed_at
                    """,
                    (signature,),
                ).fetchall()
            old_observations = [
                {
                    "run_id": row["run_id"],
                    "at": row["observed_at"],
                    "stage": row["stage"],
                    "verdict": row["verdict"],
                    "failure_code": row["failure_code"],
                    "metrics": json.loads(row["metrics_json"] or "{}"),
                }
                for row in old_rows
            ]
            entry["observations"] = [*old_observations[-98:], observation]
        else:
            entry["created_at"] = entry["updated_at"]

        with self._open() as conn:
            self._write_entry(conn, entry)
            # v3-lite: 入账 cells（显式/隐式加权 + 同桶残差）
            self._update_cell(
                conn, entry, struct,
                parent_id=parent_id,
                parent_origin=parent_origin,
                intended_motif=intended_motif,
                verdict=verdict,
                error=error,
            )
        log_step(
            "memory.record",
            f"{factor_name} verdict={verdict}",
            family=family,
            motif=intended_motif,
            parent=parent_factor,
            fail_code=failure_code,
            attempts=entry["attempts"],
        )

        # Phase 2/4: 记录编辑模式（历史兼容）
        if parent_id and struct.get("fingerprint"):
            try:
                self._record_edit_pattern_from_parent(
                    parent_id=parent_id,
                    child_id=signature,
                    child_expression=expression,
                    child_struct=struct,
                    child_metrics=entry["metrics"],
                )
            except Exception:
                pass

        return entry

    # ── 父本解析 ──

    def _resolve_parent_conn(
        self,
        conn: sqlite3.Connection,
        expression: str,
        struct: dict[str, Any],
        *,
        parent_factor: str | None = None,
        exclude_id: str | None = None,
    ) -> str | None:
        """显式父本优先；无声明时用结构相似隐式兜底（v3 子串/双父检测）。"""
        if parent_factor:
            row = conn.execute(
                "SELECT id FROM memory_entries WHERE factor_name = ? ORDER BY updated_at DESC LIMIT 1",
                (parent_factor,),
            ).fetchone()
            if row and row["id"] != exclude_id:
                return row["id"]
        return self._find_implicit_parent_conn(conn, struct, exclude_id=exclude_id)

    @staticmethod
    def _find_implicit_parent_conn(
        conn: sqlite3.Connection,
        child_struct: dict[str, Any],
        *,
        exclude_id: str | None = None,
        limit: int = 100,
    ) -> str | None:
        """结构相似度隐式父本匹配（v3：算子 Jaccard + 窗口相似，>0.5 建链）。"""
        child_fp = child_struct.get("fingerprint")
        child_ops = set(child_struct.get("operators", []))
        child_wins = child_struct.get("window_params", {})
        params: list[Any] = []
        # 注意：结构指纹会把数字归一化为 N（窗口 10→20 指纹相同），
        # 因此不能用「指纹 != child_fp」排除候选——靠 exclude_id 排除自身即可。
        where_parts = ["structure_fingerprint IS NOT NULL"]
        if exclude_id:
            where_parts.append("id != ?")
            params.append(exclude_id)
        if exclude_id:
            where_parts.append("id != ?")
            params.append(exclude_id)
        where = " AND ".join(where_parts)
        rows = conn.execute(
            f"""
            SELECT id, structure_fingerprint, operator_list_json, window_params_json
            FROM memory_entries
            WHERE {where}
            ORDER BY updated_at DESC
            LIMIT ?
            """,
            (*params, int(limit)),
        ).fetchall()
        if not rows:
            return None
        best_id: str | None = None
        best_score: float = 0.0
        for r in rows:
            parent_ops = set(json.loads(r["operator_list_json"] or "[]"))
            parent_wins = json.loads(r["window_params_json"] or "{}")
            ops_union = child_ops | parent_ops
            ops_sim = len(child_ops & parent_ops) / len(ops_union) if ops_union else 0.0
            common_ops = set(child_wins.keys()) & set(parent_wins.keys())
            if common_ops:
                win_diffs = []
                for op in common_ops:
                    cv = child_wins.get(op, [])
                    pv = parent_wins.get(op, [])
                    if cv and pv:
                        for i in range(min(len(cv), len(pv))):
                            if pv[i] != cv[i]:
                                try:
                                    max_val = max(float(pv[i]), float(cv[i]))
                                    if max_val > 0:
                                        win_diffs.append(abs(float(cv[i]) - float(pv[i])) / max_val)
                                except (TypeError, ValueError):
                                    pass
                win_sim = 1.0 - (sum(win_diffs) / len(win_diffs)) if win_diffs else 1.0
            else:
                win_sim = 0.5
            score = 0.5 * ops_sim + 0.5 * win_sim
            if score > best_score:
                best_score = score
                best_id = r["id"]
        return best_id if best_score > 0.5 else None

    def _find_implicit_parent(
        self,
        conn: sqlite3.Connection,
        child_struct: dict[str, Any],
        *,
        exclude_id: str | None = None,
        limit: int = 100,
    ) -> str | None:
        """v2 兼容别名：同 _find_implicit_parent_conn。"""
        return self._find_implicit_parent_conn(conn, child_struct, exclude_id=exclude_id, limit=limit)

    # ── cells 更新 ──
    # 注意：`_update_cell` / `_same_bucket_baseline` 的唯一生效实现在 ExperienceMixin
    # （experience.py，MRO 中更靠前）。此处历史上曾有第二份带时间衰减的实现，
    # 因从未被调用已于 2026-08-30 删除——改 cells 入账逻辑只改 experience.py。

    @staticmethod
    def _weighted_counts(row: sqlite3.Row) -> tuple[float, float]:
        """从 cells 行计算加权成功/失败（列存加权值，直接相加）。

        各列已按起源权重入账（explicit 记 1.0、implicit/invalid 记 0.5），
        因此统计口径是 direct sum，不再二次加权。
        """
        s = float(row["explicit_s"] or 0) + float(row["implicit_s"] or 0)
        f = float(row["explicit_f"] or 0) + float(row["implicit_f"] or 0)
        return s, f

    # ── 编辑模式记录（v2 兼容） ──

    def _record_edit_pattern_from_parent(
        self,
        *,
        parent_id: str,
        child_id: str,
        child_expression: str,
        child_struct: dict[str, Any],
        child_metrics: dict[str, float],
    ) -> str | None:
        from .expressions import _identify_edit_type
        with self._open() as conn:
            parent_row = conn.execute(
                "SELECT expression, structure_fingerprint, operator_list_json, window_params_json, metrics_json FROM memory_entries WHERE id = ?",
                (parent_id,),
            ).fetchone()
            if not parent_row:
                return None
            parent_expression = parent_row["expression"]
            parent_struct = _parse_expression_structure_full(parent_expression)
            parent_metrics = json.loads(parent_row["metrics_json"] or "{}")
        edit = _identify_edit_type(parent_struct, child_struct)
        if edit is None:
            return None
        edit_type = edit["edit_type"]
        edit_detail = edit["detail"]
        parent_ic = _safe_float(parent_metrics.get("ic"))
        child_ic = _safe_float(child_metrics.get("ic"))
        parent_icir = _safe_float(parent_metrics.get("icir"))
        child_icir = _safe_float(child_metrics.get("icir"))
        delta_ic = child_ic - parent_ic if (parent_ic is not None and child_ic is not None) else None
        delta_icir = child_icir - parent_icir if (parent_icir is not None and child_icir is not None) else None
        IC_SUCCESS_THRESHOLD = 0.003
        if delta_ic is not None:
            verdict = "success" if delta_ic > IC_SUCCESS_THRESHOLD else ("failure" if delta_ic < -IC_SUCCESS_THRESHOLD else "neutral")
        else:
            verdict = "neutral"
        family = classify_family(str(child_struct.get("variables", [""])), child_expression)
        pattern_id = hashlib.sha256(
            f"{edit_type}|{parent_struct.get('fingerprint', '')}|{child_struct.get('fingerprint', '')}".encode("utf-8")
        ).hexdigest()[:20]
        now = _now()
        edit_detail_json = json.dumps(edit_detail, ensure_ascii=False, separators=(",", ":"))
        with self._open() as conn:
            existing = conn.execute(
                "SELECT total_uses, success_count, confidence, vetoed FROM edit_patterns WHERE id = ?",
                (pattern_id,),
            ).fetchone()
            if existing:
                total = int(existing["total_uses"]) + 1
                succ = int(existing["success_count"]) + (1 if verdict == "success" else 0)
                if verdict == "success":
                    confidence = min(0.95, succ / (total ** 0.8)) if total else 0.5
                else:
                    failures = total - succ
                    confidence = min(0.98, failures / (total ** 0.5)) if total else 0.5
                vetoed = 1 if (verdict != "success" and confidence > 0.85) else int(existing["vetoed"])
                conn.execute(
                    """
                    UPDATE edit_patterns
                    SET total_uses = ?, success_count = ?, confidence = ?, vetoed = ?,
                        child_ic = ?, delta_ic = ?, child_icir = ?, delta_icir = ?,
                        verdict = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (total, succ, round(confidence, 4), vetoed,
                     child_ic, delta_ic, child_icir, delta_icir,
                     verdict, now, pattern_id),
                )
            else:
                confidence = 0.5
                vetoed = 0
                total = 1
                succ = 1 if verdict == "success" else 0
                conn.execute(
                    """
                    INSERT INTO edit_patterns
                        (id, parent_expression, child_expression, parent_fingerprint, child_fingerprint,
                         edit_type, edit_detail_json,
                         parent_ic, child_ic, delta_ic, parent_icir, child_icir, delta_icir,
                         verdict, confidence, total_uses, success_count, vetoed, family,
                         parent_entry_id, child_entry_id, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (pattern_id, parent_expression, child_expression,
                     parent_struct.get("fingerprint"), child_struct.get("fingerprint"),
                     edit_type, edit_detail_json,
                     parent_ic, child_ic, delta_ic, parent_icir, child_icir, delta_icir,
                     verdict, round(confidence, 4), total, succ, vetoed, family,
                     parent_id, child_id, now, now),
                )
        return pattern_id

    def query_edit_patterns(
        self,
        *,
        edit_type: str | None = None,
        verdict: str | None = None,
        family: str | None = None,
        min_confidence: float = 0.3,
        exclude_vetoed: bool = False,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """检索编辑模式，按置信度降序。"""
        clauses = ["confidence >= ?"]
        params: list[Any] = [min_confidence]
        if edit_type:
            clauses.append("edit_type = ?")
            params.append(edit_type)
        if verdict:
            clauses.append("verdict = ?")
            params.append(verdict)
        if family:
            clauses.append("family = ?")
            params.append(family)
        if exclude_vetoed:
            clauses.append("vetoed = 0")
        where = " AND ".join(clauses)
        with self._open() as conn:
            rows = conn.execute(
                f"SELECT * FROM edit_patterns WHERE {where} "
                f"ORDER BY confidence DESC, total_uses DESC LIMIT ?",
                (*params, int(limit)),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "edit_type": r["edit_type"],
                "edit_detail": json.loads(r["edit_detail_json"] or "{}"),
                "parent_fingerprint": r["parent_fingerprint"],
                "child_fingerprint": r["child_fingerprint"],
                "parent_ic": r["parent_ic"],
                "child_ic": r["child_ic"],
                "delta_ic": r["delta_ic"],
                "parent_icir": r["parent_icir"],
                "child_icir": r["child_icir"],
                "delta_icir": r["delta_icir"],
                "verdict": r["verdict"],
                "confidence": r["confidence"],
                "total_uses": r["total_uses"],
                "success_count": r["success_count"],
                "vetoed": r["vetoed"],
                "family": r["family"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]


# ── 模块内小工具 ──

def _parse_condition_structure(expression: str) -> dict[str, Any]:
    """完整结构解析（含窗口），供编辑模式记录。"""
    from .expressions import expression_features
    return expression_features(expression)


def _parse_condition_structure_full(expression: str) -> dict[str, Any]:
    return _parse_condition_structure(expression)


def _set_edit_note(conn: sqlite3.Connection, entry_id: str, edit_note: str | None) -> None:
    if edit_note is None:
        return
    conn.execute(
        "UPDATE memory_entries SET edit_note = ? WHERE id = ?",
        (edit_note, entry_id),
    )


def _extract_motif_from_exprs(conn: sqlite3.Connection, child_expr: str, parent_expr: str | None = None) -> str:
    """从表达式对推导 motif（无 edit_note 时）。"""
    if parent_expr:
        return extract_edit_motif(parent_expr, child_expr)
    return "other"