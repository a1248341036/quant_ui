# -*- coding: utf-8 -*-
"""历史 JSONL 轨迹回填与 v3 幂等迁移。"""

from __future__ import annotations

import hashlib
import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .calibration import _parent_bucket
from .constants import BASELINE_HALF_LIFE_DAYS, POSITIVE_VERDICTS
from .diagnostics import _SUCCESS_SIGNATURES, _extract_fail_detail, _match_signature, _now, _rebuild_conclusion, _safe_float
from .expressions import (
    _structure_fingerprint,
    _tokens,
    classify_family,
    expression_features,
    expression_ops,
    expression_windows,
    extract_edit_motif,
)


class BackfillMixin:
    """历史回填：UI JSONL 轨迹重放 + v1/v2→v3 幂等迁移。"""

    # ── JSONL 回填 ──

    def backfill_from_logs(self, log_root: Path) -> int:
        """从 UI JSONL 轨迹回放建库（backfill_done 防重复回放）。

        只要标记存在（无论库内是否还有条目），就不再回放历史日志。
        """
        with self._open() as conn:
            existing = conn.execute("SELECT 1 FROM memory_entries LIMIT 1").fetchone()
            done = conn.execute("SELECT v FROM store_meta WHERE k = 'backfill_done'").fetchone()
            if existing or done:
                if not done:
                    conn.execute(
                        "INSERT OR REPLACE INTO store_meta(k, v) VALUES ('backfill_done', '1')"
                    )
                return 0
        count = 0
        for log_path in sorted(Path(log_root).glob("*/run_*.jsonl")):
            try:
                lines = log_path.read_text(encoding="utf-8").splitlines()
            except OSError:
                continue
            for line in lines:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(event, dict) or event.get("event") != "tool_results":
                    continue
                for row in event.get("results") or []:
                    if isinstance(row, dict) and self.record_tool_result(run_id=log_path.parent.name, row=row):
                        count += 1
        with self._open() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO store_meta(k, v) VALUES ('backfill_done', '1')"
            )
        return count

    # ── 调仓频率回填 ──

    def backfill_freq_from_run_specs(self, log_root: Path) -> dict[str, int]:
        """按 last_run_id 关联 run 目录的 research_spec.json 快照，回填调仓频率/档位。

        只补 metrics 里没有 rebalance_freq 的条目（幂等，不覆盖真实记录）；
        写入 freq_source="derived_run_spec" 与 recorded 口径区分。
        run 目录缺失/无 spec/无 freq 的条目保持不动。
        """
        summary = {"scanned": 0, "updated": 0, "skipped_present": 0, "unresolvable": 0}
        root = Path(log_root)
        with self._open() as conn:
            rows = conn.execute("SELECT id, last_run_id, metrics_json FROM memory_entries").fetchall()
            for row in rows:
                summary["scanned"] += 1
                metrics = json.loads(row["metrics_json"] or "{}")
                # 幂等：已带真实记录（run_spec 落库）或已回填过的条目跳过
                if metrics.get("rebalance_freq") or metrics.get("freq_source") in {"run_spec", "derived_run_spec"}:
                    summary["skipped_present"] += 1
                    continue
                rid = str(row["last_run_id"] or "")
                freq: str | None = None
                mode: str | None = None
                spec_path = root / rid / "research_spec.json" if rid else None
                if spec_path and spec_path.is_file():
                    try:
                        spec = json.loads(spec_path.read_text(encoding="utf-8"))
                        gate = ((spec.get("delivery_policy") or {}).get("production") or {}).get("engine_gate") or {}
                        freq = str(gate.get("freq") or "") or None
                        mode = str(spec.get("research_mode") or "") or None
                    except (OSError, json.JSONDecodeError, AttributeError, TypeError):
                        freq = mode = None
                if freq is None and mode is None:
                    summary["unresolvable"] += 1
                    continue
                if freq is not None:
                    metrics["rebalance_freq"] = freq
                if mode is not None:
                    metrics["research_mode"] = mode
                metrics["freq_source"] = "derived_run_spec"
                conn.execute(
                    "UPDATE memory_entries SET metrics_json = ? WHERE id = ?",
                    (json.dumps(metrics, ensure_ascii=False), row["id"]),
                )
                summary["updated"] += 1
        return summary

    # ── v1/v2 → v3 幂等迁移 ──

    def _backfill_v3_conn(self, conn: sqlite3.Connection) -> None:
        """存量库重放：链接父本 family/指纹/fail_detail/tokens/parent_origin，重建 cells。

        - legacy 条目无显式父本协议 → 一律 implicit（检索时 0.5 加权）
        - 重放依据：live 路径同源 → 算子相似 0.4 + 窗口相似 0.3 + 结构指纹 0.3，阈值 0.5，同 stage
        - 同桶基线缺失时按父本 IC 回退，残差 = child IC − 基线
        - 幂等：data_version==3 时不重复迁移
        """
        version_row = conn.execute(
            "SELECT v FROM store_meta WHERE k = 'data_version'"
        ).fetchone()
        if version_row and str(version_row["v"]) == "3":
            return
        rows = conn.execute("SELECT * FROM memory_entries ORDER BY updated_at ASC").fetchall()
        summary = {"entries": 0, "cells": 0, "experience": 0}
        seen: set[str] = set()
        bucket_state: dict[tuple[str, str], dict[str, float]] = {}
        updated = 0
        for row in rows:
            eid = str(row["id"])
            if eid in seen:
                continue
            seen.add(eid)
            expression = str(row["expression"] or "")
            factor_name = str(row["factor_name"] or "")
            if not expression:
                continue
            # family 兜底
            family = str(row["family"] or "") or classify_family(factor_name, expression)
            # 结构指纹兜底
            struct = expression_features(expression)
            fingerprint = str(row["structure_fingerprint"] or "") or struct.get("fingerprint") or ""
            op_list_json = json.dumps(struct.get("operators", []), ensure_ascii=False)
            win_params_json = json.dumps(struct.get("window_params", {}), ensure_ascii=False)
            if row["structure_fingerprint"] != fingerprint or row["operator_list_json"] is None or row["window_params_json"] is None:
                conn.execute(
                    "UPDATE memory_entries SET structure_fingerprint = ?, family = ?, operator_list_json = ?, window_params_json = ? WHERE id = ?",
                    (fingerprint, family, op_list_json, win_params_json, eid),
                )
                updated += 1

            # 隐式父本链接（排除自身）
            parent_id = row["parent_id"]
            parent_origin = row["parent_origin"]
            if not parent_id:
                parent_id = self._find_implicit_parent_conn(
                    conn, struct, exclude_id=eid, limit=50
                )
                if parent_id:
                    parent_origin = "implicit"
                    parent_row = conn.execute(
                        "SELECT expression FROM memory_entries WHERE id = ?", (parent_id,)
                    ).fetchone()
                    parent_expr = parent_row["expression"] if parent_row else ""
                    motif = extract_edit_motif(parent_expr, expression)
                    conn.execute(
                        """
                        UPDATE memory_entries
                        SET parent_id = ?, parent_origin = ?, intended_motif = ?
                        WHERE id = ?
                        """,
                        (parent_id, parent_origin, motif, eid),
                    )
                    updated += 1

            # 重建 cells：桶 + origin 分列（legacy 全 implicit 加权 0.5）
            if parent_id and parent_origin:
                parent_row = conn.execute(
                    "SELECT metrics_json FROM memory_entries WHERE id = ?", (parent_id,)
                ).fetchone()
                parent_metrics = json.loads(parent_row["metrics_json"] or "{}") if parent_row else {}
                parent_ic = _safe_float(parent_metrics.get("ic"))
                bucket = _parent_bucket(parent_ic)
                # 从 DB 重新读取 intended_motif（隐式链接刚写入的值不在 row 快照里）
                motif_row = conn.execute(
                    "SELECT intended_motif FROM memory_entries WHERE id = ?", (eid,)
                ).fetchone()
                motif = str(motif_row["intended_motif"] or "") if motif_row else ""
                if not motif:
                    motif = "other"
                metrics = json.loads(row["metrics_json"] or "{}")
                child_ic = _safe_float(metrics.get("ic"))
                verdict = str(row["verdict"] or "")
                error = str(row["error"] or "")
                is_positive = verdict in POSITIVE_VERDICTS
                invalid = bool(error) or child_ic is None

                # 同桶基线（迁移时按历史顺序累积）
                state_key = (family, bucket)
                state = bucket_state.get(state_key)
                baseline = None
                if state and state["n"] >= 1:
                    baseline = state["sw"] / state["n"]
                if baseline is None and parent_ic is not None:
                    baseline = parent_ic
                residual = (child_ic - baseline) if (child_ic is not None and baseline is not None) else None

                if parent_origin == "explicit":
                    s_col, f_col = "explicit_s", "explicit_f"
                else:
                    s_col, f_col = "implicit_s", "implicit_f"
                if invalid:
                    delta = 0.5
                    is_valid = False
                else:
                    delta = 1.0 if parent_origin == "explicit" else 0.5
                    is_valid = is_positive
                col = s_col if is_valid else f_col

                cell_row = conn.execute(
                    "SELECT residuals_json FROM memory_cells WHERE family = ? AND motif = ? AND parent_bucket = ?",
                    (family, motif, bucket),
                ).fetchone()
                old_res = json.loads(cell_row["residuals_json"] or "[]") if cell_row else []
                new_res = old_res + [residual] if residual is not None else old_res
                conn.execute(
                    f"""
                    INSERT INTO memory_cells (family, motif, parent_bucket, {col}, residuals_json, updated_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                    ON CONFLICT(family, motif, parent_bucket) DO UPDATE SET
                        {col} = {col} + excluded.{col},
                        residuals_json = excluded.residuals_json,
                        updated_at = excluded.updated_at
                    """,
                    (family, motif, bucket, delta, json.dumps(new_res, ensure_ascii=False), _now()),
                )
                summary["cells"] += 1
                # 更新同桶基线状态
                if child_ic is not None:
                    st = bucket_state.get(state_key, {"n": 0.0, "sw": 0.0})
                    st["n"] += 1
                    st["sw"] += child_ic
                    bucket_state[state_key] = st

            summary["entries"] += 1

        if updated:
            conn.execute(
                "INSERT OR REPLACE INTO store_meta(k, v) VALUES ('backfill_done_v3', '1')"
            )