# -*- coding: utf-8 -*-
"""评估前 advisory、尝试查询和管理接口。"""

from __future__ import annotations

import json
import sqlite3
from typing import Any

from .calibration import _apv_gate, _eq7_confidence
from .constants import POSITIVE_VERDICTS, VERDICT_ORDER
from .diagnostics import _parse_args
from .expressions import _structure_fingerprint, classify_family, motif_from_note

# recent() 服务端排序白名单：前端排序列 → SQL 表达式（metrics 为 JSON 字段）。
# 覆盖率/年化超额存在双键（新键缺失时回退旧键），其余直接取单键。
_RECENT_SORT_EXPRS: dict[str, str] = {
    "updated_at": "updated_at",
    "factor_name": "factor_name",
    "stage": "stage",
    "attempts": "CAST(attempts AS REAL)",
    "ic": "CAST(json_extract(metrics_json, '$.ic') AS REAL)",
    "icir": "CAST(json_extract(metrics_json, '$.icir') AS REAL)",
    "coverage": (
        "COALESCE(CAST(json_extract(metrics_json, '$.coverage') AS REAL), "
        "CAST(json_extract(metrics_json, '$.factor_coverage') AS REAL))"
    ),
    "sharpe": "CAST(json_extract(metrics_json, '$.sharpe') AS REAL)",
    "excess_sharpe": "CAST(json_extract(metrics_json, '$.excess_sharpe') AS REAL)",
    "annualized_return": "CAST(json_extract(metrics_json, '$.annualized_return') AS REAL)",
    "annualized_excess_return": (
        "COALESCE(CAST(json_extract(metrics_json, '$.annualized_excess_return') AS REAL), "
        "CAST(json_extract(metrics_json, '$.long_group_annual_excess_return') AS REAL))"
    ),
    "max_drawdown": "CAST(json_extract(metrics_json, '$.max_drawdown') AS REAL)",
    "annual_turnover": "CAST(json_extract(metrics_json, '$.annual_turnover') AS REAL)",
    "daily_overlap": "CAST(json_extract(metrics_json, '$.daily_overlap') AS REAL)",
    "monotonicity": "CAST(json_extract(metrics_json, '$.monotonicity') AS REAL)",
}


# 正向 verdict 集合的 SQL IN 占位（①b 指纹正证据查重；sorted 保证参数顺序稳定）
_POSITIVE_VERDICTS = sorted(POSITIVE_VERDICTS)
_POSITIVE_PH = ",".join("?" * len(_POSITIVE_VERDICTS))


class AdvisoryMixin:
    """评估前 advisory（硬提醒）与查询/管理接口。"""

    # ── 评估前 advisory ──

    def advisory_for(self, expression: str | None, *, edit_note: str | None = None) -> dict[str, Any] | None:
        """评估前硬提醒通道（v3：指纹负证据 / 指纹正证据 / 意向编辑 APV 双门）。默认只提醒不拦截。

        返回 None（无提醒）或 {"advisories": [...], "blocked": False}。
        - duplicate_known_dead_end：同结构指纹负证据（同表达式 ≥2 次尝试）→ 已知死路，
          `hard_block_duplicates=True` 时由调用方（tools.dispatch）升级为拦截；
        - duplicate_prior_result：同结构指纹曾有正向结果（promising/入库）→ 重复劳动
          提醒（历史条目名/verdict/IC/未晋升原因），仅提醒、永不拦截——正向重复说明
          结构出过信号，正确动作是变异或核查晋升卡点，而非机械重测；
        - edit_veto：意向编辑 APV 双门否决。
        """
        if not expression:
            return None
        expression = str(expression).strip()
        if not expression:
            return None
        findings: list[dict[str, Any]] = []
        fingerprint = _structure_fingerprint(expression)
        family = classify_family("", expression)
        with self._open() as conn:
            # ① 指纹负证据：同一结构指纹的已否定条目 ≥2 次尝试 → 已知死路
            if fingerprint:
                row = conn.execute(
                    """
                    SELECT factor_name, verdict, fail_detail, attempts FROM memory_entries
                    WHERE structure_fingerprint = ?
                      AND verdict IN ('rejected', 'revise_required', 'weak')
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (fingerprint,),
                ).fetchone()
                if row and int(row["attempts"]) >= 2:
                    verdict = str(row["verdict"] or "")
                    reason = str(row["fail_detail"] or "") if row["fail_detail"] else f"已 {int(row['attempts'])} 次评估为 {verdict}"
                    findings.append({
                        "kind": "duplicate_known_dead_end",
                        "message": f"该表达式结构与历史死路相同（{row['factor_name']}，{reason}），不建议重复评估。",
                    })

            # ①b 指纹正证据：同结构曾有正向 verdict（promising/入库）→ 重复劳动提醒。
            #    与死路提醒独立并存（死路指失败变体，正向条目指值得改造的变体，
            #    互补不矛盾）；仅提醒、永不拦截——hard_block_duplicates 只作用于死路。
            if fingerprint:
                pos_row = conn.execute(
                    f"""
                    SELECT factor_name, verdict, fail_detail, attempts, metrics_json, updated_at,
                           COUNT(*) OVER () AS n_positive
                    FROM memory_entries
                    WHERE structure_fingerprint = ?
                      AND verdict IN ({_POSITIVE_PH})
                    ORDER BY updated_at DESC LIMIT 1
                    """,
                    (fingerprint, *_POSITIVE_VERDICTS),
                ).fetchone()
                if pos_row:
                    metrics = json.loads(pos_row["metrics_json"] or "{}")
                    metric_bits = []
                    for key in ("ic", "val_ic", "icir", "rank_ic"):
                        v = metrics.get(key)
                        if isinstance(v, (int, float)) and v == v:
                            metric_bits.append(f"{key}={v:+.4f}")
                    fail = str(pos_row["fail_detail"] or "").strip()
                    name = str(pos_row["factor_name"])
                    when = str(pos_row["updated_at"] or "")[:10]
                    metrics_txt = ("，" + "，".join(metric_bits)) if metric_bits else ""
                    fail_txt = f"，未晋升原因：{fail}" if fail else ""
                    findings.append({
                        "kind": "duplicate_prior_result",
                        "message": (
                            f"该表达式结构与历史条目重复：{name}（{when}，verdict={pos_row['verdict']}"
                            f"{metrics_txt}，已评估 {int(pos_row['attempts'])} 次{fail_txt}）。"
                            "同结构已测出过正向结果，勿原样重测："
                            "以其为父本做显式变异（parent_factor=该历史因子），或核查其未晋升原因后决定。"
                        ),
                        "prior_factor": name,
                        "prior_verdict": str(pos_row["verdict"]),
                        "prior_updated_at": pos_row["updated_at"],
                        "n_prior_positive": int(pos_row["n_positive"]),
                    })

            # ② 意向编辑 APV：从 edit_note 解析 motif，查 cells 统计否决
            motif = motif_from_note(edit_note) if edit_note else None
            if motif and motif != "other":
                self._edit_veto_findings(conn, family, motif, findings)

        if not findings:
            return None
        return {"advisories": findings, "blocked": False}

    def _edit_veto_findings(
        self,
        conn: sqlite3.Connection,
        family: str,
        motif: str,
        findings: list[dict[str, Any]],
    ) -> None:
        """意向编辑 APV：同 (family, motif) 各桶加权成败 + 残差置信 → 双门否决。"""
        rows = conn.execute(
            "SELECT * FROM memory_cells WHERE family = ? AND motif = ?",
            (family, motif),
        ).fetchall()
        if not rows:
            return
        s_w_total = 0.0
        f_w_total = 0.0
        residuals_all: list[float] = []
        for row in rows:
            s_w, f_w = self._weighted_counts(row)
            s_w_total += s_w
            f_w_total += f_w
            residuals_all.extend(json.loads(row["residuals_json"] or "[]"))
        if f_w_total <= 0:
            return
        conf = _eq7_confidence(residuals_all) if residuals_all else 0.0
        vetoed, severity, _ = _apv_gate(
            s_w_total, f_w_total, conf,
            tau_c=self.apv_tau_c,
            tau_v=self.apv_tau_v,
        )
        if vetoed:
            findings.append({
                "kind": "edit_veto",
                "message": (
                    f"编辑类型 {motif} 在该信号族（{family}）历史失败 "
                    f"{f_w_total:.2g} 次（成功 {s_w_total:.2g}），APV 双门否决；请换编辑方向。"
                ),
                "severity": severity,
            })

    # ── 尝试查询 ──

    def query_for_attempts(
        self,
        research_goal: str,
        recent_rows: list[dict[str, Any]],
        *,
        max_recent_attempts: int = 8,
        max_expression_chars: int = 1200,
    ) -> str:
        """构建检索查询：goal + 当前 run 最近工作的因子/表达式/错误证据块。"""
        parts = [str(research_goal or "A股日频因子挖掘")]
        for row in list(recent_rows)[-max(0, int(max_recent_attempts)):]:
            args = _parse_args(row.get("arguments_raw"))
            factor_name = str(args.get("factor_name") or "").strip() or str(row.get("factor_name") or "").strip()
            expression = str(args.get("multi_line_expr") or "").strip() or str(row.get("expression") or "").strip()
            error = str(row.get("error") or "").strip()
            if factor_name:
                parts.append(factor_name)
            if expression:
                parts.append(expression[:max_expression_chars])
            if error:
                parts.append(error[:300])
            interaction = args.get("interaction")
            if isinstance(interaction, dict):
                parts.append(str(interaction.get("interaction_type") or ""))
                parts.append(str(interaction.get("base_signal") or ""))
                parts.append(str(interaction.get("condition_signal") or ""))
                parts.append(str(interaction.get("economic_mechanism") or ""))
        return "\n".join(part for part in parts if part)

    # ── 管理接口（v2 兼容）──

    def recent(self, *, limit: int = 50, offset: int = 0,
               order: str = "recent", verdict: str = "",
               sort: str = "", dir: int = 0) -> tuple[list[dict[str, Any]], int]:
        """分页返回条目，支持服务端 verdict 过滤与按列排序（分页 UI 口径）。

        order="recent"（默认）: updated_at DESC, rowid DESC —— 最新评估在前；
        研究总结页默认口径，新 run 的条目（多为 weak/rejected）不再被
        verdict 优先级压到深部分页不可见。
        order="verdict": 正值 verdict 优先 + 时间倒序（旧口径，正证据一屏可见）。
        sort/dir 覆盖 order：sort 为 _RECENT_SORT_EXPRS 白名单键（含 metrics JSON
        字段，verdict 列按 VERDICT_ORDER 排名），dir=1 升序 / -1 降序；
        NULL 值恒排最后。未知名回落 updated_at。
        verdict 非空（须为合法 verdict）时只返回该档条目，total 为过滤后总数。

        返回 (entries, total)。
        """
        case_parts = " ".join(
            f"WHEN '{v}' THEN {rank}" for v, rank in VERDICT_ORDER.items()
        )
        sort_expr = _RECENT_SORT_EXPRS.get(sort) if sort else None
        if sort == "verdict":
            sort_expr = f"CASE verdict {case_parts} ELSE 99 END"
        where_sql, where_args = "", []
        if verdict and verdict in VERDICT_ORDER:
            where_sql, where_args = "WHERE verdict = ?", [verdict]
        if sort_expr is not None:
            direction = "DESC" if dir == -1 else "ASC"
            order_sql = (f"({sort_expr} IS NULL), {sort_expr} {direction}, "
                         f"rowid {direction}")
        elif order == "verdict":
            order_sql = f"CASE verdict {case_parts} ELSE 99 END, updated_at DESC"
        else:
            # id 是表达式指纹（十六进制串）非时间序，插入顺序用 rowid
            order_sql = "updated_at DESC, rowid DESC"
        with self._open() as conn:
            total = int(conn.execute(
                f"SELECT COUNT(*) FROM memory_entries {where_sql}",
                where_args,
            ).fetchone()[0])
            rows = conn.execute(
                f"""
                SELECT * FROM memory_entries
                {where_sql}
                ORDER BY {order_sql}
                LIMIT ? OFFSET ?
                """,
                [*where_args, max(0, int(limit)), max(0, int(offset))],
            ).fetchall()
            return self._hydrate_entries(conn, rows), total

    def list_cells(self) -> list[dict[str, Any]]:
        """全部 SSPM 单元（编辑统计层明细），供 UI 门控视图使用。"""
        with self._open() as conn:
            rows = conn.execute("SELECT * FROM memory_cells ORDER BY updated_at DESC").fetchall()
            return [
                {
                    "family": row["family"],
                    "motif": row["motif"],
                    "parent_bucket": row["parent_bucket"],
                    "explicit_s": float(row["explicit_s"] or 0),
                    "explicit_f": float(row["explicit_f"] or 0),
                    "implicit_s": float(row["implicit_s"] or 0),
                    "implicit_f": float(row["implicit_f"] or 0),
                    "residuals": json.loads(row["residuals_json"] or "[]"),
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]

    def list_experience(self, *, kind: str | None = None) -> list[dict[str, Any]]:
        """经验层条目（success_pattern / forbidden / insight）。"""
        with self._open() as conn:
            if kind:
                rows = conn.execute(
                    "SELECT * FROM memory_experience WHERE kind = ? ORDER BY updated_at DESC",
                    (kind,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT * FROM memory_experience ORDER BY updated_at DESC").fetchall()
            return [
                {
                    "id": row["id"],
                    "kind": row["kind"],
                    "name": row["name"],
                    "content": row["content"],
                    "template": row["template"],
                    "example_factors": json.loads(row["example_factors_json"] or "[]"),
                    "correlated": json.loads(row["correlated_json"] or "[]"),
                    "typical_correlation": row["typical_correlation"],
                    "occurrence_count": int(row["occurrence_count"] or 1),
                    "run_id": row["run_id"],
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
                for row in rows
            ]

    def statistics(self) -> dict[str, Any]:
        with self._open() as conn:
            entry_count = int(conn.execute("SELECT COUNT(*) FROM memory_entries").fetchone()[0])
            obs_rows = conn.execute(
                """
                SELECT e.verdict AS verdict, COUNT(*) AS n
                FROM memory_observations o
                JOIN memory_entries e ON e.id = o.entry_id
                GROUP BY e.verdict
                """
            ).fetchall()
            cells = int(conn.execute("SELECT COUNT(*) FROM memory_cells").fetchone()[0])
            experience = int(conn.execute("SELECT COUNT(*) FROM memory_experience").fetchone()[0])
        counts = {str(row["verdict"] or "unknown"): int(row["n"]) for row in obs_rows}
        attempts = sum(counts.values())
        return {
            "entries": entry_count,
            "observations": attempts,
            "verdict_counts": counts,
            "cells": cells,
            "experience": experience,
            "train_to_validated_rate": round(counts.get("validated", 0) / attempts, 4) if attempts else None,
            "production_rate": round(counts.get("production_approved", 0) / attempts, 4) if attempts else None,
        }

    def delete_entry(self, entry_id: str) -> bool:
        """删除单条记忆（级联清理 observations/FTS/cells 关联）。"""
        with self._open() as conn:
            cursor = conn.execute("DELETE FROM memory_entries WHERE id = ?", (entry_id,))
            return cursor.rowcount > 0

    def purge_factor(
        self,
        *,
        factor_names: list[str] | tuple[str, ...] = (),
        expressions: list[str] | tuple[str, ...] = (),
    ) -> int:
        """删除与指定因子相关的全部记忆条目（含 FTS 与观察记录，经级联/触发器）。"""
        ids = {self.entry_signature(e) for e in expressions if e}
        names = [n for n in factor_names if n]
        if not ids and not names:
            return 0
        deleted = 0
        with self._open() as conn:
            for eid in ids:
                cursor = conn.execute("DELETE FROM memory_entries WHERE id = ?", (eid,))
                deleted += cursor.rowcount
            for name in names:
                cursor = conn.execute("DELETE FROM memory_entries WHERE factor_name = ?", (name,))
                deleted += cursor.rowcount
        return deleted