# -*- coding: utf-8 -*-
"""编辑统计层（SSPM）与经验层蒸馏。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from typing import Any

from .calibration import _parent_bucket
from .constants import BASELINE_HALF_LIFE_DAYS
from .diagnostics import _SUCCESS_SIGNATURES, _FORBIDDEN_SIGNATURES, _match_signature, _now, _safe_float
from .expressions import classify_family, expression_features, expression_ops


class ExperienceMixin:
    """经验层：同桶基线、cells 更新（v3）和经验蒸馏（success/forbidden/insight）。"""

    # ── 加权统计 ──
    # 列存加权值（explicit 记 1.0、implicit/invalid 记 0.5），统计时直接相加。
    @staticmethod
    def _weighted_counts(row: sqlite3.Row) -> tuple[float, float]:
        s = float(row["explicit_s"] or 0) + float(row["implicit_s"] or 0)
        f = float(row["explicit_f"] or 0) + float(row["implicit_f"] or 0)
        return s, f

    def _same_bucket_baseline(
        self,
        conn: sqlite3.Connection,
        family: str,
        bucket: str,
        parent_ic: float | None = None,
    ) -> float | None:
        """AlphaMemo Eq.4 基线 = 同 (family, bucket) 历史子代 IC 的加权均值。

        状态存于 store_meta（v3_base_{family}_{bucket}），无历史时回退父本 IC。
        """
        row = conn.execute(
            "SELECT v FROM store_meta WHERE k = ?",
            (f"v3_base_{family}_{bucket}",),
        ).fetchone()
        if row:
            try:
                state = json.loads(row["v"])
                n = float(state.get("n", 0))
                if n >= 1:
                    return float(state.get("sw", 0)) / n
            except (json.JSONDecodeError, TypeError, ValueError):
                pass
        return parent_ic

    def _update_cell(
        self,
        conn: sqlite3.Connection,
        entry: dict[str, Any],
        struct: dict[str, Any],
        *,
        parent_id: str | None,
        parent_origin: str | None,
        intended_motif: str | None,
        verdict: str,
        error: str,
    ) -> None:
        """AlphaMemo ResidualMemory.update（v3）。

        - cell key = (family, motif, parent_bucket) 组合键
        - residual = child_ic − 同 (family, bucket) 时域衰减基线（半衰期 90d）
        - 成败按 explicit/implicit 分列计权：explicit 1.0 / implicit 0.5 / invalid 0.5
        - invalid 尝试只记失败，无 residual
        """
        from .constants import INVALID_WEIGHT, PARENT_ORIGIN_WEIGHT, POSITIVE_VERDICTS

        if not parent_id:
            return
        parent_row = conn.execute(
            "SELECT metrics_json FROM memory_entries WHERE id = ?", (parent_id,)
        ).fetchone()
        if not parent_row:
            return
        parent_metrics = json.loads(parent_row["metrics_json"] or "{}")
        parent_ic = _safe_float(parent_metrics.get("ic"))
        bucket = _parent_bucket(parent_ic)
        family = entry.get("family") or "other"
        motif = intended_motif or "other"
        child_ic = _safe_float(entry.get("metrics", {}).get("ic"))

        # 残差：child IC − 同桶基线
        residual: float | None = None
        if child_ic is not None:
            baseline = self._same_bucket_baseline(conn, family, bucket, parent_ic)
            if baseline is not None:
                residual = child_ic - baseline

        is_positive = verdict in POSITIVE_VERDICTS
        invalid = bool(error) or child_ic is None
        if parent_origin == "explicit":
            s_col, f_col = "explicit_s", "explicit_f"
        else:
            s_col, f_col = "implicit_s", "implicit_f"
        if invalid:
            delta = INVALID_WEIGHT
            is_valid = False
        else:
            delta = 1.0 if parent_origin == "explicit" else PARENT_ORIGIN_WEIGHT["implicit"]
            is_valid = is_positive
        col = s_col if is_valid else f_col

        old = conn.execute(
            "SELECT residuals_json FROM memory_cells WHERE family = ? AND motif = ? AND parent_bucket = ?",
            (family, motif, bucket),
        ).fetchone()
        old_res = json.loads(old["residuals_json"] or "[]") if old else []
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
        # 更新同桶基线状态（AlphaMemo Eq.4 在线更新）
        if child_ic is not None:
            state_key = f"v3_base_{family}_{bucket}"
            state_row = conn.execute(
                "SELECT v FROM store_meta WHERE k = ?", (state_key,)
            ).fetchone()
            try:
                state = json.loads(state_row["v"]) if state_row else {}
                n = float(state.get("n", 0))
                sw = float(state.get("sw", 0))
            except (json.JSONDecodeError, TypeError, ValueError):
                n, sw = 0.0, 0.0
            n += 1
            sw += child_ic
            conn.execute(
                "INSERT OR REPLACE INTO store_meta(k, v) VALUES (?, ?)",
                (state_key, json.dumps({"n": n, "sw": sw}, ensure_ascii=False)),
            )

    # ── 经验蒸馏 ──

    def _upsert_experience(
        self,
        conn: sqlite3.Connection | None,
        *,
        kind: str,
        name: str,
        content: str,
        template: str | None = None,
        evidence: dict[str, Any] | None = None,
        example_factor: str | None = None,
        correlated_with: list[str] | None = None,
        typical_correlation: float | None = None,
        run_id: str | None = None,
    ) -> str:
        """写入或更新一条经验记忆（success_pattern / forbidden / insight）。

        同 (kind, name) 去重；已存在则 occurrence_count 累加并刷新示例。
        conn 为 None 时自行打开连接。
        """
        exp_id = hashlib.sha256(f"{kind}|{name}".encode("utf-8")).hexdigest()[:20]
        now = _now()
        evidence_json = json.dumps(evidence or {}, ensure_ascii=False, separators=(",", ":"))
        correlated_json = json.dumps(correlated_with or [], ensure_ascii=False)
        example_factors_json = json.dumps([example_factor] if example_factor else [], ensure_ascii=False)
        if conn is None:
            with self._open() as conn:
                return self._upsert_experience_conn(
                    conn, exp_id, kind, name, content, template, evidence_json,
                    example_factors_json, correlated_json, typical_correlation, run_id, now,
                )
        return self._upsert_experience_conn(
            conn, exp_id, kind, name, content, template, evidence_json,
            example_factors_json, correlated_json, typical_correlation, run_id, now,
        )

    @staticmethod
    def _upsert_experience_conn(
        conn: sqlite3.Connection,
        exp_id: str,
        kind: str,
        name: str,
        content: str,
        template: str | None,
        evidence_json: str,
        example_factors_json: str,
        correlated_json: str,
        typical_correlation: float | None,
        run_id: str | None,
        now: str,
    ) -> str:
        existing = conn.execute(
            "SELECT occurrence_count FROM memory_experience WHERE id = ?", (exp_id,)
        ).fetchone()
        if existing:
            conn.execute(
                """
                UPDATE memory_experience
                SET content = ?, template = ?, evidence_json = ?,
                    example_factors_json = ?, correlated_json = ?,
                    typical_correlation = ?, occurrence_count = occurrence_count + 1,
                    updated_at = ?
                WHERE id = ?
                """,
                (content, template, evidence_json, example_factors_json,
                 correlated_json, typical_correlation, now, exp_id),
            )
        else:
            conn.execute(
                """
                INSERT INTO memory_experience (
                    id, kind, name, content, template, evidence_json,
                    example_factors_json, correlated_json, typical_correlation,
                    occurrence_count, run_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                """,
                (exp_id, kind, name, content, template, evidence_json,
                 example_factors_json, correlated_json, typical_correlation,
                 run_id, now, now),
            )
        return exp_id

    def form_memory(
        self,
        *,
        run_id: str,
        turn: int,
        batch_results: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """FactorMiner Memory Formation：从一轮评估/提交结果蒸馏结构化经验。

        batch_results 每项：factor_name / expression / metrics{ic,...} /
        admitted / rejection_reason / max_corr / correlated_with / fail_detail
        """
        formed = {"success_patterns": 0, "forbidden": 0, "insights": 0}
        if not batch_results:
            return formed
        with self._open() as conn:
            for item in batch_results:
                if not isinstance(item, dict):
                    continue
                factor_name = str(item.get("factor_name") or "")
                expression = str(item.get("expression") or "")
                metrics = item.get("metrics") if isinstance(item.get("metrics"), dict) else {}
                ic = _safe_float(metrics.get("ic"))
                verdict = str(item.get("verdict") or "")
                admitted = bool(item.get("admitted", verdict in ("validated", "production_approved", "promising")))
                conclusion = str(item.get("conclusion") or "")
                reason = str(item.get("rejection_reason") or item.get("fail_detail") or "")
                text = " ".join(filter(None, [factor_name, expression, conclusion]))
                forbidden_key = _match_signature(text, _FORBIDDEN_SIGNATURES)
                success_key = _match_signature(text, _SUCCESS_SIGNATURES)

                if forbidden_key:
                    self._upsert_experience(
                        conn,
                        kind="forbidden",
                        name=f"{factor_name or expression}",
                        content=f"禁忌方向：{forbidden_key}（{factor_name or expression}）",
                        evidence={"ic": ic, "reason": reason},
                        example_factor=factor_name or None,
                        run_id=run_id,
                    )
                    formed["forbidden"] += 1
                elif success_key and admitted:
                    self._upsert_experience(
                        conn,
                        kind="success_pattern",
                        name=f"{success_key}|{factor_name}",
                        content=f"成功模式：{success_key}（{factor_name or expression}）",
                        evidence={"ic": ic},
                        example_factor=factor_name or None,
                        run_id=run_id,
                    )
                    formed["success_patterns"] += 1
                else:
                    insight = conclusion or (f"{factor_name}: 结果 {verdict or 'unknown'}" if factor_name else None)
                    if insight:
                        self._upsert_experience(
                            conn,
                            kind="insight",
                            name=f"{factor_name or expression}",
                            content=insight,
                            evidence={"ic": ic},
                            run_id=run_id,
                        )
                        formed["insights"] += 1
        return formed

    # ── 模式层记忆 CRUD（v3 恢复：从 commit 7966fd1 移植）──

    def record_pattern(
        self,
        *,
        layer: str,
        category: str,
        content: str,
        evidence: dict[str, Any] | None = None,
    ) -> str:
        """写入或更新一条模式记忆。

        ``layer`` ∈ {"recommend", "forbid", "insight"}。
        按 ``layer|category|content`` 签名去重；已存在则 total_attempts += 1。
        """
        if layer not in ("recommend", "forbid", "insight"):
            raise ValueError(f"invalid layer: {layer}")
        pattern_id = hashlib.sha256(
            f"{layer}|{category}|{content}".encode("utf-8")
        ).hexdigest()[:20]
        now = _now()
        evidence_json = json.dumps(evidence or {}, ensure_ascii=False, separators=(",", ":"))
        with self._open() as conn:
            existing = conn.execute(
                "SELECT total_attempts, success_count FROM memory_patterns WHERE id = ?",
                (pattern_id,),
            ).fetchone()
            if existing:
                conn.execute(
                    """
                    UPDATE memory_patterns
                    SET evidence_json = ?, total_attempts = total_attempts + 1, updated_at = ?
                    WHERE id = ?
                    """,
                    (evidence_json, now, pattern_id),
                )
            else:
                conn.execute(
                    """
                    INSERT INTO memory_patterns
                        (id, layer, category, content, evidence_json,
                         total_attempts, success_count, saturation_score,
                         confidence, created_at, updated_at)
                    VALUES (?, ?, ?, ?, ?, 1, 0, 0, 0.5, ?, ?)
                    """,
                    (pattern_id, layer, category, content, evidence_json, now, now),
                )
        return pattern_id

    def update_pattern_stats(self, pattern_id: str, *, success: bool) -> None:
        """评估结果返回后更新模式的成功/失败计数和置信度。"""
        with self._open() as conn:
            row = conn.execute(
                "SELECT total_attempts, success_count FROM memory_patterns WHERE id = ?",
                (pattern_id,),
            ).fetchone()
            if not row:
                return
            total = int(row["total_attempts"]) + 1
            succ = int(row["success_count"]) + (1 if success else 0)
            rate = succ / total if total else 0.0
            conf = max(0.1, min(0.95, 1.0 - 1.0 / (total ** 0.5)))
            conn.execute(
                """
                UPDATE memory_patterns
                SET total_attempts = ?, success_count = ?, success_rate = ?,
                    confidence = ?, updated_at = ?
                WHERE id = ?
                """,
                (total, succ, round(rate, 4), round(conf, 4), _now(), pattern_id),
            )

    def query_patterns(
        self,
        *,
        layer: str | None = None,
        category: str | None = None,
        min_confidence: float = 0.3,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """检索模式记忆，按置信度降序。"""
        clauses = ["confidence >= ?"]
        params: list[Any] = [min_confidence]
        if layer:
            clauses.append("layer = ?")
            params.append(layer)
        if category:
            clauses.append("category = ?")
            params.append(category)
        where = " AND ".join(clauses)
        with self._open() as conn:
            rows = conn.execute(
                f"SELECT * FROM memory_patterns WHERE {where} "
                f"ORDER BY confidence DESC, success_rate DESC LIMIT ?",
                (*params, int(limit)),
            ).fetchall()
        return [
            {
                "id": r["id"],
                "layer": r["layer"],
                "category": r["category"],
                "content": r["content"],
                "evidence": json.loads(r["evidence_json"] or "{}"),
                "success_rate": r["success_rate"],
                "total_attempts": r["total_attempts"],
                "success_count": r["success_count"],
                "saturation_score": r["saturation_score"],
                "confidence": r["confidence"],
                "created_at": r["created_at"],
                "updated_at": r["updated_at"],
            }
            for r in rows
        ]

    # ── 每批蒸馏算子（规则式，零 LLM 成本）──

    def _group_by_family(self, results: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        """按信号族分组评估结果。"""
        families: dict[str, list[dict[str, Any]]] = {}
        for r in results:
            family = classify_family(
                str(r.get("factor_name", "")),
                str(r.get("expression", "")),
            )
            families.setdefault(family, []).append(r)
        return families

    def distill_batch_patterns(
        self,
        *,
        run_id: str,
        turn: int,
        batch_results: list[dict[str, Any]],
    ) -> list[str]:
        """从一批评估结果中蒸馏模式记忆，返回写入的 pattern_ids。

        规则蒸馏（不调用 LLM，零成本）：
        1. 同族因子 >= 3 个且全部 IC < 0.01 → forbid 模式
        2. 同族因子中有 >= 1 个 IC > 0.02 → recommend 模式
        3. 全部因子 IC < 0.005 → insight 模式
        """
        pattern_ids: list[str] = []
        if not batch_results:
            return pattern_ids

        families = self._group_by_family(batch_results)

        for family_name, members in families.items():
            ics = [
                abs(float(m.get("metrics", {}).get("ic", 0) or 0))
                for m in members
            ]
            n = len(members)
            if n < 2:
                continue

            all_weak = all(ic < 0.01 for ic in ics)
            any_promising = any(ic >= 0.02 for ic in ics)

            if all_weak and n >= 3:
                content = (
                    f"{family_name} 信号族在 {n} 次尝试中 IC 均低于 0.01，"
                    f"最高 {max(ics):.4f}。该族在当前数据/label 下可能已饱和，"
                    f"除非引入新变量或交互机制，否则不建议机械重复。"
                )
                pid = self.record_pattern(
                    layer="forbid",
                    category=family_name,
                    content=content,
                    evidence={
                        "factor_names": [m.get("factor_name") for m in members],
                        "ic_range": [round(min(ics), 6), round(max(ics), 6)],
                        "n_attempts": n,
                        "run_id": run_id,
                        "turn": turn,
                    },
                )
                pattern_ids.append(pid)

            if any_promising:
                best = max(
                    members,
                    key=lambda m: abs(float(m.get("metrics", {}).get("ic", 0) or 0)),
                )
                best_ic = float(best.get("metrics", {}).get("ic", 0) or 0)
                content = (
                    f"{family_name} 信号族中有因子 IC 达 {best_ic:+.4f}，"
                    f"其经济机制值得在邻近空间继续探索。"
                    f"建议变异方向：换窗口、换修饰算子、引入正交交互。"
                )
                pid = self.record_pattern(
                    layer="recommend",
                    category=family_name,
                    content=content,
                    evidence={
                        "best_factor": best.get("factor_name"),
                        "best_ic": round(best_ic, 6),
                        "n_attempts": n,
                        "run_id": run_id,
                        "turn": turn,
                    },
                )
                pattern_ids.append(pid)

        # 全局洞察
        all_ics = [
            abs(float(r.get("metrics", {}).get("ic", 0) or 0))
            for r in batch_results
        ]
        if len(all_ics) >= 5 and max(all_ics) < 0.005:
            content = (
                f"连续 {len(all_ics)} 个因子 IC 均低于 0.005，"
                f"当前数据/label 组合下 alpha 可能极度稀薄。"
                f"建议：切换 label 列、扩展数据源、或尝试交互信号。"
            )
            pid = self.record_pattern(
                layer="insight",
                category="global",
                content=content,
                evidence={
                    "n_consecutive": len(all_ics),
                    "max_ic": round(max(all_ics), 6),
                    "run_id": run_id,
                    "turn": turn,
                },
            )
            pattern_ids.append(pid)

        return pattern_ids

    def distill_batch_experience(
        self,
        *,
        run_id: str,
        turn: int,
        batch_results: list[dict[str, Any]],
    ) -> dict[str, int]:
        """v3 经验层批量蒸馏（规则式，零 LLM 成本），替代 v2 distill_batch_patterns。

        规则（与 v2 版一致，口径修正）：
        1. 同族 >= 3 个且全部 |IC| < 0.01 → forbidden（族饱和禁忌）
        2. 同族中有 >= 1 个 |IC| >= 0.02 → success_pattern（族机制可用；文本标注方向）
        3. 全部 >= 5 个 |IC| < 0.005 → insight（全局 alpha 稀薄预警）

        去重：同 (kind, family) 只保留一行，occurrence_count 累加（_upsert_experience），
        不再按具体 IC 值生成重复行。返回 {kind: n} 计数。
        """
        formed = {"success_patterns": 0, "forbidden": 0, "insights": 0}
        if not batch_results:
            return formed
        with self._open() as conn:
            for family_name, members in self._group_by_family(batch_results).items():
                ics = [
                    (float(m.get("metrics", {}).get("ic", 0) or 0))
                    for m in members
                ]
                abs_ics = [abs(ic) for ic in ics]
                n = len(members)
                if n < 2:
                    continue

                all_weak = all(a < 0.01 for a in abs_ics)
                any_promising = any(a >= 0.02 for a in abs_ics)

                if all_weak and n >= 3:
                    content = (
                        f"{family_name} 信号族在 {n} 次尝试中 |IC| 均低于 0.01"
                        f"（最高 {max(abs_ics):.4f}），该方向在当前数据/label 下可能已饱和；"
                        f"除非引入新变量或交互机制，不建议机械重复。"
                    )
                    self._upsert_experience(
                        conn,
                        kind="forbidden",
                        name=f"family_saturated:{family_name}",
                        content=content,
                        evidence={
                            "factor_names": [m.get("factor_name") for m in members],
                            "abs_ic_max": round(max(abs_ics), 6),
                            "n_attempts": n,
                            "run_id": run_id,
                            "turn": turn,
                        },
                        example_factor=str(members[0].get("factor_name") or "") or None,
                        run_id=run_id,
                    )
                    formed["forbidden"] += 1

                if any_promising:
                    best = max(
                        members,
                        key=lambda m: abs(float(m.get("metrics", {}).get("ic", 0) or 0)),
                    )
                    best_ic = float(best.get("metrics", {}).get("ic", 0) or 0)
                    direction = (
                        "方向为负，反向构造同样有效" if best_ic < 0 else "方向为正"
                    )
                    content = (
                        f"{family_name} 信号族中 {best.get('factor_name')} 的 |IC| 达 "
                        f"{abs(best_ic):.4f}（{direction}），其经济机制值得在邻近空间继续探索；"
                        f"建议变异方向：换窗口、换修饰算子、引入正交交互。"
                    )
                    self._upsert_experience(
                        conn,
                        kind="success_pattern",
                        name=f"family_mechanism:{family_name}",
                        content=content,
                        evidence={
                            "best_factor": best.get("factor_name"),
                            "best_abs_ic": round(abs(best_ic), 6),
                            "best_ic_signed": round(best_ic, 6),
                            "n_attempts": n,
                            "run_id": run_id,
                            "turn": turn,
                        },
                        example_factor=str(best.get("factor_name") or "") or None,
                        run_id=run_id,
                    )
                    formed["success_patterns"] += 1

            # 全局洞察
            all_abs_ics = [
                abs(float(r.get("metrics", {}).get("ic", 0) or 0))
                for r in batch_results
            ]
            if len(all_abs_ics) >= 5 and max(all_abs_ics) < 0.005:
                content = (
                    f"连续 {len(all_abs_ics)} 个因子 |IC| 均低于 0.005，"
                    f"当前数据/label 组合下 alpha 可能极度稀薄；"
                    f"建议：切换 label 列、扩展数据源、或尝试交互信号。"
                )
                self._upsert_experience(
                    conn,
                    kind="insight",
                    name="global_alpha_thin",
                    content=content,
                    evidence={
                        "n_consecutive": len(all_abs_ics),
                        "max_abs_ic": round(max(all_abs_ics), 6),
                        "run_id": run_id,
                        "turn": turn,
                    },
                    run_id=run_id,
                )
                formed["insights"] += 1
        return formed