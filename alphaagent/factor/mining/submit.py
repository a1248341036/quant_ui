"""挖掘会话内因子交付入库。"""

from __future__ import annotations

import dataclasses
import re
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from alphaagent.factor.types import IngestPolicy
from alphaagent.factor.ingest import (
    align_values_to_rows,
    clip_values,
    compute_ingest_metrics,
    ingest_factor,
    materialize_factor,
)
from alphaagent.factor.metrics import annualized_long_group_excess_return
from alphaagent.factor.mining.service import StockEvalService
from alphaagent.factor.zoo import DEFAULT_FACTORLIB_ROOT, FactorZoo, SimilarityMatrix
from alphaagent.factor.zoo.similarity import (
    SIMILARITY_BASIS_SAMPLED,
    SIMILARITY_KIND,
    cross_sectional_pearson_mean,
)
from alphaagent.factor.mining.registry_io import (
    load_mining_registry,
    set_candidate_review,
    set_candidate_promotion,
    upsert_mining_registry,
    write_candidate_registry,
)
from core import factor_categories


def _candidate_registry_similarity(
    cand_values: np.ndarray,
    panel: pd.DataFrame,
    candidate_registry_path: Path,
    *,
    exclude_factor_id: str | None = None,
    min_pairs: int = 30,
    top_k: int = 3,
) -> dict[str, Any] | None:
    """候选因子与候选 registry 中已有因子的截面 Pearson 相似度。

    候选 registry 中的因子只有 DSL 表达式、无 dense values，需要在会话域
    panel 上重新求值。返回与 SimilarityMatrix.cross_sectional_neighbor_report
    同结构的 dict，或 None（registry 为空/无可比因子时）。
    """
    from alphaagent.dsl import eval_factor
    from alphaagent.factor.align import align_series_to_panel

    registry = load_mining_registry(candidate_registry_path)
    if not registry:
        return None

    new_series = pd.Series(np.asarray(cand_values, dtype=np.float64), index=panel.index)
    corrs: list[tuple[str, float, str]] = []  # (factor_id, corr, name)

    for fid, entry in sorted(registry.items()):
        if fid == exclude_factor_id or not isinstance(entry, dict):
            continue
        expr_text = str(entry.get("expr") or "").strip()
        if not expr_text:
            continue
        try:
            old_raw = eval_factor(expr_text, panel)
            if not isinstance(old_raw, pd.Series):
                continue
            old_aligned = align_series_to_panel(old_raw, panel)
        except Exception:
            continue
        corr = cross_sectional_pearson_mean(
            np.asarray(new_series, dtype=np.float64),
            np.asarray(old_aligned, dtype=np.float64),
            panel.index,
            min_pairs=min_pairs,
        )
        if np.isfinite(corr):
            name = str(entry.get("name") or fid)
            corrs.append((fid, float(corr), name))

    if not corrs:
        return None

    corrs.sort(key=lambda p: abs(p[1]), reverse=True)
    max_abs = max(abs(c) for _, c, _ in corrs)
    top_slice = corrs[:top_k] if top_k > 0 else []

    return {
        "kind": SIMILARITY_KIND,
        "basis": "candidate_registry_reeval",
        "max_abs_corr": max_abs,
        "top_neighbors": [
            {"factor_id": fid, "name": name, "cs_corr": corr}
            for fid, corr, name in top_slice
        ],
    }


def slug_factor_id(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", s).strip("_") or "factor"


def _stage_one_stats_reasons(
    metrics: dict[str, Any],
    criteria: dict[str, Any] | None = None,
) -> list[str]:
    """海选统计门槛（IC/ICIR/coverage）。须喂 train-only 指标，防止 val 衰减被混合窗口稀释。"""
    reasons: list[str] = []
    criteria = criteria or {"min_abs_ic": 0.015, "min_icir": 0.25, "min_coverage": 0.85}
    ic = metrics.get("ic")
    if ic is None or abs(float(ic)) < float(criteria["min_abs_ic"]):
        reasons.append("ic")
    icir = metrics.get("icir")
    if icir is None or abs(float(icir)) <= float(criteria["min_icir"]):
        reasons.append("icir")
    cov = metrics.get("coverage") or metrics.get("factor_coverage")
    if cov is None or float(cov) <= float(criteria["min_coverage"]):
        reasons.append("coverage")
    return reasons


def _stage_one_turnover_reasons(
    metrics: dict[str, Any],
    criteria: dict[str, Any] | None = None,
) -> list[str]:
    """换手可行性门槛：截面自相关低于阈值 → 排名日度剧变，换手吃掉 alpha。"""
    criteria = criteria or {}
    min_ac = float(criteria.get("min_cs_autocorr", 0.18))
    if min_ac <= 0:
        return []
    ac = metrics.get("cs_pearson_autocorr")
    if ac is None or not np.isfinite(float(ac)) or float(ac) < min_ac:
        return ["cs_autocorr"]
    return []


def _stage_one_val_retention_reasons(
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    criteria: dict[str, Any] | None = None,
) -> list[str]:
    """样本外保留比门槛：|val_ic|/|train_ic| ≥ 阈值且方向不反转。

    val 窗口无数据时跳过（train-only 会话）。
    """
    criteria = criteria or {}
    min_ratio = float(criteria.get("min_val_ic_retention", 0.5))
    n_days_val = val_metrics.get("n_days") or val_metrics.get("n_instruments")
    if n_days_val is not None and int(n_days_val) == 0:
        return []
    t_ic = train_metrics.get("ic")
    v_ic = val_metrics.get("ic")
    if t_ic is None or v_ic is None:
        return ["val_ic_missing"]
    t, v = float(t_ic), float(v_ic)
    if not np.isfinite(t) or not np.isfinite(v):
        return ["val_ic_missing"]
    if t * v < 0:
        return ["val_sign_flip"]
    if abs(t) > 1e-12 and abs(v) / abs(t) < min_ratio:
        return ["val_retention"]
    return []


def _check_stage_one(
    metrics: dict[str, Any],
    similarity: dict[str, Any] | None,
    criteria: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """海选宽松池：保留逻辑候选，但不写入正式因子库。"""
    reasons: list[str] = []
    criteria = criteria or {"min_abs_ic": 0.015, "min_icir": 0.25, "min_coverage": 0.85, "max_abs_corr": 0.6}
    reasons.extend(_stage_one_stats_reasons(metrics, criteria))
    corr = (similarity or {}).get("max_abs_corr", 0.0)
    if corr is None or float(corr) >= float(criteria["max_abs_corr"]):
        reasons.append("max_cs_corr")
    return len(reasons) == 0, reasons


def _check_stage_two(
    train_metrics: dict[str, Any],
    val_metrics: dict[str, Any],
    similarity: dict[str, Any] | None,
    criteria: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """精筛统计门槛（双窗口口径，2026-08 重构）。

    - train 窗口：|IC| 与 ICIR 各自达标；
    - val 窗口：绝对水平 + 相对 train 的保留比（方向反转拦截）；
    - 截尾 IC 衰减取 train 窗口值；
    - TopN 可交易性（超额/Sharpe/回撤/尾部稳定）由 engine_gate 用完整回测
      引擎净值裁决，此处不做代理指标模拟。

    已移除的摆设门槛（研究结论 2026-08）：fmb/ls t 值在长样本上永不拦截；
    毛值 quantile 三项与可交易性脱节。
    """
    reasons: list[str] = []
    criteria = criteria or {
        "min_train_abs_ic": 0.025, "min_train_icir": 0.30,
        "min_val_abs_ic": 0.015, "min_val_ic_retention": 0.60,
        "min_val_long_excess": 0.0,
        "max_winsorized_abs_ic_decay": 0.10, "max_abs_corr": 0.4,
    }
    ic = train_metrics.get("ic")
    if ic is None or abs(float(ic)) < float(criteria.get("min_train_abs_ic", 0.025)):
        reasons.append("train_ic")
    icir = train_metrics.get("icir")
    if icir is None or abs(float(icir)) <= float(criteria.get("min_train_icir", 0.30)):
        reasons.append("train_icir")

    v_ic = val_metrics.get("ic")
    if v_ic is None or abs(float(v_ic)) < float(criteria.get("min_val_abs_ic", 0.015)):
        reasons.append("val_ic")
    reasons += _stage_one_val_retention_reasons(train_metrics, val_metrics, criteria)

    # val 多头端毛值超额（方向自适应十分组，复利年化）：
    # IC 为正 ≠ 多头组合为正——alpha 可能全在空头端/中段排名（2026-08 审计发现的盲区）。
    thr_vle = criteria.get("min_val_long_excess")
    if thr_vle is not None:
        vle = val_metrics.get("val_long_excess")
        if vle is None or not np.isfinite(float(vle)) or float(vle) <= float(thr_vle):
            reasons.append("val_long_excess")

    winsor_decay = train_metrics.get("winsorized_abs_ic_decay")
    if winsor_decay is None or float(winsor_decay) > float(criteria.get("max_winsorized_abs_ic_decay", 0.10)):
        reasons.append("winsorized_abs_ic_decay")

    corr = (similarity or {}).get("max_abs_corr", 0.0)
    if corr is None or float(corr) >= float(criteria.get("max_abs_corr", 0.4)):
        reasons.append("max_cs_corr")
    return len(reasons) == 0, reasons


class FactorSubmitService:
    """两阶段提交：海选候选池，再精筛进入正式 FactorZoo。"""

    def __init__(
        self,
        service: StockEvalService,
        *,
        factorlib_path: Path | None = None,
        registry_path: Path | None = None,
        expr_dir: Path | None = None,
        repo_root: Path,
        research_mode: str = "technical",
        max_cs_corr: float = 0.8,
        delivery_policy: dict[str, Any] | None = None,
        similar_top_k: int = 3,
        overwrite: bool = False,
        auto_realign_panel: bool = True,
    ) -> None:
        self.service = service
        self.research_mode = research_mode
        self.factorlib_path = (
            Path(factorlib_path).expanduser().resolve()
            if factorlib_path is not None
            else factor_categories.production_dir(research_mode)
        )
        self.registry_path = (
            Path(registry_path).expanduser().resolve()
            if registry_path is not None
            else factor_categories.production_registry_path(research_mode)
        )
        self.expr_dir = (
            Path(expr_dir).expanduser().resolve()
            if expr_dir is not None
            else factor_categories.production_expr_dir(research_mode)
        )
        self.repo_root = Path(repo_root).resolve()
        self.max_cs_corr = max_cs_corr
        self.delivery_policy = delivery_policy or {}
        self.similar_top_k = similar_top_k
        self.overwrite = overwrite
        self.auto_realign_panel = auto_realign_panel

    @property
    def candidate_factorlib_path(self) -> Path:
        return factor_categories.candidate_dir(self.research_mode)

    @property
    def candidate_registry_path(self) -> Path:
        return factor_categories.candidate_registry_path(self.research_mode)

    @property
    def candidate_expr_dir(self) -> Path:
        return factor_categories.candidate_expr_dir(self.research_mode)

    def submit(
        self,
        session_id: str,
        *,
        multi_line_expr: str,
        factor_name: str,
        comment: str,
        evaluation_evidence: dict[str, Any] | None = None,
        review_hook: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        orthogonality_hook: Callable[[], dict[str, Any]] | None = None,
        interaction: dict[str, Any] | None = None,
        rebalance_freq: str | None = None,
    ) -> dict[str, Any]:
        expr = multi_line_expr.strip()
        if not expr:
            return {
                "ok": False,
                "stored": False,
                "error": "multi_line_expr_required_non_empty_string",
                "error_type": "ToolArgumentsError",
            }
        if not str(comment).strip():
            return {
                "ok": False,
                "stored": False,
                "error": "comment_required_non_empty_string",
                "error_type": "ToolArgumentsError",
            }

        factor_id = slug_factor_id(factor_name)
        name = str(factor_name).strip() or factor_id

        engine_cfg = self.delivery_policy.get("production", {}).get("engine_gate") or {}
        allowed_freqs = [str(f).lower() for f in (engine_cfg.get("allowed_freqs") or ["daily"])]
        chosen_freq = str(rebalance_freq or engine_cfg.get("freq") or "daily").lower()
        if chosen_freq not in allowed_freqs:
            return {
                "ok": False,
                "stored": False,
                "error": f"invalid_rebalance_freq: {chosen_freq} (allowed={allowed_freqs})",
                "error_type": "ToolArgumentsError",
            }

        try:
            session = self.service.sessions.get(session_id)
        except KeyError:
            return {
                "ok": False,
                "stored": False,
                "error": f"session_not_found: {session_id}",
                "error_type": "SessionError",
            }

        ctx = session.ctx
        try:
            zoo = FactorZoo.open(self.factorlib_path)
        except FileNotFoundError as e:
            return {
                "ok": False,
                "stored": False,
                "error": f"factorlib_not_initialized: {self.factorlib_path}",
                "error_type": "FactorLibError",
                "detail": str(e),
            }

        # ③ 会话域复用：submit 全程在挖掘会话驻内存 panel 上评估，
        #    不再全量重载因子库域 panel（两者行集本就不同——库含全部历史；
        #    旧行为的二次构建曾触发 float64 合并的 8.69GiB OOM）。
        #    相似度抽样与最终入库通过键映射 (datetime, instrument) 对齐。
        panel = session.panel
        if panel is None or len(panel) == 0:
            return {
                "ok": False,
                "stored": False,
                "error": "session_panel_empty",
                "error_type": "SessionError",
            }

        candidate_criteria = self.delivery_policy.get("candidate", {})
        production_criteria = self.delivery_policy.get("production", {})
        stage_one_policy = IngestPolicy.from_context(
            ctx, max_cs_corr=float(candidate_criteria.get("max_abs_corr", 0.6)), similar_top_k=self.similar_top_k
        )

        # ①+③ 会话域物化：长度恒等于 panel 行数，指标直接可算。
        materialized = materialize_factor(expr, panel)
        cand_values = materialized.values
        if stage_one_policy.clip_pct is not None:
            lo_pct, hi_pct = float(stage_one_policy.clip_pct[0]), float(stage_one_policy.clip_pct[1])
            cand_values, _ = clip_values(cand_values, lower_pct=lo_pct, upper_pct=hi_pct)
        values_by_key = pd.Series(cand_values, index=panel.index)

        # ① 统计门槛前置（train-only 准入口径 + 换手可行性）：
        #    - 准入看 train 窗口，防止 val 衰减被混合窗口稀释；
        #    - cs_autocorr 硬门淘汰排名日度剧变的不可交付因子；
        #    - 任一不达标直接拒绝，跳过最贵的相似度对比。
        metrics_train = compute_ingest_metrics(
            cand_values, panel, dataclasses.replace(stage_one_policy, val_end=ctx.train_end)
        )
        gate_reasons = _stage_one_stats_reasons(metrics_train, candidate_criteria)
        gate_reasons += _stage_one_turnover_reasons(metrics_train, candidate_criteria)

        val_metrics: dict[str, Any] = {}
        if not gate_reasons and ctx.val_start > ctx.train_end:
            # 样本外保留比：|val_ic|/|train_ic| ≥ 阈值且方向不反转
            val_metrics = compute_ingest_metrics(
                cand_values, panel, dataclasses.replace(stage_one_policy, train_start=ctx.val_start)
            )
            gate_reasons += _stage_one_val_retention_reasons(metrics_train, val_metrics, candidate_criteria)
            # val 多头端毛值超额（方向自适应）：IC 为正不代表多头组合为正，
            # alpha 可能全在空头端/中段排名——纯多头可交易口径必须单独为正。
            label_col = stage_one_policy.label_col
            if label_col in panel.columns:
                dt_level = panel.index.get_level_values("datetime")
                val_rows = (dt_level >= pd.Timestamp(ctx.val_start)) & (dt_level <= pd.Timestamp(ctx.val_end))
                f_val = pd.Series(cand_values, index=panel.index)[val_rows]
                l_val = panel[label_col][val_rows]
                if len(f_val) > 0:
                    dir_sign = 1 if float(metrics_train.get("ic") or 0.0) >= 0 else -1
                    val_metrics["val_long_excess"] = annualized_long_group_excess_return(
                        f_val, l_val, direction=dir_sign
                    )

        similarity_report: dict[str, Any] | None = None
        if not gate_reasons:
            # ── 正式库相似度（抽样行快速口径）──
            zoo_report: dict[str, Any] | None = None
            if zoo.n_factors > 0:
                cand_sample = align_values_to_rows(
                    values_by_key, zoo.index.rows.iloc[zoo.index.sample_row_ids]
                )
                sim_matrix = SimilarityMatrix(zoo.paths, zoo.manifest.max_factors)
                zoo_report = sim_matrix.cross_sectional_neighbor_report(
                    zoo, None, top_k=stage_one_policy.similar_top_k,
                    candidate_sample=cand_sample,
                )

            # ── 候选 registry 相似度（在会话域 panel 上对已有候选 DSL 重新求值）──
            cand_report = _candidate_registry_similarity(
                cand_values, panel, self.candidate_registry_path,
                exclude_factor_id=factor_id,
                top_k=stage_one_policy.similar_top_k,
            )

            # ── 合并两份报告取 max_abs_corr 更大者 ──
            reports = [r for r in (zoo_report, cand_report) if r is not None]
            if reports:
                best = max(reports, key=lambda r: float(r.get("max_abs_corr", 0.0)))
                # 合并 top_neighbors
                all_neighbors: list[dict[str, Any]] = []
                seen: set[str] = set()
                for r in reports:
                    for nb in (r.get("top_neighbors") or []):
                        fid = str(nb.get("factor_id") or "")
                        if fid and fid not in seen:
                            seen.add(fid)
                            all_neighbors.append(nb)
                all_neighbors.sort(key=lambda nb: abs(float(nb.get("cs_corr", 0))), reverse=True)
                similarity_report = {
                    "kind": SIMILARITY_KIND,
                    "basis": "zoo+candidate_registry",
                    "max_abs_corr": float(best.get("max_abs_corr", 0.0)),
                    "top_neighbors": all_neighbors[:stage_one_policy.similar_top_k],
                }
            else:
                similarity_report = {
                    "kind": SIMILARITY_KIND,
                    "basis": SIMILARITY_BASIS_SAMPLED,
                    "max_abs_corr": 0.0,
                    "top_neighbors": [],
                }
        # 准入判定：统计/换手/保留比任一不过即拒；全过再叠加相似度 corr 检查。
        stage_one_ok = not gate_reasons
        stage_one_reasons = list(gate_reasons)
        if stage_one_ok:
            stage_one_ok, stage_one_reasons = _check_stage_one(
                metrics_train, similarity_report, candidate_criteria
            )

        # 上报/存档指标仍用全窗口（与历史 registry 口径一致），附 train/val 分解。
        metrics = compute_ingest_metrics(cand_values, panel, stage_one_policy)
        reported = dict(metrics)
        for src, prefix in ((metrics_train, "train"), (val_metrics, "val")):
            for key in ("ic", "icir", "rank_ic"):
                v = src.get(key)
                if v is not None and np.isfinite(float(v)):
                    reported[f"{prefix}_{key}"] = round(float(v), 6)
        t_ic, v_ic = metrics_train.get("ic"), val_metrics.get("ic")
        if t_ic is not None and v_ic is not None and abs(float(t_ic)) > 1e-12:
            reported["val_ic_retention"] = round(abs(float(v_ic) / float(t_ic)), 4)
        vle = val_metrics.get("val_long_excess")
        if vle is not None and np.isfinite(float(vle)):
            reported["val_long_excess"] = round(float(vle), 6)

        # 组合层收益指标（多头年化/夏普等）：供前端因子库与详情展示。
        if stage_one_ok:
            from alphaagent.factor.metrics import quantile_portfolio_metrics
            qp = quantile_portfolio_metrics(
                pd.Series(cand_values, index=panel.index), panel[ctx.label_col],
                n_groups=10, cost_bps=0.0,
            )
            reported["quantile_portfolio"] = {
                k: (round(float(v), 6) if isinstance(v, (int, float)) and np.isfinite(float(v)) else v)
                for k, v in qp.items()
                if k not in {"group_means"}
            }
        metrics = reported

        payload: dict[str, Any] = {
            "ok": False,
            "stored": False,
            "factor_id": factor_id,
            "factor_name": name,
            "comment": comment.strip(),
            "interaction": interaction,
            "eval_range": {"start": ctx.train_start, "end": ctx.val_end},
            "metrics": metrics,
            "similarity": similarity_report,
            "candidate_stored": False,
            "rebalance_freq": chosen_freq,
            "delivery_check": {
                "stage_one": {"passed": stage_one_ok, "fail_reasons": stage_one_reasons},
                "stage_two": {"passed": False, "fail_reasons": []},
            },
        }

        if not stage_one_ok:
            payload["skipped_reason"] = f"stage_one_failed:{','.join(stage_one_reasons)}"
            payload["error_type"] = "StageOneDeliveryCheckError"
            payload["error"] = payload["skipped_reason"]
            return payload

        # ── 正交性检查（stage_one 统一查，不再只在 approve 后触发）──
        # 所有通过统计门槛的因子（无论后续 verdict 是 approve 还是 revise）
        # 都需经过正交性检查，避免与正式库/候选池已有因子高度重复。
        orthogonality: dict[str, Any] = {
            "passed": True,
            "skipped_reason": "hook_not_configured",
        }
        if orthogonality_hook is not None:
            orthogonality = orthogonality_hook()
        payload["offline_orthogonality"] = orthogonality
        if not bool(orthogonality.get("passed")):
            reason = str(orthogonality.get("error") or orthogonality.get("skipped_reason") or "correlation_threshold")
            payload["promotion_status"] = "offline_orthogonality_blocked"
            payload["skipped_reason"] = f"offline_orthogonality_failed:{reason}"
            payload["error_type"] = "OfflineOrthogonalityError"
            payload["error"] = payload["skipped_reason"]
            return payload

        review: dict[str, Any] | None = None
        if review_hook is not None:
            review = review_hook({
                "multi_line_expr": expr,
                "factor_name": name,
                "comment": comment.strip(),
                "interaction": interaction,
                "candidate_stored": True,
                "metrics": metrics,
            })
        review_verdict = str((review or {}).get("verdict") or "").lower()
        review_status = (
            {"approve": "approved", "revise": "revise", "reject": "rejected"}.get(review_verdict, "pending_review")
        )
        payload.update(
            review_status=review_status,
        )

        if review_verdict == "reject":
            # 仅 reject 硬拦（抄袭/经典变换等垃圾候选不进任何库）；
            # revise 只是"暂缓转正式"，候选池仍按统计门槛照常收纳。
            payload["delivery_check"]["stage_two"] = {
                "passed": False,
                "fail_reasons": ["factor_review"],
            }
            payload["review"] = review
            payload["promotion_status"] = "review_rejected"
            payload["skipped_reason"] = f"factor_review:{review_verdict}"
            payload["error_type"] = "FactorReviewRejected"
            payload["error"] = f"factor_review_{review_verdict}_blocked"
            return payload

        # 统计达标 → 先入候选池（registry_only）；Reviewer 意见只影响是否继续冲正式库。
        candidate_reg, candidate_dsl = write_candidate_registry(
            self.candidate_registry_path,
            factor_id=factor_id,
            name=name,
            comment=comment.strip(),
            expr=expr,
            expr_dir=self.candidate_expr_dir,
            repo_root=self.repo_root,
            policy=stage_one_policy,
            metrics=metrics,
            similarity=similarity_report,
            source="submit_stage_one",
            evaluation_evidence=evaluation_evidence,
            interaction=interaction,
            data_fingerprint={
                "panel_path": str(ctx.panel_path),
                "index_hash": zoo.manifest.index_hash,
                "n_rows": int(zoo.manifest.n_rows),
            },
        )
        if review is not None:
            set_candidate_review(
                self.candidate_registry_path,
                factor_id=factor_id,
                review=review,
                promotion_status="pending",
            )
        payload.update(
            candidate_stored=True,
            candidate_storage="registry_only",
            candidate_registry_path=candidate_reg,
            candidate_dsl_path=candidate_dsl,
        )

        if review_verdict != "approve":
            payload["delivery_check"]["stage_two"] = {
                "passed": False,
                "fail_reasons": ["factor_review"],
            }
            payload["review"] = review
            payload["promotion_status"] = "review_blocked"
            payload["skipped_reason"] = (
                f"factor_review:{review_verdict}:stored_as_candidate_awaiting_revision"
            )
            # 候选已入库，不算交付失败；返回提示 Reviewer 意见供后续修订。
            payload["error_type"] = None
            payload["error"] = None
            return payload

        stage_two_ok, stage_two_reasons = _check_stage_two(
            metrics_train, val_metrics, similarity_report, production_criteria
        )
        payload["delivery_check"]["stage_two"] = {
            "passed": stage_two_ok,
            "fail_reasons": stage_two_reasons,
        }
        if not stage_two_ok:
            set_candidate_promotion(
                self.candidate_registry_path,
                factor_id=factor_id,
                promotion_status="stage_two_failed",
            )
            payload["skipped_reason"] = f"stage_two_failed:{','.join(stage_two_reasons)}"
            payload["error_type"] = "StageTwoDeliveryCheckError"
            payload["error"] = payload["skipped_reason"]
            return payload

        engine_gate_cfg = production_criteria.get("engine_gate")
        if isinstance(engine_gate_cfg, dict) and engine_gate_cfg.get("enabled"):
            from alphaagent.factor.mining.engine_gate import run_engine_gate
            ic_sign = 1 if float(metrics.get("ic") or 0.0) >= 0 else -1
            # 会话域回测：cand_values 与 session panel 行序一一对应。
            gate = run_engine_gate(
                panel,
                cand_values,
                val_start=ctx.val_start,
                val_end=ctx.val_end,
                direction=ic_sign,
                policy={**engine_gate_cfg, "freq": chosen_freq},
            )
            payload["engine_backtest"] = gate
            if not gate.get("passed"):
                set_candidate_promotion(
                    self.candidate_registry_path,
                    factor_id=factor_id,
                    promotion_status="engine_gate_failed",
                )
                payload["skipped_reason"] = (
                    f"engine_gate_failed:{','.join(gate.get('fail_reasons') or [])}"
                )
                payload["error_type"] = "EngineGateError"
                payload["error"] = payload["skipped_reason"]
                return payload

        # 真正入库才做一次 canonical 对齐（会话域 → 库行序），指标复用免重算。
        canonical_values = align_values_to_rows(values_by_key, zoo.index.rows)
        result = ingest_factor(
            zoo,
            factor_id=factor_id,
            name=name,
            expr=expr,
            panel=None,
            policy=IngestPolicy.from_context(ctx, max_cs_corr=float(production_criteria.get("max_abs_corr", 0.5)), similar_top_k=self.similar_top_k),
            stored_values=canonical_values,
            metrics_override=metrics,
            overwrite=self.overwrite,
        )
        if not result.stored:
            set_candidate_promotion(
                self.candidate_registry_path,
                factor_id=factor_id,
                promotion_status="production_ingest_failed",
            )
            payload["skipped_reason"] = result.skipped_reason
            payload["error_type"] = "ProductionIngestError"
            payload["error"] = result.skipped_reason or "production_ingest_failed"
            return payload

        policy = IngestPolicy.from_context(ctx, max_cs_corr=float(production_criteria.get("max_abs_corr", 0.5)), similar_top_k=self.similar_top_k)
        reg_path, dsl_path = upsert_mining_registry(
            self.registry_path,
            factor_id=factor_id,
            name=name,
            comment=comment.strip(),
            expr=expr,
            expr_dir=self.expr_dir,
            repo_root=self.repo_root,
            policy=policy,
            metrics=result.metrics,
            similarity=result.similarity,
            ingest_status="production",
            source="submit_stage_two",
            interaction=interaction,
        )
        set_candidate_promotion(
            self.candidate_registry_path,
            factor_id=factor_id,
            promotion_status="promoted",
        )
        payload.update(
            ok=True,
            stored=True,
            promotion_status="promoted",
            registry_path=reg_path,
            dsl_path=dsl_path,
            factorlib_path=str(self.factorlib_path),
            skipped_reason=None,
        )

        return payload


def default_factorlib_path(repo_root: Path) -> Path:
    _ = repo_root  # 兼容旧签名；AlphaAgent 使用固定 artifacts 路径
    return DEFAULT_FACTORLIB_ROOT
