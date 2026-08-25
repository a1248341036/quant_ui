"""挖掘会话内因子交付入库。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Callable

import numpy as np

from alphaagent.factor.types import IngestPolicy
from alphaagent.factor.ingest import ingest_factor, load_panel_for_zoo, prepare_stored_values
from alphaagent.factor.mining.service import StockEvalService
from alphaagent.factor.zoo import DEFAULT_FACTORLIB_ROOT, FactorZoo
from alphaagent.factor.zoo.realign import panel_paths_match, realign_factorlib_to_panel
from alphaagent.factor.mining.registry_io import (
    set_candidate_review,
    set_candidate_promotion,
    upsert_mining_registry,
    write_candidate_registry,
)


def slug_factor_id(name: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name).strip().lower())
    return re.sub(r"_+", "_", s).strip("_") or "factor"


def _check_stage_one(
    metrics: dict[str, Any],
    similarity: dict[str, Any] | None,
    criteria: dict[str, Any] | None = None,
) -> tuple[bool, list[str]]:
    """海选宽松池：保留逻辑候选，但不写入正式因子库。"""
    reasons: list[str] = []
    criteria = criteria or {"min_abs_ic": 0.015, "min_icir": 0.2, "min_coverage": 0.85, "max_abs_corr": 0.6}
    ic = metrics.get("ic")
    if ic is None or abs(float(ic)) < float(criteria["min_abs_ic"]):
        reasons.append("ic")
    icir = metrics.get("icir")
    if icir is None or abs(float(icir)) <= float(criteria["min_icir"]):
        reasons.append("icir")
    cov = metrics.get("coverage")
    if cov is None or float(cov) <= float(criteria["min_coverage"]):
        reasons.append("coverage")
    corr = (similarity or {}).get("max_abs_corr", 0.0)
    if corr is None or float(corr) >= float(criteria["max_abs_corr"]):
        reasons.append("max_cs_corr")
    return len(reasons) == 0, reasons


def _check_stage_two(
    metrics: dict[str, Any],
    similarity: dict[str, Any] | None,
    criteria: dict[str, Any] | None = None,
    rebalance_freq: str = "daily",
) -> tuple[bool, list[str]]:
    """精筛统计门槛。TopN 可交易性（超额/Sharpe/回撤/尾部稳定）由
    engine_gate 用完整回测引擎裁决，此处只做统计族检查。
    rebalance_freq 仅用于审计透传。"""
    _ = rebalance_freq
    reasons: list[str] = []
    criteria = criteria or {
        "min_abs_ic": 0.035, "min_icir": 0.5, "min_fmb_t_stat": 2.5,
        "min_long_group_annual_excess_return": 0.03,
        "max_winsorized_abs_ic_decay": 0.10, "max_abs_corr": 0.4,
    }
    ic = metrics.get("ic")
    if ic is None or abs(float(ic)) < float(criteria["min_abs_ic"]):
        reasons.append("ic")
    icir = metrics.get("icir")
    if icir is None or abs(float(icir)) <= float(criteria["min_icir"]):
        reasons.append("icir")
    fmb_t = (metrics.get("mls_fmb") or {}).get("nw_t_ls")
    min_t = criteria.get("min_fmb_t_stat", 0)
    if min_t and (fmb_t is None or abs(float(fmb_t)) < float(min_t)):
        reasons.append("fmb_t_stat")
    long_excess = metrics.get("long_group_annual_excess_return")
    if long_excess is None or abs(float(long_excess)) <= float(criteria["min_long_group_annual_excess_return"]):
        reasons.append("long_group_annual_excess_return")
    winsor_decay = metrics.get("winsorized_abs_ic_decay")
    if winsor_decay is None or float(winsor_decay) > float(criteria["max_winsorized_abs_ic_decay"]):
        reasons.append("winsorized_abs_ic_decay")
    corr = (similarity or {}).get("max_abs_corr", 0.0)
    if corr is None or float(corr) >= float(criteria["max_abs_corr"]):
        reasons.append("max_cs_corr")
    return len(reasons) == 0, reasons


class FactorSubmitService:
    """两阶段提交：海选候选池，再精筛进入正式 FactorZoo。"""

    def __init__(
        self,
        service: StockEvalService,
        *,
        factorlib_path: Path,
        registry_path: Path,
        expr_dir: Path,
        repo_root: Path,
        max_cs_corr: float = 0.8,
        delivery_policy: dict[str, Any] | None = None,
        similar_top_k: int = 3,
        overwrite: bool = False,
        auto_realign_panel: bool = True,
    ) -> None:
        self.service = service
        self.factorlib_path = Path(factorlib_path).expanduser().resolve()
        self.registry_path = Path(registry_path).expanduser().resolve()
        self.expr_dir = Path(expr_dir).expanduser().resolve()
        self.repo_root = Path(repo_root).resolve()
        self.max_cs_corr = max_cs_corr
        self.delivery_policy = delivery_policy or {}
        self.similar_top_k = similar_top_k
        self.overwrite = overwrite
        self.auto_realign_panel = auto_realign_panel

    @property
    def candidate_factorlib_path(self) -> Path:
        return self.factorlib_path.parent / "candidate_1d"

    @property
    def candidate_registry_path(self) -> Path:
        return self.candidate_factorlib_path / "mining_candidate_registry.json"

    @property
    def candidate_expr_dir(self) -> Path:
        return self.candidate_factorlib_path / "expressions"

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

        try:
            panel = load_panel_for_zoo(zoo, panel_path=ctx.panel_path)
        except ValueError as e:
            if not self.auto_realign_panel:
                return {
                    "ok": False,
                    "stored": False,
                    "error": str(e),
                    "error_type": "PanelMismatchError",
                }
            zoo_panel = Path(zoo.manifest.panel_path)
            if not panel_paths_match(ctx.panel_path, zoo_panel):
                return {
                    "ok": False,
                    "stored": False,
                    "error": (
                        f"{e}; panel 路径不一致: session={ctx.panel_path} zoo={zoo_panel}，"
                        "请用 --panel 与因子库相同文件，或重新 init_factorlib"
                    ),
                    "error_type": "PanelMismatchError",
                }
            try:
                from alphaagent.data.adapters.cnequity import is_cne_source, load_panel_from_cne
                if is_cne_source(ctx.panel_path):
                    full_panel = load_panel_from_cne(universe_mask=False).sort_index()
                else:
                    from alphaagent.data.panel import load_panel as _load_panel
                    full_panel = _load_panel(ctx.panel_path).sort_index()
                realign_info = realign_factorlib_to_panel(
                    self.factorlib_path,
                    panel=full_panel,
                    panel_path=ctx.panel_path,
                )
                zoo = FactorZoo.open(self.factorlib_path)
                panel = full_panel
            except Exception as exc:  # noqa: BLE001
                return {
                    "ok": False,
                    "stored": False,
                    "error": f"panel_realign_failed: {exc}",
                    "error_type": "PanelRealignError",
                }
        else:
            realign_info = None

        if realign_info is None and len(panel) != zoo.manifest.n_rows:
            return {
                "ok": False,
                "stored": False,
                "error": (
                    f"panel 行数 {len(panel)} != 库 n_rows {zoo.manifest.n_rows}；"
                    "请用相同 panel 初始化库，或仅用于调试切片"
                ),
                "error_type": "PanelMismatchError",
            }

        # First evaluate against the production library only. dry_run avoids
        # temporarily adding a weak candidate to the formal FactorZoo.
        candidate_criteria = self.delivery_policy.get("candidate", {})
        production_criteria = self.delivery_policy.get("production", {})
        stage_one_policy = IngestPolicy.from_context(
            ctx, max_cs_corr=float(candidate_criteria.get("max_abs_corr", 0.6)), similar_top_k=self.similar_top_k
        )
        prepared_values, _, _, _ = prepare_stored_values(expr, panel, zoo, stage_one_policy)
        assessment = ingest_factor(
            zoo,
            factor_id=factor_id,
            name=name,
            expr=expr,
            panel=panel,
            policy=stage_one_policy,
            stored_values=prepared_values,
            dry_run=True,
        )
        stage_one_ok, stage_one_reasons = _check_stage_one(assessment.metrics, assessment.similarity, candidate_criteria)

        payload: dict[str, Any] = {
            "ok": False,
            "stored": False,
            "factor_id": factor_id,
            "factor_name": name,
            "comment": comment.strip(),
            "interaction": interaction,
            "eval_range": {"start": ctx.train_start, "end": ctx.val_end},
            "metrics": assessment.metrics,
            "similarity": assessment.similarity,
            "candidate_stored": False,
            "rebalance_freq": chosen_freq,
            "delivery_check": {
                "stage_one": {"passed": stage_one_ok, "fail_reasons": stage_one_reasons},
                "stage_two": {"passed": False, "fail_reasons": []},
            },
        }
        if realign_info and realign_info.get("realigned"):
            payload["panel_realigned"] = realign_info

        if not stage_one_ok:
            payload["skipped_reason"] = f"stage_one_failed:{','.join(stage_one_reasons)}"
            payload["error_type"] = "StageOneDeliveryCheckError"
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
                "metrics": assessment.metrics,
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
            metrics=assessment.metrics,
            similarity=assessment.similarity,
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

        stage_two_ok, stage_two_reasons = _check_stage_two(assessment.metrics, assessment.similarity, production_criteria, rebalance_freq=chosen_freq)
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
            ic_sign = 1 if float(assessment.metrics.get("ic") or 0.0) >= 0 else -1
            gate = run_engine_gate(
                panel,
                prepared_values,
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

        result = ingest_factor(
            zoo,
            factor_id=factor_id,
            name=name,
            expr=expr,
            panel=panel,
            policy=IngestPolicy.from_context(ctx, max_cs_corr=float(production_criteria.get("max_abs_corr", 0.5)), similar_top_k=self.similar_top_k),
            stored_values=prepared_values,
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
