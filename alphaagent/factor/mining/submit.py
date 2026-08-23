"""挖掘会话内因子交付入库。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from alphaagent.factor.types import IngestPolicy
from alphaagent.factor.ingest import ingest_factor, load_panel_for_zoo
from alphaagent.factor.mining.service import StockEvalService
from alphaagent.factor.zoo import DEFAULT_FACTORLIB_ROOT, FactorZoo, init_library
from alphaagent.factor.zoo.realign import panel_paths_match, realign_factorlib_to_panel
from alphaagent.factor.mining.registry_io import upsert_mining_registry


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
    if icir is None or float(icir) <= float(criteria["min_icir"]):
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
) -> tuple[bool, list[str]]:
    """精筛入库池：满足实盘级统计、收益、稳健性和独立性要求。"""
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
    if icir is None or float(icir) <= float(criteria["min_icir"]):
        reasons.append("icir")
    fmb_t = (metrics.get("mls_fmb") or {}).get("nw_t_ls")
    min_t = criteria.get("min_fmb_t_stat", 0)
    if min_t and (fmb_t is None or abs(float(fmb_t)) < float(min_t)):
        reasons.append("fmb_t_stat")
    long_excess = metrics.get("long_group_annual_excess_return")
    if long_excess is None or float(long_excess) <= float(criteria["min_long_group_annual_excess_return"]):
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

    def _open_or_init_candidate_zoo(self, *, panel, panel_path: Path, production_zoo: FactorZoo) -> FactorZoo:
        try:
            return FactorZoo.open(self.candidate_factorlib_path)
        except FileNotFoundError:
            init_library(
                self.candidate_factorlib_path,
                panel=panel,
                panel_path=panel_path,
                n_sample_rows=production_zoo.manifest.n_sample_rows,
                max_factors=production_zoo.manifest.max_factors,
                sample_seed=production_zoo.manifest.sample_seed,
            )
            return FactorZoo.open(self.candidate_factorlib_path)

    def submit(
        self,
        session_id: str,
        *,
        multi_line_expr: str,
        factor_name: str,
        comment: str,
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
        assessment = ingest_factor(
            zoo,
            factor_id=factor_id,
            name=name,
            expr=expr,
            panel=panel,
            policy=stage_one_policy,
            dry_run=True,
        )
        stage_one_ok, stage_one_reasons = _check_stage_one(assessment.metrics, assessment.similarity, candidate_criteria)

        payload: dict[str, Any] = {
            "ok": False,
            "stored": False,
            "factor_id": factor_id,
            "factor_name": name,
            "comment": comment.strip(),
            "eval_range": {"start": ctx.train_start, "end": ctx.val_end},
            "metrics": assessment.metrics,
            "similarity": assessment.similarity,
            "candidate_stored": False,
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

        candidate_zoo = self._open_or_init_candidate_zoo(
            panel=panel, panel_path=ctx.panel_path, production_zoo=zoo
        )
        candidate_result = ingest_factor(
            candidate_zoo,
            factor_id=factor_id,
            name=name,
            expr=expr,
            panel=panel,
            policy=IngestPolicy.from_context(ctx, max_cs_corr=1.0, similar_top_k=self.similar_top_k),
            overwrite=True,
        )
        if not candidate_result.stored:
            payload["skipped_reason"] = candidate_result.skipped_reason
            payload["error_type"] = "CandidateIngestError"
            payload["error"] = candidate_result.skipped_reason or "candidate_ingest_failed"
            return payload

        candidate_reg, candidate_dsl = upsert_mining_registry(
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
            ingest_status="candidate",
            source="submit_stage_one",
        )
        payload.update(
            candidate_stored=True,
            candidate_factorlib_path=str(self.candidate_factorlib_path),
            candidate_registry_path=candidate_reg,
            candidate_dsl_path=candidate_dsl,
        )

        stage_two_ok, stage_two_reasons = _check_stage_two(assessment.metrics, assessment.similarity, production_criteria)
        payload["delivery_check"]["stage_two"] = {
            "passed": stage_two_ok,
            "fail_reasons": stage_two_reasons,
        }
        if not stage_two_ok:
            payload["skipped_reason"] = f"stage_two_failed:{','.join(stage_two_reasons)}"
            payload["error_type"] = "StageTwoDeliveryCheckError"
            payload["error"] = payload["skipped_reason"]
            return payload

        result = ingest_factor(
            zoo,
            factor_id=factor_id,
            name=name,
            expr=expr,
            panel=panel,
            policy=IngestPolicy.from_context(ctx, max_cs_corr=float(production_criteria.get("max_abs_corr", 0.5)), similar_top_k=self.similar_top_k),
            overwrite=self.overwrite,
        )
        if not result.stored:
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
        )
        payload.update(
            ok=True,
            stored=True,
            registry_path=reg_path,
            dsl_path=dsl_path,
            factorlib_path=str(self.factorlib_path),
            skipped_reason=None,
        )

        return payload


def default_factorlib_path(repo_root: Path) -> Path:
    _ = repo_root  # 兼容旧签名；AlphaAgent 使用固定 artifacts 路径
    return DEFAULT_FACTORLIB_ROOT
