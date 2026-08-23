"""股票因子挖掘评估服务：会话管理与 train/val 评估。"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from alphaagent.factor.mining.context import StockEvalContext
from alphaagent.factor.mining.env_settings import resolve_max_parallel_eval
from alphaagent.factor.mining.response import format_eval_response
from alphaagent.factor.mining.schemas import (
    EvalProfileRequest,
    EvalTrainRequest,
    EvalValRequest,
    SessionCreateRequest,
    SessionCreateResponse,
)
from alphaagent.factor.evaluation.engine import EvaluationEngine
from alphaagent.factor.evaluation.profile import EvaluationProfile, default_evaluation_profiles
from alphaagent.factor.mining.session import SessionStore
from alphaagent.factor.eval import evaluate_factor_on_split


class StockEvalService:
    """进程内评估服务，供 mining FactorEvalTools 调用。"""

    def __init__(
        self,
        *,
        sessions: SessionStore | None = None,
        max_parallel_eval: int | None = None,
        profiles: dict[str, EvaluationProfile] | None = None,
    ) -> None:
        self.sessions = sessions or SessionStore()
        self.max_parallel_eval = resolve_max_parallel_eval(max_parallel_eval)
        self._eval_semaphore = threading.Semaphore(self.max_parallel_eval)
        self.evaluation_engine = EvaluationEngine(profiles or default_evaluation_profiles())

    def create_session(self, req: SessionCreateRequest) -> SessionCreateResponse:
        from alphaagent.data.adapters.cnequity import CNE_SOURCE
        raw_path = req.panel_path
        if raw_path == CNE_SOURCE:
            # Keep the logical URI intact; Path('cne://') collapses it to 'cne:'.
            resolved_path = CNE_SOURCE
        else:
            resolved_path = Path(raw_path).expanduser().resolve()
        ctx = StockEvalContext(
            panel_path=resolved_path,
            train_start=req.train_start,
            train_end=req.train_end,
            val_start=req.val_start,
            val_end=req.val_end,
            label_col=req.label_col,
            include_fundamentals=req.include_fundamentals,
        )
        session = self.sessions.create(ctx)
        cols = list(session.panel.columns[:12])
        return SessionCreateResponse(
            session_id=session.session_id,
            panel_rows=len(session.panel),
            load_ms=float(session.meta.get("load_ms", 0)),
            columns_sample=cols,
        )

    def _run_one(
        self,
        session_id: str,
        *,
        split: str,
        multi_line_expr: str,
        factor_name: str,
        include_detail_tables: bool,
        label_quantile_n: int,
        expected_sign: int | None = None,
    ) -> dict[str, Any]:
        session = self.sessions.get(session_id)
        raw = evaluate_factor_on_split(
            session,
            split=split,
            multi_line_expr=multi_line_expr,
            factor_name=factor_name,
            include_detail_tables=include_detail_tables,
            label_quantile_n=label_quantile_n,
        )
        if include_detail_tables and raw.get("ok"):
            raw["include_detail_tables"] = True
        return format_eval_response(raw, expected_sign=expected_sign)

    def eval_train(self, req: EvalTrainRequest) -> dict[str, Any]:
        with self._eval_semaphore:
            return self._run_one(
                req.session_id,
                split="train",
                multi_line_expr=req.multi_line_expr,
                factor_name=req.factor_name,
                include_detail_tables=req.include_detail_tables,
                label_quantile_n=req.label_quantile_n,
            )

    def eval_val(self, req: EvalValRequest) -> dict[str, Any]:
        with self._eval_semaphore:
            return self._run_one(
                req.session_id,
                split="val",
                multi_line_expr=req.multi_line_expr,
                factor_name=req.factor_name,
                include_detail_tables=req.include_detail_tables,
                label_quantile_n=req.label_quantile_n,
                expected_sign=req.expected_sign,
            )

    def eval_profile(self, req: EvalProfileRequest) -> dict[str, Any]:
        with self._eval_semaphore:
            session = self.sessions.get(req.session_id)
            result = self.evaluation_engine.evaluate(
                session,
                profile_id=req.profile_id,
                multi_line_expr=req.multi_line_expr,
                factor_name=req.factor_name,
            )
            if result.get("ok"):
                record = session.candidates.record_evaluation(result)
                result["candidate"]["candidate_id"] = record.candidate_id
                result["candidate_state"] = record.state
            return result

    def record_candidate_review(self, session_id: str, candidate_id: str, review: dict[str, Any]) -> dict[str, Any] | None:
        session = self.sessions.get(session_id)
        record = session.candidates.record_review(candidate_id, review)
        if record is None:
            return None
        return {"candidate_id": record.candidate_id, "state": record.state}
