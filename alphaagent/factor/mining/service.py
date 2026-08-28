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


def _engine_result_to_legacy(raw: dict[str, Any]) -> dict[str, Any]:
    """把 EvaluationEngine 结果映射回旧 split 评估契约（format_eval_response 输入）。

    train/val 评估在收敛后走 evaluation_engine（与因子实验室同一引擎），
    但 LLM 工具契约仍要求旧结构：扁平 summary + monthly_corr_robustness
    + label_quantile_buckets。此处只做字段重组，不重算任何指标。
    """
    metrics = raw.get("metrics") or {}
    cs = metrics.get("cross_sectional_core") or {}
    mls = metrics.get("mls_fmb") or {}
    out: dict[str, Any] = {
        "ok": bool(raw.get("ok")),
        "split": raw.get("split"),
        "date_range": raw.get("date_range"),
        "label_col": raw.get("label_col"),
        "bar_interval": "1d",
        "timing_ms": raw.get("timing_ms") or {},
        "summary": {
            "ic": cs.get("ic"),
            "icir": cs.get("icir"),
            "rank_ic": cs.get("rank_ic"),
            "n_days": cs.get("n_days"),
            "n_instruments": cs.get("n_instruments"),
            "factor_coverage": cs.get("factor_coverage"),
            "factor_skewness": cs.get("factor_skewness"),
            "factor_kurtosis": cs.get("factor_kurtosis"),
            "cs_pearson_autocorr": cs.get("cs_pearson_autocorr"),
            "decile_mean_label": cs.get("decile_mean_label"),
            "mls_fmb": mls,
        },
        "monthly_corr_robustness": metrics.get("monthly_robustness") or {},
        "label_quantile_buckets": raw.get("label_quantile_buckets", []),
        "label_quantile_n": raw.get("label_quantile_n"),
        "eval_wall_seconds": (raw.get("timing_ms") or {}).get("total_ms", 0) / 1000.0,
    }
    if raw.get("by_month") is not None:
        out["by_month"] = raw["by_month"]
    if raw.get("by_symbol") is not None:
        out["by_symbol"] = raw["by_symbol"]
    return out


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
            available_columns=list(session.panel.columns),
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
        # 收敛：train/val 评估与因子实验室共用同一 EvaluationEngine + profile
        # （train_screen / validation），保持同一预处理与指标管线。
        profile_id = "train_screen" if split == "train" else "validation"
        raw = self.evaluation_engine.evaluate(
            session,
            profile_id=profile_id,
            multi_line_expr=multi_line_expr,
            factor_name=factor_name,
            label_quantile_n=label_quantile_n,
            include_detail_tables=include_detail_tables,
        )
        if not raw.get("ok"):
            return raw
        legacy = _engine_result_to_legacy(raw)
        if include_detail_tables:
            legacy["include_detail_tables"] = True
        return format_eval_response(legacy, expected_sign=expected_sign)

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

    def release_session(self, session_id: str) -> None:
        """释放一次性会话（丢弃 panel 引用），让几 GB 内存可回收。

        仅用于单次评估 API（评估完即释放，不做 LRU 缓存）；
        批量挖掘场景的会话由 run 生命周期管理，不调用此方法。
        """
        self.sessions.remove(session_id)
