"""股票因子挖掘评估服务：会话管理与 train/val 评估。"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from alphaagent.factor.evaluation.rules import evaluate_rules
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
from alphaagent.factor.mining.session import SessionStore, StockEvalSession


def _eval_lite_enabled() -> bool:
    """两段式海选开关（默认开）；``ALPHA_EVAL_LITE=0`` 退回单段全量。"""
    return (os.environ.get("ALPHA_EVAL_LITE") or "").strip() != "0"


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
            test_start=req.test_start,
            test_end=req.resolved_test_end(),
            label_col=req.label_col,
            include_fundamentals=req.include_fundamentals,
            asset_type=req.asset_type,
            focus_facets=tuple(getattr(req, "focus_facets", ()) or ()),
            engine_gate_policy=getattr(req, "engine_gate_policy", None),
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
        profile_id: str | None = None,
    ) -> dict[str, Any]:
        session = self.sessions.get(session_id)
        # 收敛：train/val 评估与因子实验室共用同一 EvaluationEngine + profile
        # （train_screen / validation），保持同一预处理与指标管线。
        if profile_id is None:
            profile_id = "train_screen" if split == "train" else "validation"
        # 挖掘批量评估跳过图表生成（逐日 IC/多空/月度分解纯为前端可视化服务，
        # 在 800 万行面板上每次评估额外耗数秒；图表仅因子实验室 eval_profile 需要）
        raw = self.evaluation_engine.evaluate(
            session,
            profile_id=profile_id,
            multi_line_expr=multi_line_expr,
            factor_name=factor_name,
            label_quantile_n=label_quantile_n,
            include_detail_tables=include_detail_tables,
            include_charts=False,
        )
        if not raw.get("ok"):
            return raw
        legacy = _engine_result_to_legacy(raw)
        if include_detail_tables:
            legacy["include_detail_tables"] = True
        return format_eval_response(legacy, expected_sign=expected_sign)

    def _eval_train_two_stage(
        self,
        req: EvalTrainRequest,
    ) -> dict[str, Any]:
        """两段式海选（2026-09-06，docs/alphaagent_对比结果_run1.md）：

        第一段跑 ``train_screen_lite``（仅 cross_sectional_core，门槛规则唯一
        数据源），过线才跑全量 ``train_screen`` 补齐 fmb/组合回测/月度稳健性
        等诊断件。并发基准实测：全量 profile 因诊断件的 GIL 段，4 路并发吞吐
        卡死 0.42 eval/s；core-only 1.99 eval/s。未过线因子占绝大多数，
        这笔诊断开销基本是白付的。引擎零改动；``ALPHA_EVAL_LITE=0`` 退回全量。
        """
        try:
            lite_profile = self.evaluation_engine.profile("train_screen_lite")
        except (KeyError, ValueError):
            lite_profile = None
        if lite_profile is None or lite_profile.rules == ():
            return self._run_one(
                req.session_id,
                split="train",
                multi_line_expr=req.multi_line_expr,
                factor_name=req.factor_name,
                include_detail_tables=req.include_detail_tables,
                label_quantile_n=req.label_quantile_n,
            )
        session = self.sessions.get(req.session_id)
        with self._eval_semaphore:
            raw_lite = self.evaluation_engine.evaluate(
                session,
                profile_id="train_screen_lite",
                multi_line_expr=req.multi_line_expr,
                factor_name=req.factor_name,
                label_quantile_n=req.label_quantile_n,
                include_detail_tables=req.include_detail_tables,
                include_charts=False,
            )
        if not raw_lite.get("ok"):
            return raw_lite
        rule_results = evaluate_rules(raw_lite.get("metrics") or {}, lite_profile.rules)
        if not all(r.get("passed") for r in rule_results):
            legacy = _engine_result_to_legacy(raw_lite)
            resp = format_eval_response(legacy, expected_sign=None)
            # format_eval_response 重建 dict，筛选标记在格式化之后附加
            resp["screen_stage"] = "lite"
            resp["screen_rules"] = rule_results
            resp["skipped_diagnostics"] = [
                m["plugin"] for m in self.evaluation_engine.profile("train_screen").metrics
                if m["plugin"] != "cross_sectional_core"
            ]
            return resp
        raw_full = self.evaluation_engine.evaluate(
            session,
            profile_id="train_screen",
            multi_line_expr=req.multi_line_expr,
            factor_name=req.factor_name,
            label_quantile_n=req.label_quantile_n,
            include_detail_tables=req.include_detail_tables,
            include_charts=False,
        )
        if not raw_full.get("ok"):
            return raw_full
        legacy = _engine_result_to_legacy(raw_full)
        if req.include_detail_tables:
            legacy["include_detail_tables"] = True
        resp = format_eval_response(legacy, expected_sign=None)
        resp["screen_stage"] = "full"
        # 引擎预演：train 过线因子附 val 窗口可交易口径（对齐用户"统计↔实盘"意图）
        self._maybe_engine_preview(session, req.multi_line_expr, resp)
        return resp

    def _maybe_engine_preview(
        self, session: StockEvalSession, multi_line_expr: str, result: dict[str, Any]
    ) -> None:
        """train 过线因子 → val 窗口引擎预演（就地注入 ``result["engine_preview"]``）。

        用户设计意图（2026-09-11 确认）：统计口径（分位组前瞻收益）与实盘口径
        （core.engine 完整约束）必须对齐——此前引擎只在 submit 时首次出现，
        LLM 整个迭代期都在用统计口径选方向，"统计好引擎差"的因子烧掉大量轮次。

        - 直接复用 ``run_engine_gate``（单一真源，零语义漂移）；
        - 窗口 = val（样本外可交易预演）；盲测段仍只属于 submit 链路；
        - 策略 = 本 run 的 engine_gate 配置（用户显式调仓/档位默认/top_pct）；
        - 异常全吞：预演是增益信息，绝不阻断评估。
        """
        try:
            policy = getattr(session.ctx, "engine_gate_policy", None)
            if not isinstance(policy, dict) or not policy.get("enabled", True):
                return
            panel = session.panel
            if panel is None or len(panel) == 0:
                return
            from alphaagent.dsl import eval_factor
            from alphaagent.factor.mining.engine_gate import run_engine_gate

            import numpy as np

            out = eval_factor(multi_line_expr, panel)
            values = out.reindex(panel.index).to_numpy(dtype=np.float64)
            cs = (result.get("metrics") or {}).get("cross_sectional_core") or {}
            ic = cs.get("ic")
            if ic is None:
                ic = (result.get("summary") or {}).get("ic")
            sign = 1 if (isinstance(ic, (int, float)) and ic >= 0) else -1
            gate = run_engine_gate(
                panel,
                values,
                val_start=session.ctx.val_start,
                val_end=session.ctx.val_end,
                direction=sign,
                policy=policy,
                asset_type=getattr(session.ctx, "asset_type", "stock"),
            )
            m = gate.get("metrics") or {}
            result["engine_preview"] = {
                "window": {"start": session.ctx.val_start, "end": session.ctx.val_end},
                "freq": policy.get("freq"),
                "passed": bool(gate.get("passed")),
                "annual_return": m.get("annual_return"),
                "excess_annual": m.get("excess_annual"),
                "sharpe": m.get("sharpe"),
                "max_drawdown": m.get("max_drawdown"),
                "fail_reasons": gate.get("fail_reasons") or [],
                "note": (
                    "val 窗口可交易口径预演：完整 T+1/涨跌停/停牌/流动性/成本约束，"
                    "选股口径同交付策略（与分位组前瞻收益统计口径不可直接比较）"
                ),
            }
        except Exception as exc:  # noqa: BLE001 — 预演绝不阻断评估
            result["engine_preview"] = {"error": f"{type(exc).__name__}: {exc}"}

    def eval_train(self, req: EvalTrainRequest) -> dict[str, Any]:
        if _eval_lite_enabled():
            return self._eval_train_two_stage(req)
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
                include_charts=req.include_charts,
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
