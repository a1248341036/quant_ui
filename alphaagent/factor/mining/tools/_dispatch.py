"""FactorEvalTools 核心：__init__ / schemas / dispatch / submit / screener。"""
from __future__ import annotations

import json
from typing import Any

from alphaagent.factor.mining.runlog import log_step
from alphaagent.factor.mining.schemas import EvalProfileRequest, EvalTrainRequest, EvalValRequest
from alphaagent.factor.mining.service import StockEvalService
from alphaagent.factor.mining.submit import FactorSubmitService
from alphaagent.factor.mining.eval.prediction import (
    GATING_OP_RE,
    build_ablation_check,
    build_prediction_check,
    normalize_prediction,
)

from ._schemas import (
    _EVAL_PARAMETERS,
    _PROFILE_EVAL_PARAMETERS,
    _SCREEN_FACTORS_PARAMETERS,
    _SUBMIT_PARAMETERS,
    _VAL_PARAMETERS,
)
from ._prefilter import _is_naive_signal_addition


def _prediction_argument_error(arguments: dict[str, Any]) -> dict[str, Any] | None:
    """prediction 必填校验：缺失/缺字段返回 ToolArgumentsError，合法时返回规范化结果。"""
    pred = normalize_prediction(arguments.get("prediction"))
    if pred is None:
        return {
            "ok": False,
            "error": (
                "prediction_required: 评估必须携带可证伪预测 prediction="
                '{"expected_shape": "monotonic_increasing|monotonic_decreasing|inverted_u|u_shape|spike_at_extreme|irregular", '
                '"expected_strong_side": "high_factor|low_factor|middle", "expected_sign": 1|-1, "falsifier": "可选"}。'
                "评估结果会自动对账注入 prediction_check——预期被证伪说明机制错误，"
                "应放弃该方向而不是调参重试。"
            ),
            "error_type": "ToolArgumentsError",
        }
    return None


def _attach_prediction_check(result: dict[str, Any], prediction: Any, *, ic: Any, decile_rows: Any) -> None:
    """评估成功后把 prediction_check 注入结果（内部调用，异常全吞）。"""
    try:
        check = build_prediction_check(prediction, ic=ic, decile_rows=decile_rows)
        if check is not None:
            result["prediction_check"] = check
    except Exception:  # noqa: BLE001 — 对账是增益信息，绝不阻断评估
        pass


def _engine_decile(result: dict[str, Any]) -> tuple[Any, Any]:
    """从引擎原生 shape（evaluate_factor 结果）取 (ic, decile_rows)。"""
    cs = (result.get("metrics") or {}).get("cross_sectional_core") or {}
    return cs.get("ic"), cs.get("decile_mean_label")


def _legacy_decile(result: dict[str, Any]) -> tuple[Any, Any]:
    """从 legacy shape（eval_on_train/val_set 结果）取 (ic, decile_rows)。"""
    summ = result.get("summary") or {}
    return summ.get("ic"), summ.get("decile_mean_label")


class _DispatchMixin:
    """dispatch / schemas / _memory_gate / _dispatch_submit / Screener 方法。"""

    service: StockEvalService
    session_id: str
    submit_service: FactorSubmitService | None
    _screener_config_dict: dict[str, Any]
    memory_store: Any

    def _memory_gate(self, expr: str, arguments: dict[str, Any]) -> Any:
        """评估/提交前查研究记忆 advisory。返回 None | advisory dict | 拦截 result（ok=False）。"""
        if self.memory_store is None:
            return None
        try:
            advisory = self.memory_store.advisory_for(str(expr or ""), edit_note=arguments.get("edit_note"))
        except Exception:
            return None
        if not advisory:
            return None
        blocked = None
        if getattr(self.memory_store, "hard_block_duplicates", False):
            for item in advisory.get("advisories", []):
                if item.get("kind") == "duplicate_known_dead_end":
                    blocked = {
                        "ok": False,
                        "error": f"memory_blocked_duplicate: {item.get('message', '')}",
                        "error_type": "MemoryAdvisoryBlock",
                        "memory_advisory": advisory,
                    }
        try:
            kinds = [item.get("kind") for item in advisory.get("advisories", [])]
            if blocked is not None:
                log_step("memory.advisory_block", f"kinds={kinds}", detail=str(advisory.get("advisories"))[:160])
            else:
                log_step("memory.advisory", f"kinds={kinds}")
        except Exception:
            pass
        return blocked if blocked is not None else advisory

    def _attach_ablation(self, expr: str, arguments: dict[str, Any], *, profile_id: str, result: dict[str, Any]) -> None:
        """E) 门控/条件结构自动消融：base-only vs full 对比注入结果。

        - 表达式含门控类算子且契约给了 base_expr → 跑 base-only 并量化条件化增量；
        - 未给 base_expr → 只附 ablation_hint 提醒（不阻断）。
        内部异常全吞，绝不影响主评估结果。
        """
        try:
            if not isinstance(result, dict) or not result.get("ok"):
                return
            if not GATING_OP_RE.search(expr):
                return
            contract = arguments.get("interaction") if isinstance(arguments.get("interaction"), dict) else {}
            base_expr = str(contract.get("base_expr") or "").strip()
            if not base_expr:
                result["ablation_hint"] = (
                    "门控/条件结构未传 interaction.base_expr，无法自动消融。"
                    "条件化可能在摧毁基信号（实测案例：年线门控把 20d 反转 IC 从 +0.039 变 -0.005）——"
                    "请补 base_expr 重跑确认门控增量。"
                )
                return
            session = self.service.sessions.get(self.session_id)
            base_raw = self.service.evaluation_engine.evaluate(
                session,
                profile_id=profile_id,
                multi_line_expr=base_expr,
                factor_name=f"{arguments.get('factor_name') or 'expr'}__baseonly",
                include_charts=False,
            )
            if not isinstance(base_raw, dict) or not base_raw.get("ok"):
                result["ablation_check"] = {
                    "verdict": "skipped",
                    "base_expr": base_expr[:200],
                    "error": str((base_raw or {}).get("error") or "base_only_eval_failed")[:200],
                }
                return
            base_cs = (base_raw.get("metrics") or {}).get("cross_sectional_core") or {}
            full_cs = (result.get("metrics") or {}).get("cross_sectional_core") or {}
            if not full_cs:
                full_summ = result.get("summary") or {}
                full_cs = {
                    "ic": full_summ.get("ic"),
                    "icir": full_summ.get("icir"),
                }
            result["ablation_check"] = build_ablation_check(base_cs, full_cs, base_expr=base_expr)
        except Exception as exc:  # noqa: BLE001
            result["ablation_check"] = {"verdict": "skipped", "error": f"{type(exc).__name__}: {str(exc)[:120]}"}

    def schemas(self) -> list[dict[str, Any]]:
        out = [
            {
                "type": "function",
                "function": {
                    "name": "evaluate_factor",
                    "description": "按冻结 EvaluationProfile 评估因子。profile 决定数据切分、风险调整、指标插件与规则门。",
                    "parameters": _PROFILE_EVAL_PARAMETERS,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "eval_on_train_set",
                    "description": "训练集评估多行因子表达式，返回 summary、monthly_corr_robustness、label_quantile_buckets。",
                    "parameters": _EVAL_PARAMETERS,
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "eval_on_val_set",
                    "description": "验证集评估；须传 expected_sign（train IC 符号 1/-1），结果含 sign_check。",
                    "parameters": _VAL_PARAMETERS,
                },
            },
        ]

        if self.submit_service is not None:
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": "submit_factor",
                        "description": (
                            "【两阶段交付】门槛数值见下方 `submit_factor` 返回的 "
                            "`delivery_check` 与 criteria（单一来源，动态渲染）："
                            + self.submit_service.criteria.to_prompt_text()
                            + " 仅正式库成功时 stored=true；候选池成功时 candidate_stored=true。"
                            " 查重失败时返回 top_neighbors 含相似因子 expr。"
                        ),
                        "parameters": _SUBMIT_PARAMETERS,
                    },
                }
            )

        # Screener（regime 感知因子筛选）工具
        screener_cfg = self._screener_config()
        if screener_cfg is not None and screener_cfg.get("enabled"):
            sc = screener_cfg
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": "screen_factors",
                        "description": (
                            "【Screener · regime 感知因子筛选】对正式库因子做市场制度感知筛选："
                            f"ADX({sc['adh_threshold']})+MA({sc['ma_period']}) 检测 regime，"
                            f"回看 {sc['lookback']} 天 Rank IC 评分，|IC|>={sc['min_ic']} 入选，"
                            f"因子间 |corr|>{sc['max_corr']} 去冗余，"
                            f"{'启用因子族 regime 偏好' if sc['use_family_boost'] else '不启用族偏好'}。"
                            " 返回当前 regime、各因子 IC/评分/权重/方向、被拒因子列表。"
                        ),
                        "parameters": _SCREEN_FACTORS_PARAMETERS,
                    },
                }
            )

        return out

    def dispatch(self, name: str, arguments: Any) -> dict[str, Any]:
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments) if arguments.strip() else {}
            except json.JSONDecodeError as e:
                return {"ok": False, "error": f"invalid_tool_arguments_json: {e}", "error_type": "JSONDecodeError"}

        if not isinstance(arguments, dict):
            return {"ok": False, "error": "tool_arguments_must_be_object", "error_type": "ToolArgumentsError"}

        if name == "submit_factor":
            return self._dispatch_submit(arguments)

        if name == "evaluate_factor":
            expr = arguments.get("multi_line_expr")
            profile_id = arguments.get("profile_id")
            if not isinstance(expr, str) or not expr.strip():
                return {"ok": False, "error": "multi_line_expr_required_non_empty_string", "error_type": "ToolArgumentsError"}
            if not isinstance(profile_id, str) or not profile_id.strip():
                return {"ok": False, "error": "profile_id_required_non_empty_string", "error_type": "ToolArgumentsError"}
            pred_error = _prediction_argument_error(arguments)
            if pred_error is not None:
                return pred_error
            # 预审：拦截"两个裸 RANK 信号简单加减"的低级因子
            if _is_naive_signal_addition(expr):
                return {
                    "ok": False,
                    "error": "naive_signal_addition_blocked: 顶层 ADD/SUBTRACT(RANK(x), RANK(y)) 是低级信号叠加，"
                             "没有经济机制上的交互。多信息源融合必须使用结构化交互算子"
                             "（GATED_SIGNAL / CS_RESIDUALIZE / DIVERGENCE_RANK / CS_GROUP_RANK / TS_CORR 等）。",
                    "error_type": "NaiveSignalAdditionBlock",
                }
            gate = self._memory_gate(expr, arguments)
            if isinstance(gate, dict) and gate.get("ok") is False:
                return gate
            result = self.service.eval_profile(
                EvalProfileRequest(
                    session_id=self.session_id,
                    profile_id=profile_id,
                    multi_line_expr=expr,
                    factor_name=str(arguments.get("factor_name") or "expr"),
                )
            )
            if isinstance(result, dict) and result.get("ok"):
                ic, decile_rows = _engine_decile(result)
                _attach_prediction_check(result, arguments.get("prediction"), ic=ic, decile_rows=decile_rows)
                self._attach_ablation(expr, arguments, profile_id=profile_id, result=result)
            if isinstance(gate, dict):
                result["memory_advisory"] = gate
            return result

        expr = arguments.get("multi_line_expr")
        if not isinstance(expr, str) or not expr.strip():
            return {"ok": False, "error": "multi_line_expr_required_non_empty_string", "error_type": "ToolArgumentsError"}

        # 预审：拦截"两个裸 RANK 信号简单加减"的低级因子
        if _is_naive_signal_addition(expr):
            return {
                "ok": False,
                "error": "naive_signal_addition_blocked: 顶层 ADD/SUBTRACT(RANK(x), RANK(y)) 是低级信号叠加，"
                         "没有经济机制上的交互。多信息源融合必须使用结构化交互算子"
                         "（GATED_SIGNAL / CS_RESIDUALIZE / DIVERGENCE_RANK / CS_GROUP_RANK / TS_CORR 等）。",
                "error_type": "NaiveSignalAdditionBlock",
            }

        factor_name = arguments.get("factor_name") or "expr"
        include_detail = bool(arguments.get("include_detail_tables", False))
        label_quantile_n = arguments.get("label_quantile_n", 10)
        if label_quantile_n is None:
            label_quantile_n = 10

        if name == "eval_on_train_set":
            pred_error = _prediction_argument_error(arguments)
            if pred_error is not None:
                return pred_error
            gate = self._memory_gate(expr, arguments)
            if isinstance(gate, dict) and gate.get("ok") is False:
                return gate
            result = self.service.eval_train(
                EvalTrainRequest(
                    session_id=self.session_id,
                    multi_line_expr=expr,
                    factor_name=factor_name,
                    include_detail_tables=include_detail,
                    label_quantile_n=int(label_quantile_n),
                )
            )
            if isinstance(result, dict) and result.get("ok"):
                ic, decile_rows = _legacy_decile(result)
                _attach_prediction_check(result, arguments.get("prediction"), ic=ic, decile_rows=decile_rows)
                self._attach_ablation(expr, arguments, profile_id="train_screen", result=result)
            if isinstance(gate, dict):
                result["memory_advisory"] = gate
            return result

        if name == "eval_on_val_set":
            expected_sign = arguments.get("expected_sign")
            if expected_sign not in (None, 1, -1):
                return {"ok": False, "error": "expected_sign_must_be_1_or_-1", "error_type": "ToolArgumentsError"}
            gate = self._memory_gate(expr, arguments)
            if isinstance(gate, dict) and gate.get("ok") is False:
                return gate
            result = self.service.eval_val(
                EvalValRequest(
                    session_id=self.session_id,
                    multi_line_expr=expr,
                    factor_name=factor_name,
                    include_detail_tables=include_detail,
                    label_quantile_n=int(label_quantile_n),
                    expected_sign=expected_sign,
                )
            )
            if isinstance(result, dict) and result.get("ok") and arguments.get("prediction") is not None:
                ic, decile_rows = _legacy_decile(result)
                _attach_prediction_check(result, arguments.get("prediction"), ic=ic, decile_rows=decile_rows)
            if isinstance(gate, dict):
                result["memory_advisory"] = gate
            return result

        if name == "screen_factors":
            return self._dispatch_screen_factors(arguments)

        return {"ok": False, "error": f"unknown_tool: {name}", "error_type": "UnknownTool"}

    def _dispatch_submit(self, arguments: dict[str, Any]) -> dict[str, Any]:
        if self.submit_service is None:
            return {
                "ok": False,
                "stored": False,
                "error": "submit_factor_disabled",
                "error_type": "SubmitDisabled",
            }

        expr = arguments.get("multi_line_expr")
        factor_name = arguments.get("factor_name")
        comment = arguments.get("comment")

        if not isinstance(factor_name, str) or not factor_name.strip():
            return {
                "ok": False,
                "stored": False,
                "error": "factor_name_required_non_empty_string",
                "error_type": "ToolArgumentsError",
            }

        if not isinstance(comment, str) or not comment.strip():
            return {
                "ok": False,
                "stored": False,
                "error": "comment_required_non_empty_string",
                "error_type": "ToolArgumentsError",
            }

        gate = self._memory_gate(str(expr or ""), arguments)
        if isinstance(gate, dict) and gate.get("ok") is False:
            return gate

        result = self.submit_service.submit(
            self.session_id,
            multi_line_expr=str(expr or ""),
            factor_name=factor_name.strip(),
            comment=comment.strip(),
            evaluation_evidence=arguments.get("evaluation_evidence"),
            review_hook=arguments.get("review_hook"),
            orthogonality_hook=arguments.get("orthogonality_hook"),
            interaction=arguments.get("interaction"),
            rebalance_freq=arguments.get("rebalance_freq"),
        )

        if isinstance(gate, dict):
            result["memory_advisory"] = gate

        return result

    # ── Screener（regime 感知因子筛选）──

    def _screener_config(self) -> dict[str, Any] | None:
        """从 submit_service.criteria 或注入的 screener_config 取 Screener 参数。"""
        # 优先用注入的 screener_config（来自 research_spec.delivery_policy.screener）
        cfg = self._screener_config_dict
        if cfg and cfg.get("enabled"):
            return cfg
        # 回退到 submit_service.criteria.screener
        if self.submit_service is not None:
            sc = self.submit_service.criteria.screener
            if sc.enabled:
                return {
                    "enabled": True,
                    "lookback": sc.lookback,
                    "min_ic": sc.min_ic,
                    "max_corr": sc.max_corr,
                    "use_family_boost": sc.use_family_boost,
                    "adx_threshold": sc.adx_threshold,
                    "ma_period": sc.ma_period,
                    "min_cross_section": sc.min_cross_section,
                }
        return None

    def _dispatch_screen_factors(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """对正式库因子做 regime 感知筛选，返回动态权重/方向。"""
        cfg = self._screener_config()
        if cfg is None or not cfg.get("enabled"):
            return {"ok": False, "error": "screener_disabled", "error_type": "ScreenerDisabled"}

        try:
            session = self.service.sessions.get(self.session_id)
        except KeyError:
            return {"ok": False, "error": f"session_not_found: {self.session_id}", "error_type": "SessionError"}

        panel = session.panel
        if panel is None or len(panel) == 0:
            return {"ok": False, "error": "session_panel_empty", "error_type": "SessionError"}

        import pandas as pd
        import numpy as np
        from core.screener import screen_factors, ScreenerConfig

        # 确定 signal_date
        ctx = session.ctx
        dt_level = panel.index.get_level_values("datetime")
        signal_date_str = arguments.get("signal_date")
        if signal_date_str:
            try:
                signal_ts = pd.Timestamp(signal_date_str)
            except Exception:
                return {"ok": False, "error": f"invalid_signal_date: {signal_date_str}", "error_type": "ToolArgumentsError"}
        else:
            # 默认用 val_end
            signal_ts = pd.Timestamp(ctx.val_end)
        # 找到 signal_date 在 panel 中的行号
        dates = pd.DatetimeIndex(sorted(dt_level.unique()))
        if signal_ts not in dates:
            # 找最近的 <= signal_ts 的日期
            valid = dates[dates <= signal_ts]
            if len(valid) == 0:
                return {"ok": False, "error": f"signal_date_before_data: {signal_ts}", "error_type": "ToolArgumentsError"}
            signal_ts = valid[-1]
        signal_idx = dates.get_loc(signal_ts)
        if isinstance(signal_idx, slice):
            signal_idx = signal_idx.start

        # 构建 close matrix (T, K)
        if "close" not in panel.columns:
            return {"ok": False, "error": "panel_missing_close_column", "error_type": "PanelError"}
        close_df = panel["close"].unstack(level="instrument")
        # 对齐 dates
        close_df = close_df.reindex(dates)

        # 构建因子矩阵
        factor_names_req = arguments.get("factor_names") or []
        if not isinstance(factor_names_req, list):
            factor_names_req = []

        # 从 submit_service.factorlib 加载因子
        if self.submit_service is None:
            return {"ok": False, "error": "submit_service_not_available", "error_type": "ScreenerDisabled"}

        from alphaagent.factor.zoo.index import FactorZoo
        try:
            zoo = FactorZoo.open(self.submit_service.factorlib_path)
        except Exception as e:
            return {"ok": False, "error": f"factorzoo_open_failed: {e}", "error_type": "FactorZooError"}

        # 获取因子列表
        all_factors = list(zoo.iter_factors())
        if factor_names_req:
            all_factors = [f for f in all_factors if f.name in factor_names_req]
        if not all_factors:
            return {"ok": True, "result": {"selected": [], "regime_label": "无因子", "weights": {}}}

        # 构建因子值矩阵
        factor_frames: dict[str, pd.DataFrame] = {}
        for f in all_factors:
            try:
                # zoo 因子值是 (datetime, instrument) 对齐的 Series
                vals = zoo.get_factor_values(f.factor_id)
                if vals is not None and len(vals) > 0:
                    fmat = vals.unstack(level="instrument").reindex(index=dates, columns=close_df.columns)
                    factor_frames[f.name] = fmat
            except Exception:
                continue

        if not factor_frames:
            return {"ok": True, "result": {"selected": [], "regime_label": "无因子值", "weights": {}}}

        # 用等权均值近似指数
        index_close_s = close_df.mean(axis=1)

        screener_cfg = ScreenerConfig(
            lookback=cfg.get("lookback", 10),
            min_ic=cfg.get("min_ic", 0.02),
            max_corr=cfg.get("max_corr", 0.7),
            use_family_boost=cfg.get("use_family_boost", True),
            adx_threshold=cfg.get("adx_threshold", 25.0),
            ma_period=cfg.get("ma_period", 60),
            min_cross_section=cfg.get("min_cross_section", 30),
        )

        result = screen_factors(
            factor_frames, close_df, signal_idx, screener_cfg,
            index_close=index_close_s,
            all_dates=dates,
        )

        return {
            "ok": True,
            "result": {
                "signal_date": str(result.signal_date.date()),
                "regime": result.regime_label,
                "selected": result.selected,
                "factor_ic": {k: round(v, 4) for k, v in result.factor_ic.items()},
                "factor_scores": {k: round(v, 4) for k, v in result.factor_scores.items()},
                "weights": {k: round(v, 4) for k, v in result.weights.items()},
                "directions": {k: "买低" if v else "买高" for k, v in result.directions.items()},
                "rejected": dict(list(result.rejected.items())[:10]),
                "regime_dist": result.regime_dist,
            },
        }
