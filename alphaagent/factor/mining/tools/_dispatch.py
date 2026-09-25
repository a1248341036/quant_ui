"""FactorEvalTools 核心：__init__ / schemas / dispatch / submit / screener。"""
from __future__ import annotations

import json
import re
from typing import Any

from alphaagent.factor.mining.runlog import log_step
from alphaagent.factor.mining.memory.expressions import facet_scope_violation
from alphaagent.factor.mining.schemas import EvalProfileRequest, EvalTrainRequest, EvalValRequest
from alphaagent.factor.mining.service import StockEvalService
from alphaagent.factor.mining.submit import FactorSubmitService

from alphaagent.factor.mining.eval.prediction import (
    GATING_OP_RE,
    build_ablation_check,
    build_prediction_check,
    describe_prediction_issues,
    normalize_prediction,
)

from ._schemas import (
    _EVAL_PARAMETERS,
    _PRECHEK_PARAMETERS,
    _PROFILE_EVAL_PARAMETERS,
    _RECOMMEND_MRMR_FACTORS_PARAMETERS,
    _SCREEN_FACTORS_PARAMETERS,
    _SUBMIT_PARAMETERS,
    _VAL_PARAMETERS,
)
from ._prefilter import (
    _is_naive_signal_addition,
    _signal_fingerprint,
    _ast_signal_fingerprint,
    _outer_transform_signature,
    _homogenization_block,
)


_PREDICTION_SOFT_LIMIT = 3

# 认知对账开关（research_spec.cognition_policy，消融 C1/C2）：缺省全开不改行为。
# prediction_check_enabled=False → 不注入 prediction_check、缺失软门不再升级拦截；
# ablation_check_enabled=False → 不注入 ablation_check/ablation_hint。
_DEFAULT_COGNITION_POLICY = {"prediction_check_enabled": True, "ablation_check_enabled": True}
# 算子黑名单（research_spec.operator_policy.blacklist，消融 D2）：命中即在
# 评估/提交前拦截并引导 LLM 改用基础算子；缺省空清单不拦截。
_DEFAULT_OPERATOR_POLICY = {"blacklist": ()}

# 数据面聚焦硬锁定作用的工具：三个评估入口 + 提交入口（都带 multi_line_expr）。
_FACET_LOCK_TOOLS = frozenset({
    "evaluate_factor",
    "eval_on_train_set",
    "eval_on_val_set",
    "submit_factor",
})

# LLM 可用的评估 profile 白名单（dispatch 层硬性收口）。
# production_delivery / 其它 split=full 口径含盲测段（2025+），只允许 submit
# 链路内部使用——LLM 探索期直接评估 = 按盲测数据选因子，盲测门禁即被架空。
_MINING_ALLOWED_PROFILES = frozenset({
    "train_screen",
    "validation",
    "size_neutral_validation",
})


def _prediction_argument_error(arguments: dict[str, Any], *, enabled: bool = True) -> dict[str, Any] | None:
    """prediction 参数校验：携带但字段非法时返回 ToolArgumentsError。

    缺失走软门（``_DispatchMixin._prediction_gate``）：放行 + 结果附
    prediction_warning，同工具累计缺失 ≥ ``_PREDICTION_SOFT_LIMIT`` 次才拦截
    ——GLM 系 provider 不稳定遵守 schema required，硬拦截会导致整轮并发
    tool_calls 作废重试（实测一次 run 白烧 ~19 分钟），代价远大于纪律收益。

    2026-09-05 两层改进：①枚举别名归一（"D10"→high_factor 等，语义对但
    词汇错的输入直接放行——上下文满屏 D1~D10，强求切换词汇是逆 LLM 天性）；
    ②真正非法时错误信息逐字段回显"收到什么/合法值是什么"，省掉盲猜重试
    （实测 7 次调用因错误信息不带实际值而连续失败）。

    ``enabled=False``（消融 C1）：校验整体短路——携带了也不拦，缺失不记账。
    """
    if not enabled:
        return None
    pred = normalize_prediction(arguments.get("prediction"))
    if arguments.get("prediction") is None:
        return None  # 缺失 → 软门
    if pred is None:
        issues = describe_prediction_issues(arguments.get("prediction"))
        return {
            "ok": False,
            "error": (
                f"prediction_invalid: {issues or '字段非法'}——"
                '{"expected_shape": "monotonic_increasing|monotonic_decreasing|inverted_u|u_shape|spike_at_extreme|irregular|conditional_subgroup", '
                '"expected_strong_side": "high_factor|low_factor|middle（conditional_subgroup 可省略）", '
                '"expected_sign": 1|-1, "falsifier": "可选"}。'
                "组内/门控条件式预期直接传 expected_shape=conditional_subgroup。"
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


# near_miss 阈值单一真源在 memory/constants.py（schema._classify 共用）
from alphaagent.factor.mining.memory.constants import (
    NEAR_MISS_COVERAGE,
    NEAR_MISS_ICIR_SOFT,
    NEAR_MISS_IC_RATIO,
)
# train 段 |IC| 高于此值且属财务/慢标签口径时提示 PIT 伪影嫌疑（实测
# fundamental 档 train IC 0.06~0.08 的因子几乎全部 val 阵亡——阶梯函数语义陷阱）
_PIT_SUSPICION_IC = 0.045


def _turnover_gate_limit_of(tools: Any) -> float | None:
    """取当前 run 的分档换手硬门（tools.submit_service.criteria.turnover_gate_limit）。

    供 diagnostics 运行时诊断用——缺失时返回 None，诊断层回落 defaults 分档。
    """
    criteria = getattr(getattr(tools, "submit_service", None), "criteria", None)
    return getattr(criteria, "turnover_gate_limit", None)


def _attach_yield_hints(
    result: dict[str, Any],
    expr: str,
    arguments: dict[str, Any],
    session: Any | None = None,
    recent_evals: list[dict[str, Any]] | None = None,
    turnover_gate_limit: float | None = None,
) -> None:
    """按评估结果注入产出率提示（委托给独立的诊断与仲裁引擎）。"""
    from alphaagent.factor.mining.diagnostics import apply_diagnostics_to_result
    apply_diagnostics_to_result(
        result, expr, arguments,
        session=session, recent_evals=recent_evals,
        turnover_gate_limit=turnover_gate_limit,
    )


def _ic_bar_from_rules(rules: Any) -> float | None:
    """取屏幕规则里的 |IC| 期望值（与档位 override 同源），无则 None。"""
    if not isinstance(rules, list):
        return None
    for row in rules:
        if isinstance(row, dict) and str(row.get("metric") or "").endswith("cross_sectional_core.ic"):
            try:
                return float(row.get("expected"))
            except (TypeError, ValueError):
                return None
    return None


def _candidate_ic_bar(research_mode: str | None) -> float:
    """候选池 |IC| 准入线兜底（唯一真源：DEFAULT_RESEARCH_SPEC + 模式 override）。"""
    try:
        from alphaagent.factor.mining.research_spec import DEFAULT_RESEARCH_SPEC
        from core.research_modes import RESEARCH_MODES

        bar = float(DEFAULT_RESEARCH_SPEC["delivery_policy"]["candidate"]["min_abs_ic"])
        spec = RESEARCH_MODES.get(str(research_mode or "technical"))
        if spec is not None:
            bar = float(getattr(spec, "candidate_overrides", {}).get("min_abs_ic", bar))
        return bar
    except Exception:  # noqa: BLE001
        return 0.020


def _near_miss_verdict(metrics: dict[str, Any]) -> bool:
    """IC 达门槛 80%、ICIR/coverage 达标但未过线 → near_miss（供 memory._classify 复用）。"""
    ic = metrics.get("ic")
    icir = metrics.get("icir")
    coverage = metrics.get("factor_coverage", metrics.get("coverage"))
    try:
        ic_f = abs(float(ic)) if ic is not None else None
        icir_f = float(icir) if icir is not None else None
        cov_f = float(coverage) if coverage is not None else None
    except (TypeError, ValueError):
        return False
    if ic_f is None:
        return False
    th = _ic_bar_from_rules(metrics.get("screen_rules")) or _candidate_ic_bar(
        metrics.get("research_mode")
    )
    return bool(
        NEAR_MISS_IC_RATIO * th <= ic_f < th
        and (icir_f is not None and icir_f > NEAR_MISS_ICIR_SOFT)
        and (cov_f is not None and cov_f > NEAR_MISS_COVERAGE)
    )


def _engine_decile(result: dict[str, Any]) -> tuple[Any, Any]:
    """从评估结果取 (ic, decile_rows)，兼容 engine 原生 shape 与 legacy shape。

    历史坑（2026-09-15 消融实验实测定位）：train_screen 走两段式 eval_train，
    返回的是 legacy 扁平 shape（summary.decile_mean_label），而本函数原先只读
    engine 原生 shape（metrics.cross_sectional_core.decile_mean_label）——
    metrics 键不存在 ⇒ 恒返回 (None, None) ⇒ prediction_check 全部
    unverifiable，**预测对账机制在 train 路径静默失效**（实测 23/23 全
    unverifiable）。此处补 legacy 回退，恢复对账。
    """
    cs = (result.get("metrics") or {}).get("cross_sectional_core") or {}
    ic = cs.get("ic")
    decile_rows = cs.get("decile_mean_label")
    if decile_rows is None:
        summ = result.get("summary") or {}
        if ic is None:
            ic = summ.get("ic")
        decile_rows = summ.get("decile_mean_label")
    return ic, decile_rows


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
    focus_facets: tuple[str, ...] = ()
    _missing_prediction_counts: dict[str, int]

    def _facet_lock_block(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any] | None:
        """数据面聚焦硬锁定：勾选聚焦面后，越界/未触面表达式在执行前直接拦截。

        只作用于表达式类工具（三个 eval + submit_factor）；聚焦未启用时恒放行。
        拦截结果带 facet_lock 结构化字段，供日志与前端定位越界面。
        """
        focus = tuple(getattr(self, "focus_facets", None) or ())
        if not focus or tool_name not in _FACET_LOCK_TOOLS:
            return None
        expr = arguments.get("multi_line_expr") if isinstance(arguments, dict) else None
        if not isinstance(expr, str) or not expr.strip():
            return None
        vio = facet_scope_violation(expr, focus)
        if vio is None:
            return None
        return {
            "ok": False,
            "error": vio["message"],
            "error_type": "ToolArgumentsError",
            "facet_lock": {k: v for k, v in vio.items() if k != "message"},
        }

    def _operator_blacklist_block(self, tool_name: str, expr: Any) -> dict[str, Any] | None:
        """算子黑名单拦截（research_spec.operator_policy.blacklist，消融 D2）。

        表达式命中黑名单算子 → 评估/提交前直接拦截，错误信息列明被禁算子
        并引导改用基础 TS/CS 算子；黑名单为空（默认）恒放行，不改行为。
        """
        blacklist = getattr(self, "operator_policy", {}).get("blacklist") or ()
        if not blacklist or not isinstance(expr, str) or not expr.strip():
            return None
        used = sorted(
            {tok.upper() for tok in re.findall(r"\b([A-Z][A-Z_]{2,})\b", expr) if tok.upper() in set(blacklist)}
        )
        if not used:
            return None
        return {
            "ok": False,
            "error": (
                f"operator_blacklisted: 本次运行已禁用高级算子 {', '.join(used)}。"
                "请改用基础时序/截面滚动算子（TS_MEAN / TS_STD / DELTA / RANK / CS_ZSCORE / TS_CORR 等）"
                "重新构造同机制的表达式。"
            ),
            "error_type": "OperatorBlacklistBlock",
            "blocked_operators": used,
        }

    def _prediction_gate(self, tool_name: str, arguments: dict[str, Any], result: dict[str, Any]) -> dict[str, Any] | None:
        """prediction 软门：缺失放行但记账 + 附 prediction_warning；累计缺失
        ≥ ``_PREDICTION_SOFT_LIMIT`` 次升级拦截。返回拦截 result 或 None
        （result 原地注入 warning）。
        """
        if not self.cognition_policy.get("prediction_check_enabled", True):
            return None
        if arguments.get("prediction") is None:
            counts = getattr(self, "_missing_prediction_counts", None)
            if counts is None:
                counts = {}
                self._missing_prediction_counts = counts
            counts[tool_name] = counts.get(tool_name, 0) + 1
            n = counts[tool_name]
            if n >= _PREDICTION_SOFT_LIMIT:
                counts.pop(tool_name, None)  # 拦截后重新计数，给 provider 自纠机会
                return {
                    "ok": False,
                    "error": (
                        f"prediction_required: 已连续 {n} 次调用 {tool_name} 未携带 prediction。"
                        '每次评估必须传 prediction={"expected_shape": ..., "expected_strong_side": ..., '
                        '"expected_sign": 1|-1, "falsifier": 可选}——不可证伪的评估没有研究价值。'
                    ),
                    "error_type": "ToolArgumentsError",
                }
            result["prediction_warning"] = (
                f"本次未携带 prediction，结果未做预期对账（第 {n}/{_PREDICTION_SOFT_LIMIT} 次，"
                "累计缺失将拦截）。后续每次 evaluate 必须传 prediction="
                '{"expected_shape": ..., "expected_strong_side": ..., "expected_sign": 1|-1}。'
            )
        return None

    def _attach_signature_hints(self, expr: str, result: dict[str, Any]) -> None:
        """传参错误自愈：DSL 求值因参数个数/名字错误失败时，把表达式里出现的
        算子的真实签名附进错误结果——LLM 按提示修正即可，不必回避没见过的算子。
        """
        try:
            err = str(result.get("error") or "")
            if not any(k in err for k in ("positional argument", "keyword argument", "unexpected keyword", "missing ")):
                return
            from alphaagent.dsl.catalog import _slim_signature
            from alphaagent.dsl.registry import build_operator_namespace

            ns = build_operator_namespace()
            used = sorted(
                {tok.upper() for tok in re.findall(r"\b([A-Z][A-Z_]{2,})\b", expr) if tok.upper() in ns}
            )
            if not used:
                return
            result["operator_signatures"] = {name: _slim_signature(ns[name]) for name in used[:8]}
            result["error"] = err + "（正确签名见 operator_signatures 字段——按参数顺序传参修正后重试）"
        except Exception:  # noqa: BLE001 — 自愈是增益信息，绝不改变失败语义
            pass

    def _record_eval_signature(self, result: dict[str, Any], expr: str) -> None:
        """记录最近一次训练评估的结构指纹 / 换手 / 是否含平滑 / 信号根指纹，供熔断器消费。"""
        try:
            from alphaagent.dsl.core.ast import all_smoothing_ops, structure_fingerprint

            # evaluate 返回扁平结构（format_eval_response 输出），换手在顶层；
            # 兼容引擎原生 metrics.quantile_portfolio 形状
            qp = (result.get("metrics") or {}).get("quantile_portfolio") or {}
            t_val = qp.get("avg_daily_side_turnover")
            if t_val is None:
                t_val = result.get("avg_daily_side_turnover")
            sig: dict[str, Any] = {
                "fingerprint": structure_fingerprint(expr),
                "turnover": float(t_val) if t_val is not None else None,
                "has_smoothing": bool(all_smoothing_ops(expr)),
                "signal_fingerprint": _signal_fingerprint(expr),
                "signal_fingerprint_ast": _ast_signal_fingerprint(expr),
                "outer_transform_signature": _outer_transform_signature(expr),
            }
            recent = getattr(self, "_recent_evals", None)
            if recent is None:
                return
            recent.append(sig)
            # 上限与 window_size 联动（__init__ 里算好 _recent_evals_cap = max(20, window_size*2)），
            # 防 window_size 超配时 recent_evals 静默截断窗口
            cap = getattr(self, "_recent_evals_cap", 20)
            if len(recent) > cap:
                recent.pop(0)
        except Exception:  # noqa: BLE001 — 观测失败不影响评估
            pass

    def _memory_gate(self, expr: str, arguments: dict[str, Any]) -> Any:
        """评估/提交前查研究记忆 advisory。返回 None | advisory dict | 拦截 result（ok=False）。"""
        if self.memory_store is None:
            return None
        try:
            advisory = self.memory_store.advisory_for(
                str(expr or ""),
                edit_note=arguments.get("edit_note"),
                current_run_id=getattr(self, "run_id", None),
                enable_advisory_cache=bool(getattr(self.memory_store, "enable_advisory_cache", True)),
            )
        except Exception:
            return None
        if not advisory:
            return None
        blocked = None
        if getattr(self.memory_store, "hard_block_duplicates", False):
            for item in advisory.get("advisories", []):
                if item.get("kind") == "duplicate_known_dead_end" and not item.get("exempt_from_block", False):
                    blocked = {
                        "ok": False,
                        "error": f"memory_blocked_duplicate: {item.get('message', '')}",
                        "error_type": "MemoryAdvisoryBlock",
                        "memory_advisory": advisory,
                    }
                    break
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

        out.append(
            {
                "type": "function",
                "function": {
                    "name": "recommend_mrmr_factors",
                    "description": (
                        "【mRMR 因子子集推荐】基于最大相关最小冗余算法，在统一因子库中挑选高质量且低相关冗余"
                        "的互补因子子集，输出 Top-k 推荐清单、IC 表现与冗余度。"
                    ),
                    "parameters": _RECOMMEND_MRMR_FACTORS_PARAMETERS,
                },
            }
        )

        out.append(
            {
                "type": "function",
                "function": {
                    "name": "precheck_expression",
                    "description": (
                        "【结构风险静态预检】对 DSL 表达式做纯 AST 分析，不触发评估："
                        "门控 threshold 过高（分箱坍缩）、分段中间区过宽、稀疏字段 FILLNA(0) 零簇、"
                        "已知死路结构。评估/提交前调用可避免在注定坍缩的结构上空转。"
                    ),
                    "parameters": _PRECHEK_PARAMETERS,
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

        # 数据面聚焦硬锁定：勾选聚焦面后，越界/未触面表达式在评估与提交前一律拦截
        if name in _FACET_LOCK_TOOLS and isinstance(arguments.get("multi_line_expr"), str):
            facet_block = self._facet_lock_block(name, arguments)
            if facet_block is not None:
                return facet_block

        # 算子黑名单（消融 D2）：评估/提交表达式命中即拦截
        if name in _FACET_LOCK_TOOLS:
            op_block = self._operator_blacklist_block(name, arguments.get("multi_line_expr"))
            if op_block is not None:
                return op_block

        if name == "submit_factor":
            return self._dispatch_submit(arguments)

        if name == "evaluate_factor":
            expr = arguments.get("multi_line_expr")
            profile_id = arguments.get("profile_id")
            if not isinstance(expr, str) or not expr.strip():
                return {"ok": False, "error": "multi_line_expr_required_non_empty_string", "error_type": "ToolArgumentsError"}
            if not isinstance(profile_id, str) or not profile_id.strip():
                return {"ok": False, "error": "profile_id_required_non_empty_string", "error_type": "ToolArgumentsError"}
            # prediction 软门：携带但字段非法 → 拦截；缺失 → 放行记账（见 _prediction_gate）
            pred_error = _prediction_argument_error(
                arguments, enabled=self.cognition_policy.get("prediction_check_enabled", True)
            )
            if pred_error is not None:
                return pred_error
            # profile 白名单（2026-09-10）：production_delivery 等"全区间/交付复检"
            # 口径的 split=full 含盲测段（2025+），LLM 在探索期直接调用会按盲测数据
            # 筛选迭代（多重检验烧盲测段）。全区间复检只属于 submit_factor 内部链路；
            # deepseek 实测一轮就摸到并使用了 19 次，必须硬性收口。
            if profile_id not in _MINING_ALLOWED_PROFILES:
                return {
                    "ok": False,
                    "error": (
                        f"profile_not_allowed_for_mining: {profile_id} —— 全区间/交付复检口径"
                        "（split=full，含盲测段）由 submit_factor 内部自动执行，挖掘期不得直接评估。"
                        "训练集海选用 profile_id='train_screen'，样本外验证用 eval_on_val_set。"
                    ),
                    "error_type": "ToolArgumentsError",
                }
            # 预审：拦截"两个裸 RANK 信号简单加减"的低级因子
            if _is_naive_signal_addition(expr):
                return {
                    "ok": False,
                    "error": "naive_signal_addition_blocked: 顶层 ADD/SUBTRACT(RANK(x), RANK(y)) 是低级信号叠加，"
                             "没有经济机制上的交互。多信息源融合必须使用结构化交互算子"
                             "（GATED_SIGNAL / CS_RESIDUALIZE / DIVERGENCE_RANK / CS_GROUP_RANK / TS_CORR 等）。",
                    "error_type": "NaiveSignalAdditionBlock",
                }
            # 同质化平滑变体动态熔断：滑动窗口内同根+套平滑累计 N 次 → 拦截评估
            homo_policy = getattr(self, "homogenization_policy", None) or {}
            homo_block = _homogenization_block(
                expr,
                getattr(self, "_recent_evals", None),
                max_consecutive=int(homo_policy.get("max_consecutive", 3)),
                window_size=int(homo_policy.get("window_size", 10)),
                enabled=bool(homo_policy.get("enabled", True)),
                max_outer_variants=int(homo_policy.get("max_outer_variants", 6)),
            )
            if homo_block is not None:
                try:
                    log_step("homogenization.block", f"signal_fingerprint_ast={_ast_signal_fingerprint(expr)} same_root_smooth>=homo_policy.max_consecutive")
                except Exception:
                    pass
                return homo_block
            gate = self._memory_gate(expr, arguments)
            if isinstance(gate, dict) and gate.get("ok") is False:
                return gate
            if profile_id == "train_screen":
                # 挖掘海选统一走两段式（lite 先筛）：LLM 无论选 evaluate_factor
                # 还是 eval_on_train_set，train 评估都享受同一吞吐改造
                # （2026-09-06：run2 实测 LLM 选 evaluate_factor 时走 eval_profile
                # 全量+图表，单次 60-78s，两段式被完全绕过）。
                result = self.service.eval_train(
                    EvalTrainRequest(
                        session_id=self.session_id,
                        multi_line_expr=expr,
                        factor_name=str(arguments.get("factor_name") or "expr"),
                        include_detail_tables=False,
                        label_quantile_n=10,
                    )
                )
            else:
                result = self.service.eval_profile(
                    EvalProfileRequest(
                        session_id=self.session_id,
                        profile_id=profile_id,
                        multi_line_expr=expr,
                        factor_name=str(arguments.get("factor_name") or "expr"),
                        include_charts=False,
                    )
                )
            if isinstance(result, dict) and result.get("ok"):
                ic, decile_rows = _engine_decile(result)
                if self.cognition_policy.get("prediction_check_enabled", True):
                    _attach_prediction_check(result, arguments.get("prediction"), ic=ic, decile_rows=decile_rows)
                session_obj = self.service.sessions.get(self.session_id) if (self.service and hasattr(self.service, "sessions")) else None
                _attach_yield_hints(result, expr, arguments, session=session_obj, recent_evals=getattr(self, "_recent_evals", None), turnover_gate_limit=_turnover_gate_limit_of(self))
                self._record_eval_signature(result, expr)
                if self.cognition_policy.get("ablation_check_enabled", True):
                    self._attach_ablation(expr, arguments, profile_id=profile_id, result=result)
                pred_block = self._prediction_gate("evaluate_factor", arguments, result)
                if pred_block is not None:
                    return pred_block
            elif isinstance(result, dict):
                self._attach_signature_hints(expr, result)
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
        # 同质化平滑变体动态熔断（eval_on_train_set / eval_on_val_set 共用）。
        # val 验证不参与熔断：val 是同一因子的样本外验证，不是新变体探索，
        # 不应被"同根变体过多"误伤（2026-09-23 实测 wma4_sz 的 val 被拦）。
        if name != "eval_on_val_set":
            homo_policy = getattr(self, "homogenization_policy", None) or {}
            homo_block = _homogenization_block(
                expr,
                getattr(self, "_recent_evals", None),
                max_consecutive=int(homo_policy.get("max_consecutive", 3)),
                window_size=int(homo_policy.get("window_size", 10)),
                enabled=bool(homo_policy.get("enabled", True)),
                max_outer_variants=int(homo_policy.get("max_outer_variants", 6)),
            )
            if homo_block is not None:
                try:
                    log_step("homogenization.block", f"signal_fingerprint_ast={_ast_signal_fingerprint(expr)} tool={name}")
                except Exception:
                    pass
                return homo_block

        factor_name = arguments.get("factor_name") or "expr"
        include_detail = bool(arguments.get("include_detail_tables", False))
        label_quantile_n = arguments.get("label_quantile_n", 10)
        if label_quantile_n is None:
            label_quantile_n = 10

        if name == "eval_on_train_set":
            pred_error = _prediction_argument_error(
                arguments, enabled=self.cognition_policy.get("prediction_check_enabled", True)
            )
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
                if self.cognition_policy.get("prediction_check_enabled", True):
                    _attach_prediction_check(result, arguments.get("prediction"), ic=ic, decile_rows=decile_rows)
                if self.cognition_policy.get("ablation_check_enabled", True):
                    self._attach_ablation(expr, arguments, profile_id="train_screen", result=result)
                session_obj = self.service.sessions.get(self.session_id) if (self.service and hasattr(self.service, "sessions")) else None
                _attach_yield_hints(result, expr, arguments, session=session_obj, recent_evals=getattr(self, "_recent_evals", None), turnover_gate_limit=_turnover_gate_limit_of(self))
                self._record_eval_signature(result, expr)
                pred_block = self._prediction_gate("eval_on_train_set", arguments, result)
                if pred_block is not None:
                    return pred_block
            elif isinstance(result, dict):
                self._attach_signature_hints(expr, result)
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
            if isinstance(result, dict) and result.get("ok") and arguments.get("prediction") is not None \
                    and self.cognition_policy.get("prediction_check_enabled", True):
                ic, decile_rows = _legacy_decile(result)
                _attach_prediction_check(result, arguments.get("prediction"), ic=ic, decile_rows=decile_rows)
            if isinstance(gate, dict):
                result["memory_advisory"] = gate
            return result

        if name == "screen_factors":
            return self._dispatch_screen_factors(arguments)

        if name == "recommend_mrmr_factors":
            return self._dispatch_recommend_mrmr_factors(arguments)

        if name == "precheck_expression":
            return self._dispatch_precheck_expression(arguments)

        return {"ok": False, "error": f"unknown_tool: {name}", "error_type": "UnknownTool"}

    def _dispatch_precheck_expression(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """S1：结构风险静态预检（纯 AST，不触发评估、不触达盲测段）。"""
        from alphaagent.factor.mining.tools._precheck import precheck_expression

        expr = arguments.get("multi_line_expr")
        if not isinstance(expr, str) or not expr.strip():
            return {"ok": False, "error": "multi_line_expr_required_non_empty_string", "error_type": "ToolArgumentsError"}
        try:
            return precheck_expression(expr)
        except Exception as exc:  # noqa: BLE001 — 预检失败不阻断
            return {
                "ok": True,
                "result": {
                    "risks": [{"kind": "precheck_error", "risk": "unknown",
                               "hint": f"{type(exc).__name__}: {str(exc)[:160]}"}],
                    "risk_count": 1,
                    "max_risk": "unknown",
                    "summary": "结构预检异常，请以评估结果为准。",
                },
            }

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

        # 同质化平滑变体动态熔断（提交前同样拦截，防 LLM 绕过评估直接提交）
        homo_policy = getattr(self, "homogenization_policy", None) or {}
        homo_block = _homogenization_block(
            str(expr or ""),
            getattr(self, "_recent_evals", None),
            max_consecutive=int(homo_policy.get("max_consecutive", 3)),
            window_size=int(homo_policy.get("window_size", 10)),
            enabled=bool(homo_policy.get("enabled", True)),
            max_outer_variants=int(homo_policy.get("max_outer_variants", 6)),
        )
        if homo_block is not None:
            try:
                log_step("homogenization.block", f"signal_fingerprint_ast={_ast_signal_fingerprint(str(expr or ''))} tool=submit_factor")
            except Exception:
                pass
            return homo_block

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

    def _dispatch_recommend_mrmr_factors(self, arguments: dict[str, Any]) -> dict[str, Any]:
        """基于 mRMR 算法从候选池中推荐高质量且低冗余的因子子集。"""
        import pandas as pd
        from alphaagent.factor.cache import FactorValueCache
        from alphaagent.factor.stacking import build_stacking_dataset, collect_factor_entries
        from alphaagent.factor.stacking.model import mrmr_rank_features
        from alphaagent.factor.window_config import DEFAULT_TRAIN_END, DEFAULT_VAL_END

        k = int(arguments.get("k") or 8)
        k = max(1, min(k, 30))
        max_corr = float(arguments.get("max_corr") or 0.6)
        no_candidate = bool(arguments.get("no_candidate", False))
        include = arguments.get("include_factors")

        entries = collect_factor_entries(
            include_candidate=not no_candidate, include_production=True
        )
        if include and isinstance(include, list):
            wanted = {str(n).strip() for n in include if str(n).strip()}
            if wanted:
                entries = [e for e in entries if e.name in wanted]

        if len(entries) < 2:
            return {
                "ok": False,
                "error": "insufficient_factors",
                "message": f"可用因子数量不足（当前仅 {len(entries)} 个，需至少 2 个）",
            }

        panel = None
        if hasattr(self, "session_id") and self.session_id and hasattr(self, "service"):
            try:
                sess = self.service.sessions.get(self.session_id)
                if sess and sess.panel is not None and len(sess.panel) > 0:
                    panel = sess.panel
            except Exception:
                panel = None

        mining_end = pd.Timestamp(DEFAULT_TRAIN_END)
        if panel is None:
            from alphaagent.data.adapters.cnequity import load_panel_from_cne
            panel_start = mining_end - pd.DateOffset(months=12) - pd.DateOffset(days=250)
            end = pd.Timestamp(DEFAULT_VAL_END)
            panel = load_panel_from_cne(start=panel_start, end=end, include_fundamentals=True)

            from alphaagent.factor.cache import get_default_cache
            cache = get_default_cache()
        dataset = build_stacking_dataset(
            panel,
            entries,
            label_days=5,
            mining_end=mining_end,
            size_neutral=True,
            max_corr=max_corr,
            cache=cache,
            decay_months=12,
        )

        if len(dataset.feature_names) < 2:
            return {
                "ok": False,
                "error": "insufficient_valid_features",
                "message": f"去重后有效特征不足（仅 {len(dataset.feature_names)} 个）",
            }

        dts = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
        rec_start = mining_end - pd.DateOffset(months=12)
        ranking = mrmr_rank_features(
            dataset.feature_matrix,
            dataset.label,
            pd.Series(dts),
            dataset.feature_names,
            window_start=rec_start,
            window_end=mining_end,
            k=k,
            beta=0.7,
        )

        name_lib_map = {e.name: e.library for e in entries}
        for item in ranking:
            item["library"] = "正式" if "production" in name_lib_map.get(item["name"], "") else "候选"

        return {
            "ok": True,
            "tool": "recommend_mrmr_factors",
            "k": k,
            "n_recommended": len(ranking),
            "ranking": ranking,
            "recommended_names": [r["name"] for r in ranking],
            "window": f"[{rec_start.date()} ~ {mining_end.date()}]",
            "summary": f"已通过 mRMR 选出 {len(ranking)} 个互补因子（Top: {', '.join(r['name'] for r in ranking[:3])}）",
        }
