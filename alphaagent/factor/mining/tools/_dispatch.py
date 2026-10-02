"""FactorEvalTools 核心：__init__ / schemas / dispatch / submit / screener。"""
from __future__ import annotations

import json
import logging
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


_PREDICTION_SOFT_LIMIT = 3   # 兜底默认值；真源 evaluation_policy.prediction_soft_limit（P5 收口）


def _prediction_soft_limit() -> int:
    """prediction 缺失软门次数：读配置中心（原硬编码常量，2026-10-02 P5 收口）。"""
    try:
        from alphaagent.factor.evaluation.defaults import DEFAULT_EVALUATION_POLICY

        return int(DEFAULT_EVALUATION_POLICY.get("prediction_soft_limit", _PREDICTION_SOFT_LIMIT))
    except Exception:  # noqa: BLE001
        return _PREDICTION_SOFT_LIMIT

# 认知对账开关（research_spec.cognition_policy，消融 C1/C2）：缺省全开不改行为。
# prediction_check_enabled=False → 不注入 prediction_check、缺失软门不再升级拦截；
# ablation_check_enabled=False → 不注入 ablation_check/ablation_hint。
_DEFAULT_COGNITION_POLICY = {
    "prediction_check_enabled": True,
    "ablation_check_enabled": True,
    # 2026-09-25：train 海选过线（promising 线）后系统自动跑样本外验证。
    # promising 语义是"训练样本过线"而非"质量结论"，强制 val 把"过线→验证"
    # 变成系统行为，不再依赖模型自觉（实测 21 promising / 0 submit）。
    "force_val_on_promising": True,
}
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

# 判定侧需要、但工具对象上**没有**的研报配置键（2026-10-01 修）：
# `self.report_policy` 全仓从未被赋值（`getattr(self, "report_policy", None)` 恒为 None），
# 导致复现判定读不到 report_policy —— 本夜实测：保真度门槛被静默跳过（0 条
# `report_fidelity_check` 日志），同时 reproduce_min_abs_ic/icir 也一直吃硬编码默认值。
# 现改由 run gate（判定侧唯一可靠通道）携带这些键。
_GATE_CARRIED_POLICY_KEYS = (
    "reproduce_min_abs_ic",
    "reproduce_min_icir",
    "reproduce_shape_requirement",
    "reproduce_fidelity",
    "factor_records_file",
    "question_queue_file",
    "inject_factor_records",
)


def _effective_report_policy(tool_policy: dict[str, Any] | None,
                             gate: dict[str, Any] | None) -> dict[str, Any]:
    """合并「工具对象上的 report_policy」与「gate 携带的判定配置」（gate 优先）。

    纯函数，便于单测：gate 里同名的非 None 值覆盖工具侧取值。
    """
    merged: dict[str, Any] = dict(tool_policy or {})
    for key in _GATE_CARRIED_POLICY_KEYS:
        val = (gate or {}).get(key)
        if val is not None:
            merged[key] = val
    return merged


def _reproduce_strength(ic: float, icir: float, *, min_ic: float, min_icir: float):
    """复现「强度」判定：**IC 与 ICIR 一律按绝对值比较**（2026-10-01 修）。

    为什么：负 IC / 负 ICIR 是**正常信号**（只是方向相反，强度等价）。系统里方向相关
    判定已有三处且都是"方向无关"口径：①`prediction` 形态/方向对账（`eval/prediction.py`，
    模型自己声明 expected_sign，负方向同样可 confirmed）；②train↔val 方向一致性
    （`require_sign_consistency`，只要求两段同号，不要求为正）；③正式库门槛 `abs_gte`
    （`evaluation/profile.py:158`）。
    唯独复现判定原先 `abs(ic)` 与**带符号** `icir` 混用 → 方向反向但强度足够的机制
    被判"未过线"（实测 3 次：|IC| 0.0125~0.0225、|ICIR| 0.104~0.234，例
    `rq045_vwap_bias_to_sz_extreme` |IC|=0.0225/|ICIR|=0.2335），并与正式门槛口径漂移。
    修法：两侧都取绝对值，方向不参与强度判定。

    返回 ``(是否达标, {"ic": 原始IC, "icir": 原始ICIR, "abs_ic":…, "abs_icir":…})``，
    保留原始带符号值供日志与父本体检展示（避免"日志看起来符号矛盾"）。
    """
    abs_ic = abs(float(ic))
    abs_icir = abs(float(icir))
    return (abs_ic >= min_ic and abs_icir >= min_icir), {
        "ic": float(ic), "icir": float(icir),
        "abs_ic": abs_ic, "abs_icir": abs_icir,
    }


def _reproduce_passes(ic: float, icir: float, shape_verdict: str, strong_side_match: bool, *,
                      min_ic: float, min_icir: float, shape_requirement: str = "strong_side"):
    """复现通过判据（唯一实现）：**【指标】 ∧ 【形态要求】**（2026-10-01 两次定调后的形态）。

    演进：`指标 OR 形态` → `底线 ∧ 指标 ∧ confirmed` → **`指标 ∧ 形态要求`**（本实现）：
      · 底线 `reproduce_floor_abs_ic` **已删除** —— 它与 `reproduce_min_abs_ic` 同值时被
        后者完全包含（判据为「且」后零影响），是冗余键；
      · 形态项由硬编码 `confirmed` 改为**可配档位**（`reproduce_shape_requirement`）：
          - `strong_side`（默认）：实际强侧 == 声明强侧。**经济含义强且独立于 IC** ——
            IC 只看全体股票相关性，看不见"收益集中在中间组、整体相关仍为正"的因子
            （多空组合赚不到钱）；实测该档与 `confirmed` 质量等价
            （|IC| 0.0179/0.206 vs 0.0180/0.210）却少拦 40 条同强度母本；
          - `confirmed`：prediction_check 完全一致（强侧+形态+方向），最严，备选；
          - `off`：仅按指标，最松。
        `actual_shape/expected_shape` 字符串**恒作诊断信息**输出，不受档位影响。

    - **指标**：|IC| ≥ min_ic 且 |ICIR| ≥ min_icir（都按 **abs**；负值=方向相反、强度等价）。
    - **形态**：见上，由 `shape_requirement` 决定。

    阈值唯一真源：``research_spec.DEFAULT_RESEARCH_SPEC['report_policy']``
    （reproduce_min_abs_ic / reproduce_min_icir / reproduce_shape_requirement），
    随 run gate 传到判定侧（``agentscope_run._gate_state``）。
    返回 ``(是否通过, 明细)``；明细含原始带符号值、abs 强度与各项命中情况，供日志/诊断。
    """
    by_metrics, detail = _reproduce_strength(ic, icir, min_ic=min_ic, min_icir=min_icir)
    by_shape = (
        shape_requirement == "off"
        or (shape_requirement == "strong_side" and strong_side_match)
        or (shape_requirement == "confirmed" and shape_verdict == "confirmed")
    )
    return (by_metrics and by_shape), {
        **detail,
        "by_metrics": by_metrics, "by_shape": by_shape,
        "strong_side_match": strong_side_match, "shape_verdict": shape_verdict,
        "shape_requirement": shape_requirement,
    }


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
        # 强度口径：IC / ICIR 一律 abs（负值=方向相反，强度等价；2026-10-01 统一）
        ic_f = abs(float(ic)) if ic is not None else None
        icir_f = abs(float(icir)) if icir is not None else None
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
            if n >= _prediction_soft_limit():
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
                f"本次未携带 prediction，结果未做预期对账（第 {n}/{_prediction_soft_limit()} 次，"
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

    def _auto_val_verify(self, result: dict[str, Any], expr: str, factor_name: str) -> None:
        """训练海选过线（promising 线）后**系统自动**跑一次样本外验证。

        2026-09-25（feat/promising-to-val-gate）：promising 语义 = "训练样本海选
        过线"，不是质量结论。此前模型拿到 promising 后继续同根变异而非推 val
        （实测 21 promising / 0 submit），现把"过线 → val"变成系统行为：
        评估成功且达 promising 线时自动调用 eval_val，结果合并进
        ``result["val_verification"]``；val 不过 → ``result["val_failed"]=True``
        并附强提示。异常绝不阻断主评估（增强信息）。

        阈值与 ``memory._classify`` 同源（evaluation_policy），val 门槛与
        ``DeliveryCriteria.candidate`` 同源——避免口径漂移。
        """
        # 研报模式复现判定：**先于所有 early-return**（2026-09-30 修：原先放在
        # force_val_on_promising / 指标解析之后，且指标键取 metrics.cross_sectional_core——
        # 实测结果里该键不存在（指标在 summary），导致判定几乎从不触发）。
        # 2026-09-30 review 修：本块原先插在 docstring **之前**，使 """...""" 沦为
        # 死字符串语句（__doc__ 为空）；现 docstring 归位到 def 之后。
        try:
            self._report_reproduce_judge(result, factor_name, expr=expr)
        except Exception as _e:  # noqa: BLE001
            logging.info("report_reproduce_judge_exception %s", _e)

        try:
            if not self.cognition_policy.get("force_val_on_promising", True):
                return
            cs = (result.get("metrics") or {}).get("cross_sectional_core") or result.get("summary") or {}
            ic = cs.get("ic")
            icir = cs.get("icir")
            cov = cs.get("factor_coverage", cs.get("coverage"))
            try:
                # 强度口径统一 abs（IC 与 ICIR 同口径；负值只是方向相反）
                ic_f = abs(float(ic))
                icir_f = abs(float(icir))
                cov_f = float(cov)
            except (TypeError, ValueError):
                return
            from alphaagent.factor.mining.research_spec import DEFAULT_RESEARCH_SPEC

            ep = DEFAULT_RESEARCH_SPEC["evaluation_policy"]
            if not (
                ic_f >= float(ep["min_train_abs_ic"])
                and icir_f > float(ep.get("min_train_icir_soft", 0.2))
                and cov_f > float(ep["min_train_coverage"])
            ):
                return
            # 防抖：同表达式（归一化）只自动 val 一次（会话级）
            key = re.sub(r"\s+", "", expr)
            done = getattr(self, "_auto_val_done", None)
            if done is None:
                done = self._auto_val_done = set()
            if key in done:
                return
            done.add(key)

            from alphaagent.factor.mining.eval.schemas import EvalValRequest

            sign = 1 if float(ic) >= 0 else -1
            val = self.service.eval_val(
                EvalValRequest(
                    session_id=self.session_id,
                    multi_line_expr=expr,
                    factor_name=f"{factor_name}__auto_val",
                    expected_sign=sign,
                )
            )
            if not isinstance(val, dict) or not val.get("ok"):
                return
            vcs = (val.get("metrics") or {}).get("cross_sectional_core") or val.get("summary") or {}
            v_ic = vcs.get("ic")
            v_icir = vcs.get("icir")
            v_cov = vcs.get("factor_coverage", vcs.get("coverage"))
            v_sign = (val.get("sign_check") or {}).get("matches_expected_sign")

            from alphaagent.factor.mining.delivery_criteria import DeliveryCriteria

            crit = DeliveryCriteria.defaults()
            retention = abs(float(v_ic) / float(ic)) if v_ic is not None and float(ic) else None
            val_ok = bool(
                v_ic is not None
                and abs(float(v_ic)) >= float(crit.candidate.min_val_abs_ic)
                and (retention is None or retention >= 0.5)
                and v_sign is not False
            )
            result["val_verification"] = {
                "auto": True,
                "val_ic": round(float(v_ic), 6) if v_ic is not None else None,
                "val_icir": round(float(v_icir), 6) if v_icir is not None else None,
                "val_coverage": round(float(v_cov), 4) if v_cov is not None else None,
                "val_retention": round(float(retention), 4) if retention is not None else None,
                "sign_consistent": v_sign,
                "passed": val_ok,
                "min_val_abs_ic": float(crit.candidate.min_val_abs_ic),
                "guidance": (
                    "训练样本海选过线且样本外验证通过，可直接 submit_factor 走交付门槛，"
                    "或作为父本向相邻机制扩展（不要继续堆同根变体）。"
                    if val_ok
                    else "训练样本海选过线但**样本外验证未通过**：结构在 val 段不成立，"
                    "禁止继续同根变异或提交，应更换信号根/机制。"
                ),
            }
            if not val_ok:
                result["val_failed"] = True
        except Exception:  # noqa: BLE001 — 自动 val 是增强信息，失败绝不影响主评估
            pass

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

        # 研报模式网关：取调用参数携带的一份（跨边界最可靠）。2026-09-30 review 修：
        # (1) 原提取块在 JSON 解析**之前** → arguments 是 JSON 字符串时永远取不到网关；
        # (2) 每次调用无携带网关时**显式重置**为 None → 否则上一次 dispatch 留下的旧网关
        #     会压掉本轮新网关（判定会拿着旧题的 qid/phase 跑）。
        try:
            _incoming_gate = arguments.get("_report_gate")
            self._report_gate_arg = (
                dict(_incoming_gate) if isinstance(_incoming_gate, dict) and _incoming_gate else None
            )
        except Exception:  # noqa: BLE001
            self._report_gate_arg = None

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
                self._auto_val_verify(result, expr, str(arguments.get("factor_name") or "expr"))
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
                self._auto_val_verify(result, expr, factor_name)
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

    def _report_reproduce_judge(self, result: dict[str, Any], factor_name: str,
                                expr: str = "") -> None:
        """研报模式复现判定（唯一实现）。

        通过条件（**全部满足**，且须过"原文锚"保真度校验）：
        ① |IC| ≥ reproduce_min_abs_ic 且 |ICIR| ≥ reproduce_min_icir（指标，均按绝对值）；
        ② 形态要求 `reproduce_shape_requirement`（默认 strong_side = 强侧匹配；
           可选 confirmed / off）。
        指标**兼容** ``metrics.cross_sectional_core`` 与 ``summary`` 两种结构（实测引擎输出为后者）。
        每次判定都写日志，便于事后归因。判据唯一实现在 ``_reproduce_passes``。
        """
        from alphaagent.factor.mining.report_channels import get_run_gate

        # 显式 None 哨兵区分「本次调用未携带网关」与「携带了空网关」：只有前者回落到
        # 进程内全局网关，避免 `or` 把空字典当未设置、或旧实例属性压掉本轮新网关
        # （2026-09-30 review 修）。
        _gate_arg = getattr(self, "_report_gate_arg", None)
        gate = _gate_arg if _gate_arg is not None else (
            get_run_gate() or getattr(self, "report_reproduce_gate", None) or {}
        )
        phase = str(gate.get("phase") or "")
        qid = str(gate.get("qid") or "")
        if phase != "reproduce" or not qid:
            log_step("report_reproduce_judge",
                     f"factor={factor_name} 跳过 phase={phase or '-'} qid={qid or '-'}")
            return
        rp = _effective_report_policy(getattr(self, "report_policy", None), gate)
        min_ic = float(rp.get("reproduce_min_abs_ic", 0.010))
        min_icir = float(rp.get("reproduce_min_icir", 0.10))
        # 形态要求档位（2026-10-01）：strong_side（默认，经济含义强且独立于 IC）/
        # confirmed（最严）/ off（仅指标）。底线已删除（与 min_abs_ic 同值属冗余）。
        shape_req = str(rp.get("reproduce_shape_requirement", "strong_side") or "strong_side")
        cs = (result.get("metrics") or {}).get("cross_sectional_core") or result.get("summary") or {}
        try:
            ic_s = float(cs.get("ic"))
            icir_s = float(cs.get("icir"))
        except (TypeError, ValueError):
            log_step("report_reproduce_judge", f"qid={qid} factor={factor_name} 指标缺失 -> 跳过")
            return
        # 强度：IC 与 ICIR **都按绝对值**（负值是正常方向，见 _reproduce_strength 说明）
        _pc = result.get("prediction_check") or {}
        shape = str(_pc.get("verdict") or "")
        # 强侧匹配：实际强侧 == 声明强侧（IC 看不见的可交易性维度）
        _act_side = str((_pc.get("actual") or {}).get("strong_side") or "")
        _exp_side = str((_pc.get("expected") or {}).get("expected_strong_side") or "")
        strong_side_match = bool(_act_side and _exp_side and _act_side == _exp_side)
        _passed, _stg = _reproduce_passes(
            ic_s, icir_s, shape, strong_side_match,
            min_ic=min_ic, min_icir=min_icir, shape_requirement=shape_req,
        )
        by_metrics = _stg["by_metrics"]
        by_shape = _stg["by_shape"]
        if not _passed:
            # 日志同时给「真实带符号值」「判定用绝对值」与「底线/通道命中情况」，
            # 避免 ic/icir 符号看起来矛盾，也便于区分是"没过底线"还是"两条通道都没走通"
            log_step(
                "report_reproduce_judge",
                f"qid={qid} factor={factor_name} ic={ic_s:+.4f} icir={icir_s:+.4f} "
                f"|IC|={_stg['abs_ic']:.4f} |ICIR|={_stg['abs_icir']:.4f} "
                f"metrics={'过' if by_metrics else '未过'} "
                f"shape={shape or '-'}"
                f"({_stg['strong_side_match'] and '强侧匹配' or (_act_side + '≠' + _exp_side).strip('≠')}) "
                f"要求={shape_req} -> 未过线",
            )
            return
        # ── 复现保真度「原文锚」硬门槛（2026-10-01）────────────────────────
        # 依据：昨夜实测 10 个"有原文公式"的课题，复现公式与原文的字段 Jaccard 平均
        # 0.062、算子 Jaccard 0.085——模型靠题面约束脱稿自由探索，用容易过线的通用因子
        # （5 日反转 / Amihud / PE 动量）顶替原文机制。此处对"过线"增设机械校验：
        # 复现公式必须与原文公式共享 ≥N 个**报告特有字段**（通用行情字段不算）。
        # 仅研报模式：本函数只由 report gate 触发、配置在 report_policy 下。
        # 客观锚通道：校验结果写进 result["report_fidelity"]（模型可见）+ steps.log。
        _fid_cfg = rp.get("reproduce_fidelity") or {}
        if bool(_fid_cfg.get("enabled", False)) and expr:
            try:
                from alphaagent.factor.mining.agent.question_queue import (
                    check_reproduce_fidelity,
                    find_question,
                )

                _fid_spec = {"report_policy": rp}
                _q = find_question(qid, _fid_spec)
                _fid = check_reproduce_fidelity(_q, expr, spec=_fid_spec)
                result["report_fidelity"] = _fid
                log_step(
                    "report_fidelity_check",
                    f"qid={qid} factor={factor_name} passed={_fid.get('passed')} "
                    f"ref_n={_fid.get('n_records')} "
                    f"shared_specific={_fid.get('shared_specific_fields')} "
                    f"fj={_fid.get('field_jaccard')} oj={_fid.get('op_jaccard')} "
                    f"reason={_fid.get('reason')}",
                )
                if not _fid.get("passed"):
                    log_step(
                        "report_reproduce_judge",
                        f"qid={qid} factor={factor_name} 原文锚未达标"
                        f"（{_fid.get('reason')}）-> 不予过线",
                    )
                    return
            except Exception as _fe:  # noqa: BLE001
                # 校验异常绝不静默变成"放行"：记录后按未过线处理（宁严勿松）
                log_step("report_fidelity_check",
                         f"qid={qid} factor={factor_name} 校验异常 -> 不予过线: "
                         f"{type(_fe).__name__}: {str(_fe)[:120]}")
                return

        from alphaagent.factor.mining.agent.question_state import mark_reproduce_ok

        _parts = []
        if by_shape:
            _parts.append(f"shape={shape or 'confirmed'}")
        if by_metrics:
            _parts.append("signal=ok")
        _parts.append(f"ic={ic_s:+.4f} icir={icir_s:+.4f}")
        detail = " | ".join(_parts)
        # 父本"体检报告"：供发散题面给针对性改进建议（ic/icir/cov/换手/形态对账）
        _prof: dict = {}
        for _k in ("ic", "icir"):
            if cs.get(_k) is not None:
                try:
                    _prof[_k] = float(cs[_k])
                except (TypeError, ValueError):
                    pass
        _cov = cs.get("factor_coverage", cs.get("coverage"))
        if _cov is not None:
            try:
                _prof["cov"] = float(_cov)
            except (TypeError, ValueError):
                pass
        _ac = cs.get("cs_pearson_autocorr")
        if _ac is not None:
            try:
                _prof["autocorr"] = float(_ac)
            except (TypeError, ValueError):
                pass
        for _tk in ("avg_daily_side_turnover", "avg_turnover"):
            if result.get(_tk) is not None:
                try:
                    _prof["turnover"] = float(result[_tk])
                    break
                except (TypeError, ValueError):
                    pass
        _pc = result.get("prediction_check") or {}
        if _pc.get("verdict"):
            _prof["shape_verdict"] = str(_pc["verdict"])
        if (_pc.get("actual") or {}).get("shape"):
            _prof["actual_shape"] = str(_pc["actual"]["shape"])
        if (_pc.get("expected") or {}).get("expected_shape"):
            _prof["expected_shape"] = str(_pc["expected"]["expected_shape"])
        try:
            _lock_rounds = max(0, int(gate.get("lock_rounds") or 3))
        except (TypeError, ValueError) as _lr_err:
            # 不再静默吞掉：非法值要能在日志里与"正常默认"区分开（第二轮 review 修）
            logging.warning("lock_rounds 非法（%r），回退默认 3: %s", gate.get("lock_rounds"), _lr_err)
            _lock_rounds = 3
        mark_reproduce_ok(
            str(gate.get("mode") or "report"), qid, _lock_rounds,
            factor=factor_name, detail=detail, profile=_prof,
        )
        # 判定通过后把本轮阶段推进到 diverge。**不得原地改 gate**：它很可能就是进程内
        # 全局网关对象，原地改会把 phase 永久留成 diverge，后续新题的复现判定被静默跳过
        # （2026-09-30 review 修：改经 set_run_gate 写回新字典）。
        _next_gate = {**gate, "phase": "diverge"}
        try:
            from alphaagent.factor.mining.report_channels import set_run_gate

            set_run_gate(_next_gate)
            # 全局网关已是权威通道：清掉实例缓存，避免「下一次 dispatch 之前」仍有旧
            # phase=diverge 压住新一轮判定（2026-09-30 review 修）。
            self._report_gate_arg = None
        except Exception as _se:  # noqa: BLE001
            # 不做"写实例属性兜底"：dispatch() 每次调用都会重置该属性，兜底值在下次
            # dispatch 之前就被覆盖，等于无效保护（第二轮 review 指出）→ 只留痕。
            logging.warning("set_run_gate(diverge) 失败（不影响本次判定结果）: %s", _se)
        log_step("report_reproduce_judge", f"qid={qid} factor={factor_name} PASS [{detail}] -> reproduce_ok")
