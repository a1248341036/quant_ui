"""因子挖掘评估结果诊断与反馈子系统。

采用策略模式 (Strategy) 实现各维度的独立诊断，并通过中央仲裁者 (DiagnosticMediator)
协调反馈建议的优先级，消除大模型在不同门禁与先验之间的规则打架与自相矛盾。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from alphaagent.factor.mining.delivery.delivery_criteria import DeliveryCriteria
from alphaagent.factor.mining.research_spec import DEFAULT_RESEARCH_SPEC


@dataclass
class DiagnosticContext:
    """因子评估诊断输入上下文。"""

    expr: str
    result: dict[str, Any]
    arguments: dict[str, Any]
    metrics: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)
    split: str = "train"
    passed: bool | None = None
    ic: float | None = None
    abs_ic: float | None = None
    icir: float | None = None
    abs_icir: float | None = None
    coverage: float | None = None
    turnover: float | None = None
    research_mode: str | None = None
    screen_rules: list[dict[str, Any]] = field(default_factory=list)
    prediction_check: dict[str, Any] | None = None
    recent_signatures: list[str] = field(default_factory=list)
    session: Any | None = None

    @classmethod
    def from_eval_result(
        cls, result: dict[str, Any], expr: str, arguments: dict[str, Any]
    ) -> "DiagnosticContext":
        metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
        summary = result.get("summary") if isinstance(result.get("summary"), dict) else {}
        cs = metrics.get("cross_sectional_core") or {
            k: v for k, v in summary.items() if k in ("ic", "icir", "rank_ic", "factor_coverage")
        }

        ic_val = cs.get("ic")
        icir_val = cs.get("icir")
        cov_val = cs.get("factor_coverage", cs.get("coverage"))

        try:
            ic_f = float(ic_val) if ic_val is not None else None
            abs_ic_f = abs(ic_f) if ic_f is not None else None
        except (TypeError, ValueError):
            ic_f = abs_ic_f = None

        try:
            icir_f = float(icir_val) if icir_val is not None else None
            abs_icir_f = abs(icir_f) if icir_f is not None else None
        except (TypeError, ValueError):
            icir_f = abs_icir_f = None

        try:
            cov_f = float(cov_val) if cov_val is not None else None
        except (TypeError, ValueError):
            cov_f = None

        # 换手率：优先从顶层取（format_eval_response 后的结构，权威值），
        # 回退到 metrics.quantile_portfolio（引擎原始结构）。
        # 2026-09-19 修复：此前只从 metrics.quantile_portfolio 取，但
        # apply_diagnostics_to_result 收到的是 format_eval_response 后的 result，
        # 换手率在顶层 avg_daily_side_turnover，导致 ctx.turnover 永远为 None，
        # 换手率诊断器从不触发。
        turnover_val = result.get("avg_daily_side_turnover")
        if turnover_val is None:
            qp = metrics.get("quantile_portfolio") or {}
            turnover_val = qp.get("avg_daily_side_turnover") if isinstance(qp, dict) else None
        try:
            turnover_f = float(turnover_val) if turnover_val is not None else None
        except (TypeError, ValueError):
            turnover_f = None

        passed_val = result.get("passed")
        rules = result.get("screen_rules")
        rule_list = rules if isinstance(rules, list) else []
        if passed_val is None and rule_list:
            passed_val = all(bool(r.get("passed")) for r in rule_list)
        # 2026-09-19 修复：两段式海选下，screen_stage="full" 意味着 lite 已过线
        # 才跑全量 profile，此时 passed=True。此前格式化后的 result 既无 passed
        # 也无 screen_rules（仅 lite 未过线时才有 screen_rules），导致 ctx.passed
        # 永远为 None，换手率诊断器（要求 ctx.passed）从不触发。
        if passed_val is None and result.get("screen_stage") == "full":
            passed_val = True

        return cls(
            expr=expr or "",
            result=result,
            arguments=arguments or {},
            metrics=metrics,
            summary=summary,
            split=str(result.get("split") or "train"),
            passed=passed_val,
            ic=ic_f,
            abs_ic=abs_ic_f,
            icir=icir_f,
            abs_icir=abs_icir_f,
            coverage=cov_f,
            turnover=turnover_f,
            research_mode=result.get("research_mode"),
            screen_rules=rule_list,
            prediction_check=result.get("prediction_check") if isinstance(result.get("prediction_check"), dict) else None,
        )


@dataclass
class DiagnosticOpinion:
    """单个诊断器输出的结构化意见。"""

    source: str
    opinion_type: str  # 'block_submit' | 'encourage_submit' | 'near_miss' | 'warning' | 'collapse_risk'
    title: str
    message: str
    reasons: list[str] = field(default_factory=list)
    hints: list[str] = field(default_factory=list)
    priority: int = 50  # 数值越大优先级越高 (0~100)
    extra_fields: dict[str, Any] = field(default_factory=dict)


class BaseDiagnostic(ABC):
    """因子诊断策略抽象基类。"""

    @property
    @abstractmethod
    def name(self) -> str:
        """诊断器唯一名称。"""

    @abstractmethod
    def evaluate(self, ctx: DiagnosticContext) -> DiagnosticOpinion | None:
        """根据上下文给出诊断意见。返回 None 表示不触发该诊断。"""


class PitSuspicionDiagnostic(BaseDiagnostic):
    """异常高 IC 警告诊断（防 PIT 泄漏与阶梯函数陷阱，区分基本面与价量因子）。"""

    name = "pit_suspicion"
    _THRESHOLD: float = 0.045

    def evaluate(self, ctx: DiagnosticContext) -> DiagnosticOpinion | None:
        if ctx.passed:
            return None
        # P1-3 动态细分：仅对基本面/慢标签因子或触及 funda_ 列的因子警示 PIT 伪影
        expr_str = str(ctx.expr or "").lower()
        is_funda = "funda_" in expr_str or str(ctx.research_mode) == "fundamental"
        if not is_funda:
            return None

        if ctx.abs_ic is not None and ctx.abs_ic >= self._THRESHOLD:
            msg = (
                f"训练 |IC|={ctx.abs_ic:.4f} 异常高（≥{self._THRESHOLD:.3f}）——财务/慢标签因子的 train IC 虚高"
                "通常来自阶梯函数语义陷阱或 PIT 泄漏（实测该类因子 val 保留率极低）。"
                "提交前先核查：①因子是否只用披露日之前的数据；②窗口对齐是否引入未来函数；"
                "③建议直接 eval_on_val_set 看真实保留比，勿按 train IC 定预期。"
            )
            return DiagnosticOpinion(
                source=self.name,
                opinion_type="warning",
                title="PIT 泄漏嫌疑",
                message=msg,
                priority=90,
                extra_fields={"pit_warning": msg},
            )
        return None


class NearMissDiagnostic(BaseDiagnostic):
    """接近海选线诊断（差临门一脚引导）。"""

    name = "near_miss"
    _RATIO: float = 0.80

    def evaluate(self, ctx: DiagnosticContext) -> DiagnosticOpinion | None:
        if ctx.passed or ctx.abs_ic is None:
            return None

        # 从屏幕规则或真源获取 threshold
        th = self._get_threshold(ctx)
        default_ep = DEFAULT_RESEARCH_SPEC["evaluation_policy"]
        icir_soft = float(default_ep.get("min_train_icir_soft", 0.2))
        cov_thr = float(default_ep.get("min_train_coverage", 0.85))

        if (
            self._RATIO * th <= ctx.abs_ic < th
            and ctx.abs_icir is not None
            and ctx.abs_icir > icir_soft
            and ctx.coverage is not None
            and ctx.coverage > cov_thr
        ):
            # P1-2 边际贡献分析（十分位分桶分析）
            dml = ctx.metrics.get("decile_mean_label") or ctx.summary.get("decile_mean_label") or []
            marginal_hint = ""
            if isinstance(dml, list) and len(dml) >= 10:
                try:
                    means = [float(r.get("mean_label") or 0.0) for r in dml]
                    mid_spread = abs(means[6] - means[3])
                    end_spread = abs(means[-1] - means[0])
                    if end_spread > 0 and mid_spread / end_spread < 0.2:
                        marginal_hint = "（边际分析：主要拖累在中盘 D4-D7 组缺乏区分度，可考虑加入波动率或市值过滤）"
                except Exception:
                    pass

            msg = (
                f"接近海选线（|IC|={ctx.abs_ic:.4f}，ICIR={ctx.abs_icir:.3f}，coverage={ctx.coverage:.2f}）——"
                f"差临门一脚{marginal_hint}。建议动作："
                "①若机制置信度高，直接 eval_on_val_set 验证方向保留性；"
                "②或微调窗口长度/扩展正交信息源。"
            )
            return DiagnosticOpinion(
                source=self.name,
                opinion_type="near_miss",
                title="接近海选线",
                message=msg,
                priority=60,
                extra_fields={"near_miss_hint": msg},
            )
        return None

    def _get_threshold(self, ctx: DiagnosticContext) -> float:
        for row in ctx.screen_rules:
            if isinstance(row, dict) and str(row.get("metric") or "").endswith("cross_sectional_core.ic"):
                try:
                    exp = float(row.get("expected"))
                    if exp > 0:
                        return exp
                except (TypeError, ValueError):
                    pass
        from core.research_modes import RESEARCH_MODES
        bar = float(DEFAULT_RESEARCH_SPEC["delivery_policy"]["candidate"]["min_abs_ic"])
        spec = RESEARCH_MODES.get(str(ctx.research_mode or "technical"))
        if spec is not None:
            bar = float(getattr(spec, "candidate_overrides", {}).get("min_abs_ic", bar))
        return bar


class TurnoverDiagnostic(BaseDiagnostic):
    """换手率超标诊断（硬门与建议红线，增加输入列自相关因果归因）。"""

    name = "turnover"
    _REDLINE: float = 0.40

    def evaluate(self, ctx: DiagnosticContext) -> DiagnosticOpinion | None:
        if not ctx.passed or ctx.split != "train":
            return None

        gate_limit = DeliveryCriteria.defaults().candidate.max_avg_daily_side_turnover
        if ctx.turnover is not None and ctx.turnover > self._REDLINE:
            if ctx.turnover >= gate_limit:
                gate_line = f"必被 stage_one 换手硬门槛拦截（>{gate_limit:.2f}）"
            else:
                gate_line = f"超建议红线（{self._REDLINE:.2f}），且此区间多半止步精筛/engine_gate"

            # P1-1 动态换手来源归因
            attribution_text = ""
            if ctx.session is not None and hasattr(ctx.session, "get_column_autocorr"):
                try:
                    import re
                    cols = re.findall(r"\$([a-zA-Z_][a-zA-Z0-9_]*)", ctx.expr or "")
                    col_corrs = {c: ctx.session.get_column_autocorr(c) for c in set(cols)}
                    if col_corrs:
                        worst_col = min(col_corrs.items(), key=lambda x: x[1])
                        if worst_col[1] < 0.6:
                            attribution_text = f"（诊断分析：换手主要源于 ${worst_col[0]} 日度自相关偏低 {worst_col[1]:.2f} 引起剧烈翻转）"
                except Exception:
                    pass

            msg = (
                f"训练已过海选线，但日单边换手 {ctx.turnover:.2f} 超标——{gate_line}{attribution_text}，"
                "调用 submit_factor 纯浪费算力。请勿提交：推荐降噪路径："
                "①换用慢信息源（如基本面 PIT 数据 $funda_*$、筹码周频变量）替代高频抖动变量；"
                "②外层加 CS_ZSCORE(x) 压制极端尾部与日度排名抖动（不压缩分布）；"
                "③缩短信号窗口或换 TS_QUANTILE 降低日度翻转频率。"
                "注意：末位 EMA/WMA/TS_MEAN 平滑会压缩因子值分布导致十分位塌陷，不建议盲目堆叠平滑。"
            )
            hints = [
                "换用基本面 PIT 或慢速筹码变量（降换手不压缩分布）",
                "CS_ZSCORE 截面秩变换（压制尾部不塌分布）",
                "缩短信号窗口或 TS_QUANTILE 降翻转频率",
            ]
            return DiagnosticOpinion(
                source=self.name,
                opinion_type="block_submit",
                title="换手率超标",
                message=msg,
                hints=hints,
                priority=85,  # 高优先级阻断
                extra_fields={
                    "submit_decision_required": msg,
                    "turnover_reduction_hints": hints,
                    "submit_suppressed": True,  # P0-1 标记让下游 advisory 知晓
                },
            )
        return None


class IcirDiagnostic(BaseDiagnostic):
    """ICIR 预检诊断（防止盲目提交被 stage_one 拦截，含 P1-4 月度波动归因）。"""

    name = "icir_gate"

    def evaluate(self, ctx: DiagnosticContext) -> DiagnosticOpinion | None:
        if not ctx.passed or ctx.split != "train":
            return None

        gate = DeliveryCriteria.defaults().candidate.min_icir
        if ctx.abs_icir is not None and ctx.abs_icir < gate:
            # P1-4 波动来源归因：分析月度 IC 稳健性 / 反转月份
            attribution_text = ""
            monthly = (
                ctx.result.get("monthly_corr_robustness")
                or ctx.metrics.get("monthly_corr_robustness")
                or {}
            )
            if isinstance(monthly, dict):
                try:
                    pos_ratio = monthly.get("positive_ratio")
                    neg_months = monthly.get("negative_months")
                    if pos_ratio is not None and float(pos_ratio) < 0.60:
                        attribution_text = (
                            f"（波动归因：月度正相关比例仅 {float(pos_ratio):.0%}，方向不稳定）"
                        )
                    elif neg_months:
                        attribution_text = f"（波动归因：存在反转月份 {str(neg_months)[:40]} 拖累信噪比）"
                except (TypeError, ValueError):
                    pass

            msg = (
                f"训练已过海选线（promising），但 ICIR={ctx.abs_icir:.3f} < stage_one 门槛 {gate:.2f}{attribution_text}——"
                "调用 submit_factor 大概率被 stage_one 拦截（纯浪费算力）。请勿提交，改为升 ICIR："
                "换窗（如 20→10/40）、去极值压缩日度波动、换用更高信噪比的数据面——IC 方向已对，"
                "先让 IR 过线再交。注意：末位 EMA/TS_MEAN 平滑会压缩分布导致十分位塌陷，不建议用。"
            )
            return DiagnosticOpinion(
                source=self.name,
                opinion_type="block_submit",
                title="ICIR 未达门槛",
                message=msg,
                priority=80,
                extra_fields={"submit_decision_required": msg},
            )
        return None


class AntiHomogenizationDiagnostic(BaseDiagnostic):
    """同质化探索熔断器 (P1-7 Anti-Homogenization Circuit Breaker)。

    检测当前 run 中是否连续 >=3 次在同结构上微调平滑算子且换手率持续未降破 0.50。
    触发熔断硬告警，阻断大模型无脑试 WMA/EMA 窗口的死循环，强迫切换信号源/结构。
    """

    name = "anti_homogenization"

    def evaluate(self, ctx: DiagnosticContext) -> DiagnosticOpinion | None:
        expr = ctx.expr or str(ctx.arguments.get("multi_line_expr") or "")
        if not expr or not ctx.recent_signatures:
            return None

        try:
            from alphaagent.dsl.core.ast import all_smoothing_ops, structure_fingerprint

            current_has_sm = bool(all_smoothing_ops(expr))
            if not current_has_sm:
                return None
            current_fp = structure_fingerprint(expr)

            # 同质化判定：连续 >=3 次均为「同结构指纹 + 含平滑 + 换手 > 0.50」
            consecutive_smoothing_high_turnover = 0
            for item in reversed(ctx.recent_signatures):
                if isinstance(item, dict):
                    t = item.get("turnover")
                    has_sm = item.get("has_smoothing")
                    fp = item.get("fingerprint")
                    if has_sm and (t is not None and t > 0.50) and fp == current_fp:
                        consecutive_smoothing_high_turnover += 1
                    else:
                        break

            if consecutive_smoothing_high_turnover >= 3:
                msg = (
                    f"⚠️ 同质化探索熔断警告：已连续 {consecutive_smoothing_high_turnover} 次在同结构上微调平滑算子降低换手，"
                    "但换手率始终高于 0.50 门槛！实证表明：平滑无法改变信号源本身的高频本质，继续微调窗口纯属浪费算力。"
                    "请立即停止参数微调，改换慢速数据源（如基本面 PIT 或周频筹码）或彻底更换信号根结构。"
                )
                return DiagnosticOpinion(
                    source=self.name,
                    opinion_type="block_submit",
                    title="同质化平滑调参熔断",
                    message=msg,
                    priority=88,  # 高于普通换手诊断，优先提醒熔断
                    extra_fields={"homogenization_breaker_warning": msg},
                )
        except Exception:
            pass
        return None


class DecileCollapseDiagnostic(BaseDiagnostic):
    """十分位塌陷风险诊断（末位平滑算子检测）。"""

    name = "decile_collapse"

    def evaluate(self, ctx: DiagnosticContext) -> DiagnosticOpinion | None:
        expr = ctx.expr or str(ctx.arguments.get("multi_line_expr") or ctx.arguments.get("expr") or "")
        if not expr:
            return None

        try:
            from alphaagent.dsl.core.ast import all_smoothing_ops, terminal_smoothing

            sm_ops = all_smoothing_ops(expr)
            is_terminal = terminal_smoothing(expr)
            if is_terminal:
                hint = (
                    f"末位平滑算子 {is_terminal} 压缩因子值分布，十分位 D1-D3/D8-D10 区分度不足——"
                    "建议去掉末位平滑，改用 CS_ZSCORE/RANK 做截面连续化（不压缩分布）。"
                )
                return DiagnosticOpinion(
                    source=self.name,
                    opinion_type="collapse_risk",
                    title="十分位分布塌陷高风险",
                    message=hint,
                    priority=75,
                    extra_fields={
                        "collapse_risk": "high",
                        "collapse_risk_hint": hint,
                    },
                )
            if len(sm_ops) >= 2:
                hint = (
                    f"表达式含 {len(sm_ops)} 个平滑算子（{', '.join(sm_ops[:3])}），"
                    "分布压缩风险中等——若十分位区分度不足，考虑减少平滑层数。"
                )
                return DiagnosticOpinion(
                    source=self.name,
                    opinion_type="collapse_risk",
                    title="平滑算子堆叠风险",
                    message=hint,
                    priority=40,
                    extra_fields={
                        "collapse_risk": "medium",
                        "collapse_risk_hint": hint,
                    },
                )
        except Exception:
            pass
        return None


class DiagnosticMediator:
    """中央诊断仲裁者：收集各策略意见并执行优先级裁决与冲突消解。"""

    def __init__(self, diagnostics: list[BaseDiagnostic] | None = None) -> None:
        self.diagnostics = diagnostics or [
            PitSuspicionDiagnostic(),
            AntiHomogenizationDiagnostic(),
            TurnoverDiagnostic(),
            IcirDiagnostic(),
            NearMissDiagnostic(),
            DecileCollapseDiagnostic(),
        ]

    def arbitrate(self, ctx: DiagnosticContext) -> dict[str, Any]:
        """执行仲裁，产出最终挂载到 result 的诊断字段字典。"""
        opinions: list[DiagnosticOpinion] = []
        for diag in self.diagnostics:
            try:
                op = diag.evaluate(ctx)
                if op is not None:
                    opinions.append(op)
            except Exception:
                continue

        # 排序：按优先级降序
        opinions.sort(key=lambda x: -x.priority)

        final_fields: dict[str, Any] = {}

        # 冲突裁决 1: 如果预测对账被证伪 (contradicted)，拥有最高否决权 (P0-5)
        is_contradicted = (
            ctx.prediction_check is not None
            and str(ctx.prediction_check.get("verdict")) == "contradicted"
        )

        # 冲突裁决 2: 如果存在 block_submit (如换手率超标或 ICIR 不足)
        has_block = any(op.opinion_type == "block_submit" for op in opinions)

        # 提取各诊断字段
        for op in opinions:
            # 如果被证伪，强制抑制 near_miss 建议微调
            if is_contradicted and op.opinion_type in ("near_miss", "encourage_submit"):
                continue
            for k, v in op.extra_fields.items():
                if k not in final_fields:
                    final_fields[k] = v

        # 冲突裁决 3: 如果被证伪，强制覆盖决策要求，禁止调参
        if is_contradicted:
            final_fields["submit_decision_required"] = (
                "预测形态对账已被实际截面严重证伪 (verdict=contradicted)。"
                "禁止对被证伪的机制结构做微调参数或盲目提交；必须更换经济因果机制或放弃该结构。"
            )
            final_fields["submit_suppressed"] = True
            final_fields.pop("near_miss_hint", None)
        elif ctx.passed and ctx.split == "train" and not has_block:
            # 如果训练通过且没有任何 block 意见，给出标准的承诺决策提示
            if "submit_decision_required" not in final_fields:
                final_fields["submit_decision_required"] = (
                    "训练已过海选线（promising）。两个动作二选一，不得沉默跳过："
                    "①立即调用 submit_factor 走入库门槛（正交检查/审查/精筛会自动裁决）；"
                    "②给出不提交的明确理由（正交顾虑/机制疑点/PIT 疑点），并以此指导下一轮变异方向。"
                )

        # P1-8 统一反馈收口器 (Feedback Synthesizer):
        # 将零碎分散的提示收敛为 3 个标准字段 (diagnostic_verdict, bottleneck, actionable_guidance)
        verdict = "PROCEED_TO_SUBMIT"
        bottleneck = "NONE"
        guidance = "指标健康，建议直接调用 submit_factor 走入库门槛。"

        if is_contradicted:
            verdict = "REJECT_MECHANISM"
            bottleneck = "SHAPE_CONTRADICTED"
            guidance = "形态对账被证伪，严禁微调窗口，必须更换经济假设与信号根。"
        elif has_block:
            verdict = "RESTRUCTURE_REQUIRED"
            if ctx.turnover is not None and ctx.turnover > 0.40:
                bottleneck = "HIGH_TURNOVER"
                guidance = f"日单边换手 {ctx.turnover:.2f} 严重超标，请勿直接提交；优先使用慢信息源或加 CS_ZSCORE 连续化降噪。"
            elif ctx.abs_icir is not None and ctx.abs_icir < 0.28:
                bottleneck = "LOW_ICIR"
                guidance = f"ICIR={ctx.abs_icir:.3f} 稳定性不足，建议去极值或换窗降低日度收益波动后再交。"
        elif not ctx.passed:
            if "near_miss_hint" in final_fields:
                verdict = "NEAR_MISS_EXPLORE"
                bottleneck = "IC_BORDERLINE"
                guidance = "接近海选线，可直接推 val 验证或拓展正交信息源。"
            else:
                verdict = "SEARCH_ANOTHER_DIRECTION"
                bottleneck = "SIGNAL_INSUFFICIENT"
                guidance = "信号较弱，建议切换至其他未尝试的信号族或新数据面。"

        final_fields["diagnostic_verdict"] = verdict
        final_fields["bottleneck"] = bottleneck
        final_fields["actionable_guidance"] = guidance

        return final_fields


def apply_diagnostics_to_result(
    result: dict[str, Any],
    expr: str,
    arguments: dict[str, Any],
    session: Any | None = None,
    recent_evals: list[dict[str, Any]] | None = None,
) -> None:
    """门面入口：构建上下文，执行仲裁，回写进 result 字典。"""
    try:
        ctx = DiagnosticContext.from_eval_result(result, expr, arguments)
        ctx.session = session
        if recent_evals:
            ctx.recent_signatures = recent_evals
        mediator = DiagnosticMediator()
        fields = mediator.arbitrate(ctx)
        result.update(fields)
    except Exception:
        pass
