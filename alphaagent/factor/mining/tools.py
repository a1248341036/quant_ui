"""LLM 评估与交付工具：train/val 评估 + submit_factor 入库。"""



from __future__ import annotations



import json

from pathlib import Path

from typing import Any



from alphaagent.factor.mining.schemas import EvalProfileRequest, EvalTrainRequest, EvalValRequest

from alphaagent.factor.mining.service import StockEvalService

from alphaagent.factor.mining.submit import FactorSubmitService
from alphaagent.factor.mining.interactions import INTERACTION_TYPES


_INTERACTION_PARAMETER: dict[str, Any] = {
    "type": "object",
    "description": "多因子交互契约；结构化交互算子和 MULTIPLY 必须提供。",
    "properties": {
        "interaction_type": {
            "type": "string",
            "enum": sorted(INTERACTION_TYPES),
        },
        "base_signal": {"type": "string"},
        "condition_signal": {"type": "string"},
        "economic_mechanism": {
            "type": "string",
            "description": ">=20字的因果机制，而非指标复述。",
        },
        "expected_subgroup_pattern": {},
        "ablation_required": {"type": "boolean", "default": True},
    },
    "required": ["interaction_type", "base_signal", "condition_signal", "economic_mechanism"],
}



_EVAL_PARAMETERS: dict[str, Any] = {

    "type": "object",

    "properties": {

        "multi_line_expr": {

            "type": "string",

            "description": "多行因子 DSL：可含赋值行，最后一行为因子值；列用 $列名 引用，算子大写。",

        },

        "factor_name": {"type": "string", "description": "因子列逻辑名，默认 expr。"},

        "include_detail_tables": {

            "type": "boolean",

            "description": "true 时额外返回 by_month / by_symbol 明细；默认 false 仅返回 summary。",

            "default": False,

        },

        "label_quantile_n": {

            "type": "integer",

            "description": "按因子值等频分位分桶，输出每桶 label 均值；0 表示不计算。默认 10。",

            "default": 10,

        },


        "interaction": _INTERACTION_PARAMETER,

    },

    "required": ["multi_line_expr"],

    "additionalProperties": False,

}



_VAL_PARAMETERS: dict[str, Any] = {

    "type": "object",

    "properties": {

        **_EVAL_PARAMETERS["properties"],

        "expected_sign": {

            "type": "integer",

            "description": "train summary.ic 的符号（1=正、-1=负）；传入后返回 sign_check。",

            "enum": [1, -1],

        },


        "interaction": _INTERACTION_PARAMETER,

    },

    "required": ["multi_line_expr"],

    "additionalProperties": False,

}

_PROFILE_EVAL_PARAMETERS: dict[str, Any] = {
    "type": "object",
    "properties": {
        "multi_line_expr": _EVAL_PARAMETERS["properties"]["multi_line_expr"],
        "factor_name": _EVAL_PARAMETERS["properties"]["factor_name"],
        "profile_id": {
            "type": "string",
            "description": "冻结的 EvaluationProfile ID；决定 split、transform、metrics 与 rule gate。",
        },
        "interaction": _INTERACTION_PARAMETER,
    },
    "required": ["multi_line_expr", "profile_id"],
    "additionalProperties": False,
}



_SUBMIT_PARAMETERS: dict[str, Any] = {

    "type": "object",

    "properties": {

        "multi_line_expr": {

            "type": "string",

            "description": "与 train/val 评估一致的多行因子 DSL。",

        },

        "factor_name": {

            "type": "string",

            "description": "因子唯一逻辑名（蛇形英文），将写入因子库 factor_id。",

        },

        "comment": {

            "type": "string",

            "description": "因子含义说明：经济直觉、关键算子与窗口、预期 IC 方向等，供后续查阅。",

        },

        "interaction": _INTERACTION_PARAMETER,

        "rebalance_freq": {

            "type": "string",

            "enum": ["daily", "weekly", "monthly"],

            "description": "交付调仓频率：对比 evaluate 结果中 topn_portfolio.by_freq 三种频率的收益/换手/重合率后选择；缺省 daily。",

        },

    },

    "required": ["multi_line_expr", "factor_name", "comment"],

    "additionalProperties": False,

}



TOOL_NAMES = ("evaluate_factor", "eval_on_train_set", "eval_on_val_set", "submit_factor")





class FactorEvalTools:

    """持有一个已建会话，向 LLM 暴露 eval_on_train_set / eval_on_val_set / submit_factor。"""



    def __init__(

        self,

        service: StockEvalService,

        session_id: str,

        *,

        submit_service: FactorSubmitService | None = None,

    ) -> None:

        self.service = service

        self.session_id = session_id

        self.submit_service = submit_service



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

                            "【两阶段交付】候选池：train 窗口 |IC|>=0.015、ICIR>0.25、coverage>0.85、"
                            "cs_autocorr>=0.18、val 保留比>=0.5、max_cs_corr<0.6。"
                            "正式库：train |IC|>=0.025 且 ICIR>0.30、val |IC|>=0.015 且保留比>=0.6、"
                            "截尾 IC 衰减<=10%、max_cs_corr<0.5，最后 engine_gate 净值回测裁决"
                            "（weekly 全约束：净超额年化>=3%、超额夏普>=0.5、回撤<=40%、重合率>=0.5）。"

                            "仅正式库成功时 stored=true；候选池成功时 candidate_stored=true。"

                            "查重失败时返回 top_neighbors 含相似因子 expr。"

                        ),

                        "parameters": _SUBMIT_PARAMETERS,

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
            return self.service.eval_profile(
                EvalProfileRequest(
                    session_id=self.session_id,
                    profile_id=profile_id,
                    multi_line_expr=expr,
                    factor_name=str(arguments.get("factor_name") or "expr"),
                )
            )



        expr = arguments.get("multi_line_expr")

        if not isinstance(expr, str) or not expr.strip():

            return {"ok": False, "error": "multi_line_expr_required_non_empty_string", "error_type": "ToolArgumentsError"}



        factor_name = arguments.get("factor_name") or "expr"

        include_detail = bool(arguments.get("include_detail_tables", False))

        label_quantile_n = arguments.get("label_quantile_n", 10)

        if label_quantile_n is None:

            label_quantile_n = 10



        if name == "eval_on_train_set":

            return self.service.eval_train(

                EvalTrainRequest(

                    session_id=self.session_id,

                    multi_line_expr=expr,

                    factor_name=factor_name,

                    include_detail_tables=include_detail,

                    label_quantile_n=int(label_quantile_n),

                )

            )

        if name == "eval_on_val_set":

            expected_sign = arguments.get("expected_sign")

            if expected_sign not in (None, 1, -1):

                return {"ok": False, "error": "expected_sign_must_be_1_or_-1", "error_type": "ToolArgumentsError"}

            return self.service.eval_val(

                EvalValRequest(

                    session_id=self.session_id,

                    multi_line_expr=expr,

                    factor_name=factor_name,

                    include_detail_tables=include_detail,

                    label_quantile_n=int(label_quantile_n),

                    expected_sign=expected_sign,

                )

            )

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

        return self.submit_service.submit(

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



    # ── 批次内历史（类变量，跨工具调用累积，进程级） ──
    _batch_history: list[dict[str, Any]] = []

    @classmethod
    def _record_batch(cls, factor_name: str, expr: str, ic: float | None, icir: float | None, passed: bool) -> None:
        cls._batch_history.append({
            "factor_name": factor_name,
            "expr_head": expr.strip().split("\n")[0][:80] if expr else "",
            "ic": ic, "icir": icir, "passed": passed,
        })
        if len(cls._batch_history) > 30:
            cls._batch_history = cls._batch_history[-30:]

    @classmethod
    def _batch_summary(cls) -> str:
        if not cls._batch_history:
            return ""
        lines = ["\n--- 本轮批次内已评估因子汇总 ---"]
        for i, h in enumerate(cls._batch_history[-8:], 1):
            status = "PASS" if h["passed"] else "fail"
            ic_str = f"{h['ic']:+.4f}" if h["ic"] is not None else "N/A"
            icir_str = f"{h['icir']:+.3f}" if h["icir"] is not None else "N/A"
            lines.append(f"  {i}. [{status}] {h['factor_name']}: IC={ic_str} ICIR={icir_str} | {h['expr_head']}")
        # 同质化检测
        expr_heads = [h["expr_head"] for h in cls._batch_history[-8:]]
        if len(expr_heads) >= 3:
            from collections import Counter
            common = Counter(expr_heads).most_common(1)[0]
            if common[1] >= 3:
                lines.append(f"  ⚠ 同质化警告: '{common[0]}' 已出现 {common[1]} 次，请换一个完全不同的信号维度！")
        ics = [h["ic"] for h in cls._batch_history[-8:] if h["ic"] is not None]
        if len(ics) >= 3:
            avg_ic = sum(ics) / len(ics)
            lines.append(f"  📊 近 {len(ics)} 个因子平均 IC={avg_ic:+.4f}")
        return "\n".join(lines)

    @staticmethod
    def _extract_metrics(result: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
        """统一提取 cross_sectional / monthly / mls / long_short。

        兼容三种返回结构：
        1. evaluate_factor: result['metrics']['cross_sectional_core']  （嵌套）
        2. eval_on_train_set: result['summary']  （扁平）
        3. 旧格式: result['metrics'] 本身就是扁平
        """
        metrics = result.get("metrics", {})
        if not isinstance(metrics, dict):
            metrics = {}
        summary = result.get("summary", {})
        if not isinstance(summary, dict):
            summary = {}

        # cross_sectional_core
        cs = metrics.get("cross_sectional_core", {}) if isinstance(metrics.get("cross_sectional_core"), dict) else {}
        if not cs:
            # 尝试 summary（eval_on_train_set 格式）
            if "ic" in summary:
                cs = summary
            elif "ic" in metrics:
                cs = metrics

        # monthly_robustness
        monthly = metrics.get("monthly_robustness", {}) if isinstance(metrics.get("monthly_robustness"), dict) else {}
        if not monthly:
            if "share_months_ic_positive" in summary:
                monthly = summary
            elif "share_months_ic_positive" in metrics:
                monthly = metrics

        # mls_fmb
        mls = metrics.get("mls_fmb", {}) if isinstance(metrics.get("mls_fmb"), dict) else {}
        if not mls:
            if "nw_t_ls" in summary:
                mls = summary
            elif "nw_t_ls" in metrics:
                mls = metrics

        # long_short_portfolio
        ls_port = metrics.get("long_short_portfolio", {}) if isinstance(metrics.get("long_short_portfolio"), dict) else {}
        if not ls_port:
            if "net_ir_annual" in summary:
                ls_port = summary
            elif "net_ir_annual" in metrics:
                ls_port = metrics

        return cs, monthly, mls, ls_port

    @staticmethod
    def _diagnose(result: dict[str, Any]) -> str:
        """从评估结果生成结构化诊断建议。"""
        if not result.get("ok"):
            err = str(result.get("error") or result.get("error_type") or "unknown")
            return f"❌ 评估失败: {err}。请检查 DSL 语法和列名。"

        cs, monthly, mls, _ls_port = FactorEvalTools._extract_metrics(result)

        ic = cs.get("ic")
        icir = cs.get("icir")
        coverage = cs.get("factor_coverage")
        share_pos = monthly.get("share_months_ic_positive")
        nw_t_ls = mls.get("nw_t_ls")

        rule_results = result.get("rule_results", [])
        passed = result.get("passed")
        if passed is None:
            passed = bool(ic is not None and abs(ic) >= 0.02 and icir is not None and icir > 0.25 and coverage is not None and coverage > 0.85)

        tips: list[str] = []

        if passed:
            tips.append("✅ 通过 train_screen 门槛，建议用 validation profile 验证泛化。")
        else:
            # 分析失败原因
            failed_rules = [r for r in rule_results if not r.get("passed")] if rule_results else []
            if failed_rules:
                for r in failed_rules:
                    metric = r.get("metric", "")
                    actual = r.get("actual")
                    expected = r.get("expected")
                    op = r.get("op", "")
                    tips.append(f"❌ {metric}={actual:.4f} 未达 {op} {expected}")
            else:
                if ic is not None and abs(ic) < 0.02:
                    tips.append(f"IC={ic:+.4f} 偏低（需 |IC|≥0.02），信号太弱或方向有误。")
                if icir is not None and icir < 0.25:
                    tips.append(f"ICIR={icir:+.3f} 偏低（需≥0.25），IC 日间波动太大，考虑平滑(TS_MEAN/EMA)或换窗口。")
                if coverage is not None and coverage < 0.85:
                    tips.append(f"Coverage={coverage:.3f} 偏低，因子缺失太多，检查数据条件或放宽过滤。")

        # 月度稳健性诊断
        if share_pos is not None:
            if ic is not None and ic > 0 and share_pos < 0.7:
                tips.append(f"月度IC+占比={share_pos:.1%} 偏低（正IC需>70%），信号不稳定。")
            elif ic is not None and ic < 0 and share_pos > 0.3:
                tips.append(f"月度IC+占比={share_pos:.1%} 偏高（负IC需<30%），信号不稳定。")

        # 方向建议
        if ic is not None and abs(ic) < 0.01:
            tips.append("💡 IC 接近 0，当前信号无预测力。请换一个完全不同的经济假设或信息维度。")
        elif ic is not None and abs(ic) < 0.015:
            tips.append("💡 IC 微弱，尝试：① 检查经济机制是否成立 ② 换信息维度 ③ 用门控/组内排名/残差化等可解释交互")

        return " ".join(tips) if tips else ""

    @staticmethod
    def result_to_content(result: dict[str, Any]) -> str:
        """精简评估结果为结构化文本，附带诊断建议和批次汇总。"""
        # Submit has a different payload shape; preserving its delivery evidence
        # is essential for the next LLM turn.
        if "delivery_check" in result or "candidate_stored" in result:
            metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
            similarity = result.get("similarity") if isinstance(result.get("similarity"), dict) else {}
            delivery = result.get("delivery_check") if isinstance(result.get("delivery_check"), dict) else {}
            stage1 = delivery.get("stage_one") if isinstance(delivery.get("stage_one"), dict) else {}
            stage2 = delivery.get("stage_two") if isinstance(delivery.get("stage_two"), dict) else {}
            lines = [
                f"提交结果: {result.get('factor_name', 'expr')}",
                f"candidate_stored={bool(result.get('candidate_stored'))} stored={bool(result.get('stored'))}",
                (
                    f"IC={metrics.get('ic')} ICIR={metrics.get('icir')} "
                    f"rankIC={metrics.get('rank_ic')} coverage={metrics.get('coverage')} "
                    f"max_cs_corr={similarity.get('max_abs_corr')}"
                ),
                f"stage1: passed={stage1.get('passed')} fail_reasons={stage1.get('fail_reasons', [])}",
                f"stage2: passed={stage2.get('passed')} fail_reasons={stage2.get('fail_reasons', [])}",
            ]
            if result.get("skipped_reason"):
                lines.append(f"skipped_reason={result['skipped_reason']}")
            review = result.get("factor_review")
            if isinstance(review, dict) and review:
                lines.append(f"Reviewer: verdict={review.get('verdict')} novelty={review.get('novelty')}")
            return "\n".join(lines)

        if not result.get("ok"):
            err = str(result.get("error") or result.get("error_type") or "unknown")
            return f"❌ 评估失败: {err}"

        cs, monthly, mls, ls_port = FactorEvalTools._extract_metrics(result)

        ic = cs.get("ic")
        icir = cs.get("icir")
        rank_ic = cs.get("rank_ic")
        coverage = cs.get("factor_coverage")
        share_pos = monthly.get("share_months_ic_positive")
        nw_t_ls = mls.get("nw_t_ls")
        net_ir_annual = ls_port.get("net_ir_annual")

        # factor_name 和 expression 兼容两种结构
        candidate = result.get("candidate", {})
        if isinstance(candidate, dict):
            factor_name = candidate.get("factor_name") or result.get("factor_name") or "expr"
            expr = candidate.get("expression", "") or result.get("multi_line_expr", "")
        else:
            factor_name = result.get("factor_name", "expr")
            expr = ""
        split = result.get("split", "")
        passed = result.get("passed")
        if passed is None:
            passed = bool(ic is not None and abs(ic) >= 0.02 and icir is not None and icir > 0.25 and coverage is not None and coverage > 0.85)

        # 记录到批次历史
        FactorEvalTools._record_batch(factor_name, expr, ic, icir, bool(passed))

        # 构建精简文本
        lines = [
            f"因子: {factor_name} | split={split} | passed={passed}",
            f"IC={ic:+.4f} ICIR={icir:+.3f} rankIC={rank_ic:+.4f} cov={coverage:.3f}" if all(v is not None for v in [ic, icir, rank_ic, coverage]) else f"IC={ic} ICIR={icir} rankIC={rank_ic} cov={coverage}",
        ]
        if share_pos is not None:
            lines.append(f"月IC+占比={share_pos:.1%}")
        if nw_t_ls is not None:
            lines.append(f"NW-t(LS)={nw_t_ls:.2f}")
        if net_ir_annual is not None:
            lines.append(f"多空年化IR={net_ir_annual:.2f}")

        # rule_results 精简
        rule_results = result.get("rule_results", [])
        if rule_results:
            rule_strs = []
            for r in rule_results:
                status = "✓" if r.get("passed") else "✗"
                metric = r.get("metric", "").replace("cross_sectional_core.", "")
                actual = r.get("actual")
                expected = r.get("expected")
                rule_strs.append(f"{status} {metric}={actual:.4f}({expected})" if actual is not None else f"{status} {metric}({expected})")
            lines.append(" | ".join(rule_strs))

        # 诊断
        diag = FactorEvalTools._diagnose(result)
        if diag:
            lines.append(f"诊断: {diag}")

        # 预审警告
        preflight_warn = result.get("preflight_warning")
        if preflight_warn:
            lines.append(f"⚠ 预审: {preflight_warn}")

        # 批次汇总
        batch = FactorEvalTools._batch_summary()
        if batch:
            lines.append(batch)

        # factor_review（如有）
        review = result.get("factor_review")
        if isinstance(review, dict) and review:
            verdict = review.get("verdict", "")
            novelty = review.get("novelty", "")
            reasons = review.get("reasons", [])
            required = review.get("required_changes", [])
            lines.append(f"Reviewer: verdict={verdict} novelty={novelty}")
            if reasons:
                lines.append(f"  理由: {'; '.join(reasons[:2])}")
            if required:
                lines.append(f"  要求: {'; '.join(required[:2])}")

        # submit 相关（如有）
        if "stored" in result:
            lines.append(f"入库: stored={result.get('stored')} candidate_stored={result.get('candidate_stored')}")
            delivery = result.get("delivery_check", {})
            if isinstance(delivery, dict):
                s1 = delivery.get("stage_one", {})
                s2 = delivery.get("stage_two", {})
                if isinstance(s1, dict):
                    lines.append(f"  stage1: {s1.get('passed')} {s1.get('fail_reasons', '')}")
                if isinstance(s2, dict):
                    lines.append(f"  stage2: {s2.get('passed')} {s2.get('fail_reasons', '')}")

        return "\n".join(lines)



