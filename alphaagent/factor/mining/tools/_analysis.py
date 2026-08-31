"""FactorEvalTools 静态分析：批次历史、指标提取、诊断、result_to_content。"""
from __future__ import annotations

from collections import Counter
from typing import Any


class _AnalysisMixin:
    """批次历史记录、指标提取、诊断建议、result_to_content。"""

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
    def _rule_thresholds(result: dict[str, Any]) -> tuple[float | None, float | None, float | None]:
        """从评估 profile 规则提取 IC / ICIR / Coverage 门槛（与真实门禁同源）。

        规则可能在 rule_results（已求值）或 profile.rules（未求值）中；
        找不到时返回 None，调用方回落默认值。避免提示词硬编码与门禁脱节。
        """
        rules = result.get("rule_results")
        if not rules:
            profile = result.get("profile") if isinstance(result.get("profile"), dict) else {}
            rules = profile.get("rules") or []
        ic_thr = icir_thr = cov_thr = None
        for rule in rules:
            if not isinstance(rule, dict):
                continue
            metric = str(rule.get("metric") or "")
            expected = rule.get("expected")
            if expected is None:
                continue
            try:
                value = float(expected)
            except (TypeError, ValueError):
                continue
            if metric.endswith("cross_sectional_core.ic"):
                ic_thr = value
            elif metric.endswith("cross_sectional_core.icir"):
                icir_thr = value
            elif metric.endswith("cross_sectional_core.factor_coverage"):
                cov_thr = value
        return ic_thr, icir_thr, cov_thr

    @staticmethod
    def _diagnose(result: dict[str, Any]) -> str:
        """从评估结果生成结构化诊断建议。"""
        from ._analysis import _AnalysisMixin

        if not result.get("ok"):
            err = str(result.get("error") or result.get("error_type") or "unknown")
            return f"❌ 评估失败: {err}。请检查 DSL 语法和列名。"

        cs, monthly, mls, _ls_port = _AnalysisMixin._extract_metrics(result)

        ic = cs.get("ic")
        icir = cs.get("icir")
        coverage = cs.get("factor_coverage")
        share_pos = monthly.get("share_months_ic_positive")
        nw_t_ls = mls.get("nw_t_ls")

        rule_results = result.get("rule_results", [])
        passed = result.get("passed")
        if passed is None:
            ic_thr, icir_thr, cov_thr = _AnalysisMixin._rule_thresholds(result)
            passed = bool(
                ic is not None and abs(ic) >= (ic_thr or 0.02)
                and icir is not None and icir > (icir_thr or 0.25)
                and coverage is not None and coverage > (cov_thr or 0.85)
            )

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
                ic_thr, icir_thr, cov_thr = _AnalysisMixin._rule_thresholds(result)
                if ic is not None and abs(ic) < (ic_thr or 0.02):
                    tips.append(f"IC={ic:+.4f} 偏低（需 |IC|≥{ic_thr or 0.02}），信号太弱或方向有误。")
                if icir is not None and icir < (icir_thr or 0.25):
                    tips.append(f"ICIR={icir:+.3f} 偏低（需≥{icir_thr or 0.25}），IC 日间波动太大，考虑平滑(TS_MEAN/EMA)或换窗口。")
                if coverage is not None and coverage < (cov_thr or 0.85):
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
        from ._analysis import _AnalysisMixin

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

        cs, monthly, mls, ls_port = _AnalysisMixin._extract_metrics(result)

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
            ic_thr, icir_thr, cov_thr = _AnalysisMixin._rule_thresholds(result)
            passed = bool(
                ic is not None and abs(ic) >= (ic_thr or 0.02)
                and icir is not None and icir > (icir_thr or 0.25)
                and coverage is not None and coverage > (cov_thr or 0.85)
            )

        # 记录到批次历史
        _AnalysisMixin._record_batch(factor_name, expr, ic, icir, bool(passed))

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
        diag = _AnalysisMixin._diagnose(result)
        if diag:
            lines.append(f"诊断: {diag}")

        # 预审警告
        preflight_warn = result.get("preflight_warning")
        if preflight_warn:
            lines.append(f"⚠ 预审: {preflight_warn}")

        # 批次汇总
        batch = _AnalysisMixin._batch_summary()
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
