"""将 FactorEvalTools 包装为 AgentScope FunctionTool。"""

from __future__ import annotations

import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from agentscope.message import TextBlock
from agentscope.tool import FunctionTool, Toolkit, ToolChunk

from alphaagent.factor.mining.tools import FactorEvalTools

_EXECUTOR: ThreadPoolExecutor | None = None


def _executor(max_workers: int) -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=max(1, max_workers))
    return _EXECUTOR


# ── 因子逻辑预审 ──────────────────────────────────────────────

_CONFLICTING_OPERATORS = {
    # 算子名 → 对应的信号维度
    "CHIP_PEAK_LOC": "chip", "CHIP_ENTROPY": "chip", "CHIP_COM_W_GAP": "chip",
    "CHIP_MASS_ASYM": "chip", "CHIP_DENSITY": "chip",
    "TS_PCTCHANGE": "pctchange", "TS_DELTA": "pctchange",
    "TS_STD": "volatility", "TS_VAR": "volatility",
    "TS_RANK": "rank",
}


def _parse_signal_dims(expr: str) -> set[str]:
    """粗略提取表达式涉及的信号维度。"""
    dims: set[str] = set()
    upper = expr.upper()
    # 检测各种信号族
    if "TS_PCTCHANGE" in upper or "TS_DELTA" in upper:
        dims.add("reversal_or_momentum")
    if "TS_STD" in upper or "TS_VAR" in upper:
        dims.add("volatility")
    if "TS_CORR" in upper or "CORR" in upper:
        dims.add("correlation")
    if "CHIP_" in upper:
        dims.add("chip")
    if "VWAP" in upper or "adj_vwap" in expr.lower():
        dims.add("vwap")
    if "ADJ_OPEN" in upper and ("DELAY" in upper or "PREV" in upper or "SUBTRACT" in upper):
        dims.add("overnight_gap")
    if "TS_MEAN" in upper and ("RET" in upper or "PCTCHANGE" in upper):
        dims.add("momentum")
    if "VOLUME" in upper or "AMOUNT" in upper:
        dims.add("volume")
    if "FLOAT_CAP" in upper or "TOT_CAP" in upper or "LOG($FLOAT" in upper:
        dims.add("size")
    return dims


def _preflight_check(multi_line_expr: str, factor_name: str) -> dict[str, Any] | None:
    """因子逻辑预审：在跑 DSL 之前快速检测常见逻辑错误。

    返回 None 表示通过，返回 dict 表示拦截（含 warning 和 suggestion）。
    """
    expr = multi_line_expr.strip()
    if not expr:
        return None

    upper = expr.upper()

    # 获取最后一行（因子值）
    lines = [l.strip() for l in expr.split("\n") if l.strip() and not l.strip().startswith("#")]
    if not lines:
        return None
    final_line = lines[-1].upper()

    # ── 检查1: ADD 中混合负相关信号 ──
    # 找所有 ADD(...) 调用，检查里面是否同时含 chip + intraday_return 之类的冲突组合
    add_pattern = r'ADD\s*\('
    add_matches = re.findall(add_pattern, upper)
    if add_matches:
        # 检测是否有 CHIP + intraday_return 在同一个 ADD 中
        has_chip = "CHIP_" in upper
        has_intraday = "SUBTRACT($ADJ_CLOSE, $ADJ_OPEN" in upper or "DIVIDE(SUBTRACT($ADJ_CLOSE, $ADJ_OPEN" in upper
        has_reversal = "NEG(TS_PCTCHANGE" in upper or "TS_PCTCHANGE($ADJ_CLOSE" in upper

        if has_chip and has_intraday:
            return {
                "blocked": False,
                "warning": "逻辑冲突: ADD(筹码信号, 日内收益) — 二者天然负相关(~-0.3~-0.5)，ADD 会抵消信号。",
                "suggestion": "改用 MULTIPLY 做条件过滤，或用残差化(CS_RESIDUALIZE)剥离重叠部分。筹码信号本身已是强因子，考虑直接单独使用。",
            }
        if has_chip and has_reversal:
            return {
                "blocked": False,
                "warning": "逻辑冲突: ADD(筹码信号, 反转信号) — 筹码形态和价格反转都反映'价格位置'，ADD 会导致信号重复计算。",
                "suggestion": "改用 MULTIPLY 做条件过滤(筹码确认反转)，或只保留筹码信号。",
            }
        # 检测 volatility + reversal 在 ADD 中
        has_vol = "TS_STD($RET" in upper or "TS_VAR(" in upper
        if has_vol and has_reversal and "MULTIPLY" not in final_line:
            return {
                "blocked": False,
                "warning": "逻辑冲突: ADD(波动率, 反转) — 波动率和反转都源自价格变动，可能存在共线性。",
                "suggestion": "考虑用 MULTIPLY 做波动率调整反转，或用残差化剥离。",
            }

    # ── 检查2: 纯 $float_cap 无变换 ──
    if re.search(r'(?:^|\(|,\s*)\$FLOAT_CAP\s*(?:\)|,)', upper) and "LOG" not in upper:
        return {
            "blocked": True,
            "warning": "纯市值因子: 直接使用 $float_cap 无 LOG 变换，截面分布极端右偏。",
            "suggestion": "至少包一层 LOG($float_cap)，或使用 CS_BUCKET(LOG($float_cap), 10) 做分组中性化。",
        }

    # ── 检查3: 两个相同信号简单相加（同质化） ──
    # 如果最终行是 ADD(X, Y)，且 X 和 Y 来自同一个信号族
    if final_line.startswith("ADD(") or "ADD(CS_ZSCORE" in final_line:
        dims = _parse_signal_dims(expr)
        # 检测是否有两个 CS_ZSCORE 相加且变量来源相同
        cszscore_count = upper.count("CS_ZSCORE")
        if cszscore_count >= 2:
            # 获取所有 $ 变量引用
            vars_in_expr = set(re.findall(r'\$[a-z_]+', expr.lower()))
            if len(vars_in_expr) <= 2:
                return {
                    "blocked": False,
                    "warning": f"两个 CS_ZSCORE 相加但只用了 {len(vars_in_expr)} 个变量({vars_in_expr})，信号维度可能重叠。",
                    "suggestion": "确保相加的信号来自不同的经济维度（如量+价、短期+长期），否则用 MULTIPLY 做交互。",
                }

    return None


def _dispatch_sync(tools: FactorEvalTools, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    result = tools.dispatch(name, arguments)
    elapsed = round(time.perf_counter() - t0, 4)
    return result if isinstance(result, dict) else {"ok": False, "error": str(result)}, elapsed


def build_factor_eval_toolkit(
    tools: FactorEvalTools,
    *,
    max_workers: int = 4,
    reviewer: Any | None = None,
) -> Toolkit:
    """构建与 OpenAI 版一致的 eval / submit 工具集。"""

    async def eval_on_train_set(
        multi_line_expr: str,
        factor_name: str = "expr",
        include_detail_tables: bool = False,
        label_quantile_n: int = 10,
    ) -> ToolChunk:
        """训练集评估多行因子表达式，返回 summary、monthly_corr_robustness、label_quantile_buckets。"""
        loop = __import__("asyncio").get_running_loop()
        result, _elapsed = await loop.run_in_executor(
            _executor(max_workers),
            _dispatch_sync,
            tools,
            "eval_on_train_set",
            {
                "multi_line_expr": multi_line_expr,
                "factor_name": factor_name,
                "include_detail_tables": include_detail_tables,
                "label_quantile_n": label_quantile_n,
            },
        )
        if reviewer is not None:
            reviewer.record_evaluation(
                "train",
                {"multi_line_expr": multi_line_expr, "factor_name": factor_name},
                result,
            )
        result.setdefault("factor_name", factor_name)
        return ToolChunk(content=[TextBlock(text=tools.result_to_content(result))])

    async def evaluate_factor(
        multi_line_expr: str,
        profile_id: str,
        factor_name: str = "expr",
    ) -> ToolChunk:
        """按已冻结 EvaluationProfile 执行 DSL 评估；profile 控制 split、transform、指标与规则。"""
        # ── 因子逻辑预审 ──
        preflight = _preflight_check(multi_line_expr, factor_name)
        if preflight is not None:
            warning = preflight["warning"]
            suggestion = preflight["suggestion"]
            blocked = preflight.get("blocked", False)
            prefix = "⛔ 预审拦截" if blocked else "⚠ 预审警告"
            content = (
                f"{prefix}: {warning}\n"
                f"建议: {suggestion}\n"
                f"(表达式: {multi_line_expr[:120]}...)\n"
            )
            if blocked:
                content += "请修改后重新调用 evaluate_factor。"
                return ToolChunk(content=[TextBlock(text=content)])
            content += "仍可继续评估，但强烈建议先修改表达式。"
            # 不拦截，但把警告附加到结果后面
        loop = __import__("asyncio").get_running_loop()
        args = {"multi_line_expr": multi_line_expr, "profile_id": profile_id, "factor_name": factor_name}
        result, _elapsed = await loop.run_in_executor(
            _executor(max_workers), _dispatch_sync, tools, "evaluate_factor", args
        )
        if preflight is not None and not preflight.get("blocked", False):
            result["preflight_warning"] = preflight["warning"]
        if reviewer is not None and result.get("ok", True):
            reviewer_result = _profile_result_for_reviewer(result)
            split = result.get("split")
            if split == "train":
                reviewer.record_evaluation("train", args, reviewer_result)
            elif split == "val":
                reviewer.record_evaluation("val", args, reviewer_result)
            if split == "val" and "validation" in reviewer.review_on:
                result["factor_review"] = await reviewer.review(
                    {"multi_line_expr": multi_line_expr, "factor_name": factor_name, "comment": ""},
                    turn=getattr(reviewer, "current_turn", 0),
                )
                candidate_id = result.get("candidate", {}).get("candidate_id")
                if candidate_id:
                    state = tools.service.record_candidate_review(tools.session_id, candidate_id, result["factor_review"])
                    if state is not None:
                        result["candidate_state"] = state["state"]
        result.setdefault("factor_name", factor_name)
        return ToolChunk(content=[TextBlock(text=tools.result_to_content(result))])

    async def eval_on_val_set(
        multi_line_expr: str,
        factor_name: str = "expr",
        include_detail_tables: bool = False,
        label_quantile_n: int = 10,
        expected_sign: int | None = None,
    ) -> ToolChunk:
        """验证集评估；须传 expected_sign（train IC 符号 1/-1），结果含 sign_check。"""
        loop = __import__("asyncio").get_running_loop()
        args: dict[str, Any] = {
            "multi_line_expr": multi_line_expr,
            "factor_name": factor_name,
            "include_detail_tables": include_detail_tables,
            "label_quantile_n": label_quantile_n,
        }
        if expected_sign is not None:
            args["expected_sign"] = expected_sign
        result, _elapsed = await loop.run_in_executor(
            _executor(max_workers),
            _dispatch_sync,
            tools,
            "eval_on_val_set",
            args,
        )
        if reviewer is not None:
            reviewer.record_evaluation("val", args, result)
            if result.get("ok", True) and "validation" in reviewer.review_on:
                result["factor_review"] = await reviewer.review(
                    {
                        "multi_line_expr": multi_line_expr,
                        "factor_name": factor_name,
                        "comment": "",
                    },
                    turn=getattr(reviewer, "current_turn", 0),
                )
        result.setdefault("factor_name", factor_name)
        return ToolChunk(content=[TextBlock(text=tools.result_to_content(result))])

    func_tools: list[FunctionTool] = [
        FunctionTool(evaluate_factor, name="evaluate_factor", is_read_only=True),
        FunctionTool(eval_on_train_set, name="eval_on_train_set", is_read_only=True),
        FunctionTool(eval_on_val_set, name="eval_on_val_set", is_read_only=True),
    ]

    if tools.submit_service is not None:

        async def submit_factor(
            multi_line_expr: str,
            factor_name: str,
            comment: str,
        ) -> ToolChunk:
            """【正式交付】先经独立原创性审核，再将保留级候选入库 factorzoo。"""
            review: dict[str, Any] | None = None
            if reviewer is not None and "pre_submit" in reviewer.review_on:
                review = await reviewer.review(
                    {"multi_line_expr": multi_line_expr, "factor_name": factor_name, "comment": comment},
                    turn=getattr(reviewer, "current_turn", 0),
                )
                if review.get("verdict") != "approve":
                    result = {
                        "ok": False,
                        "stored": False,
                        "candidate_stored": False,
                        "error": "factor_review_rejected",
                        "error_type": "FactorReviewRejected",
                        "factor_review": review,
                    }
                    return ToolChunk(content=[TextBlock(text=tools.result_to_content(result))])
            loop = __import__("asyncio").get_running_loop()
            result, _elapsed = await loop.run_in_executor(
                _executor(max_workers),
                _dispatch_sync,
                tools,
                "submit_factor",
                {
                    "multi_line_expr": multi_line_expr,
                    "factor_name": factor_name,
                    "comment": comment,
                },
            )
            if review is not None:
                result["factor_review"] = review
            result.setdefault("factor_name", factor_name)
            return ToolChunk(content=[TextBlock(text=tools.result_to_content(result))])

        func_tools.append(FunctionTool(submit_factor, name="submit_factor"))

    return Toolkit(tools=func_tools)


def _profile_result_for_reviewer(result: dict[str, Any]) -> dict[str, Any]:
    """Adapt generic evidence to the reviewer compatibility shape."""
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    return {
        "ok": result.get("ok", False),
        "summary": metrics.get("cross_sectional_core", {}),
        "monthly_corr_robustness": metrics.get("monthly_robustness", {}),
        "profile_hash": result.get("profile_hash"),
        "rule_results": result.get("rule_results", []),
    }


def context_to_openai_messages(agent_context: Any) -> list[dict[str, Any]]:
    """将 AgentScope context 快照为 OpenAI 风格 messages（便于与旧日志格式对齐）。"""
    out: list[dict[str, Any]] = []
    for msg in agent_context:
        role = getattr(msg, "role", None) or getattr(msg, "name", "unknown")
        content = getattr(msg, "content", None)
        if isinstance(content, list):
            text_parts = []
            for block in content:
                if hasattr(block, "text"):
                    text_parts.append(block.text)
                elif isinstance(block, dict) and block.get("text"):
                    text_parts.append(str(block["text"]))
            content = "\n".join(text_parts)
        out.append({"role": str(role), "content": content})
    return out
