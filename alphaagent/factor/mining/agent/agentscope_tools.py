"""将 FactorEvalTools 包装为 AgentScope FunctionTool。"""

from __future__ import annotations

import asyncio
import functools
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import pandas as pd

from agentscope.message import TextBlock
from agentscope.tool import FunctionTool, Toolkit, ToolChunk

from alphaagent.factor.mining.tools import FactorEvalTools
from alphaagent.factor.mining.infra.jsonutil import json_safe
from alphaagent.factor.mining.interactions import lint_expression_interaction
from alphaagent.factor.mining.population import screen_population

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


_FLOAT_CAP_TOKEN = "$FLOAT_CAP"
# 这些场景下 $float_cap 不是信号本体，无需 LOG：
# - CHIP_*/CROWD_* 算子的市值参数（按设计吃流通市值）
# - LOG/LOG1P 变换本身
# - DIVIDE 的任一位置（比值输出，如 amount/float_cap=换手）
# - CS_NEUTRALIZE/CS_RESIDUALIZE/CS_BUCKET 的分组/条件位置（arg>=1，按秩不变）
_FLOAT_CAP_ALLOW_OPS = {"LOG", "LOG1P"}
_FLOAT_CAP_ALLOW_NONFIRST = {"CS_NEUTRALIZE", "CS_RESIDUALIZE", "CS_BUCKET", "DIVIDE"}
_FLOAT_CAP_BLOCK_ALL_ARGS = {"ADD", "SUBTRACT", "MULTIPLY"}


def _float_cap_signal_use(multi_line_expr: str) -> tuple[bool, str]:
    """判断 $float_cap 是否被当作信号值使用（而非算子参数/分组变量）。

    返回 (is_signal_use, detail)。信号用法（裸用、四则运算、值算子第一参数）
    才要求 LOG 包裹；CHIP_*/CROWD_* 的市值参数、分组位置、比值分母属合法用法。
    """
    code_lines = []
    for raw in multi_line_expr.splitlines():
        code = raw.split("#", 1)[0]
        if code.strip():
            code_lines.append(code)
    upper = "\n".join(code_lines).upper()
    if _FLOAT_CAP_TOKEN not in upper:
        return False, ""

    # 扫描每个 $float_cap 出现位置：记录完整调用链（内层→外层的 (算子, 参数下标)）
    stack: list[list] = []  # [op_name, next_arg_index]
    usages: list[list[tuple[str, int]]] = []
    i, n = 0, len(upper)
    while i < n:
        ch = upper[i]
        if upper.startswith(_FLOAT_CAP_TOKEN, i):
            # stack[0] 是最外层调用；reversed 让 chain[0] = 最内层
            usages.append([(op, idx) for op, idx in reversed(stack)])
            i += len(_FLOAT_CAP_TOKEN)
            continue
        if ch == "(":
            j = i - 1
            while j >= 0 and (upper[j].isalnum() or upper[j] == "_"):
                j -= 1
            stack.append([upper[j + 1:i], 0])
            i += 1
            continue
        if ch == ",":
            if stack:
                stack[-1][1] += 1
            i += 1
            continue
        if ch == ")":
            if stack:
                stack.pop()
            i += 1
            continue
        i += 1

    for chain in usages:
        if not chain:
            return True, "裸用 $float_cap（未经过任何变换或算子参数位置）"
        # 链上任一环处于"条件/分组位置"（如 CS_NEUTRALIZE 的第二参数）→ 整体是条件变量
        for op, idx in chain[1:]:
            if op in _FLOAT_CAP_ALLOW_NONFIRST and idx >= 1:
                break
        else:
            inner_op, inner_idx = chain[0]
            if inner_op.startswith(("CHIP_", "CROWD_")) or inner_op in _FLOAT_CAP_ALLOW_OPS:
                continue
            if inner_op in _FLOAT_CAP_BLOCK_ALL_ARGS:
                return True, f"{inner_op} 中把 $float_cap 当运算值（市值量纲直接进入信号）"
            if inner_op in _FLOAT_CAP_ALLOW_NONFIRST:
                if inner_idx == 0:
                    return True, f"{inner_op} 的信号输入直接用 $float_cap"
                continue
            # 其它算子：第一参数 = 信号本体；非第一参数视为窗口/分组等参数位置
            if inner_idx <= 0:
                return True, f"{inner_op} 的信号输入直接用 $float_cap"
    return False, ""


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

    # ── 检查2: $float_cap 被当信号值使用且无变换 ──
    # （CHIP_*/CROWD_* 市值参数、LOG 变换、分组位置、比值分母属合法用法，不拦）
    cap_signal_use, cap_detail = _float_cap_signal_use(expr)
    if cap_signal_use:
        return {
            "blocked": True,
            "warning": f"纯市值因子: {cap_detail}，无 LOG 变换，截面分布极端右偏。",
            "suggestion": (
                "对信号用法至少包一层 LOG($float_cap)，或用 CS_BUCKET(LOG($float_cap), 10) 做分组中性化。"
                "注意：CHIP_*/CROWD_* 算子的市值参数、CS_NEUTRALIZE/CS_BUCKET 的分组位置、"
                "DIVIDE 的分母属合法用法，直接用即可，无需 LOG。"
            ),
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


def _interaction_contract(
    multi_line_expr: str,
    interaction: dict[str, Any] | str | None,
    policy: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
    """Return ``(normalized_contract, warning, blocked_error)``."""
    try:
        return lint_expression_interaction(
            multi_line_expr,
            interaction,
            policy=policy,
        )
    except Exception as exc:
        return None, None, {
            "ok": False,
            "blocked": True,
            "warning": "interaction 契约校验失败。",
            "suggestion": str(exc),
            "error_type": type(exc).__name__,
        }


# ── 离线正交预判 ──────────────────────────────────────────────

_ORTHO_N_DATES = 5       # 随机抽样锚点数
_ORTHO_BLOCK_DAYS = 20   # 每个锚点向前取的连续交易日数，保留 TS_* 窗口语义
_ORTHO_MAX_CORR = 0.7    # Spearman 相关性阈值


def _sample_orthogonality_panel(panel: pd.DataFrame) -> pd.DataFrame:
    """Sample contiguous date blocks so TS operators retain real lookback context."""
    dates = panel.index.get_level_values("datetime").unique().sort_values()
    required = _ORTHO_N_DATES * _ORTHO_BLOCK_DAYS
    if len(dates) < required:
        return panel

    rng = np.random.default_rng(42)
    anchors = rng.choice(
        np.arange(_ORTHO_BLOCK_DAYS - 1, len(dates)),
        size=_ORTHO_N_DATES,
        replace=False,
    )
    selected: set[pd.Timestamp] = set()
    for anchor in anchors:
        selected.update(dates[max(0, int(anchor) - _ORTHO_BLOCK_DAYS + 1):int(anchor) + 1])
    return panel[panel.index.get_level_values("datetime").isin(selected)]


def _orthogonality_check(tools: FactorEvalTools, multi_line_expr: str) -> dict[str, Any]:
    """Post-review sampled check against production/candidate zoos and registry candidates.

    2026-09-01 起附带 top-3 相似因子清单（similar_factors），供评估结果直接
    回传"和谁相似、多相似"——LLM 在评估阶段即可看到与库内因子的相近程度，
    不必等到 submit 被正交门拦下才知道。
    """
    result = {
        "passed": True,
        "skipped_reason": None,
        "threshold": _ORTHO_MAX_CORR,
        "max_abs_corr": 0.0,
        "compared_factors": 0,
        "blocked_factor_id": None,
        "similar_factors": [],
    }
    try:
        session = tools.service.sessions.get(tools.session_id)
        sampled_panel = _sample_orthogonality_panel(session.panel)

        from alphaagent.core.paths import FACTORZOO_DIR
        from core import factor_categories
        from alphaagent.dsl import eval_factor
        from alphaagent.factor.align import align_series_to_panel
        from alphaagent.factor.metrics import spearman_ic
        from alphaagent.factor.zoo import FactorZoo

        # 统一大库（2026-09-03）：production_main 两模式共享；FACTORZOO_DIR 指向
        # 已改名的 production_technical（迁移遗留），仅作兜底保留。
        # 未初始化的库 = 没有可比较对象，跳过该库；不能让单个库缺失把整个
        # 正交检查 fail-closed（曾把大库合并后的所有提交全部拒掉）。
        roots: list = []
        for root in (
            factor_categories.production_dir("technical"),
            factor_categories.production_dir("fundamental"),
            factor_categories.candidate_dir("technical"),
            factor_categories.candidate_dir("fundamental"),
            FACTORZOO_DIR,
        ):
            if root not in roots:
                roots.append(root)
        zoos = []
        for root in roots:
            try:
                zoo = FactorZoo.open(root)
            except FileNotFoundError:
                continue
            except OSError:
                continue
            zoos.append(zoo)
        registry_path = factor_categories.candidate_registry_path("technical")
        registry = json.loads(registry_path.read_text(encoding="utf-8")) if registry_path.is_file() else {}
        if sum(zoo.n_factors for zoo in zoos) == 0 and not registry:
            result["skipped_reason"] = "empty_factor_libraries"
            return result

        raw = eval_factor(multi_line_expr, sampled_panel)
        if not isinstance(raw, pd.Series):
            raise TypeError(f"factor_output_must_be_series:{type(raw)!r}")
        # align_series_to_panel 返回 ndarray（panel index 序），须包成 Series 才能 reindex
        compared: set[str] = set()
        corr_pairs: list[tuple[str, float]] = []
        sampled_keys = sampled_panel.index
        # align_series_to_panel 返回 ndarray（panel index 序），包成 Series 对齐抽样键
        new_values = np.asarray(
            pd.Series(
                np.asarray(align_series_to_panel(raw, sampled_panel), dtype=np.float64),
                index=sampled_panel.index,
            ).reindex(sampled_keys),
            dtype=np.float64,
        )
        for zoo in zoos:
            rows = zoo.index.rows
            selected_dates = set(sampled_panel.index.get_level_values("datetime"))
            subset = rows[rows["datetime"].isin(selected_dates)]
            if subset.empty:
                continue

            row_ids = subset["row_id"].to_numpy(dtype=np.int64)

            for factor_id in zoo.catalog.list_factor_ids()[:20]:
                if factor_id in compared:
                    continue
                compared.add(factor_id)
                try:
                    subset_keys = pd.MultiIndex.from_arrays(
                        [
                            pd.to_datetime(subset["datetime"]).to_numpy(),
                            subset["instrument"].astype(str).to_numpy(),
                        ],
                        names=["datetime", "instrument"],
                    )
                    old_by_key = pd.Series(
                        np.asarray(zoo.read_factor(factor_id)[row_ids], dtype=np.float64),
                        index=subset_keys,
                    )
                    old_values = np.asarray(old_by_key.reindex(sampled_keys), dtype=np.float64)
                except Exception:
                    continue
                valid = np.isfinite(old_values) & np.isfinite(new_values)
                if int(valid.sum()) < 30:
                    continue
                corr = abs(float(spearman_ic(old_values[valid], new_values[valid], min_pairs=30)))
                if np.isfinite(corr):
                    corr_pairs.append((factor_id, corr))
                if np.isfinite(corr) and corr > result["max_abs_corr"]:
                    result["max_abs_corr"] = corr
                    result["blocked_factor_id"] = factor_id

        # Registry-only candidates have no dense values, so compare their DSL on
        # the same sampled panel.
        for factor_id, entry in sorted(registry.items()):
            if factor_id in compared or not isinstance(entry, dict):
                continue
            expr = str(entry.get("expr") or "").strip()
            if not expr:
                continue
            compared.add(factor_id)
            try:
                old_raw = eval_factor(expr, sampled_panel)
                # registry 分支同样：align 返回 ndarray，包 Series 后对齐
                old_values = np.asarray(
                    pd.Series(
                        np.asarray(align_series_to_panel(old_raw, sampled_panel), dtype=np.float64),
                        index=sampled_panel.index,
                    ).reindex(sampled_keys),
                    dtype=np.float64,
                )
            except Exception:
                continue
            valid = np.isfinite(old_values) & np.isfinite(new_values)
            if int(valid.sum()) < 30:
                continue
            corr = abs(float(spearman_ic(old_values[valid], new_values[valid], min_pairs=30)))
            if np.isfinite(corr):
                corr_pairs.append((factor_id, corr))
            if np.isfinite(corr) and corr > result["max_abs_corr"]:
                result["max_abs_corr"] = corr
                result["blocked_factor_id"] = factor_id

        result["compared_factors"] = len(compared)
        corr_pairs.sort(key=lambda p: -p[1])
        result["similar_factors"] = [
            {"factor_id": fid, "corr": round(c, 4)} for fid, c in corr_pairs[:3]
        ]
        result["passed"] = result["max_abs_corr"] < _ORTHO_MAX_CORR
        return result
    except Exception as exc:  # noqa: BLE001
        # This is now an explicit post-review gate, so an unverifiable check fails closed.
        return {
            **result,
            "passed": False,
            "skipped_reason": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _dispatch_sync(tools: FactorEvalTools, name: str, arguments: dict[str, Any]) -> tuple[dict[str, Any], float]:
    t0 = time.perf_counter()
    result = tools.dispatch(name, arguments)
    elapsed = round(time.perf_counter() - t0, 4)
    result = result if isinstance(result, dict) else {"ok": False, "error": str(result)}
    # 错误串可能携带整段重复索引等巨量调试信息（曾撑出 166MB 日志/上下文），统一截断。
    err = result.get("error")
    if isinstance(err, str) and len(err) > 2000:
        result["error"] = err[:2000] + f" …[truncated, original {len(err)} chars]"
    return result, elapsed


# 单个因子评估的超时秒数：numba JIT 首次编译可能很慢，
# submit 时需加载 CNE panel + realign + 重算，给更充裕的时间。
# topn_portfolio 指标接入 core.engine 全量回测引擎（daily/weekly/monthly 三频率），
# 单次评估计算量显著增加，300s 已不够用。
_EVAL_TIMEOUT_SECONDS = 600


async def _dispatch_with_timeout(
    loop: asyncio.AbstractEventLoop,
    executor: ThreadPoolExecutor,
    tools: FactorEvalTools,
    name: str,
    args: dict[str, Any],
    *,
    timeout: float = _EVAL_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any], float]:
    """带超时的 dispatch，超时返回错误而非永久阻塞。"""
    try:
        result, elapsed = await asyncio.wait_for(
            loop.run_in_executor(executor, _dispatch_sync, tools, name, args),
            timeout=timeout,
        )
        return result, elapsed
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"评估超时（>{timeout:.0f}s），算子可能首次 JIT 编译或计算量过大，已自动跳过"}, timeout


def _result_tool_chunk(result: dict[str, Any]) -> ToolChunk:
    """Return machine-readable output; the UI observer parses this payload."""
    return ToolChunk(content=[TextBlock(text=json.dumps(json_safe(result), ensure_ascii=False, default=str))])


def _expr_key(expr: str) -> str:
    return re.sub(r"\s+", "", expr or "")


def _compact_evaluation(result: dict[str, Any]) -> dict[str, Any]:
    """Keep reviewer evidence small; dense tables never belong in candidate registry."""
    profile = result.get("profile") if isinstance(result.get("profile"), dict) else {}
    return {
        "split": result.get("split"),
        "passed": result.get("passed"),
        "summary": result.get("summary") if isinstance(result.get("summary"), dict) else {},
        "monthly_corr_robustness": result.get("monthly_corr_robustness"),
        "label_quantile_buckets": result.get("label_quantile_buckets"),
        "rule_results": result.get("rule_results"),
        "profile_id": profile.get("profile_id"),
    }


def _evaluation_evidence(reviewer: Any | None, expr: str) -> dict[str, Any] | None:
    if reviewer is None:
        return None
    evaluations = reviewer.evaluations.get(_expr_key(expr), {})
    evidence = {
        split: [_compact_evaluation(row) for row in rows[-2:]]
        for split, rows in evaluations.items()
    }
    return evidence or None


def build_factor_eval_toolkit(
    tools: FactorEvalTools,
    *,
    max_workers: int = 4,
    reviewer: Any | None = None,
    interaction_policy: dict[str, Any] | None = None,
    population_max: int = 0,
) -> Toolkit:
    """构建与 OpenAI 版一致的 eval / submit / typed-interaction 工具集。

    ``population_max > 0`` 时注册种群批量工具 `propose_population`（路径 B）；
    0 表示关闭，工具从模型可见列表中整体移除。
    """

    def _gate_interaction(
        expr: str,
        interaction: dict[str, Any] | str | None,
    ) -> tuple[dict[str, Any] | None, str | None, ToolChunk | None]:
        spec, warning, error = _interaction_contract(expr, interaction, interaction_policy)
        if error is not None:
            content = (
                f"⛔ 交互契约拦截: {error.get('warning') or error.get('error')}\n"
                f"建议: {error.get('suggestion')}\n"
                f"(表达式: {expr[:160]}...)"
            )
            return spec, warning, ToolChunk(content=[TextBlock(text=content)])
        return spec, warning, None

    async def eval_on_train_set(
        multi_line_expr: str,
        factor_name: str = "expr",
        include_detail_tables: bool = False,
        label_quantile_n: int = 10,
        interaction: dict[str, Any] | str | None = None,
        prediction: dict[str, Any] | None = None,
        parent_factor: str | None = None,
        edit_note: str | None = None,
    ) -> ToolChunk:
        """训练集评估多行因子表达式，返回 summary、monthly_corr_robustness、label_quantile_buckets。"""
        loop = __import__("asyncio").get_running_loop()
        contract, interaction_warning, blocked = _gate_interaction(multi_line_expr, interaction)
        if blocked is not None:
            return blocked
        args: dict[str, Any] = {
            "multi_line_expr": multi_line_expr,
            "factor_name": factor_name,
            "include_detail_tables": include_detail_tables,
            "label_quantile_n": label_quantile_n,
            "interaction": contract,
        }
        if prediction is not None:
            args["prediction"] = prediction
        if parent_factor:
            args["parent_factor"] = parent_factor
        if edit_note:
            args["edit_note"] = edit_note
        result, _elapsed = await _dispatch_with_timeout(
            loop, _executor(max_workers), tools, "eval_on_train_set",
            args,
        )
        if contract is not None:
            result["interaction"] = contract
        if interaction_warning:
            result.setdefault("preflight_warning", interaction_warning)
        if reviewer is not None:
            reviewer.record_evaluation(
                "train",
                {
                    "multi_line_expr": multi_line_expr,
                    "factor_name": factor_name,
                    "interaction": contract,
                },
                result,
            )
        result.setdefault("factor_name", factor_name)
        return _result_tool_chunk(result)

    async def evaluate_factor(
        multi_line_expr: str,
        profile_id: str,
        factor_name: str = "expr",
        interaction: dict[str, Any] | str | None = None,
        prediction: dict[str, Any] | None = None,
        parent_factor: str | None = None,
        edit_note: str | None = None,
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

        contract, interaction_warning, blocked = _gate_interaction(multi_line_expr, interaction)
        if blocked is not None:
            return blocked

        # ── 精确重复评估拦截（2026-09-05）：表达式与历史正向条目逐字相同时，
        # 重跑评估零信息增量——直接回放历史结果，不烧评估轮次。
        # 仅拦"逐字相同"：任何变异（参数/算子/修饰）都会改变指纹或表达式，不受影响。
        # （prior_result 提醒在 advisory 层永不拦截是设计；但"原样重测"连提醒里
        #   建议的显式变异都没做，实测 LLM 会无视软提醒——此处升级为硬拦。）
        try:
            dup = tools.memory_store.exact_duplicate_prior(multi_line_expr) if tools.memory_store else None
        except Exception:  # noqa: BLE001 — 拦截失效不阻断评估
            dup = None
        if dup is not None:
            ic_txt = f"{dup['ic']:+.4f}" if isinstance(dup.get("ic"), (int, float)) else "N/A"
            content = (
                f"⛔ 重复评估拦截：该表达式与历史条目 {dup['factor_name']} 逐字相同"
                f"（{str(dup['updated_at'])[:10]}，verdict={dup['verdict']}，IC={ic_txt}），"
                "评估结果不会改变，已跳过本次执行。\n"
                "正确动作：①以其为父本做显式变异（parent_factor="
                f"{dup['factor_name']} + edit_note 说明改动点）；"
                "②若要推进，直接对它调用 submit_factor 走入库门槛；"
                "③若认为历史结论已过时，改用因子实验室人工复核。"
            )
            return ToolChunk(content=[TextBlock(text=content)])

        # Orthogonality is enforced by ingest similarity; offline re-evaluation
        # is outside the tool timeout and can hang on heavy JIT operators.
        loop = __import__("asyncio").get_running_loop()
        args = {
            "multi_line_expr": multi_line_expr,
            "profile_id": profile_id,
            "factor_name": factor_name,
            "interaction": contract,
        }
        if prediction is not None:
            args["prediction"] = prediction
        if parent_factor:
            args["parent_factor"] = parent_factor
        if edit_note:
            args["edit_note"] = edit_note
        result, _elapsed = await _dispatch_with_timeout(
            loop, _executor(max_workers), tools, "evaluate_factor", args
        )
        if preflight is not None and not preflight.get("blocked", False):
            result["preflight_warning"] = preflight["warning"]
        if contract is not None:
            result["interaction"] = contract
        if interaction_warning:
            result.setdefault("preflight_warning", interaction_warning)
        if reviewer is not None and result.get("ok", True):
            reviewer_result = _profile_result_for_reviewer(result)
            split = result.get("split")
            if split == "train":
                reviewer.record_evaluation("train", args, reviewer_result)
            elif split == "val":
                reviewer.record_evaluation("val", args, reviewer_result)
            if split == "val" and "validation" in reviewer.review_on:
                result["factor_review"] = await reviewer.review(
                    {
                        "multi_line_expr": multi_line_expr,
                        "factor_name": factor_name,
                        "comment": "",
                        "interaction": contract,
                    },
                    turn=getattr(reviewer, "current_turn", 0),
                )
                # validation 阶段 reviewer 只给建议，不阻断 submit
                # LLM 可根据 review意见改进因子，但 verdict != approve 不阻止提交候选池
                candidate_id = result.get("candidate", {}).get("candidate_id")
                if candidate_id:
                    state = tools.service.record_candidate_review(tools.session_id, candidate_id, result["factor_review"])
                    if state is not None:
                        result["candidate_state"] = state["state"]
        # ── 相似因子召回（2026-09-01）：评估达标后立即对比库内已有因子 ──
        # 只对通过训练筛的因子跑（弱因子反正不会提交，省算力）；结果附
        # similar_existing（top-3 相似清单），高相似时直接在结果里警告，
        # LLM 不必等 submit 被正交门拦下才发现白跑。
        if result.get("ok") and result.get("passed"):
            try:
                ortho = await asyncio.wait_for(
                    loop.run_in_executor(
                        _executor(max_workers), _orthogonality_check, tools, multi_line_expr,
                    ),
                    timeout=120,
                )
                similar = ortho.get("similar_factors") or []
                if ortho.get("skipped_reason"):
                    result["similar_existing"] = {"skipped": ortho["skipped_reason"]}
                elif similar:
                    result["similar_existing"] = {
                        "max_abs_corr": ortho.get("max_abs_corr"),
                        "top": similar,
                        "compared_factors": ortho.get("compared_factors"),
                    }
                    if (ortho.get("max_abs_corr") or 0) >= _ORTHO_MAX_CORR:
                        tops = ", ".join(f"{s['factor_id']}({s['corr']:.2f})" for s in similar[:2])
                        result["similarity_warning"] = (
                            f"⚠ 与已有因子高度相似: {tops} —— 提交将被正交门拦截。"
                            "建议换信号机制，或直接以这些因子为父本做正交化改造。"
                        )
            except Exception as exc:  # noqa: BLE001 — 召回是增益信息，失败不影响评估
                result["similar_existing"] = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
        result.setdefault("factor_name", factor_name)
        return _result_tool_chunk(result)

    async def propose_population(
        skeletons: list[dict[str, Any]],
        max_population: int = 24,
        screen_end: str | None = None,
    ) -> ToolChunk:
        """种群批量筛选：提交 1~3 个参数化骨架（DSL 模板 + 参数网格），引擎一次性
        展开并轻量评估全部候选，返回按 |ICIR| 排序的 top 表与死因直方图。

        使用规范：
        - 每轮至多调用一次；max_population 默认 24、上限 36，网格别铺满；
        - 模板占位符写 {param}，如 TS_MEAN(over_gap,{w})；grid 给候选值列表；
        - 快筛口径只含 IC/ICIR/RankIC/coverage/autocorr（train 侧），不含 mls/月度——
          幸存者须再用 evaluate_factor(train_screen) 复核后走 submit_factor；
        - 骨架避免使用需 interaction 契约的算子（GATED_SIGNAL/CS_GROUP_RANK 等），
          否则后续提交会被契约拦截；
        - 用途：参数敏感性扫描与机制邻域探索。纯双因子四则组合请先给出
          与父本正交的新息来源，否则会被 Reviewer 打回。
        """
        import asyncio

        # 容忍模型传参漂移：skeletons 可能是 JSON 字符串或单个对象
        if isinstance(skeletons, str):
            try:
                parsed = json.loads(skeletons)
                skeletons = parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                return _result_tool_chunk({"ok": False, "error": "skeletons_must_be_valid_json_array"})
        elif isinstance(skeletons, dict):
            skeletons = [skeletons]
        skeletons = [s for s in (skeletons or []) if isinstance(s, dict)]
        if not skeletons:
            return _result_tool_chunk({
                "ok": False,
                "error": "skeletons_empty",
                "hint": '每个骨架形如 {"name": str, "template": "DSL含{param}占位符", "grid": {"param": [值...]}}',
            })

        loop = __import__("asyncio").get_running_loop()

        def _run() -> dict[str, Any]:
            session = tools.service.sessions.get(tools.session_id)
            return screen_population(
                session,
                skeletons,
                max_population=max_population,
                screen_end=screen_end,
            )

        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(_executor(max_workers), _run), timeout=1800.0
            )
        except asyncio.TimeoutError:
            result = {"ok": False, "error": f"population_timeout:>1800s (n={max_population})"}
        except Exception as exc:  # noqa: BLE001
            result = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:400]}
        result["elapsed_seconds"] = round(time.perf_counter() - t0, 1)
        return _result_tool_chunk(result)

    async def eval_on_val_set(
        multi_line_expr: str,
        factor_name: str = "expr",
        include_detail_tables: bool = False,
        label_quantile_n: int = 10,
        expected_sign: int | None = None,
        interaction: dict[str, Any] | str | None = None,
        prediction: dict[str, Any] | None = None,
        profile_id: str | None = None,
        parent_factor: str | None = None,
        edit_note: str | None = None,
    ) -> ToolChunk:
        """验证集评估；须传 expected_sign（train IC 符号 1/-1），结果含 sign_check。"""
        # 模型常从 evaluate_factor 习惯性带入 profile_id：显式接受并校验，避免 TypeError。
        if profile_id is not None and profile_id != "validation":
            return ToolChunk(content=[TextBlock(
                text=(
                    f"eval_on_val_set 固定使用冻结的 validation profile；"
                    f"收到 profile_id={profile_id!r}。如需其他 split/规则，"
                    f"请改用 evaluate_factor(profile_id=...)。"
                )
            )])
        loop = __import__("asyncio").get_running_loop()
        contract, interaction_warning, blocked = _gate_interaction(multi_line_expr, interaction)
        if blocked is not None:
            return blocked
        args: dict[str, Any] = {
            "multi_line_expr": multi_line_expr,
            "factor_name": factor_name,
            "include_detail_tables": include_detail_tables,
            "label_quantile_n": label_quantile_n,
            "interaction": contract,
        }
        if expected_sign is not None:
            args["expected_sign"] = expected_sign
        if prediction is not None:
            args["prediction"] = prediction
        if parent_factor:
            args["parent_factor"] = parent_factor
        if edit_note:
            args["edit_note"] = edit_note
        result, _elapsed = await _dispatch_with_timeout(
            loop, _executor(max_workers), tools, "eval_on_val_set", args,
        )
        if reviewer is not None:
            reviewer.record_evaluation("val", args, result)
            if result.get("ok", True) and "validation" in reviewer.review_on:
                result["factor_review"] = await reviewer.review(
                    {
                        "multi_line_expr": multi_line_expr,
                        "factor_name": factor_name,
                        "comment": "",
                        "interaction": contract,
                    },
                    turn=getattr(reviewer, "current_turn", 0),
                )
        result.setdefault("factor_name", factor_name)
        if contract is not None:
            result["interaction"] = contract
        if interaction_warning:
            result.setdefault("preflight_warning", interaction_warning)
        return _result_tool_chunk(result)

    func_tools: list[FunctionTool] = [
        FunctionTool(evaluate_factor, name="evaluate_factor", is_read_only=True),
        FunctionTool(eval_on_train_set, name="eval_on_train_set", is_read_only=True),
        FunctionTool(eval_on_val_set, name="eval_on_val_set", is_read_only=True),
    ]
    if population_max and population_max > 0:
        func_tools.append(FunctionTool(propose_population, name="propose_population", is_read_only=True))

    if tools.submit_service is not None:

        async def submit_factor(
            multi_line_expr: str,
            factor_name: str,
            comment: str,
            interaction: dict[str, Any] | str | None = None,
            rebalance_freq: str | None = None,
            parent_factor: str | None = None,
            edit_note: str | None = None,
            **_legacy_kwargs: Any,
        ) -> ToolChunk:
            """【正式交付】统计数据通过即写候选池；reviewer approve 才写正式 factorzoo。"""
            # ── 先执行 submit（stage_one 候选池 + stage_two 正式库统计门槛） ──
            loop = __import__("asyncio").get_running_loop()
            contract, interaction_warning, blocked = _gate_interaction(multi_line_expr, interaction)
            if blocked is not None:
                return blocked
            review_hook = None
            if reviewer is not None:
                def review_hook(candidate: dict[str, Any]) -> dict[str, Any]:
                    future = asyncio.run_coroutine_threadsafe(
                        reviewer.review(
                            candidate, turn=getattr(reviewer, "current_turn", 0), stage="pre_submit"
                        ),
                        loop,
                    )
                    return future.result()
            submit_args: dict[str, Any] = {
                "multi_line_expr": multi_line_expr,
                "factor_name": factor_name,
                "comment": comment,
                "interaction": contract,
                "rebalance_freq": rebalance_freq,
                "evaluation_evidence": _evaluation_evidence(reviewer, multi_line_expr),
                "review_hook": review_hook,
                "orthogonality_hook": lambda: _orthogonality_check(tools, multi_line_expr),
            }
            if parent_factor:
                submit_args["parent_factor"] = parent_factor
            if edit_note:
                submit_args["edit_note"] = edit_note
            result, _elapsed = await _dispatch_with_timeout(
                loop, _executor(max_workers), tools, "submit_factor",
                submit_args,
                # 提交含全区间复检 + 首次 JIT 编译，600s 会白白失败一次（重试靠热缓存才过）。
                timeout=900,
            )
            if _legacy_kwargs:
                result["ignored_arguments"] = sorted(_legacy_kwargs)
            if contract is not None:
                result["interaction"] = contract
            if interaction_warning:
                result.setdefault("preflight_warning", interaction_warning)
            result.setdefault("factor_name", factor_name)
            return _result_tool_chunk(result)

        func_tools.append(FunctionTool(submit_factor, name="submit_factor"))

    # Screener（regime 感知因子筛选）——开关在 research_spec.delivery_policy.screener.enabled
    if tools._screener_config() is not None and tools._screener_config().get("enabled"):

        async def screen_factors(
            factor_names: list[str] | None = None,
            signal_date: str | None = None,
            **_legacy_kwargs: Any,
        ) -> ToolChunk:
            """【Screener · regime 感知筛选】对正式库因子做市场制度感知筛选，输出动态权重/方向。"""
            result, _elapsed = await _dispatch_with_timeout(
                asyncio.get_running_loop(), _executor(max_workers), tools, "screen_factors",
                {
                    "factor_names": factor_names or [],
                    "signal_date": signal_date,
                },
                timeout=120,
            )
            return _result_tool_chunk(result)

        func_tools.append(FunctionTool(screen_factors, name="screen_factors"))

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
