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
from alphaagent.factor.mining.runlog import log_step
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


def _collapse_preflight(multi_line_expr: str) -> dict[str, Any] | None:
    """分箱塌缩静态预检（2026-09-27）：常数簇结构硬拦。

    此前 ``precheck_expression`` 只作为独立工具暴露，需 LLM 自愿调用——实测模型
    几乎不调用，于是反复写出 ``CS_ZSCORE(GATED_SIGNAL(..., 0))`` 这类必然塌缩的
    结构，在训练集拿到虚高 IC 后自旋一两小时，直到 ``submit_factor`` 才被
    ``decile_collapse`` 拦下。此处内联到 evaluate 路径，把反馈提前到第一次评估之前。

    返回 ``None`` 表示无阻断风险；否则返回 ``_preflight_check`` 同构的拦截 dict。
    逃生阀 ``ALPHA_PRECHECK_BLOCK_COLLAPSE=0`` 降级为仅警告（回退旧行为）。
    """
    try:
        from alphaagent.factor.mining.tools._precheck import precheck_expression

        res = (precheck_expression(multi_line_expr) or {}).get("result") or {}
    except Exception:  # noqa: BLE001 — 预检失败绝不阻断正常评估
        return None
    if not res.get("blocked"):
        return None
    hard = [r for r in (res.get("risks") or []) if r.get("blocked")]
    detail = "；".join(str(r.get("hint") or r.get("kind")) for r in hard[:2])
    return {
        "blocked": True,
        "warning": (
            f"分箱塌缩结构（{', '.join(str(r.get('kind')) for r in hard)}）："
            "该结构在等频十分位下组数必然 < 8，stage_one 必失败，不必浪费评估轮次。"
        ),
        "suggestion": detail,
    }


def _preflight_check(multi_line_expr: str, factor_name: str) -> dict[str, Any] | None:
    """因子逻辑预审：在跑 DSL 之前快速检测常见逻辑错误。

    返回 None 表示通过，返回 dict 表示拦截（含 warning 和 suggestion）。
    """
    expr = multi_line_expr.strip()
    if not expr:
        return None

    # ── 检查0: 分箱塌缩常数簇（硬拦，放最前——结构性问题、无歧义）──
    collapse = _collapse_preflight(expr)
    if collapse is not None:
        return collapse

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
_ORTHO_MAX_CORR = 0.7    # 兜底默认值；真源 evaluation_policy.orthogonality_max_corr（2026-10-02 P5 收口）


def _ortho_max_corr(tools: Any = None) -> float:
    """离线正交门阈值：优先取 run 网关（研报模式随 _gate_state 传递），否则取配置中心默认。

    2026-10-02 P5 收口：原为模块级硬编码常量（3 处引用），违反项目"阈值收口配置中心"纪律。
    """
    gate = getattr(tools, "report_reproduce_gate", None) or {}
    value = gate.get("orthogonality_max_corr")
    if value is None:
        try:
            from alphaagent.factor.evaluation.defaults import DEFAULT_EVALUATION_POLICY

            value = DEFAULT_EVALUATION_POLICY.get("orthogonality_max_corr", _ORTHO_MAX_CORR)
        except Exception:  # noqa: BLE001
            value = _ORTHO_MAX_CORR
    try:
        return float(value)
    except (TypeError, ValueError):
        return _ORTHO_MAX_CORR


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


def _visible_panel(session: Any) -> pd.DataFrame | None:
    """挖掘期 LLM 可见区间面板（train ∪ val，**剔除盲测段 test**）。

    ``session.panel`` 覆盖 train ∪ val ∪ test（见 ``StockEvalContext.coverage_range``），
    盲测隔离依赖下游按 split 切片；而正交召回的结论会回流给 LLM，故必须显式切到
    ``visible_range()``。

    取不到区间时返回 ``None``：调用方应跳过检查——既不放行泄漏（回落到全区间），
    也不因异常而 fail-closed 误判为"高度相似"。
    """
    panel = getattr(session, "panel", None)
    ctx = getattr(session, "ctx", None)
    if panel is None or ctx is None:
        return None
    try:
        start, end = ctx.visible_range()
    except Exception:  # noqa: BLE001 — 区间不可知 = 不做检查，绝不回落到全区间
        return None
    from alphaagent.data.panel import slice_panel

    return slice_panel(panel, start=start, end=end)


def _orthogonality_check(tools: FactorEvalTools, multi_line_expr: str) -> dict[str, Any]:
    """Post-review sampled check against production/candidate zoos and registry candidates.

    2026-09-01 起附带 top-3 相似因子清单（similar_factors），供评估结果直接
    回传"和谁相似、多相似"——LLM 在评估阶段即可看到与库内因子的相近程度，
    不必等到 submit 被正交门拦下才知道。
    """
    result = {
        "passed": True,
        "skipped_reason": None,
        "threshold": _ortho_max_corr(tools),
        "max_abs_corr": 0.0,
        "compared_factors": 0,
        "blocked_factor_id": None,
        "similar_factors": [],
    }
    try:
        session = tools.service.sessions.get(tools.session_id)
        visible_panel = _visible_panel(session)
        if visible_panel is None:
            # 可见区间不可知 → 不做正交召回（宁缺勿泄漏）
            result["skipped_reason"] = "visible_range_unavailable"
            return result
        sampled_panel = _sample_orthogonality_panel(visible_panel)

        from alphaagent.core.paths import FACTORZOO_DIR
        from core import factor_categories
        from alphaagent.dsl import eval_factor
        from alphaagent.factor.align import align_series_to_panel
        from alphaagent.factor.metrics import spearman_ic
        from alphaagent.factor.zoo import FactorZoo

        # 统一大库（2026-09-03）：production_main 与 candidate_main 全模式共享。
        # 未初始化的库 = 没有可比较对象，跳过该库；不能让单个库缺失把整个
        # 正交检查 fail-closed。
        roots: list = []
        for root in (
            factor_categories.production_dir("technical"),
            factor_categories.candidate_dir("technical"),
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
        result["passed"] = result["max_abs_corr"] < _ortho_max_corr(tools)
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


# 默认运行参数由 MiningConfig 统一提供真源
from alphaagent.factor.mining.infra.config import MiningConfig

_DEFAULT_MINING_CONFIG = MiningConfig()
_runtime_config: MiningConfig = _DEFAULT_MINING_CONFIG


def set_runtime_config(config: MiningConfig | None) -> None:
    """设置全局运行时 MiningConfig 配置。"""
    global _runtime_config
    _runtime_config = config or _DEFAULT_MINING_CONFIG


async def _dispatch_with_timeout(
    loop: asyncio.AbstractEventLoop,
    executor: ThreadPoolExecutor,
    tools: FactorEvalTools,
    name: str,
    args: dict[str, Any],
    *,
    timeout: float | None = None,
) -> tuple[dict[str, Any], float]:
    """带超时的 dispatch，超时返回错误而非永久阻塞。"""
    # 研报模式网关随参数一起进入评估引擎（跨线程池/未来跨进程都可靠）
    try:
        from alphaagent.factor.mining.report_channels import get_run_gate as _grg

        _g = _grg()
        if _g and isinstance(args, dict):
            args["_report_gate"] = dict(_g)
    except Exception:  # noqa: BLE001
        pass
    actual_timeout = float(timeout if timeout is not None else _runtime_config.eval_timeout_seconds)
    try:
        result, elapsed = await asyncio.wait_for(
            loop.run_in_executor(executor, _dispatch_sync, tools, name, args),
            timeout=actual_timeout,
        )
        return result, elapsed
    except asyncio.TimeoutError:
        return {"ok": False, "error": f"评估超时（>{actual_timeout:.0f}s），算子可能首次 JIT 编译或计算量过大，已自动跳过", "error_type": "EvalTimeout"}, actual_timeout


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


def _lineage_key(text: Any) -> str:
    """血统比对键：忽略大小写与下划线。

    同一课题号在因子名里有 `rq<qid>`（如 `rq17eebb_*`）与 `rq_<qid>`（如 `rq_addf97_*`）
    两种写法，逐字比较会把"同课题"误判成"跨课题"。
    """
    return str(text or "").lower().replace("_", "")


def _report_lineage_block(tools: Any, parent_factor: Any) -> str | None:
    """研报模式血统门禁（复现 / 发散）：返回 ⛔ 文案表示拦截，``None`` 表示放行。

    2026-10-02：原实现把门禁内联在 ``evaluate_factor`` / ``submit_factor`` 里，而**批量评估入口**
    ``eval_on_train_set``（实测 108 次/run，是主路径）与 ``eval_on_val_set`` 没有装 →
    34/154（22%）因子绕过门禁挂在其它课题血统上（rq13_*/rq17_*）。此处抽成共享助手统一挂载，
    文案与原实现逐字一致（只补覆盖，不改口径）。

    2026-10-03：比对改成**忽略大小写与下划线**（`_lineage_key`）。实测（4 个 run、38 条门禁
    拦截）里 14 条是"同一课题被误杀"：课题号是 `RQ_17eebb` / `RQ_addf97`，而因子名有两种写法
    `rq17eebb_rev_timed_turn_slope`（无下划线）与 `rq_addf97_growth_surprise_core`（有下划线）
    → 原实现 `qid not in pf` 逐字比较必然不匹配，**与拦截文案自己声明的口径「或至少包含
    `<课题号>`」相矛盾**，白烧轮次。跨课题血统（如把 `rqffd0ba_*` 传给 RQ_addf97）仍照旧硬拦。
    """
    g = getattr(tools, "report_reproduce_gate", None) or {}
    phase = str(g.get("phase") or "")
    if phase == "diverge" and g.get("diverge_parent"):
        pname = str(g.get("parent_name") or "")
        if pname and _lineage_key(pname) not in _lineage_key(parent_factor):
            return (f"⛔ 发散门禁：本轮是课题 {g.get('qid')} 的发散轮，`parent_factor` 必须指向复现版 "
                    f"`{pname}`（并只改一个维度）。若认为该机制在本池无效，请明确放弃该课题，"
                    "不要在同一轮里另起炉灶。")
    if g.get("required") and phase == "reproduce":
        qid = str(g.get("qid") or "")
        pf = str(parent_factor or "")
        if qid and _lineage_key(qid) not in _lineage_key(pf):
            return (f"⛔ 复现门禁：当前处于研报复现阶段（课题 {qid}），"
                    f"`parent_factor` 必须写成 `reproduce_of:{qid}`（或至少包含 `{qid}`），"
                    f"当前传入={pf or '(空)'}。\n"
                    "请先忠实复现研报机制（只允许字段同族替换与算子落地，不得改变机制语义），"
                    "通过 train 后才进入发散阶段。")
    return None


def _report_lineage_fill(tools: Any, parent_factor: Any) -> str | None:
    """研报模式血统**自动补全**：漏传 `parent_factor` 时按当前阶段补齐（2026-10-03）。

    实测依据（run `afb3d7707264`，全 run 25 条 `tool_failed` 里 13+ 条是这个）：
    复现门禁拦下的都是「`parent_factor` 必须写成 `reproduce_of:RQ_xxx`，当前传入=(空)」——
    模型**已经在正确的课题与正确的阶段**里干活，只是忘了填血统字段，于是白烧一轮。
    漏传 = 忘写，补全即本意；**传了非空的错血统仍按原样硬拦**（不覆盖、不猜测，避免污染血统）。

    两种阶段各自的正确补全值：
      · `phase=reproduce`（`required` 开）→ `reproduce_of:<qid>`；
      · `phase=diverge`（`diverge_parent` 开）→ 该课题复现版因子名 `parent_name`。

    开关 `report_policy.parent_autofill`（默认开，随研报网关传递）；关掉即回到
    「血统必须由模型显式声明」的严格口径。补全成功会落一条 `report_lineage_autofill` 到 steps.log。
    """
    pf = str(parent_factor or "").strip()
    if pf:
        return pf
    g = getattr(tools, "report_reproduce_gate", None) or {}
    if not g.get("parent_autofill", True):
        return None
    phase = str(g.get("phase") or "")
    filled: str | None = None
    if phase == "reproduce" and g.get("required"):
        qid = str(g.get("qid") or "")
        filled = f"reproduce_of:{qid}" if qid else None
    elif phase == "diverge" and g.get("diverge_parent"):
        filled = str(g.get("parent_name") or "") or None
    if filled:
        try:
            log_step("report_lineage_autofill",
                     f"phase={phase} qid={g.get('qid')} parent={filled}")
        except Exception:  # noqa: BLE001 — 日志永不阻断评估
            pass
    return filled


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
        **_legacy_kwargs: Any,
    ) -> ToolChunk:
        """训练集评估多行因子表达式，返回 summary、monthly_corr_robustness、label_quantile_buckets。"""
        parent_factor = _report_lineage_fill(tools, parent_factor)
        _lineage_block = _report_lineage_block(tools, parent_factor)
        if _lineage_block:
            return ToolChunk(content=[TextBlock(text=_lineage_block)])
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
        if _legacy_kwargs:
            result["ignored_arguments"] = sorted(_legacy_kwargs)
        return _result_tool_chunk(result)

    async def evaluate_factor(
        multi_line_expr: str,
        profile_id: str,
        factor_name: str = "expr",
        interaction: dict[str, Any] | str | None = None,
        prediction: dict[str, Any] | None = None,
        parent_factor: str | None = None,
        edit_note: str | None = None,
        **_legacy_kwargs: Any,
    ) -> ToolChunk:
        """按已冻结 EvaluationProfile 执行 DSL 评估；profile 控制 split、transform、指标与规则。"""
        # ── 研报模式血统门禁（复现 / 发散，2026-10-02 抽为共享助手，见 _report_lineage_block）──
        parent_factor = _report_lineage_fill(tools, parent_factor)
        _lineage_block = _report_lineage_block(tools, parent_factor)
        if _lineage_block:
            return ToolChunk(content=[TextBlock(text=_lineage_block)])

        # （复现门禁已由首部 _report_lineage_block 统一处理）

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
                f"⛔ 重复评估拦截：与历史条目 {dup['factor_name']} 逐字相同"
                f"（{str(dup['updated_at'])[:10]}，verdict={dup['verdict']}，IC={ic_txt}），已跳过。\n"
                "行动建议：以其为父本做显式变异（填 parent_factor/edit_note）；或若指标达标且换手合规直接 submit。"
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
                    timeout=_runtime_config.backtest_timeout_seconds,
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
                    if (ortho.get("max_abs_corr") or 0) >= _ortho_max_corr(tools):
                        tops = ", ".join(f"{s['factor_id']}({s['corr']:.2f})" for s in similar[:2])
                        result["similarity_warning"] = (
                            f"⚠ 与已有因子高度相似: {tops} —— 提交将被正交门拦截。"
                            "建议换信号机制，或直接以这些因子为父本做正交化改造。"
                        )
            except Exception as exc:  # noqa: BLE001 — 召回是增益信息，失败不影响评估
                result["similar_existing"] = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
        result.setdefault("factor_name", factor_name)
        if _legacy_kwargs:
            result["ignored_arguments"] = sorted(_legacy_kwargs)
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
                return _result_tool_chunk({"ok": False, "error": "skeletons_must_be_valid_json_array", "error_type": "ToolArgumentsError"})
        elif isinstance(skeletons, dict):
            skeletons = [skeletons]
        skeletons = [s for s in (skeletons or []) if isinstance(s, dict)]
        if not skeletons:
            return _result_tool_chunk({
                "ok": False,
                "error": "skeletons_empty",
                "hint": '每个骨架形如 {"name": str, "template": "DSL含{param}占位符", "grid": {"param": [值...]}}',
            })

        # 数据面聚焦硬锁定：种群骨架同样只允许聚焦面列族（越界模板不展开、不评估）
        focus = tuple(getattr(tools, "focus_facets", None) or ())
        if focus:
            from alphaagent.factor.mining.memory.expressions import facet_scope_violation

            for sk in skeletons:
                template = str(sk.get("template") or "")
                vio = facet_scope_violation(template, focus) if template.strip() else None
                if vio is not None:
                    sk_name = str(sk.get("name") or "?")[:40]
                    detail = str(vio["message"]).split(": ", 1)[-1]
                    return _result_tool_chunk({
                        "ok": False,
                        "error": f"facet_lock_violation: 种群骨架「{sk_name}」越界——{detail}",
                        "error_type": "ToolArgumentsError",
                        "facet_lock": {k: v for k, v in vio.items() if k != "message"},
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
        _pop_timeout = _runtime_config.population_timeout_seconds
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(_executor(max_workers), _run), timeout=_pop_timeout
            )
        except asyncio.TimeoutError:
            result = {"ok": False, "error": f"population_timeout:>{_pop_timeout:.0f}s (n={max_population})", "error_type": "EvalTimeout"}
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
        **_legacy_kwargs: Any,
    ) -> ToolChunk:
        """验证集评估；须传 expected_sign（train IC 符号 1/-1），结果含 sign_check。"""
        parent_factor = _report_lineage_fill(tools, parent_factor)
        _lineage_block = _report_lineage_block(tools, parent_factor)
        if _lineage_block:
            return ToolChunk(content=[TextBlock(text=_lineage_block)])
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
        if _legacy_kwargs:
            result["ignored_arguments"] = sorted(_legacy_kwargs)
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
            # ── 研报模式血统门禁（复现 / 发散）：统一走共享助手，避免双份口径漂移（OCR 2026-10-02）──
            parent_factor = _report_lineage_fill(tools, parent_factor)
            _lineage_block = _report_lineage_block(tools, parent_factor)
            if _lineage_block:
                return ToolChunk(content=[TextBlock(text=_lineage_block)])

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
                # 提交含全区间复检 + 首次 JIT 编译 + 正交 hook 重评 registry 候选
                # （~22 条 × 10s）+ 深夜慢速 reviewer LLM（超时重试可达 +180s），
                # 实测 908s 撞穿 900s 旧上限 → stage_one 过线因子全部丢失。
                timeout=_runtime_config.population_timeout_seconds,
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
                timeout=_runtime_config.submit_timeout_seconds,
            )
            return _result_tool_chunk(result)

        func_tools.append(FunctionTool(screen_factors, name="screen_factors"))

    async def precheck_expression(
        multi_line_expr: str,
        **_legacy_kwargs: Any,
    ) -> ToolChunk:
        """【结构风险静态预检】纯 AST 分析，不触发评估、不触达盲测段。"""
        result, _elapsed = await _dispatch_with_timeout(
            asyncio.get_running_loop(), _executor(max_workers), tools, "precheck_expression",
            {"multi_line_expr": multi_line_expr},
            timeout=_runtime_config.submit_timeout_seconds,
        )
        return _result_tool_chunk(result)

    func_tools.append(FunctionTool(precheck_expression, name="precheck_expression"))

    return Toolkit(tools=func_tools)


def _profile_result_for_reviewer(result: dict[str, Any]) -> dict[str, Any]:
    """Adapt generic evidence to the reviewer compatibility shape.

    兼容两种 result 结构（train/val 评估经引擎收敛后统一走 legacy 扁平口径，
    但本函数历史上只认引擎原生结构）：
    - legacy 扁平结构（eval/service._engine_result_to_legacy）：summary 在顶层
      ``result["summary"]``，月度稳健性在 ``result["monthly_corr_robustness"]``；
    - 引擎原生结构：指标嵌在 ``result["metrics"]`` 下。

    2026-09-24 修复：此前只读引擎原生结构，令 legacy 口径的 train 证据 summary
    恒为空 ``{}``。_metric_precheck 读到空 summary 后 train IC/ICIR/coverage 全取
    0（且 ``0 × val_ic = 0 ≤ 0`` 误判方向不一致），于是所有 val 验证必然产出
    「训练集指标未达标 + 方向不一致」四条 revise 理由 → verdict 恒为
    revise_required（覆盖数值达标的 validated 分支）→「验证通过」永不出现，
    因子永远不进候选池。
    """
    metrics = result.get("metrics") if isinstance(result.get("metrics"), dict) else {}
    summary = result.get("summary")
    if not isinstance(summary, dict) or not summary:
        summary = metrics.get("cross_sectional_core", {})
    monthly = result.get("monthly_corr_robustness")
    if not isinstance(monthly, dict) or not monthly:
        monthly = metrics.get("monthly_robustness", {})
    return {
        "ok": result.get("ok", False),
        "summary": summary,
        "monthly_corr_robustness": monthly,
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
