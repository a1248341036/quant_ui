"""种群批量筛选：LLM 产出参数化骨架，引擎展开网格做轻量批量评估。

设计边界：
- 只做 train 侧快筛（IC / ICIR / RankIC / coverage / 截面自相关），
  不含 mls_fmb / monthly / 组合指标——那些交给既有 submit 链路裁决；
- 不写库、不做相似度去重（与正式库的去重仍在 submit 四道门）；
- 对既有挖掘链路纯增量：注册为独立工具，不改任何既有路径。
"""
from __future__ import annotations

import itertools
import re
import time
from typing import Any

import numpy as np
import pandas as pd

from alphaagent.dsl import eval_factor
from alphaagent.dsl.core.errors import MultiLineFactorEvalError
from alphaagent.factor.align import align_series_to_panel
from alphaagent.factor.metrics import (
    coverage,
    cross_sectional_ic,
    cross_sectional_lag1_pearson_autocorr,
    cross_sectional_rank_ic,
    cs_ic_summary,
)

MAX_POPULATION = 36


# 模型对骨架内层键名会自由发挥，这里按“角色”做泛化匹配。
_TEMPLATE_KEYS = ("template", "dsl", "expr", "expression", "multi_line_expr", "formula")
_NAME_KEYS = ("name", "factor_name_template", "factor_name_template_str", "factor_name", "id")
_GRID_KEYS = ("grid", "params", "parameter_grid")


def _pick(sk: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for k in keys:
        v = sk.get(k)
        if v:
            return v
    return None


def expand_skeletons(
    skeletons: list[dict[str, Any]],
    *,
    max_population: int = MAX_POPULATION,
) -> tuple[list[tuple[str, str, dict[str, Any]]], list[str]]:
    """骨架 × 参数网格 → [(候选名, 表达式, 参数)]；附带越界/格式错误列表。

    模板占位符写作 ``{param}``，用字符串替换展开（不用 str.format，
    避免 DSL 中意外花括号触发 KeyError）。骨架内层键名按角色泛化匹配
    （template/dsl/multi_line_expr 等），容忍模型的键名漂移。
    """
    errors: list[str] = []
    out: list[tuple[str, str, dict[str, Any]]] = []
    budget = max_population
    for sk in skeletons:
        if not isinstance(sk, dict):
            errors.append(f"骨架须为对象: {str(sk)[:60]}")
            continue
        raw_template = _pick(sk, _TEMPLATE_KEYS)
        raw_grid = _pick(sk, _GRID_KEYS)
        raw_name = _pick(sk, _NAME_KEYS)
        template = str(raw_template or "")
        if not template.strip() or not isinstance(raw_grid, dict) or not raw_grid:
            errors.append(
                f"骨架缺 template/grid: keys={sorted(k for k in sk.keys() if k != 'grid')[:6]}"
            )
            continue
        # 名字里的占位符（如 w{w}）去掉花括号段，避免污染候选名
        base_name = re.sub(r"\{[^}]*\}", "", str(raw_name or "pop")).strip("._- ") or "pop"
        keys = list(raw_grid.keys())
        combos = list(itertools.product(*[list(raw_grid[k]) for k in keys]))
        if len(out) + len(combos) > budget:
            combos = combos[: max(0, budget - len(out))]
            if not combos:
                errors.append(f"{base_name}: 超出 max_population={budget}，已跳过")
                break
        for combo in combos:
            expr = template
            tag = []
            params: dict[str, Any] = {}
            for k, v in zip(keys, combo):
                expr = expr.replace("{" + str(k) + "}", str(v))
                tag.append(f"{k}{v}")
                params[k] = v
            out.append((f"{base_name}_" + "_".join(tag), expr, params))
    return out, errors


def screen_expr(expr: str, panel: pd.DataFrame, label_col: str, *, min_pairs: int = 5) -> dict[str, Any]:
    """单条轻量筛：物化 + IC/ICIR/RankIC/coverage/自相关（同一口径于 eval._build_summary 子集）。"""
    t0 = time.perf_counter()
    try:
        out = eval_factor(expr, panel)
    except MultiLineFactorEvalError as exc:
        return {"ok": False, "error": str(exc)[:300]}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:300]}
    if not isinstance(out, pd.Series):
        return {"ok": False, "error": "factor_output_must_be_series"}

    values = align_series_to_panel(out, panel)
    factor = pd.Series(values, index=panel.index, name="cand", dtype=np.float32)
    label = panel[label_col]
    daily_ic = cross_sectional_ic(factor, label, min_pairs=min_pairs)
    daily_ric = cross_sectional_rank_ic(factor, label, min_pairs=min_pairs)
    cs = cs_ic_summary(daily_ic, daily_ric)
    n_inst = int(factor.index.get_level_values("instrument").nunique())
    finite_vals = values[np.isfinite(values)]
    return {
        "ok": True,
        "ic": cs.get("ic"),
        "icir": cs.get("icir"),
        "rank_ic": cs.get("rank_ic"),
        "n_days": cs.get("n_days"),
        "coverage": float(coverage(values)) if len(finite_vals) else 0.0,
        "cs_pearson_autocorr": cross_sectional_lag1_pearson_autocorr(
            factor, min_pairs=min(30, max(n_inst - 1, 2))
        ),
        "secs": round(time.perf_counter() - t0, 1),
    }


def screen_population(
    session: Any,
    skeletons: list[dict[str, Any]],
    *,
    max_population: int = MAX_POPULATION,
    label_col: str | None = None,
    screen_end: str | None = None,
    autocorr_gate: float = 0.18,
) -> dict[str, Any]:
    """批量筛选入口：展开 → 逐条轻量筛 → 排序 + 死因直方图。

    返回紧凑结构（≤ max_population 行小表可内联回对话）；
    ``screen_end`` 可截短快窗（如只筛到 2021-06-30），进一步提速。
    """
    panel = session.panel
    lc = label_col or session.ctx.label_col
    if screen_end:
        from alphaagent.data.panel import slice_panel
        panel = slice_panel(panel, end=screen_end)
    if panel.empty:
        return {"ok": False, "error": "screen_window_empty"}

    candidates, expand_errors = expand_skeletons(skeletons, max_population=max_population)
    if not candidates:
        return {"ok": False, "error": "no_candidates", "errors": expand_errors}

    t0 = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for cname, expr, params in candidates:
        r = screen_expr(expr, panel, lc)
        rows.append({
            "name": cname,
            "params": params,
            "expr_sha": hash(expr) & 0xFFFF,
            **r,
        })
    wall = round(time.perf_counter() - t0, 1)

    valid = [r for r in rows if r.get("ok")]
    valid.sort(key=lambda r: (abs(float(r.get("icir") or 0)), abs(float(r.get("ic") or 0))), reverse=True)

    dead: dict[str, int] = {}
    for r in rows:
        if not r.get("ok"):
            key = str(r.get("error", "?")).split(":")[0][:48]
            dead[key] = dead.get(key, 0) + 1
            continue
        reasons = []
        ac = r.get("cs_pearson_autocorr")
        if ac is None or not np.isfinite(float(ac)) or float(ac) < autocorr_gate:
            reasons.append(f"autocorr<{autocorr_gate}")
        icir = r.get("icir")
        if icir is None or abs(float(icir)) <= 0.25:
            reasons.append("icir<=0.25")
        cov = r.get("coverage")
        if cov is None or float(cov) <= 0.85:
            reasons.append("coverage<=0.85")
        for key in reasons:
            dead[key] = dead.get(key, 0) + 1

    compact_keys = ("name", "ic", "icir", "rank_ic", "coverage", "cs_pearson_autocorr", "secs")
    top = [{k: r.get(k) for k in compact_keys} for r in valid[:10]]
    all_rows = [
        {**{k: r.get(k) for k in compact_keys}, "dead": None}
        for r in rows
    ]

    return {
        "ok": True,
        "window": {"start": str(panel.index.get_level_values('datetime').min())[:10],
                   "end": str(panel.index.get_level_values('datetime').max())[:10]},
        "n_candidates": len(candidates),
        "n_valid": len(valid),
        "wall_seconds": wall,
        "ranking": "by |ICIR| then |IC| (train-side screening only)",
        "top": top,
        "rows": all_rows,
        "dead_ends_histogram": dict(sorted(dead.items(), key=lambda kv: -kv[1])),
        "expand_errors": expand_errors,
        "next_step_hint": (
            "对 top 候选先用 evaluate_factor(train_screen) 复核完整指标，"
            "再走 submit_factor；复核时记得补 interaction 契约（如骨架用了 GATED/GROUP_RANK 等）。"
        ),
    }
