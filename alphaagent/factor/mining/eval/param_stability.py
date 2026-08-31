"""参数邻域稳定性证据（stage_two 附加门禁）。

真实 alpha 不应依赖某一个神奇窗口（knife-edge parameter）。本模块对 DSL 中
TS_* 算子第二参数位的整数窗口做整体 ±offset 邻域扰动，逐变体在 train 窗口
重评估，产出表现分布证据（positive_fraction / worst_icir），阈值判定在
``DeliveryChecker.param_stability``（证据与裁决分离，与两层治理一致）。

设计要点：
- 窗口扫描为文本级（balanced-paren 顶层参数切分），不依赖 Python AST——
  DSL 含 ``$ref`` / ``@freq`` 等非 Python 语法，AST 解析会破坏原文定位；
- 只扰动「第二个参数位置的纯整数字面量」（TS_* 家族约定 TS_OP(x, window, ...)），
  非 int / 动态窗变量 / 第三位窗口（如 TS_COEFFICIENT(a, b, w)）不覆盖；
- 同一窗口值的全部出现位置整体平移（全局参数扰动，而非逐出现独立扰动）；
- 变体评估报错按非正计入（邻域边缘失效是真实风险）；全部报错 → 跳过门禁；
- 方向调整：adj = metric * sign(base_train_ic)，负 IC 的稳定因子不受惩罚。
"""

from __future__ import annotations

import re
from typing import Any, Callable

import numpy as np

# TS_* 家族：第二参数位约定为窗口（TS_OP(x, window, ...)）。
_WINDOW_CALL_RE = re.compile(r"\b(TS_[A-Z][A-Z0-9_]*)\s*\(")
_INT_TOKEN_RE = re.compile(r"[+-]?\d+")

DEFAULT_WINDOW_OFFSETS: tuple[int, ...] = (-1, 1, -2, 2)


def _blank_comments(text: str) -> str:
    """把 # 注释替换为等长空格，保持所有字符偏移不变。"""
    return re.sub(
        r"#[^\n]*",
        lambda m: " " * (m.end() - m.start()),
        text,
    )


def _split_top_args(src: str) -> list[str]:
    """按顶层逗号切分参数串（忽略嵌套括号内的逗号）。"""
    args: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in src:
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            if depth == 0:
                break
            depth -= 1
            cur.append(ch)
        elif ch == "," and depth == 0:
            args.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    if cur or args:
        args.append("".join(cur))
    return args


def find_window_params(expr: str) -> dict[int, list[tuple[int, int]]]:
    """扫描 DSL 中 TS_* 调用第二参数位的整数字面量窗口。

    Returns
    -------
    dict
        ``{窗口值: [(start, end), ...]}``，span 为该整数字面量在原文中的
        字符区间（end 不含）。同一窗口值的多处出现聚合在一起。
    """
    text = _blank_comments(expr)
    occ: dict[int, list[tuple[int, int]]] = {}
    for m in _WINDOW_CALL_RE.finditer(text):
        open_idx = m.end() - 1
        depth = 0
        close_idx: int | None = None
        for j in range(open_idx, len(text)):
            if text[j] == "(":
                depth += 1
            elif text[j] == ")":
                depth -= 1
                if depth == 0:
                    close_idx = j
                    break
        if close_idx is None:
            continue
        args = _split_top_args(text[open_idx + 1 : close_idx])
        if len(args) < 2:
            continue
        raw = args[1]
        token = raw.strip()
        if not _INT_TOKEN_RE.fullmatch(token):
            continue
        # 第二参数在原文中的位置：开括号 + 第一参数长度 + 1 个逗号 + 前导空白
        start = open_idx + 1 + len(args[0]) + 1 + (len(raw) - len(raw.lstrip()))
        end = start + len(token)
        value = int(token)
        occ.setdefault(value, []).append((start, end))
    return occ


def _replace_spans(expr: str, spans: list[tuple[int, int]], token: str) -> str:
    out = expr
    for start, end in sorted(spans, key=lambda s: s[0], reverse=True):
        out = out[:start] + token + out[end:]
    return out


def build_window_variants(
    expr: str,
    *,
    window_offsets: tuple[int, ...] = DEFAULT_WINDOW_OFFSETS,
    min_window: int = 2,
    max_window_values: int = 2,
    max_variants: int = 6,
) -> list[dict[str, Any]]:
    """生成窗口邻域变体表达式。

    每个变体只扰动一个窗口值（该值全部出现位置整体平移一个 offset）。
    窗口值按出现次数降序、绝对值升序优先；变体总数不超过 max_variants。
    """
    occ = find_window_params(expr)
    if not occ:
        return []
    values = sorted(occ.keys(), key=lambda v: (-len(occ[v]), abs(v)))
    if max_window_values > 0:
        values = values[:max_window_values]
    variants: list[dict[str, Any]] = []
    for v in values:
        for off in window_offsets:
            if len(variants) >= max_variants:
                break
            try:
                new_v = v + int(off)
            except (TypeError, ValueError):
                continue
            if int(off) == 0 or new_v < max(1, int(min_window)) or new_v == v:
                continue
            variants.append({
                "base_window": int(v),
                "offset": int(off),
                "expr": _replace_spans(expr, occ[v], str(new_v)),
            })
        if len(variants) >= max_variants:
            break
    return variants


def evaluate_param_stability(
    expr: str,
    evaluate_variant: Callable[[str], dict[str, Any]],
    *,
    base_ic: float | None = None,
    window_offsets: tuple[int, ...] = DEFAULT_WINDOW_OFFSETS,
    min_window: int = 2,
    max_window_values: int = 2,
    max_variants: int = 6,
) -> dict[str, Any]:
    """计算参数邻域稳定性证据。

    Parameters
    ----------
    expr : str
        原始 DSL 表达式。
    evaluate_variant : callable
        ``variant_expr -> {"ic": float, "icir": float}``；任何异常按该变体
        评估失败处理（计入非正，不中断整体）。
    base_ic : float, optional
        原因子 train 窗口 IC，用于方向调整（sign）；None/0 视为正向。

    Returns
    -------
    dict
        ``{direction, n_variants, variants: [...], positive_fraction,
        worst_icir, skipped_reason}``。skipped_reason 存在时其余统计为 None，
        门禁跳过（no_integer_window_params / all_variants_errored）。
    """
    direction = 1 if base_ic is None or float(base_ic) >= 0 else -1
    variants = build_window_variants(
        expr,
        window_offsets=window_offsets,
        min_window=min_window,
        max_window_values=max_window_values,
        max_variants=max_variants,
    )
    if not variants:
        return {
            "direction": direction,
            "skipped_reason": "no_integer_window_params",
            "n_variants": 0,
            "variants": [],
            "positive_fraction": None,
            "worst_icir": None,
        }

    results: list[dict[str, Any]] = []
    for v in variants:
        try:
            m = evaluate_variant(v["expr"]) or {}
            ic = float(m.get("ic"))
            icir = float(m.get("icir"))
            if not (np.isfinite(ic) and np.isfinite(icir)):
                raise ValueError(f"non-finite ic/icir: {ic}/{icir}")
        except Exception as exc:  # noqa: BLE001
            results.append({
                **v,
                "error": str(exc)[:200],
                "adj_ic": None,
                "adj_icir": None,
                "positive": False,
            })
        else:
            results.append({
                **v,
                "ic": round(ic, 6),
                "icir": round(icir, 6),
                "adj_ic": round(ic * direction, 6),
                "adj_icir": round(icir * direction, 6),
                "positive": bool(ic * direction > 0),
            })

    n = len(results)
    if all("error" in r for r in results):
        return {
            "direction": direction,
            "skipped_reason": "all_variants_errored",
            "n_variants": n,
            "variants": results,
            "positive_fraction": None,
            "worst_icir": None,
        }
    positive_fraction = sum(1 for r in results if r.get("positive")) / n
    worst_icir = min(r["adj_icir"] for r in results if r.get("adj_icir") is not None)
    return {
        "direction": direction,
        "skipped_reason": None,
        "n_variants": n,
        "variants": results,
        "positive_fraction": round(positive_fraction, 4),
        "worst_icir": round(worst_icir, 6),
    }
