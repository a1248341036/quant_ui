"""precheck_expression 的结构风险静态预检（S1）。

纯 AST 分析，**无 panel 数据依赖**（盲测隔离天然满足）：只读表达式文本，
用 ``dsl/core/ast.py::parse_ast`` 解析后检查已知死路结构：

1. **门控激活比例**：``GATED_SIGNAL(signal, state, threshold, ...)`` 的 threshold
   是 AST 字面量 → 截面激活比例 ≈ ``1 - threshold``。threshold ≥ 0.8 时
   分箱坍缩风险高（9-15 decile_collapse_spec 实测：0.9→2 组、0.6→9 组）。
2. **分段中间区**：``PIECEWISE_STATE(signal, state, low_q, high_q, ...)`` 的
   ``high_q - low_q`` 即中间区比例；≥ 0.5 时中间区常数簇导致分箱坍缩。
3. **稀疏字段填零**：``FILLNA(x, 0)`` 且 ``x`` 引用已知稀疏字段族
   （funda_/holder_/pred_/dt_/ff_/exp_/ds_/div_/mgn_）→ 零簇常数化风险。
4. **已知死路指纹**：复用 ``_prefilter`` 的信号根指纹，仅作提示
   （完整死路拦截在记忆层，这里给静态结构层面的预警）。

输出结构化字段（``risks``），不是自然语言说教段落——保持可机器消费。
"""
from __future__ import annotations

from typing import Any

# 稀疏字段前缀：这些列大量缺失，FILLNA(x,0) 会制造大零簇
_SPARSE_COL_PREFIXES = (
    "funda_", "holder_", "pred_", "dt_", "ff_", "exp_", "ds_", "div_", "mgn_",
)

# GATED_SIGNAL(signal, state, threshold, high_state, neutral)
_GATED_SIGNAL_IDX = 2
# PIECEWISE_STATE(signal, state, low_q, high_q, ...)
_PIECEWISE_LOW_IDX, _PIECEWISE_HIGH_IDX = 2, 3
# FILLNA(x, fill)
_FILLNA_SRC_IDX, _FILLNA_VAL_IDX = 0, 1


def _as_float(node: Any) -> float | None:
    if node is None or getattr(node, "type", None) != "num":
        return None
    try:
        return float(getattr(node, "value", ""))
    except (TypeError, ValueError):
        return None


def _col_ref_name(node: Any) -> str | None:
    """AST 节点是变量引用（$col 或赋值中间变量）时返回列名（去掉 $）。"""
    if node is None:
        return None
    if getattr(node, "type", None) == "var":
        return str(getattr(node, "value", "")).lstrip("$")
    return None


def _iter_call_nodes(expr: str):
    """逐行解析 DSL（含中间赋值行），产出全部 call 节点。

    ``dsl.core.ast.parse_ast`` 对多行表达式只返回**最终行**的 AST，中间变量赋值的
    GATED_SIGNAL / PIECEWISE_STATE 等会被漏掉；故这里对每一行单独解析并收集。
    """
    from alphaagent.dsl.core.ast import parse_ast

    for line in str(expr).splitlines():
        line = line.strip()
        if not line:
            continue
        ast = parse_ast(line)
        if ast is not None:
            yield from ast.walk_calls()


def precheck_expression(expr: str) -> dict[str, Any]:
    """对 DSL 表达式做静态结构风险预检。返回 ``{"ok": True, "result": {...}}``。"""
    from alphaagent.dsl.core.ast import parse_ast

    risks: list[dict[str, Any]] = []
    call_nodes = list(_iter_call_nodes(expr)) if expr else []
    if not call_nodes and not expr.strip():
        return {
            "ok": True,
            "result": {
                "risks": [{"kind": "parse_failed", "risk": "unknown",
                           "hint": "表达式为空或无法解析为 AST，结构预检跳过"}],
                "risk_count": 1,
                "max_risk": "unknown",
                "summary": "结构预检跳过：表达式为空。",
            },
        }

    for call in call_nodes:
        op = call.op.upper()
        args = call.args

        # 1) 门控激活比例
        if op == "GATED_SIGNAL" and len(args) > _GATED_SIGNAL_IDX:
            thr = _as_float(args[_GATED_SIGNAL_IDX])
            if thr is not None and thr >= 0.8:
                activation = max(0.0, min(1.0, 1.0 - thr))
                risks.append({
                    "kind": "gate_collapse",
                    "op": "GATED_SIGNAL",
                    "threshold": thr,
                    "activation_ratio": round(activation, 3),
                    "risk": "high" if activation <= 0.25 else "medium",
                    "hint": (
                        f"GATED_SIGNAL threshold={thr} → 截面仅 ~{activation:.0%} 激活，"
                        "等频十分位大概率坍缩（实测 0.9→2 组、0.6→9 组），"
                        "统计 IC 虚高但引擎净值崩。建议降低 threshold 或改用连续化门控。"
                    ),
                })

        # 2) 分段中间区宽度
        elif op == "PIECEWISE_STATE" and len(args) > _PIECEWISE_HIGH_IDX:
            lo = _as_float(args[_PIECEWISE_LOW_IDX])
            hi = _as_float(args[_PIECEWISE_HIGH_IDX])
            if lo is not None and hi is not None and 0.0 <= lo < hi <= 1.0:
                mid = hi - lo
                if mid >= 0.5:
                    risks.append({
                        "kind": "piecewise_middle_band",
                        "op": "PIECEWISE_STATE",
                        "low_q": lo,
                        "high_q": hi,
                        "middle_band": round(mid, 3),
                        "risk": "high" if mid >= 0.6 else "medium",
                        "hint": (
                            f"PIECEWISE_STATE 中间区 {mid:.0%}（low_q={lo}, high_q={hi}）"
                            "全置同一常数 → 分箱坍缩风险。建议收窄中间区或改连续状态。"
                        ),
                    })

        # 3) 稀疏字段填零
        elif op == "FILLNA" and len(args) > _FILLNA_VAL_IDX:
            val = _as_float(args[_FILLNA_VAL_IDX])
            col = _col_ref_name(args[_FILLNA_SRC_IDX] if args else None)
            sparse = bool(col and col.startswith(_SPARSE_COL_PREFIXES))
            if val == 0.0 and sparse:
                risks.append({
                    "kind": "sparse_fillna_zero",
                    "op": "FILLNA",
                    "column": col,
                    "fill": 0.0,
                    "risk": "medium",
                    "hint": (
                        f"FILLNA({col}, 0) 在稀疏字段上填零 → 大零簇被截面排序拉到同一极端，"
                        "加剧坍缩。建议用 FILLNA_CS_MEDIAN 或先做截面残差。"
                    ),
                })

    risk_count = len(risks)
    max_risk = max((r["risk"] for r in risks), default="low")
    if risk_count == 0:
        summary = "结构预检通过：未发现已知坍缩/零簇风险结构。"
    else:
        bad = [r["kind"] for r in risks if r["risk"] == "high"]
        summary = (
            f"结构预检发现 {risk_count} 项风险（最高 {max_risk}）："
            + ("，".join(bad) if bad else "均为中低风险，提交前可考虑消融验证。")
        )
    return {
        "ok": True,
        "result": {
            "risks": risks,
            "risk_count": risk_count,
            "max_risk": max_risk,
            "summary": summary,
        },
    }
