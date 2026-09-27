"""precheck_expression 的结构风险静态预检（S1）。

纯 AST 分析，**无 panel 数据依赖**（盲测隔离天然满足）：只读表达式文本，
用 ``dsl/core/ast.py::parse_ast`` 解析后检查已知死路结构：

1. **门控常数簇**：``GATED_SIGNAL`` / ``PIECEWISE_STATE`` 未激活端被压成**同一个常数**
   （``neutral`` / ``mid_value``），等频十分位分箱会把重合边界 drop 掉，组数不足即塌缩。

   **2026-09-27 重新标定（实测）**：常数簇占比 → ``pd.qcut(x, 10, duplicates="drop")`` 组数

   =========  ==========  ==========
   常数簇占比   实际组数     判定(>=8)
   =========  ==========  ==========
   0%          10          OK
   10%         10          OK
   20%          9          OK
   **30%**     **8**       **OK（临界点）**
   40%          7          COLLAPSE
   50%          6          COLLAPSE
   60%          5          COLLAPSE
   70%          4          COLLAPSE
   80%          3          COLLAPSE
   85%          2          COLLAPSE
   =========  ==========  ==========

   推论：``GATED_SIGNAL`` 的合法 threshold ∈ [0.5, 0.85] 对应常数簇 15%~50%，
   **每一次合法调用都必然塌缩**（旧版只在 threshold>=0.8 报警，理由是「0.6→9 组」，
   该实测值有误——0.6 实际是 5 组）。故本检查改为**按常数簇判定，与 threshold 无关**。
   ``PIECEWISE_STATE`` 中间区宽度临界点同为 0.3。

   注：``CS_ZSCORE`` / ``RANK`` 等单调变换**无法**打散常数簇（线性变换后仍同值），
   旧版 behavior_rules 教的「末位包 CS_ZSCORE」在数学上无效。

2. **分段中间区**：``PIECEWISE_STATE(signal, state, low_q, high_q, ...)`` 的
   ``high_q - low_q`` 即中间常数区占比；> 0.3 时组数不足。
3. **稀疏字段填零**：``FILLNA(x, 0)`` 且 ``x`` 引用已知稀疏字段族
   （funda_/holder_/pred_/dt_/ff_/exp_/ds_/div_/mgn_）→ 零簇常数化风险。
4. **已知死路指纹**：复用 ``_prefilter`` 的信号根指纹，仅作提示
   （完整死路拦截在记忆层，这里给静态结构层面的预警）。

输出结构化字段（``risks`` + ``blocked``），不是自然语言说教段落——保持可机器消费。
``blocked=True`` 表示存在**必然导致 stage_one 失败**的结构（high 风险），
由 tool 层硬拦并回传可直接抄写的替代写法。
"""
from __future__ import annotations

import os

from typing import Any

# 稀疏字段前缀：这些列大量缺失，FILLNA(x,0) 会制造大零簇
_SPARSE_COL_PREFIXES = (
    "funda_", "holder_", "pred_", "dt_", "ff_", "exp_", "ds_", "div_", "mgn_",
)

# GATED_SIGNAL(signal, state, threshold, high_state, neutral)
_GATED_SIGNAL_IDX = 2
_GATED_SIGNAL_NEUTRAL_IDX = 4
# PIECEWISE_STATE(signal, state, low_q, high_q, low_sign, high_sign, mid_value)
_PIECEWISE_LOW_IDX, _PIECEWISE_HIGH_IDX = 2, 3
_PIECEWISE_MID_IDX = 6
# FILLNA(x, fill)
_FILLNA_SRC_IDX, _FILLNA_VAL_IDX = 0, 1

# 分箱安全边界（实测标定，见模块 docstring）：常数簇占比 / 中间区宽度上限。
# 超过即 < 8 组，stage_one 的 decile 门槛必失败。
_MAX_CONSTANT_CLUSTER = 0.30

# 逃生阀：置 0 时本预检只警告不硬拦（回退旧行为，供对照实验用）。
_BLOCK_ENV = "ALPHA_PRECHECK_BLOCK_COLLAPSE"


def _block_enabled() -> bool:
    return str(os.environ.get(_BLOCK_ENV, "1")).strip().lower() not in ("0", "false", "no")


def _is_finite_constant(node: Any) -> bool:
    """节点是否"有限常数"（num 字面量）。None（未传参）按默认常数处理 → True。"""
    if node is None:
        return True
    if getattr(node, "type", None) != "num":
        return False
    val = _as_float(node)
    if val is None:
        return False
    import math as _math

    return _math.isfinite(val)


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

        # 1) 门控常数簇（判定依据是常数簇占比，与 threshold 松紧无关）
        if op == "GATED_SIGNAL":
            neutral_node = args[_GATED_SIGNAL_NEUTRAL_IDX] if len(args) > _GATED_SIGNAL_NEUTRAL_IDX else None
            if _is_finite_constant(neutral_node):
                thr = _as_float(args[_GATED_SIGNAL_IDX]) if len(args) > _GATED_SIGNAL_IDX else None
                # 常数簇占比 = 未激活比例（threshold 已知时用精确值，否则按合法区间估）
                const_share = (1.0 - thr) if (thr is not None and 0.0 <= thr <= 1.0) else None
                share_txt = f"{const_share:.0%}" if const_share is not None else "15%~50%（合法 threshold 区间）"
                risks.append({
                    "kind": "gate_collapse",
                    "op": "GATED_SIGNAL",
                    "threshold": thr,
                    "neutral": _as_float(neutral_node) if neutral_node is not None else 0.0,
                    "constant_cluster_share": round(const_share, 3) if const_share is not None else None,
                    "risk": "high",
                    "blocked": True,
                    "hint": (
                        f"GATED_SIGNAL 把未激活端压成同一个常数，常数簇约 {share_txt}；"
                        f"等频十分位分箱的组数 = 10×激活率，实测 0.5/0.6/0.8/0.85 → 6/5/3/2 组，"
                        f"而 stage_one 要求 ≥ 8 组（临界点 threshold=0.3，即常数簇 30%）。"
                        "CS_ZSCORE/RANK 是单调变换，无法打散常数簇。"
                        "请改用 SOFT_GATE(signal, state, strength=0.5) 做连续加权，"
                        "或 CS_GROUP_RANK(signal, CS_BUCKET(state, 5)) 做组内排名。"
                    ),
                })

        # 2) 分段中间区宽度（中间区全置 mid_value → 常数簇）
        elif op == "PIECEWISE_STATE" and len(args) > _PIECEWISE_HIGH_IDX:
            lo = _as_float(args[_PIECEWISE_LOW_IDX])
            hi = _as_float(args[_PIECEWISE_HIGH_IDX])
            if lo is not None and hi is not None and 0.0 <= lo < hi <= 1.0:
                mid = hi - lo
                mid_node = args[_PIECEWISE_MID_IDX] if len(args) > _PIECEWISE_MID_IDX else None
                const_mid = _is_finite_constant(mid_node)
                # 容差防浮点误差（如 0.65-0.35=0.30000000000000004 恰好等于临界值仍算 OK）
                if mid > _MAX_CONSTANT_CLUSTER + 1e-6 and const_mid:
                    risks.append({
                        "kind": "piecewise_middle_band",
                        "op": "PIECEWISE_STATE",
                        "low_q": lo,
                        "high_q": hi,
                        "middle_band": round(mid, 3),
                        "risk": "high",
                        "blocked": True,
                        "hint": (
                            f"PIECEWISE_STATE 中间区 {mid:.0%}（low_q={lo}, high_q={hi}）被置为同一常数，"
                            f"常数簇 > 30% → 分箱组数不足 8（实测 0.4/0.6/0.7 → 7/5/4 组）。"
                            "建议收窄中间区到 0.3 以内，或改用 SOFT_GATE 做连续状态倾斜。"
                        ),
                    })
                elif mid > 0.0 and const_mid and mid > _MAX_CONSTANT_CLUSTER * 0.8:
                    risks.append({
                        "kind": "piecewise_middle_band",
                        "op": "PIECEWISE_STATE",
                        "low_q": lo,
                        "high_q": hi,
                        "middle_band": round(mid, 3),
                        "risk": "medium",
                        "blocked": False,
                        "hint": (
                            f"PIECEWISE_STATE 中间区 {mid:.0%} 置常数，已逼近 30% 临界点"
                            "（越过即分箱组数 < 8）。建议收窄或用 SOFT_GATE。"
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
    # blocked：存在必然导致 stage_one 失败的塌缩结构（high）且逃生阀未关闭。
    hard = [r for r in risks if r.get("blocked")]
    blocked = bool(hard) and _block_enabled()
    if risk_count == 0:
        summary = "结构预检通过：未发现已知坍缩/零簇风险结构。"
    else:
        bad = [r["kind"] for r in risks if r["risk"] == "high"]
        summary = (
            f"结构预检发现 {risk_count} 项风险（最高 {max_risk}）："
            + ("，".join(bad) if bad else "均为中低风险，提交前可考虑消融验证。")
        )
        if blocked:
            summary = "⛔ " + summary + "（塌缩结构将导致 stage_one 必失败，表达式已被拦截）"
    return {
        "ok": True,
        "result": {
            "risks": risks,
            "risk_count": risk_count,
            "max_risk": max_risk,
            "blocked": blocked,
            "summary": summary,
        },
    }
