"""预审：拦截"两个裸信号简单加减"的低级因子 + 同质化平滑变体动态熔断。"""
from __future__ import annotations

import re
from typing import Any

from alphaagent.dsl.core.ast import parse_ast, _parse_var_table

# 检测顶层 ADD/SUBTRACT(RANK(x), RANK(y)) 且 x/y 均为简单 TS_/$field 变换（无结构化交互算子）。
# 例外：如果某个操作数本身含 CS_RESIDUALIZE / CS_NEUTRALIZE / GATED_SIGNAL / CS_GROUP_RANK /
# DIVERGENCE_RANK / PIECEWISE_STATE / IF_THEN_ELSE / TS_CORR / TS_RANKCORR 等结构化算子，则放行。

_STRUCTURED_OPS = frozenset({
    "CS_RESIDUALIZE", "CS_NEUTRALIZE", "GATED_SIGNAL", "CS_GROUP_RANK",
    "DIVERGENCE_RANK", "PIECEWISE_STATE", "IF_THEN_ELSE",
    "TS_CORR", "TS_RANKCORR", "TS_COV", "MUTUAL_INFO_LAG",
})

_SIMPLE_TS_RE = re.compile(r"\bTS_[A-Z]+\s*\(")
_RANK_RE = re.compile(r"\bRANK\s*\(")
_CS_RANK_RE = re.compile(r"\bCS_RANK\s*\(")
_CS_ZSCORE_RE = re.compile(r"\bCS_ZSCORE\s*\(")
_ZSCORE_RE = re.compile(r"\bZSCORE\s*\(")


def _is_naive_signal_addition(expr: str) -> bool:
    """检测表达式是否为"两个裸信号简单加减"的低级因子。

    判定逻辑：
    1. 顶层（最后一行）的算子是 ADD 或 SUBTRACT
    2. 整个表达式中至少 2 个截面标准化调用（RANK/CS_RANK/CS_ZSCORE/ZSCORE）
    3. 整个表达式中不含结构化交互算子（GATED_SIGNAL / CS_RESIDUALIZE 等）
    → 返回 True 表示应拦截

    对带赋值的多行表达式，检查顶层是否 ADD/SUBTRACT，但信号计数覆盖所有行。
    """
    if not expr or not expr.strip():
        return False

    full_text = expr.strip()
    full_upper = full_text.upper()

    # 取最后一行（因子值行），跳过注释行
    lines = full_text.split("\n")
    last_line = lines[-1].strip()
    while last_line.startswith("#") and lines:
        lines = lines[:-1]
        if not lines:
            return False
        last_line = lines[-1].strip()

    # 去掉前导赋值 "var = ..."
    if "=" in last_line and not last_line.upper().startswith(("ADD(", "SUBTRACT(")):
        parts = last_line.split("=", 1)
        if len(parts) == 2:
            last_line = parts[1].strip()

    # 检查顶层是否 ADD( 或 SUBTRACT(
    last_upper = last_line.upper()
    if not (last_upper.startswith("ADD(") or last_upper.startswith("SUBTRACT(")):
        return False

    # 在整个表达式中检查截面标准化调用数量
    # RANK / CS_RANK / CS_ZSCORE / ZSCORE 都算"裸信号"标记
    signal_count = (
        len(_RANK_RE.findall(full_upper))
        + len(_CS_RANK_RE.findall(full_upper))
        + len(_CS_ZSCORE_RE.findall(full_upper))
        + len(_ZSCORE_RE.findall(full_upper))
    )
    if signal_count < 2:
        return False

    # 检查是否含结构化交互算子——含则放行
    for op in _STRUCTURED_OPS:
        if op in full_upper:
            return False

    return True


# ── 同质化平滑变体动态熔断 ──
# 目的：不针对具体信号（如隔夜/vwap），动态识别"同一信号根结构 + 套平滑"的连续重复，
# 达阈值即拦截评估，强迫 LLM 换信号根结构。阈值由 research_spec.homogenization_policy 控制。
#
# 信号根指纹 = 表达式中所有算子去掉平滑/截面归一/基础运算后剩下的核心信号算子
#              + 引用的 $ 字段集合（去掉平滑/归一内部包裹的字段）。
# 例如 vwap 反转 `dev = DIVIDE(SUBTRACT($adj_close, $adj_vwap), $adj_vwap); sm = WMA(dev, 6); CS_ZSCORE(sm)`
#   → 信号根指纹 = (frozenset(), frozenset({"$adj_close", "$adj_vwap"}))
#      （DIVIDE/SUBTRACT 是基础运算剔除，WMA 是平滑剔除，CS_ZSCORE 是归一剔除；
#       核心算子为空但字段集合区分了 vwap 反转 vs 隔夜因子）
# 隔夜因子 `overnight = DIVIDE(SUBTRACT($adj_open, DELAY($adj_close,1)), DELAY($adj_close,1)); sm = WMA(overnight, 6); CS_ZSCORE(sm)`
#   → 信号根指纹 = (frozenset(), frozenset({"$adj_open", "$adj_close"}))
# 两者字段集合不同 → 不同信号根，不会互相误判；但同族参数变体（换窗口）字段相同 → 熔断。

# 平滑算子（与 dsl/core/ast.py SMOOTHING_OPS 同源）
_SMOOTHING_OPS = frozenset({
    "EMA", "WMA", "TS_MEAN", "TS_MEDIAN", "TS_DECAY_LINEAR", "SMA",
})
# 截面归一/中性化（不算信号算子）
_NORMALIZE_OPS = frozenset({
    "RANK", "CS_RANK", "CS_ZSCORE", "CS_WINSORIZE", "CS_DEMEAN",
    "CS_QUANTILE", "CS_NEUTRALIZE", "ZSCORE", "WINSORIZE",
    "DEMEAN", "QUANTILE", "NORMALIZE", "CS_RESIDUALIZE", "CS_GROUP_RANK",
})
# 基础运算（不算信号算子）
_BASIC_OPS = frozenset({
    "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "NEG", "ABS", "LOG", "EXP",
    "MAX", "MIN", "MAXIMUM", "MINIMUM", "DELAY", "FILLNA",
})

_OP_NAME_RE = re.compile(r"\b([A-Z_][A-Z0-9_]+)\s*\(")
_FIELD_RE = re.compile(r"\$([a-zA-Z_][a-zA-Z0-9_]*)")

# 辅助字段（中性化/残差化/分组用，不是信号源）——从信号根指纹剔除，避免因加不同中性化键
# 而误判为不同信号根。如 vwap 反转套 CS_RESIDUALIZE(..., LOG($float_cap)) 不应因 $float_cap
# 而与不带中性化的版本被判不同族。
_AUX_FIELDS = frozenset({
    "$float_cap", "$amount", "$industry_sw_l1", "$industry_sw_l2",
    "$industry_sw_l3", "$industry_em",
})

# 中性化键别名：外层变换签名里把常见中性化键归一成稳定短签名，
# 使 CS_NEUTRALIZE(x, $float_cap) 与 CS_NEUTRALIZE(x, CS_BUCKET(LOG($float_cap), 10))
# 视为同一中性化维度（市值），而市值 vs 行业视为不同维度。
_AUX_KEY_ALIASES = {
    "$float_cap": "cap",
    "$industry_sw_l1": "ind_l1",
    "$industry_sw_l2": "ind_l2",
    "$industry_sw_l3": "ind_l3",
    "$industry_em": "ind_em",
}

# 剥外层包装（归一化/平滑/符号）——信号根指纹与外层变换签名共用
_STRIP_OPS = _NORMALIZE_OPS | _SMOOTHING_OPS | {"NEG", "ABS"}


def _ast_signal_fingerprint(expr: str) -> str:
    """基于 AST 的信号根指纹（归一化结构字符串）。

    比 _signal_fingerprint（正则版）更精确：按真实树拓扑递归归一化，无误匹配。
    核心改进：先剥掉外层归一化/平滑/基础运算包装，只对**信号根子树**算指纹。
    这样 CS_ZSCORE(CS_WINSORIZE(WMA(dev,5))) 与 CS_ZSCORE(NEG(WMA(dev,5))) 都剥到 dev，
    指纹相同——它们本质是同一个 vwap 反转信号根，只是归一化层排列不同。

    归一化规则：
    - 外层归一化（CS_ZSCORE/CS_WINSORIZE/RANK/CS_RESIDUALIZE/...）→ 剥掉
    - 外层平滑（WMA/EMA/TS_MEAN/...）→ 剥掉（信号根在平滑之下）
    - 外层基础运算（NEG/ABS）→ 剥掉（符号/绝对值不改信号本质）
    - 信号根子树：数字 → N，$ 字段保留原名（辅助字段 → AUX），中间变量回溯，
      平滑算子统一成 SMOOTH（让换平滑算子的变体指纹相同）

    返回归一化结构字符串。空串表示解析失败（回落到正则版）。

    示例：
      CS_ZSCORE(WMA(dev,5)) where dev=SUBTRACT($adj_close,$adj_vwap)
        → 剥 CS_ZSCORE+WMA → 信号根 = SUBTRACT($adj_close,$adj_vwap)
        → "(SUBTRACT,$adj_close,$adj_vwap)"
      CS_ZSCORE(NEG(WMA(dev,10))) 同 dev
        → 剥 CS_ZSCORE+NEG+WMA → 信号根 = SUBTRACT($adj_close,$adj_vwap)
        → "(SUBTRACT,$adj_close,$adj_vwap)"  ← 同指纹，熔断
      CS_ZSCORE(WMA(overnight,6)) where overnight=SUBTRACT($adj_open,DELAY($adj_close,1))
        → 信号根 = SUBTRACT($adj_open,(DELAY,$adj_close,N))
        → "(SUBTRACT,$adj_open,(DELAY,$adj_close,N))"  ← 字段不同，不熔断
    """
    if not expr:
        return ""
    ast = parse_ast(expr)
    if ast is None:
        return ""
    var_table = _parse_var_table(expr) if "\n" in expr else {}

    def _resolve_var(node):
        """回溯中间变量到赋值表达式的 AST。"""
        seen: set[str] = set()
        while node is not None and node.type == "var" and node.value in var_table:
            if node.value in seen:
                return None
            seen.add(node.value)
            node = parse_ast(var_table[node.value])
        return node

    # 剥外层归一化/平滑/基础运算（NEG/ABS），找信号根子树
    signal_node = ast
    while signal_node is not None and signal_node.is_call() and signal_node.op.upper() in _STRIP_OPS:
        signal_node = signal_node.args[0] if signal_node.args else None
        signal_node = _resolve_var(signal_node)
    if signal_node is None:
        return ""

    def _norm(node) -> str:
        if node is None:
            return ""
        if node.type == "num":
            return "N"
        if node.type == "var":
            # 回溯中间变量（var_table 里的 key 是无 $ 前缀的变量名）
            if var_table and node.value in var_table:
                sub = parse_ast(var_table[node.value])
                if sub is not None:
                    return _norm(sub)
            # 剩余 var 节点都是 $ 字段（parse_ast 把 $adj_close 存成 value='adj_close'）
            # 保留字段名以区分信号源（$adj_vwap vs $adj_open），辅助字段归一成 AUX
            val = "$" + node.value
            return "AUX" if val.lower() in _AUX_FIELDS else val
        if node.type == "call":
            op = node.op.upper()
            # 平滑算子统一替换成 SMOOTH（信号根子树内部的平滑也算同族变体）
            if op in _SMOOTHING_OPS:
                op = "SMOOTH"
            parts = [op] + [_norm(a) for a in node.args]
            return "(" + ",".join(parts) + ")"
        if node.type == "binop":
            parts = [node.op] + [_norm(a) for a in node.args]
            return "(" + ",".join(parts) + ")"
        if node.type == "unary":
            return "(" + node.op + "," + _norm(node.args[0]) + ")"
        return node.type

    return _norm(signal_node)


def _neutralize_key(node: Any) -> str:
    """中性化键归一化：市值桶/行业/其他字段 → 稳定短签名。

    处理 CS_NEUTRALIZE/CS_RESIDUALIZE/CS_GROUP_RANK 的第二参数：
    - $float_cap / LOG($float_cap) / CS_BUCKET(LOG($float_cap), N) → "cap"
    - $industry_sw_l1 → "ind_l1"（其余行业键同理）
    - 其他 $ 字段 → 保留字段名；其他算子 → 算子名
    """
    if node is None:
        return "none"
    if node.type == "var":
        val = "$" + node.value
        return _AUX_KEY_ALIASES.get(val.lower(), val)
    if node.type == "call":
        op = node.op.upper()
        # 剥掉市值桶/对数/延迟等包装，只留中性化键本体
        if op in ("CS_BUCKET", "LOG", "DELAY", "NEG", "ABS") and node.args:
            return _neutralize_key(node.args[0])
        return op
    if node.type == "num":
        return "N"
    return node.type


def _outer_transform_signature(expr: str) -> str:
    """信号根之上的变换链签名（从外到内），区分中性化/截面变换变体。

    与 _ast_signal_fingerprint 互补：指纹只认信号根（剥掉所有外层变换），
    本签名记录被剥掉的外层变换链。同根 + 同外层变换 = 参数微调（熔断）；
    同根 + 不同外层变换（换中性化键/换截面变换）= 正交化探索（放行）。

    示例（同一信号根 x）：
      CS_NEUTRALIZE(CS_ZSCORE(CS_WINSORIZE(x, 0.01, 0.99)), CS_BUCKET(LOG($float_cap), 10))
        → "CS_NEUTRALIZE:cap|CS_ZSCORE|CS_WINSORIZE"
      CS_NEUTRALIZE(CS_ZSCORE(CS_WINSORIZE(x, 0.01, 0.99)), $industry_sw_l1)
        → "CS_NEUTRALIZE:ind_l1|CS_ZSCORE|CS_WINSORIZE"  ← 与市值中性不同，不累计
      RANK(CS_WINSORIZE(x, 0.01, 0.99))
        → "RANK|CS_WINSORIZE"  ← 与 ZSCORE 变体不同，不累计
      WMA(x, 8) 与 WMA(x, 4)（同外层）
        → "SMOOTH" == "SMOOTH"  ← 同外层，换窗口仍累计熔断
    """
    if not expr:
        return ""
    ast = parse_ast(expr)
    if ast is None:
        return ""
    var_table = _parse_var_table(expr) if "\n" in expr else {}

    def _resolve_var(node):
        seen: set[str] = set()
        while node is not None and node.type == "var" and node.value in var_table:
            if node.value in seen:
                return None
            seen.add(node.value)
            node = parse_ast(var_table[node.value])
        return node

    parts: list[str] = []
    node = ast
    while node is not None and node.is_call() and node.op.upper() in _STRIP_OPS:
        op = node.op.upper()
        if op in _SMOOTHING_OPS:
            parts.append("SMOOTH")
        elif op in ("CS_NEUTRALIZE", "CS_RESIDUALIZE", "CS_GROUP_RANK") and len(node.args) >= 2:
            parts.append(f"{op}:{_neutralize_key(node.args[1])}")
        else:
            parts.append(op)
        node = node.args[0] if node.args else None
        node = _resolve_var(node)
    return "|".join(parts)


def _signal_fingerprint(expr: str) -> tuple[frozenset[str], frozenset[str]]:
    """提取表达式的信号根指纹（核心信号算子集合, 信号字段集合）。

    用于同质化检测：两个表达式信号根指纹相同 + 都含平滑 → 同质化变体。
    返回 (signal_ops, fields) 二元组，便于精确比较。
    - signal_ops: 去掉平滑/归一/基础运算后的核心算子（如 CHIP_MASS_ASYM/DIVERGENCE_RANK）
    - fields: 表达式引用的 $ 字段集合，剔除辅助字段（$float_cap/$amount/$industry_* 等中性化键）
      保留信号源字段（如 $adj_vwap/$adj_open/$adj_close/$volume/$ret）
    两者都为空 → 纯平滑/归一组合（无信号根），不参与熔断。
    """
    if not expr:
        return (frozenset(), frozenset())
    upper = expr.upper()
    ops = set(_OP_NAME_RE.findall(upper))
    signal_ops = ops - _SMOOTHING_OPS - _NORMALIZE_OPS - _BASIC_OPS
    fields = frozenset(f"${m.lower()}" for m in _FIELD_RE.findall(expr)) - _AUX_FIELDS
    return (frozenset(signal_ops), fields)


# 兼容旧调用名（_record_eval_signature 已改用 _signal_fingerprint）
def _signal_ops_fingerprint(expr: str) -> frozenset[str]:
    """旧接口：只返回信号算子集合（不含字段）。已弃用，保留兼容。"""
    return _signal_fingerprint(expr)[0]


def _has_smoothing(expr: str) -> bool:
    """表达式是否含平滑算子。"""
    if not expr:
        return False
    ops = set(_OP_NAME_RE.findall(expr.upper()))
    return bool(ops & _SMOOTHING_OPS)


def _homogenization_block(
    expr: str,
    recent_evals: list[dict[str, Any]] | None,
    *,
    max_consecutive: int = 3,
    window_size: int = 10,
    enabled: bool = True,
    max_outer_variants: int = 6,
) -> dict[str, Any] | None:
    """同质化平滑变体预检：滑动窗口内 ≥max_consecutive 次"同一信号根+同外层变换+含平滑"→ 拦截评估。

    判定逻辑（动态，不针对具体信号）：
    1. 当前表达式含平滑算子；
    2. 提取当前表达式的信号根指纹（核心算子集合 + 引用字段集合）与外层变换签名
       （信号根之上的归一化/中性化/截面变换链）；
    3. 从最近评估倒序遍历最近 window_size 条，统计"信号根指纹相同 + 外层变换签名相同
       + 含平滑"的次数（不因中间夹了其他根/非平滑条目而中断——防 LLM 换根轮换绕过熔断器）；
    4. 同根同外层累计次数 ≥ max_consecutive → 返回拦截 result（ok=False）。
    5. 防绕过：同根但外层变换不同的变体（换中性化键/换截面变换）不累计，但计入变体
       多样性；窗口内同根变体数 ≥ max_outer_variants → 同样拦截，防 LLM 用"换中性化键"
       无限微调同一信号根。

    信号根指纹 = (核心信号算子, 引用字段)。vwap 反转和隔夜因子字段不同 → 不会互相误判；
    同族参数变体（换 WMA 窗口）字段相同 → 熔断。

    与 AntiHomogenizationDiagnostic 的区别：
    - 诊断层熔断器要求换手>0.50（只拦高换手同质化），且只 block_submit（仍允许评估）；
    - 本预检在 dispatch 层评估前拦截，不依赖换手，直接拒绝评估同质化变体，
      强迫 LLM 换信号根结构（不是换平滑窗口）。

    参数：
        expr: 当前表达式
        recent_evals: 最近评估签名列表（_record_eval_signature 记录的 dict）
        max_consecutive: 窗口内同根同外层累计次数阈值（默认 3）
        window_size: 滑动窗口大小（默认 10，只统计最近 N 次评估）
        enabled: 开关（research_spec.homogenization_policy.enabled）
        max_outer_variants: 同根外层变换变体多样性上限（默认 6，防换中性化键绕过）
    """
    if not enabled or not expr or not recent_evals:
        return None
    if not _has_smoothing(expr):
        return None
    # 优先用 AST 指纹（精确嵌套结构 + 平滑算子统一 + 字段名保留），
    # 解析失败回落到正则版 _signal_fingerprint
    current_ast = _ast_signal_fingerprint(expr)
    if current_ast:
        current_sig: Any = current_ast
        sig_kind = "ast"
    else:
        current_sig = _signal_fingerprint(expr)
        # 信号根为空（无核心算子且无字段）→ 不参与熔断，避免误伤
        if not current_sig[0] and not current_sig[1]:
            return None
        sig_kind = "regex"
    current_outer = _outer_transform_signature(expr)

    # 滑动窗口内同根累计：不因中间夹了其他根/非平滑条目而中断
    count = 0
    outer_variants: set[str] = set()
    for item in reversed(recent_evals[-window_size:]):
        if not isinstance(item, dict):
            continue
        if not item.get("has_smoothing"):
            continue
        # 优先比对 AST 指纹，其次比对正则指纹
        prev_ast = item.get("signal_fingerprint_ast")
        prev_sig = item.get("signal_fingerprint")
        matched = False
        if sig_kind == "ast" and isinstance(prev_ast, str) and prev_ast:
            matched = prev_ast == current_sig
        elif sig_kind == "regex" and isinstance(prev_sig, tuple):
            matched = prev_sig == current_sig
        if not matched:
            continue
        # 同根但外层变换不同 → 正交化探索（换中性化键/换截面变换），不累计
        prev_outer = item.get("outer_transform_signature")
        if prev_outer is not None and prev_outer != current_outer:
            outer_variants.add(prev_outer)
            continue
        count += 1

    if count >= max_consecutive:
        if sig_kind == "ast":
            sig_desc = current_sig[:200]
        else:
            ops_str = ", ".join(sorted(current_sig[0])) or "(无核心算子)"
            fields_str = ", ".join(sorted(current_sig[1])) or "(无字段)"
            sig_desc = f"算子: {ops_str}; 字段: {fields_str}"
        return {
            "ok": False,
            "error": (
                f"homogenization_smoothing_block: 最近 {window_size} 次评估中已有 {count} 次"
                f"在相同信号根（{sig_desc}）上套平滑算子做变体。"
                f"平滑只降换手不改信号本质，继续微调窗口纯属浪费算力。"
                f"请更换信号根结构（换数据源/换核心算子/换机制），不要再用同一信号根+平滑。"
                f"不用平滑的降换手动作（详见行为准则 rule 2）："
                f"  ① 换慢源（基本面 $funda_* ρ_f≈0.95+ / 筹码 CHIP_* ρ_f≈0.85+ / 周线 @1w ρ_f≈0.95+）；"
                f"  ② 信号积分 F'=λ·F+(1-λ)·DELAY(F,1)，λ≤(0.85-ρ_f)/(1-ρ_f)"
                f"（不塌分布，优于末位 EMA/WMA）；"
                f"  ③ 截面变换 CS_ZSCORE/RANK 压尾部，不压缩分布。"
                f"ρ_f<0.6 的高频源（如 1d 反转）降换手必丢 IC——直接换源，不要硬降。"
            ),
            "error_type": "HomogenizationSmoothingBlock",
        }
    if current_outer and len(outer_variants) + 1 > max_outer_variants:
        if sig_kind == "ast":
            sig_desc = current_sig[:200]
        else:
            ops_str = ", ".join(sorted(current_sig[0])) or "(无核心算子)"
            fields_str = ", ".join(sorted(current_sig[1])) or "(无字段)"
            sig_desc = f"算子: {ops_str}; 字段: {fields_str}"
        return {
            "ok": False,
            "error": (
                f"homogenization_smoothing_block: 同一信号根（{sig_desc}）在最近 {window_size} 次评估中"
                f"已尝试 {len(outer_variants) + 1} 种外层变换变体（中性化键/截面变换），超过上限"
                f" {max_outer_variants}。换中性化键/截面变换不改变信号根本质，继续在同一根上"
                f"做正交化变体收益递减。请更换信号根结构（换数据源/换核心算子/换机制）。"
            ),
            "error_type": "HomogenizationSmoothingBlock",
        }
    return None
