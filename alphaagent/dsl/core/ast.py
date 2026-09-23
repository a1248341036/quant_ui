# -*- coding: utf-8 -*-
"""DSL 表达式 AST 解析器：独立于 parser.py，用于结构分析。

不参与 DSL→Python 编译，只做结构提取：
- 族分类（AST 根算子 → family）
- 末位子树模式检测（terminal smoothing / terminal rank）
- 结构指纹（subtree_hash → 替代 regex 版 _structure_fingerprint）

设计约束：
- 纯 Python 递归下降解析，不依赖 pyparsing（parser.py 已有，但 AST 是独立关注点）
- 解析失败返回 None，不抛异常（调用方做 None 检查）
- 多行表达式：取最终表达式行（与 parse_multi_line_expression 同口径）
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any


# ── AST 节点 ──

@dataclass(frozen=True)
class ASTNode:
    """DSL 表达式 AST 节点。

    type: "call" | "var" | "num" | "binop" | "unary"
    op: 算子名（call 时）或运算符（binop/unary 时）
    args: 子节点列表
    value: 叶子节点的值（var 名 / num 值）
    """

    type: str
    op: str = ""
    args: tuple["ASTNode", ...] = ()
    value: str = ""

    def is_call(self, *names: str) -> bool:
        """是否是调用节点且算子名在 names 中（大小写不敏感）。"""
        if self.type != "call":
            return False
        if not names:
            return True
        return self.op.upper() in {n.upper() for n in names}

    def root_op(self) -> str:
        """AST 根算子名（call 节点的 op；非 call 返回 type）。"""
        return self.op if self.type == "call" else self.type

    def walk(self) -> list["ASTNode"]:
        """前序遍历所有节点。"""
        out = [self]
        for child in self.args:
            out.extend(child.walk())
        return out

    def walk_calls(self) -> list["ASTNode"]:
        """前序遍历所有 call 节点。"""
        return [n for n in self.walk() if n.type == "call"]

    def subtree_hash(self, var_table: dict[str, str] | None = None) -> str:
        """子树结构哈希（归一化：变量→VAR，数字→N，忽略参数名）。

        与 expressions._structure_fingerprint 语义等价但更精确：
        - regex 版可能误匹配嵌套结构（如 SUBSTRING 内的算子名）
        - AST 版按真实树拓扑递归归一化，无误匹配

        用于：
        - 记忆库指纹去重（替代 _structure_fingerprint）
        - advisory 死路检测（同结构指纹已失败 ≥2 次）

        若传入 var_table，遇到变量节点时回溯到赋值表达式的 AST，
        使 `CS_ZSCORE(smooth)`（smooth=EMA(...)）与 `CS_ZSCORE(EMA(...))`
        的指纹一致——检测更多同构变体。
        """
        if self.type == "num":
            return "N"
        if self.type == "var":
            if var_table and self.value in var_table:
                sub_ast = parse_ast(var_table[self.value])
                if sub_ast is not None:
                    return sub_ast.subtree_hash(var_table)
            return "VAR"
        if self.type == "call":
            parts = [self.op.upper()] + [a.subtree_hash(var_table) for a in self.args]
            return "(" + ",".join(parts) + ")"
        if self.type == "binop":
            parts = [self.op] + [a.subtree_hash(var_table) for a in self.args]
            return "(" + ",".join(parts) + ")"
        if self.type == "unary":
            return "(" + self.op + "," + self.args[0].subtree_hash(var_table) + ")"
        return self.type


# ── 平滑 / 归一化 / 信号算子集合 ──

SMOOTHING_OPS = frozenset({
    "EMA", "WMA", "TS_MEAN", "TS_MEDIAN", "TS_DECAY_LINEAR",
    "SMA",
})

NORMALIZE_OPS = frozenset({
    "RANK", "CS_RANK", "CS_ZSCORE", "CS_WINSORIZE", "CS_DEMEAN",
    "CS_QUANTILE", "CS_NEUTRALIZE", "ZSCORE", "WINSORIZE",
    "DEMEAN", "QUANTILE", "NORMALIZE",
})

_SIGNAL_OP_FAMILY: dict[str, str] = {
    # 动量/反转
    "TS_PCTCHANGE": "momentum", "TS_DELTA": "momentum", "DELTA": "momentum",
    "TS_RANK": "momentum", "NEG": "reversal",
    # 波动率
    "TS_STD": "volatility", "TS_VAR": "volatility",
    "TS_SKEWNESS": "volatility", "TS_KURTOSIS": "volatility",
    # 量价关系
    "TS_CORR": "correlation", "TS_COV": "correlation",
    # 筹码
    "CHIP_ENTROPY": "chip", "CHIP_DAILY": "chip",
    "CHIP_PEAK_LOC": "chip", "CHIP_MASS_ASYM": "chip",
    "CHIP_DISTRIBUTION_SKEW": "chip", "CHIP_DISTRIBUTION_KURT": "chip",
    "CHIP_WEIGHTED_PRICE": "chip", "CHIP_CONCENTRATION": "chip",
    # 拥挤
    "CROWD_MEAN_RATIO": "crowd", "CROWD_RANK_WEIGHTED": "crowd",
    # 交互结构
    "GATED_SIGNAL": "gated", "PIECEWISE_STATE": "piecewise",
    "DIVERGENCE_RANK": "divergence", "CS_GROUP_RANK": "group_rank",
    "CS_RESIDUALIZE": "residual", "IF_THEN_ELSE": "conditional",
    # 量能
    "TS_SUM": "volume",
    # K线形态
    "KLINE_GEOMETRY": "kline",
    # 基本面
    "FILLNA": "fundamental", "FILLNA_CS_MEDIAN": "fundamental",
    # 安全除法（CLV 类一字板防护）
    "SAFE_DIVIDE": "safe_divide",
    # 流动性
    "MAXIMUM": "liquidity",
}


# ── 解析器 ──

def _parse_var_table(expression: str) -> dict[str, str]:
    """从多行表达式中提取变量赋值表 {var_name: expr_text}。

    用于 terminal_smoothing 等需要回溯中间变量的分析。
    """
    table: dict[str, str] = {}
    if not expression:
        return table
    for line in str(expression).split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$', line)
        if m:
            table[m.group(1)] = m.group(2).strip()
    return table


def parse_ast(expression: str) -> ASTNode | None:
    """将 DSL 表达式文本解析为 AST 树。

    支持多行表达式（中间变量赋值 → 最终表达式）。
    返回最终表达式的 AST；解析失败返回 None。
    """
    if not expression:
        return None
    text = str(expression).strip()
    if not text:
        return None

    # 多行表达式：取最终表达式行（与 parse_multi_line_expression 同口径）
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    final_expr = None
    for line in lines:
        m = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(.+)$', line)
        if m:
            final_expr = m.group(2).strip()
        else:
            final_expr = line

    if not final_expr:
        return None

    # 去掉外层冗余括号
    final_expr = _strip_outer_parens(final_expr)
    if not final_expr:
        return None

    try:
        return _parse_expr(final_expr)
    except Exception:
        return None


def _strip_outer_parens(text: str) -> str:
    """去掉包裹整个表达式的外层括号。"""
    while text.startswith("(") and text.endswith(")"):
        depth = 0
        ok = True
        for i, c in enumerate(text):
            if c == "(":
                depth += 1
            elif c == ")":
                depth -= 1
                if depth == 0 and i < len(text) - 1:
                    ok = False
                    break
        if ok:
            text = text[1:-1].strip()
        else:
            break
    return text


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _parse_expr(text: str) -> ASTNode:
    """递归下降解析单个表达式为 AST。"""
    text = text.strip()
    if not text:
        raise ValueError("empty expression")

    # 中缀运算符（按优先级从低到高）
    for op_pattern in (r'\|\|', r'&&', r'==|!=', r'>=|<=|>|<', r'[+\-]', r'[*/]'):
        result = _split_binop(text, op_pattern)
        if result is not None:
            left, op, right = result
            return ASTNode(type="binop", op=op, args=(_parse_expr(left), _parse_expr(right)))

    # 一元负号（不是数字的负号）
    if text.startswith("-") and not _is_number(text):
        inner = text[1:].strip()
        if inner:
            return ASTNode(type="unary", op="-", args=(_parse_expr(inner),))

    # 函数调用：NAME(args)
    m = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.*)\)$', text, re.DOTALL)
    if m:
        op_name = m.group(1)
        args_str = m.group(2).strip()
        args = _parse_args(args_str)
        return ASTNode(type="call", op=op_name, args=tuple(args))

    # 变量：$name 或 name 或 name@freq
    if re.match(r'^\$?[a-zA-Z_][a-zA-Z0-9_]*(@[a-zA-Z0-9_]+)?$', text):
        return ASTNode(type="var", value=text.lstrip("$"))

    # 数字（含科学计数法）
    if _is_number(text):
        return ASTNode(type="num", value=text)

    # 布尔字面量
    if text.lower() in ("true", "false"):
        return ASTNode(type="var", value=text.lower())

    # 字符串字面量
    if (text.startswith("'") and text.endswith("'")) or \
       (text.startswith('"') and text.endswith('"')):
        return ASTNode(type="var", value=text)

    raise ValueError(f"cannot parse: {text[:80]}")


def _parse_args(args_str: str) -> list[ASTNode]:
    """解析函数参数列表（逗号分隔，处理嵌套括号和字符串）。"""
    if not args_str:
        return []
    args: list[ASTNode] = []
    depth = 0
    current: list[str] = []
    in_string = False
    string_char = ""

    for c in args_str:
        if in_string:
            current.append(c)
            if c == string_char:
                in_string = False
            continue
        if c in ("'", '"'):
            in_string = True
            string_char = c
            current.append(c)
            continue
        if c == "(":
            depth += 1
            current.append(c)
        elif c == ")":
            depth -= 1
            current.append(c)
        elif c == "," and depth == 0:
            arg_str = "".join(current).strip()
            if arg_str:
                args.append(_parse_expr(arg_str))
            current = []
        else:
            current.append(c)

    if current:
        arg_str = "".join(current).strip()
        if arg_str:
            args.append(_parse_expr(arg_str))

    return args


def _split_binop(text: str, op_pattern: str) -> tuple[str, str, str] | None:
    """在文本中找最外层的中缀运算符，返回 (left, op, right)。

    从右向左找（左结合），跳过括号内和字符串内的运算符。
    跳过一元运算符（前面是运算符或左括号或开头的 +/-）。
    """
    depth = 0
    in_string = False
    string_char = ""
    i = len(text) - 1

    while i >= 0:
        c = text[i]
        if in_string:
            if c == string_char:
                in_string = False
            i -= 1
            continue
        if c in ("'", '"'):
            in_string = True
            string_char = c
            i -= 1
            continue
        if c == ")":
            depth += 1
            i -= 1
            continue
        if c == "(":
            depth -= 1
            i -= 1
            continue
        if depth == 0:
            for op_len in (2, 1):
                if i + op_len <= len(text):
                    candidate = text[i:i + op_len]
                    if re.match(f'^{op_pattern}$', candidate):
                        left = text[:i].strip()
                        # 跳过一元运算符：前面是空/运算符/左括号
                        if not left or left[-1] in "(+-*/&|<>=":
                            break
                        right = text[i + op_len:].strip()
                        if right:
                            return (left, candidate, right)
        i -= 1
    return None


# ── 结构分析 API ──

def terminal_smoothing(expression: str) -> dict[str, Any]:
    """检测末位平滑子树。

    从 AST 根开始向下扫描：
    - 跳过归一化层（RANK/CS_ZSCORE/CS_WINSORIZE/...）
    - 遇到平滑算子（EMA/WMA/TS_MEAN/...）→ 计数 + 继续向下
    - 遇到信号算子 → 停止

    返回::
        {
            "has_terminal_smoothing": bool,   # 末位（归一化之下）是平滑算子
            "smoothing_layers": int,          # 连续平滑层数
            "smoothing_ops": list[str],       # 出现的平滑算子名
            "root_op": str,                   # AST 根算子
            "signal_root": str,               # 剥掉归一化+平滑后的信号根算子
            "total_smoothing_ops": int,       # 全表达式平滑算子总数
            "collapse_risk": str,             # "high" / "medium" / "low" / "none"
        }
    """
    ast = parse_ast(expression)
    if ast is None:
        return _empty_smoothing()

    var_table = _parse_var_table(expression)

    def _resolve_var(node: ASTNode) -> ASTNode | None:
        """遇到变量节点时，回溯到变量赋值的 AST。"""
        seen: set[str] = set()
        while node is not None and node.type == "var" and node.value in var_table:
            if node.value in seen:
                return None  # 循环引用保护
            seen.add(node.value)
            node = parse_ast(var_table[node.value])
        return node

    smoothing_ops_found: list[str] = []
    smoothing_layers = 0

    node: ASTNode | None = ast
    while node is not None:
        if node.is_call() and node.op.upper() in NORMALIZE_OPS:
            node = node.args[0] if node.args else None
            node = _resolve_var(node)
            continue
        if node.is_call() and node.op.upper() in SMOOTHING_OPS:
            smoothing_ops_found.append(node.op.upper())
            smoothing_layers += 1
            node = node.args[0] if node.args else None
            node = _resolve_var(node)
            continue
        break

    root_op = ast.root_op()
    has_terminal = smoothing_layers > 0

    # 信号根 = 剥掉归一化和平滑后的第一个算子
    signal_node: ASTNode | None = ast
    while signal_node is not None:
        if signal_node.is_call() and signal_node.op.upper() in (NORMALIZE_OPS | SMOOTHING_OPS):
            signal_node = signal_node.args[0] if signal_node.args else None
            signal_node = _resolve_var(signal_node)
            continue
        break
    signal_root = signal_node.root_op() if signal_node else root_op

    # 全表达式平滑算子计数（含变量回溯）
    total_smoothing = len(all_smoothing_ops(expression))

    if has_terminal and smoothing_layers >= 2:
        risk = "high"
    elif has_terminal:
        risk = "medium"
    elif total_smoothing >= 2:
        risk = "medium"
    elif total_smoothing >= 1:
        risk = "low"
    else:
        risk = "none"

    return {
        "has_terminal_smoothing": has_terminal,
        "smoothing_layers": smoothing_layers,
        "smoothing_ops": smoothing_ops_found,
        "root_op": root_op,
        "signal_root": signal_root,
        "total_smoothing_ops": total_smoothing,
        "collapse_risk": risk,
    }


def _empty_smoothing() -> dict[str, Any]:
    return {
        "has_terminal_smoothing": False,
        "smoothing_layers": 0,
        "smoothing_ops": [],
        "root_op": "",
        "signal_root": "",
        "total_smoothing_ops": 0,
        "collapse_risk": "none",
    }


def classify_family_ast(expression: str) -> str:
    """基于 AST 信号根算子分类信号族。

    规则：
    1. 剥掉归一化层（RANK/CS_ZSCORE/...）——含变量回溯
    2. 剥掉平滑层（EMA/WMA/TS_MEAN/...）——含变量回溯
    3. 取信号根算子 → 按算子语义分类
    4. 信号根是 panel 列引用 → 按列名前缀分类（$close→price_volume, $funda_*→fundamental）
    5. 全是归一化+平滑（无信号算子）→ "smoothing"
    6. 空表达式 → ""
    """
    if not expression or not expression.strip():
        return ""
    ast = parse_ast(expression)
    if ast is None:
        return ""

    var_table = _parse_var_table(expression)

    def _resolve_var(node: ASTNode) -> ASTNode | None:
        seen: set[str] = set()
        while node is not None and node.type == "var" and node.value in var_table:
            if node.value in seen:
                return None
            seen.add(node.value)
            node = parse_ast(var_table[node.value])
        return node

    node: ASTNode | None = ast
    while node is not None:
        node = _resolve_var(node)
        if node is None:
            break
        if node.is_call() and (node.op.upper() in NORMALIZE_OPS or node.op.upper() in SMOOTHING_OPS):
            node = node.args[0] if node.args else None
            continue
        break

    if node is None:
        return "smoothing"

    # 信号根是算子调用 → 按算子语义分类
    if node.type == "call":
        signal_op = node.op.upper()
        return _SIGNAL_OP_FAMILY.get(signal_op, "other")

    # 信号根是 panel 列引用 → 按列名前缀分类
    if node.type == "var":
        col = node.value.lower()
        if col.startswith("funda_") or col.startswith("fin_"):
            return "fundamental"
        if col == "vwap":
            return "vwap"
        if col in ("close", "open", "high", "low", "adj_close", "adj_open",
                    "adj_high", "adj_low", "amount", "turnover"):
            return "price_volume"
        if col in ("volume", "adj_volume", "vol"):
            return "volume"
        if col.startswith("chip_") or col.startswith("csr_"):
            return "chip"
        if col in ("float_cap", "total_cap", "market_cap"):
            return "size"
        return "other"

    return "other"


def all_smoothing_ops(expression: str) -> list[str]:
    """返回表达式中所有平滑算子（按 AST 前序遍历顺序，含变量回溯）。"""
    if not expression:
        return []
    var_table = _parse_var_table(expression)
    ops: list[str] = []
    _collect_smoothing_ops(expression, var_table, ops, set())
    return ops


def _collect_smoothing_ops(expr_text: str, var_table: dict[str, str], out: list[str], seen: set[str]) -> None:
    """递归收集平滑算子，遇到变量时回溯到赋值表达式。"""
    ast = parse_ast(expr_text)
    if ast is None:
        return
    for node in ast.walk_calls():
        if node.op.upper() in SMOOTHING_OPS:
            out.append(node.op.upper())
    # 回溯变量
    for node in ast.walk():
        if node.type == "var" and node.value in var_table and node.value not in seen:
            seen.add(node.value)
            _collect_smoothing_ops(var_table[node.value], var_table, out, seen)


def structure_fingerprint(expression: str) -> str:
    """AST 版结构指纹（替代 expressions._structure_fingerprint）。

    语义等价：变量→VAR，数字→N，算子保留。
    实现差异：AST 递归归一化 vs regex 字符串替换——AST 版无误匹配风险。

    变量回溯：遇到中间变量时回溯到赋值表达式的 AST，
    使 `CS_ZSCORE(smooth)`（smooth=EMA(...)）与 `CS_ZSCORE(EMA(...))`
    的指纹一致——检测更多同构变体。
    """
    ast = parse_ast(expression)
    if ast is None:
        return ""
    var_table = _parse_var_table(expression)
    raw = ast.subtree_hash(var_table if var_table else None)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


# -----------------------------------------------------------------------------
# SSPM 粗族分类映射 (9大粗族，降低 cells 碎片化)
# -----------------------------------------------------------------------------

_FINE_TO_COARSE_FAMILY: dict[str, str] = {
    # 动量/反转/隔夜跳空
    "momentum": "momentum_reversal",
    "reversal": "momentum_reversal",
    "gap_overnight": "momentum_reversal",
    "price_volume": "momentum_reversal",
    # 波动/形态
    "volatility": "volatility",
    "kline": "volatility",
    # 量/流动性/拥挤
    "volume": "volume_liquidity",
    "liquidity": "volume_liquidity",
    "crowd": "volume_liquidity",
    "breadth": "volume_liquidity",
    "size": "volume_liquidity",
    # 相关性
    "correlation": "correlation",
    # 筹码
    "chip": "chip",
    # 基本面
    "fundamental": "fundamental",
    # 交互/复合算子
    "gated": "interaction",
    "piecewise": "interaction",
    "divergence": "interaction",
    "group_rank": "interaction",
    "residual": "interaction",
    "conditional": "interaction",
    # 均价偏离独立族 (高样本密度)
    "vwap": "vwap",
    # 兜底
    "smoothing": "other",
    "safe_divide": "other",
    "other": "other",
}


def classify_family_coarse(
    family_or_fusion_key: str,
    facets: set[str] | list[str] | None = None,
    expression: str = "",
) -> str:
    """将细粒度族名或融合族键映射为 9 大粗族 (SSPM cells 专用)。

    A1. AST/keyword 细族 → 粗族:
        momentum/reversal/gap_overnight → momentum_reversal
        volatility/kline → volatility
        volume/liquidity/crowd/breadth/size → volume_liquidity
        correlation → correlation
        chip → chip
        fundamental → fundamental
        gated/piecewise/divergence/group_rank/residual/conditional → interaction
        vwap → vwap
        other/safe_divide/smoothing → other

    A2. 融合族键（如 '价量面×基本面'）→ 粗族（最慢信息面主导）:
        含基本面/股东/机构面 → fundamental
        含筹码面 → chip
        含事件/资金/两融/披露/分红/业绩面 → interaction
        仅行情内部跨面 → 回落按 AST 信号根走 A1
    """
    key = str(family_or_fusion_key or "").strip()
    facets_set = set(facets or [])

    # 如果键是融合键（含 '×'）或传了跨面 facets
    if "×" in key or len(facets_set) >= 2:
        # 1. 慢信息面优先：基本面 / 股东 / 机构
        if any(f in key for f in ("基本面", "股东", "机构")) or any(
            f in facets_set for f in ("基本面", "股东", "机构", "fundamental", "holder")
        ):
            return "fundamental"
        # 2. 筹码面主导
        if "筹码" in key or any(f in facets_set for f in ("筹码", "chip")):
            return "chip"
        # 3. 事件 / 资金 / 两融 / 披露 / 分红 / 业绩
        if any(f in key for f in ("事件", "资金", "两融", "披露", "分红", "业绩")) or any(
            f in facets_set for f in ("事件", "资金", "两融", "披露", "分红", "业绩", "event", "capital")
        ):
            return "interaction"
        # 4. 仅行情内部跨面（价量×量能×拥挤等），走 AST 信号根或下方的 fine mapping

    # 优先查 fine_to_coarse 字典
    if key in _FINE_TO_COARSE_FAMILY:
        return _FINE_TO_COARSE_FAMILY[key]

    # 如果未直接命中预设细族，但提供了 expression，尝试用 AST 现算
    if expression:
        ast_fam = classify_family_ast(expression)
        if ast_fam in _FINE_TO_COARSE_FAMILY:
            return _FINE_TO_COARSE_FAMILY[ast_fam]

    return "other"

