# -*- coding: utf-8 -*-
"""表达式结构分析：token 化、结构指纹、编辑类型（motif）、信号族分类。"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# ── 正则与已知集合 ──

_TOKEN_RE = re.compile(r"[a-zA-Z][a-zA-Z0-9_]{2,}")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
_STOPWORDS = {"add", "subtract", "multiply", "divide", "ts", "cs", "mean", "std", "rank", "expr", "factor"}

_KNOWN_OPERATORS = frozenset({
    # 时序算子
    "ts_mean", "ts_sum", "ts_std", "ts_var", "ts_median", "ts_max", "ts_min",
    "ts_rank", "ts_quantile", "ts_delta", "ts_delay", "ts_advance",
    "ts_corr", "ts_cov", "ts_regression", "ts_decay_linear", "ts_skewness",
    "ts_kurtosis", "ts_arg_max", "ts_arg_min", "ts_pct_change",
    # 截面算子
    "cs_rank", "cs_zscore", "cs_winsorize", "cs_demean", "cs_quantile",
    "cs_residualize", "cs_neutralize",
    # 算术
    "add", "subtract", "multiply", "divide", "abs", "log", "sign",
    "max", "min", "power", "sqrt", "reciprocal", "negate",
    # 其他
    "rank", "zscore", "winsorize", "normalize", "demean", "quantile",
    "residualize", "neutralize", "if_else", "cond",
})

_KNOWN_VARIABLES = frozenset({
    "close", "open", "high", "low", "volume", "amount", "vwap", "adj_open",
    "adj_close", "adj_high", "adj_low", "ret", "returns", "turnover",
    "market_cap", "market_value", "float_share", "total_share",
    "amt", "adv20", "adv60", "adv120",
    # 财务字段
    "funda_roe", "funda_roa", "funda_eps", "funda_bps", "funda_revenue",
    "funda_net_profit", "funda_gross_margin", "funda_net_margin",
    # 资金流/事件
    "ff_net_flow", "ff_main_flow", "ff_large_order",
    "pred_net_profit", "pred_eps",
    "holder_count", "holder_change",
    "dt_big_order", "bt_block_trade",
})

_FUNC_CALL_RE = re.compile(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*")
_VAR_RE = re.compile(r"\$?([a-zA-Z_][a-zA-Z0-9_]*)")
_NUM_RE = re.compile(r"\b(\d+(?:\.\d+)?)\b")

# ── 编辑 motif 定义 ──

MOTIFS = (
    "window_rescale",
    "feature_swap",
    "operator_substitute",
    "operator_swap",
    "condition_gate",
    "composition_add",
    "normalize_change",
    "decorrelation_add",
    "interaction_add",
    "compound_edit",
    "other",
)

# ── 因子族分类规则 ──

_FAMILY_RULES: dict[str, tuple[str, ...]] = {
    "gap_overnight": ("gap", "overnight", "open_close"),
    "vwap": ("vwap",),
    "chip": ("chip", "peak", "entropy"),
    "momentum": ("momentum", "pctchange", "ma_w", "ma20", "ma_dev"),
    "reversal": ("reversal", "neg_ts"),
    "volume": ("volume", "amount", "turnover"),
    "volatility": ("std", "var", "vol_"),
    "fundamental": ("funda_", "roe", "roa", "growth", "quality", "value"),
    "correlation": ("corr", "cov", "rankcorr"),
    "liquidity": ("liquidity", "float", "amihud"),
    "breadth": ("breadth", "advance", "decline"),
}


# ── 中文分词 ──

_jieba_cut = None
try:  # 可选依赖：装 jieba 时用标准中文分词，否则回退 bigram
    import jieba  # type: ignore
    _jieba_cut = jieba.lcut_for_search
except Exception:  # pragma: no cover - 环境无 jieba 时走 bigram
    pass


def _cjk_tokens(text: str) -> set[str]:
    """抽取中文字段为可检索 token。

    优先使用 jieba（安装后生效），否则按连续中文串生成 bigram。
    """
    if not text:
        return set()
    if _jieba_cut is not None:
        words = {w.strip().lower() for w in _jieba_cut(text) if len(w.strip()) >= 2 and _CJK_RE.fullmatch(w.strip())}
        if words:
            return words
    out: set[str] = set()
    for seq in _CJK_RE.findall(text):
        if len(seq) == 2:
            out.add(seq)
            continue
        out.update(seq[i:i + 2] for i in range(len(seq) - 1))
    return out


def _tokens(*values: Any) -> list[str]:
    """从英文表达式/字段与中文结论中抽取去重 token，供 BM25 检索。"""
    found: set[str] = set()
    for value in values:
        text = str(value or "")
        for token in _TOKEN_RE.findall(text):
            token = token.lower()
            found.add(token)
            # 下划线连接的复合词拆开（momentum_reversal -> momentum + reversal），
            # 否则查询侧 "momentum reversal"（空格分词）与条目侧对不上
            found.update(part for part in token.split("_") if len(part) >= 2)
        found.update(_cjk_tokens(text))
    return sorted(found - _STOPWORDS)


# ── 结构指纹 ──

def _structure_fingerprint(expression: str) -> str:
    """计算表达式结构指纹：变量替换为 VAR，数字替换为 N，算子保留。

    示例::
        rank(subtract(ts_mean($close, 5), ts_mean($close, 20)))
        → hash("rank(subtract(ts_mean(VAR,N),ts_mean(VAR,N)))")
    """
    if not expression:
        return ""
    text = str(expression).strip()
    # 替换 $variable → VAR
    text = re.sub(r"\$[a-zA-Z_][a-zA-Z0-9_]*", "VAR", text)
    # 替换 bare variables → VAR (只替换已知变量，避免误替换算子)
    for var in sorted(_KNOWN_VARIABLES, key=len, reverse=True):
        text = re.sub(rf"\b{re.escape(var)}\b", "VAR", text)
    # 替换 funda_*/ff_*/pred_*/holder_*/dt_*/bt_* 前缀变量
    text = re.sub(r"\b(?:funda_|ff_|pred_|holder_|dt_|bt_)[a-zA-Z0-9_]+", "VAR", text)
    # 替换数字 → N
    text = re.sub(r"\d+(?:\.\d+)?", "N", text)
    # 规范化空格和换行
    text = re.sub(r"\s+", "", text)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


# ── 表达式结构解析 ──

def _parse_expression_structure(expression: str) -> dict[str, Any]:
    """轻量级 regex 解析器，提取表达式的算子/变量/窗口结构。

    返回::
        {
            "operators": ["rank", "subtract", "ts_mean", "ts_mean"],
            "variables": ["close"],
            "window_params": {"ts_mean": [5, 20]},
            "constants": [5, 20],
            "fingerprint": "hash(...)",
        }
    """
    if not expression:
        return {"operators": [], "variables": [], "window_params": {}, "constants": [], "fingerprint": ""}

    text = str(expression).strip()

    # 提取算子（函数名）
    operators: list[str] = []
    for match in _FUNC_CALL_RE.finditer(text):
        name = match.group(1).lower()
        if name in _KNOWN_OPERATORS:
            operators.append(name)

    # 提取变量（$close → close）
    variables: list[str] = []
    seen_vars: set[str] = set()
    for match in _VAR_RE.finditer(text):
        name = match.group(1).lower()
        # 排除算子名和纯数字
        if name in _KNOWN_OPERATORS:
            continue
        if name in {"true", "false", "null", "none", "nan", "inf"}:
            continue
        if name in _KNOWN_VARIABLES or name.startswith("funda_") or name.startswith("ff_") \
                or name.startswith("pred_") or name.startswith("holder_") \
                or name.startswith("dt_") or name.startswith("bt_"):
            if name not in seen_vars:
                variables.append(name)
                seen_vars.add(name)

    # 提取数字常量
    constants: list[int | float] = []
    for match in _NUM_RE.finditer(text):
        val = match.group(1)
        constants.append(int(val) if "." not in val else float(val))

    # 窗口参数：将算子与其后续常量关联
    window_params: dict[str, list[int | float]] = {}
    tokens = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)|(\d+(?:\.\d+)?)", text)
    op_queue: list[str] = []
    for name_tok, num_tok in tokens:
        if name_tok:
            lower = name_tok.lower()
            if lower in _KNOWN_OPERATORS:
                op_queue.append(lower)
        elif num_tok and op_queue:
            val = int(num_tok) if "." not in num_tok else float(num_tok)
            # 最后出现的算子取这个常量
            last_op = op_queue[-1]
            window_params.setdefault(last_op, []).append(val)

    # 结构指纹
    fingerprint = _structure_fingerprint(text)

    return {
        "operators": operators,
        "variables": variables,
        "window_params": window_params,
        "constants": constants,
        "fingerprint": fingerprint,
    }


# ── 表达式特征快捷函数 ──

def expression_features(expression: str) -> dict[str, Any]:
    """返回表达式结构特征（算子/变量/窗口/指纹）。"""
    return _parse_expression_structure(expression)


def expression_ops(expression: str) -> list[str]:
    """返回表达式中的算子列表。"""
    return _parse_expression_structure(expression).get("operators", [])


def expression_windows(expression: str) -> dict[str, list[int | float]]:
    """返回表达式中的窗口参数。"""
    return _parse_expression_structure(expression).get("window_params", {})


def template_from_expression(expression: str) -> str:
    """表达式 → 参数槽模板：数字字面量按出现顺序替换为 {w1}/{w2}/...

    FactorMiner 式"可模仿形状"：RANK(SUBTRACT($adj_close, TS_MEAN($vwap, 20)))
    → RANK(SUBTRACT($adj_close, TS_MEAN($vwap, {w1})))。
    模板用于经验层注入（成功模式可照抄骨架换参、死路结构可整条禁掉）。
    """
    text = str(expression or "").strip()
    if not text:
        return ""
    counter = 0

    def _sub(_m: re.Match) -> str:
        nonlocal counter
        counter += 1
        return f"{{w{counter}}}"

    return re.sub(r"\d+(?:\.\d+)?", _sub, text)


# ── 数据面识别（跨面融合引导用）────────────────────────────────
# 面板实际接入的数据列族（与 data/adapters/plugins 一致）：
# 价量(stock_daily_wide) / 筹码(CHIP_*) / 拥挤(CROWD_*) / 基本面(funda_*) /
# 股东(holder_*) / 业绩预告(forecast) / 事件(event_faces: 龙虎榜+大宗) / 资金流(fund_flow)
FACET_DEFS: list[tuple[str, tuple[str, ...]]] = [
    ("价量面", ("$adj_", "$close", "$open", "$high", "$low", "$ret", "$vwap")),
    ("量能面", ("$volume", "$amount", "$turnover")),
    ("筹码面", ("chip_",)),
    ("拥挤面", ("crowd_",)),
    ("基本面", ("funda_",)),
    ("股东面", ("holder_",)),
    ("事件面", ("forecast", "dragon", "event_", "$event")),
    ("资金面", ("fund_flow", "$inflow", "$outflow")),
]

# 数据源分组：融合（family 用面对组合键）只在跨组时成立。
# 同组多面（如 价量面+量能面 都来自 stock_daily_wide 行情面板）是普通单因子
# 的常见形态，判成"融合"会把族分类从细粒度退化为粗粒度对键，污染记忆桶。
# $float_cap 是行情面板列，已从股东面识别键移出，不参与面判定。
FACET_GROUPS: dict[str, tuple[str, ...]] = {
    "行情组": ("价量面", "量能面", "筹码面", "拥挤面"),
    "基本面组": ("基本面", "股东面"),
    "事件资金组": ("事件面", "资金面"),
}

_FACET_ORDER: dict[str, int] = {name: i for i, (name, _) in enumerate(FACET_DEFS)}
_FACET_TO_GROUP: dict[str, str] = {
    facet: group for group, facets in FACET_GROUPS.items() for facet in facets
}


def expr_facets(expression: str) -> set[str]:
    """识别一个表达式触及的数据面（按列/算子前缀匹配）。"""
    low = str(expression or "").lower()
    out: set[str] = set()
    for name, keys in FACET_DEFS:
        if any(k.lower() in low for k in keys):
            out.add(name)
    return out


def facet_groups(facets: set[str]) -> set[str]:
    """一个面集合覆盖的数据源组。"""
    return {_FACET_TO_GROUP[f] for f in facets if f in _FACET_TO_GROUP}


def is_cross_group_fusion(facets: set[str]) -> bool:
    """≥2 个面且跨数据源组才算融合（同组多面不算，防线见 FACET_GROUPS 注释）。"""
    return len(facets) >= 2 and len(facet_groups(facets)) >= 2


def fusion_family_key(facets: set[str]) -> str:
    """融合因子的族键：面名按 FACET_DEFS 稳定排序 × 连接（如 基本面×价量面）。"""
    ordered = sorted((f for f in facets if f in _FACET_ORDER),
                     key=lambda f: _FACET_ORDER[f])
    return "×".join(ordered)


# ── 编辑类型识别 ──

def _identify_edit_type(
    parent_struct: dict[str, Any],
    child_struct: dict[str, Any],
) -> dict[str, Any] | None:
    """对比两个因子结构，识别编辑类型和详情。

    返回 ``None`` 表示无法识别有意义的编辑（结构完全不同或完全相同）。
    """
    if not parent_struct or not child_struct:
        return None

    p_ops = parent_struct.get("operators", [])
    c_ops = child_struct.get("operators", [])
    p_vars = set(parent_struct.get("variables", []))
    c_vars = set(child_struct.get("variables", []))
    p_wins = parent_struct.get("window_params", {})
    c_wins = child_struct.get("window_params", {})

    # 结构指纹相同不代表无编辑：指纹归一化数字（窗口 10→20 指纹相同），
    # 但 window_params 保留了具体值。先检查窗口差异，再判定是否"完全相同"。
    win_changes: list[dict[str, Any]] = []
    common_op_wins = set(p_wins.keys()) & set(c_wins.keys())
    for op in sorted(common_op_wins):
        p_vals = p_wins.get(op, [])
        c_vals = c_wins.get(op, [])
        if p_vals != c_vals:
            for i, (pv, cv) in enumerate(zip(p_vals, c_vals)):
                if pv != cv:
                    if isinstance(pv, (int, float)) and isinstance(cv, (int, float)):
                        if cv > pv:
                            win_changes.append({
                                "operator": op, "position": i,
                                "from": pv, "to": cv, "direction": "extend",
                            })
                        else:
                            win_changes.append({
                                "operator": op, "position": i,
                                "from": pv, "to": cv, "direction": "shrink",
                            })

    # 算子集合差异
    p_ops_set = set(p_ops)
    c_ops_set = set(c_ops)
    ops_added = c_ops_set - p_ops_set
    ops_removed = p_ops_set - c_ops_set

    # 变量差异
    vars_added = c_vars - p_vars
    vars_removed = p_vars - c_vars

    # 真正无编辑：指纹相同 + 无窗口/算子/变量差异
    if (
        parent_struct.get("fingerprint") == child_struct.get("fingerprint")
        and not win_changes
        and not ops_added
        and not ops_removed
        and not vars_added
        and not vars_removed
    ):
        return None  # 结构完全相同，无编辑

    # 判定编辑类型（优先级从高到低）

    # 1. interaction_add
    if len(p_vars) < len(c_vars) and len(ops_added) > 0:
        return {
            "edit_type": "interaction_add",
            "detail": {
                "new_variables": sorted(vars_added),
                "new_operators": sorted(ops_added),
            },
        }

    # 2. composition_add
    if len(c_ops) > len(p_ops) and vars_added == set() and vars_removed == set():
        return {
            "edit_type": "composition_add",
            "detail": {"new_operators": sorted(ops_added)},
        }

    # 3. variable_replace
    if vars_added and vars_removed and ops_added == set() and ops_removed == set() and not win_changes:
        return {
            "edit_type": "variable_replace",
            "detail": {
                "from": sorted(vars_removed),
                "to": sorted(vars_added),
            },
        }

    # 4. window_extend / window_shrink
    if win_changes and not vars_added and not vars_removed and ops_added == set() and ops_removed == set():
        all_extend = all(c["direction"] == "extend" for c in win_changes)
        all_shrink = all(c["direction"] == "shrink" for c in win_changes)
        if all_extend:
            return {"edit_type": "window_extend", "detail": {"changes": win_changes}}
        if all_shrink:
            return {"edit_type": "window_shrink", "detail": {"changes": win_changes}}
        return {"edit_type": "window_change", "detail": {"changes": win_changes}}

    # 5. operator_swap
    if ops_added and ops_removed and not vars_added and not vars_removed and not win_changes:
        return {
            "edit_type": "operator_swap",
            "detail": {"from": sorted(ops_removed), "to": sorted(ops_added)},
        }

    # 6. normalize_change
    if ops_added and ops_removed:
        norm_ops = {"rank", "zscore", "winsorize", "normalize", "demean", "quantile",
                    "cs_rank", "cs_zscore", "cs_winsorize", "cs_demean", "cs_quantile"}
        if ops_added.issubset(norm_ops) and ops_removed.issubset(norm_ops):
            return {
                "edit_type": "normalize_change",
                "detail": {"from": sorted(ops_removed), "to": sorted(ops_added)},
            }

    # 7. decorrelation_add
    decorr_ops = {"residualize", "neutralize", "cs_residualize", "cs_neutralize"}
    if ops_added & decorr_ops:
        return {
            "edit_type": "decorrelation_add",
            "detail": {"new_operators": sorted(ops_added & decorr_ops)},
        }

    # 8. compound_edit
    changes: list[str] = []
    if win_changes:
        changes.append("window_change")
    if vars_added or vars_removed:
        changes.append("variable_change")
    if ops_added or ops_removed:
        changes.append("operator_change")
    if changes:
        return {
            "edit_type": "compound_edit",
            "detail": {
                "changes": changes,
                "vars_added": sorted(vars_added),
                "vars_removed": sorted(vars_removed),
                "ops_added": sorted(ops_added),
                "ops_removed": sorted(ops_removed),
                "win_changes": win_changes,
            },
        }

    return None


# ── 编辑 motif 提取（v3-lite 高层 API）──

def extract_edit_motif(parent_expr: str, child_expr: str) -> str:
    """从父本→子本表达式对中提取编辑 motif 名称。

    返回 MOTIFS 中的一个字符串。
    """
    parent_struct = _parse_expression_structure(parent_expr)
    child_struct = _parse_expression_structure(child_expr)
    edit = _identify_edit_type(parent_struct, child_struct)
    if edit is None:
        return "other"
    et = edit["edit_type"]
    # 映射 _identify_edit_type 的 edit_type 到 MOTIFS
    if et in ("window_extend", "window_shrink", "window_change"):
        return "window_rescale"
    if et == "variable_replace":
        return "feature_swap"
    if et == "operator_swap":
        return "operator_swap"
    if et == "composition_add":
        # 如果新增了 if_else/cond 算子，归为 condition_gate
        detail = edit.get("detail", {})
        new_ops = set(detail.get("new_operators", []))
        if new_ops & {"if_else", "cond"}:
            return "condition_gate"
        return "composition_add"
    if et == "normalize_change":
        return "normalize_change"
    if et == "decorrelation_add":
        return "decorrelation_add"
    if et == "interaction_add":
        return "interaction_add"
    if et == "compound_edit":
        return "compound_edit"
    return "other"


def motif_from_note(edit_note: str) -> str | None:
    """从 edit_note 字符串中解析 motif 名称。

    edit_note 格式: ``edit=<motif> <参数变化>``

    返回 MOTIFS 之一；空串/None 返回 None；未识别返回 "other"。
    """
    if not edit_note:
        return None
    text = str(edit_note).strip()
    # 尝试从 "edit=<motif> ..." 格式提取
    if text.startswith("edit="):
        rest = text[5:].strip()
        # 第一个 token 是 motif
        parts = rest.split(None, 1)
        if parts:
            candidate = parts[0].strip().lower()
            if candidate in MOTIFS:
                return candidate
    # 尝试直接匹配（含中文别名）
    text_lower = text.lower()
    for motif in MOTIFS:
        if motif in text_lower:
            return motif
    # 中文别名：窗口/参数变异
    if any(kw in text_lower for kw in ("窗口", "参数")):
        return "window_rescale"
    if "算子" in text_lower and ("替换" in text_lower or "换" in text_lower or "substitut" in text_lower):
        return "operator_substitute"
    return "other"


# ── 信号族分类 ──

def classify_family_ex(factor_name: str, expression: str) -> tuple[str, set[str]]:
    """分类到信号族，返回 (family, facets)。

    facets 跨数据源组 ≥2 面（is_cross_group_fusion）→ family = 面对组合键
    （如 基本面×价量面），融合因子在记忆桶/饱和度中按融合对聚合；
    否则走 _FAMILY_RULES 细粒度规则（单面/同组多面行为与历史一致）。
    facets 无论是否融合都返回，供 facets_json 与检索亲和使用。
    """
    text = (str(factor_name or "") + " " + str(expression or "")).lower()
    facets = expr_facets(text)
    if is_cross_group_fusion(facets):
        return fusion_family_key(facets), facets
    for family, keywords in _FAMILY_RULES.items():
        if any(kw in text for kw in keywords):
            return family, facets
    return "other", facets


def classify_family(factor_name: str, expression: str) -> str:
    """根据因子名和表达式启发式分类到信号族（classify_family_ex 的薄包装）。"""
    return classify_family_ex(factor_name, expression)[0]
