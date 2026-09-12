# -*- coding: utf-8 -*-
"""模块 07 · operator_catalog：可用算子清单（机制分组 + 高频/聚焦分层瘦身）。"""

from alphaagent.dsl.catalog import operator_catalog_markdown

NAME = "operator_catalog"
TITLE = "可用算子目录"
ORDER = 70
REQUIRED = False
SEP_BEFORE = "\n\n---\n\n"

# 数据面 → 聚焦时注入完整签名的算子族前缀（与 expressions.FACET_DEFS 对齐）
_FACET_FAMILY_PREFIXES: dict[str, tuple[str, ...]] = {
    "筹码面": ("CHIP_",),
    "拥挤面": ("CROWD_",),
    "价量面": ("PRICE_", "WICK_", "KLINE_"),
    "量能面": ("VOLUME_", "MUTUAL_"),
}


def _focused_prefixes(focus_facets) -> tuple[str, ...]:
    prefixes: list[str] = []
    for facet in focus_facets or ():
        prefixes.extend(_FACET_FAMILY_PREFIXES.get(str(facet), ()))
    return tuple(prefixes)


def _excluded_prefixes(focus_facets) -> tuple[str, ...]:
    """未选面的专属算子族整体隐藏（工具层会拦其表达式，留着只会诱导越界）。"""
    focus = {str(f) for f in (focus_facets or ()) if f}
    if not focus:
        return ()
    excluded: list[str] = []
    for facet, prefixes in _FACET_FAMILY_PREFIXES.items():
        if facet not in focus:
            excluded.extend(prefixes)
    return tuple(excluded)


def render(ctx) -> str:  # noqa: ANN001
    focused = _focused_prefixes(getattr(ctx, "focus_facets", ()))
    phase = getattr(ctx, "prompt_phase", "full")
    focused_note = ""
    if focused and ctx.include_operator_catalog:
        focused_note = f"本轮聚焦数据面：{'/'.join(focused)} 开头的算子已附完整签名，优先在聚焦族内构建机制。"
    elif getattr(ctx, "focus_facets", ()) and ctx.include_operator_catalog:
        focused_note = "本轮已按聚焦数据面隐藏未选面的专属算子族（其表达式会被工具层拦截）。"

    # ── explore 阶段：不注入完整目录，只给一行提示 + 高频算子名 ──
    if phase == "explore" and ctx.include_operator_catalog and not focused:
        from alphaagent.dsl.catalog import build_operator_namespace, _FREQUENT_OPERATORS
        ns = build_operator_namespace()
        freq = [n for n in sorted(ns) if n in _FREQUENT_OPERATORS]
        freq_text = " ".join(f"`{n}`" for n in freq)
        return f"""### 可用算子（探索阶段精简）

算子均为**大写**（如 `TS_MEAN`、`DELTA`）。高频算子（已附签名见报错自愈）：{freq_text}。
冷门算子传参出错时错误信息会附真实签名，按提示修正即可。深耕阶段恢复完整目录。{focused_note}"""

    catalog = (
        operator_catalog_markdown(
            focused_prefixes=focused,
            excluded_prefixes=_excluded_prefixes(getattr(ctx, "focus_facets", ())),
        )
        if ctx.include_operator_catalog
        else "（本次未注入算子清单）"
    )
    explore_hint = ""
    if phase == "explore" and ctx.include_operator_catalog:
        explore_hint = " 探索阶段优先使用高频算子（已附完整签名），冷门算子签名见报错自愈。"
    return f"""### 可用算子

算子均为**大写**（如 `TS_MEAN`、`DELTA`）。支持位置参数，也支持关键字参数语法 `name=value`（关键字参数必须在位置参数之后）。按机制分节列出；签名省略类型标注，**参数顺序即语义**（位置传参必须严格按签名顺序）；语义自明的基础四则/比较/初等函数折叠在末行。选算子前先想机制（见「A 股市场机制与 alpha 分布」），再按节定位。低频算子未附签名——传参报错时错误信息会附真实签名，按提示修正即可，不必回避。{focused_note}{explore_hint}

{catalog}"""