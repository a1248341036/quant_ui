"""DSL 算子清单：供 eval、挖掘 prompt 等模块共享。"""

from __future__ import annotations

import inspect
from collections.abc import Callable

from alphaagent.dsl.registry import build_operator_namespace


def list_operator_names() -> list[str]:
    return sorted(build_operator_namespace())


def _slim_signature(fn) -> str:
    """签名瘦身：去掉类型标注（对 LLM 是纯噪音），保留参数名顺序与默认值。

    `TS_MEAN(df: 'pd.DataFrame', window: 'Window')` → `TS_MEAN(df, window)`；
    `CHIP_PEAK_LOC(..., nbins: 'int' = 64, method: 'str' = 'cyq')` → `(..., nbins=64, method='cyq')`。
    位置参数顺序与真实签名严格一致（DSL 按位置传参，顺序即语义）。

    keyword-only 参数（`*` 后）显式渲染 `*` 分隔符：`MUTUAL_INFO_LAG(df, volume,
    window, lag, *, n_bins=8)`——LLM 曾把 n_bins 按第 5 个位置参数传入而 exec
    报 "takes 4 positional arguments but 5 were given"，catalog 平铺签名是根因。
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return "(...)"
    parts: list[str] = []
    star_done = False
    for p in sig.parameters.values():
        if p.kind is inspect.Parameter.KEYWORD_ONLY and not star_done:
            parts.append("*")
            star_done = True
        if p.default is inspect.Parameter.empty:
            parts.append(p.name)
            continue
        d = repr(p.default) if isinstance(p.default, str) else str(p.default)
        if len(d) > 24:
            d = d[:21] + "..."
        parts.append(f"{p.name}={d}")
    return "(" + ", ".join(parts) + ")"


# ── 目录分组（机制视角，与挖掘提示词的交互契约/数据面语言对齐）─────────────
# 有序规则：首个命中的组生效；每组内保持字母序。
# 未命中任何组的算子 = 语义自明的基础件，默认折叠成一行说明。

_CATALOG_GROUPS: list[tuple[str, Callable[[str], bool]]] = [
    ("结构化交互（门控/残差/分歧/分段——配合 interaction 契约使用）",
     lambda n: n in {"GATED_SIGNAL", "CS_GROUP_RANK", "CS_RESIDUALIZE",
                     "DIVERGENCE_RANK", "PIECEWISE_STATE", "IF_THEN_ELSE"}),
    ("时序滚动（TS_；其中 TS_CORR/TS_COV/TS_RANKCORR 属交互类，须传契约）",
     lambda n: n.startswith("TS_")),
    ("时序基础（差分/滞后/均线——高频主力，窗口语义见签名）",
     lambda n: n in {"DELAY", "DELTA", "EMA", "SMA", "WMA"}),
    ("截面变换与分组（CS_、RANK）", lambda n: n.startswith("CS_") or n == "RANK"),
    ("筹码分布（CHIP_）", lambda n: n.startswith("CHIP_")),
    ("拥挤度（CROWD_）", lambda n: n.startswith("CROWD_")),
    ("缺口结构（PRICE_）", lambda n: n.startswith("PRICE_")),
    ("K线几何与影线", lambda n: n.startswith(("WICK_", "KLINE_"))),
    ("量钟与信息流", lambda n: n.startswith(("VOLUME_", "MUTUAL_"))),
    ("回归拟合", lambda n: n in {"REGRESI", "REGBETA", "SLOPE", "RESI", "SEQUENCE"}),
]

_FOLD_SUMMARY = (
    "- 基础四则/比较/初等函数（语义自明，直用）："
    "`ADD/SUBTRACT/MULTIPLY/DIVIDE(df1, df2)` 逐元素四则；"
    "`GT/LT/GE/LE/EQ/NE(df1, df2)` 比较得 0/1 面板；`AND/OR` 组合布尔；"
    "`MAXIMUM/MINIMUM/MAX/MIN(x, y, z=None)` 逐元素极值（支持标量广播）；"
    "`ABS/SIGN/LOG/EXP/POW/SQRT/INV/NEG(df)` 初等函数；"
    "`CAST(df, dtype)` 面板类型转换；`FILLNA(df, value=0.0)` 非有限值替换。"
)

# hybrid 档保留完整签名的算子清单。数据依据（2026-09-03，research_memory
# 2773 条评估记录的 operator_list_json 统计）：历史实际使用 ≥5 次的 24 个
# 全收录；交互契约算子（strategy_tracks 硬约束）无论频率全收录；其余为
# prompt 文档明确推荐的入门算子。
_FREQUENT_OPERATORS = frozenset({
    # 历史高频（usage≥5）
    "TS_MEAN", "DIVIDE", "RANK", "CS_WINSORIZE", "CS_ZSCORE", "LOG",
    "SUBTRACT", "CS_NEUTRALIZE", "ADD", "CS_RESIDUALIZE", "TS_STD",
    "MULTIPLY", "TS_MEDIAN", "MAX", "ABS", "TS_RANK", "TS_SUM",
    "TS_MAX", "TS_CORR", "MIN", "TS_QUANTILE", "SIGN", "TS_MIN",
    # 交互契约算子（strategy_tracks 硬约束，签名必须随目录提供）
    "GATED_SIGNAL", "CS_GROUP_RANK", "DIVERGENCE_RANK",
    "PIECEWISE_STATE", "IF_THEN_ELSE", "TS_RANKCORR", "TS_COV",
    "MUTUAL_INFO_LAG", "NEG",
    # 数据面/时序基础
    "DELTA", "DELAY", "EMA", "SMA", "WMA",
    # 交叉结构（CHIP 主力样例，签名形状族内可参照）
    "CHIP_PEAK_LOC", "CHIP_ENTROPY",
})


def _summary_of(fn, max_chars: int) -> str:
    doc = (inspect.getdoc(fn) or "").strip().splitlines()
    summary = doc[0].strip() if doc else ""
    if max_chars and len(summary) > max_chars:
        summary = summary[: max_chars - 1].rstrip() + "…"
    return summary


def operator_catalog_markdown(
    *,
    max_summary_chars: int = 60,
    include_basic: bool = False,
    tier: str = "hybrid",
    focused_prefixes: tuple[str, ...] = (),
) -> str:
    """按机制分组渲染算子目录。

    - 家族分节（交互/时序/截面/筹码/拥挤/缺口/几何/量钟/回归），节内字母序；
    - 摘要截断到 ``max_summary_chars``（签名永不截断——传参顺序即语义）；
    - 语义自明的基础四则/比较/初等函数默认折叠为一行（``include_basic=True``
      恢复逐个渲染，含未归组的算子全量平铺）。

    tier
    ----
    - ``"hybrid"``（默认）：高频算子（历史挖掘实际使用，见 _FREQUENT_OPERATORS）
      与交互契约算子渲染完整签名行；其余冷门算子按节压成纯名字清单（无签名、
      无摘要）——数据依据：2773 条历史评估记录仅触及 28 个算子（2026-09 统计），
      未用过的签名占目录篇幅 60%+。冷门算子传参出错时 dispatch 会在错误信息
      中附真实签名（自愈路径）。
    - ``"full"``：全部算子完整签名行（旧行为）。

    focused_prefixes：按数据面聚焦动态注入签名的族前缀（如选中筹码面 →
    ``("CHIP_",)``）——匹配的算子在 hybrid 档也渲染完整签名行。探索路径
    不因瘦身堵死：聚焦哪一面，哪一面的专业算子签名就常驻。
    """
    ns = build_operator_namespace()
    folded: list[str] = []
    groups: dict[str, list[str]] = {}
    for name in sorted(ns):
        for title, match in _CATALOG_GROUPS:
            if match(name):
                groups.setdefault(title, []).append(name)
                break
        else:
            folded.append(name)

    def _render(name: str) -> str:
        fn = ns[name]
        line = f"- `{name}{_slim_signature(fn)}`"
        summary = _summary_of(fn, max_summary_chars)
        if summary:
            line += f" — {summary}"
        return line

    hybrid = tier == "hybrid"
    lines: list[str] = []
    for title, _match in _CATALOG_GROUPS:
        names = groups.get(title)
        if not names:
            continue
        if lines:
            lines.append("")
        lines.append(f"**{title}**")
        if hybrid:
            full = [
                n for n in names
                if n in _FREQUENT_OPERATORS
                or n.startswith(tuple(focused_prefixes))
            ]
            cold = [n for n in names if n not in full]
            lines.extend(_render(n) for n in full)
            # 冷门算子：纯名字清单，提示签名走报错自愈
            if cold:
                names_text = " ".join(f"`{n}`" for n in cold)
                lines.append(f"- 其余（签名见报错自愈，慎用）：{names_text}")
        else:
            lines.extend(_render(n) for n in names)
    if include_basic:
        if lines:
            lines.append("")
        lines.extend(_render(n) for n in folded)
    elif folded:
        if lines:
            lines.append("")
        lines.append(_FOLD_SUMMARY)
    return "\n".join(lines).strip()
