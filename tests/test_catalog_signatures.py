# -*- coding: utf-8 -*-
"""catalog 签名渲染回归：keyword-only 参数必须显式渲染 * 分隔符。

背景：MUTUAL_INFO_LAG 的 n_bins 是 keyword-only，catalog 平铺渲染导致 LLM
按第 5 个位置参数传入，exec 报 "takes 4 positional arguments but 5 were given"。
"""
import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphaagent.dsl.catalog import _slim_signature, operator_catalog_markdown
from alphaagent.dsl.registry import build_operator_namespace


def test_mutual_info_lag_signature_marks_keyword_only():
    ns = build_operator_namespace()
    sig = _slim_signature(ns["MUTUAL_INFO_LAG"])
    assert sig == "(df_close, df_volume, window, lag, *, n_bins=8, min_pairs=None)", sig


def test_positional_only_operators_have_no_star():
    """常规位置参数算子不应被误加 *。"""
    ns = build_operator_namespace()
    assert _slim_signature(ns["TS_MEAN"]) == "(df, window)"
    assert _slim_signature(ns["CROWD_SHARE"]) == (
        "(dimension, weight, window, side_or_nbuckets='high', split_or_bucket_idx=0.5)"
    )


def test_all_keyword_only_operators_render_star_once():
    """全库扫描：凡有 keyword-only 参数的算子，渲染文本含且仅含一个 *。"""
    ns = build_operator_namespace()
    checked = 0
    for name, fn in ns.items():
        has_kw_only = any(
            p.kind is inspect.Parameter.KEYWORD_ONLY
            for p in inspect.signature(fn).parameters.values()
        )
        if not has_kw_only:
            continue
        checked += 1
        sig = _slim_signature(fn)
        assert sig.count("*") == 1, f"{name}: {sig}"
        assert " *," in sig or sig.contains("(*)") or ", *," in sig or "*, " in sig, f"{name}: {sig}"
    assert checked >= 3  # MUTUAL_INFO_LAG / WICK_EFFICIENCY / KLINE_GEOMETRY


def test_catalog_markdown_contains_star_for_mutual_info():
    md = operator_catalog_markdown()
    line = next(l for l in md.splitlines() if "MUTUAL_INFO_LAG(" in l)
    assert "*, n_bins=8" in line


# ── hybrid 分层：高频全签名 + 冷门名字清单 + 聚焦族注入 ──

def test_hybrid_tier_keeps_frequent_and_folds_cold_names():
    """hybrid 档：高频算子有签名行，冷门算子只出现在名字清单行。"""
    md = operator_catalog_markdown()  # 默认 hybrid
    lines = md.splitlines()
    # 高频：TS_MEAN 全签名
    assert any(l.startswith("- `TS_MEAN(") for l in lines)
    # 交互契约算子必须全签名（无论频率）
    assert any(l.startswith("- `GATED_SIGNAL(") for l in lines)
    assert any(l.startswith("- `PIECEWISE_STATE(") for l in lines)
    # 冷门：CHIP_BIMODAL_SCORE 不应有签名行，只出现在名字清单
    assert not any(l.startswith("- `CHIP_BIMODAL_SCORE(") for l in lines)
    assert any("CHIP_BIMODAL_SCORE" in l and l.startswith("- 其余") for l in lines)
    # full 档恢复
    md_full = operator_catalog_markdown(tier="full")
    assert any(l.startswith("- `CHIP_BIMODAL_SCORE(") for l in md_full.splitlines())


def test_hybrid_tier_focused_prefixes_get_signatures():
    """聚焦面联动：传 focused_prefixes 后对应族算子升级为全签名行。"""
    md = operator_catalog_markdown(focused_prefixes=("CHIP_",))
    assert any(l.startswith("- `CHIP_PEAK_LOC(") for l in md.splitlines())
    assert any(l.startswith("- `CHIP_BIMODAL_SCORE(") for l in md.splitlines())
    # 未聚焦的族仍折叠（CROWD_ 不在 _FREQUENT_OPERATORS）
    assert not any(l.startswith("- `CROWD_SHARE(") for l in md.splitlines())


# ── 机制分组 + 摘要截断 + 基础件折叠 ──

def test_catalog_markdown_grouped_by_mechanism():
    """目录按机制分节：交互算子聚首节，TS_ 家族成节，基础件折叠成一行。"""
    md = operator_catalog_markdown()
    lines = md.splitlines()
    # 分节标题存在且交互节在最前
    assert lines[0].startswith("**结构化交互")
    assert any(l.startswith("**时序滚动") for l in lines)
    assert any(l.startswith("**筹码分布") for l in lines)
    # 交互算子都在交互节标题之后、时序节之前（标题行本身不含算子行）
    idx_inter = next(i for i, l in enumerate(lines) if l.startswith("**结构化交互"))
    idx_ts = next(i for i, l in enumerate(lines) if l.startswith("**时序滚动"))
    inter_body = "\n".join(lines[idx_inter:idx_ts])
    assert "GATED_SIGNAL(" in inter_body
    assert "GATED_SIGNAL(" not in "\n".join(lines[idx_ts:])
    # 基础件不再逐个渲染（以列表行开头的独立条目）
    rendered_names = [l for l in lines if l.startswith("- `")]
    assert not any(l.startswith("- `ADD(") or l.startswith("- `GT(") or l.startswith("- `LOG(") for l in rendered_names)
    assert "基础四则/比较/初等函数" in md


def test_catalog_markdown_summary_truncated():
    """长摘要截断到 max_summary_chars 且以 … 结尾；签名部分永不截断（full 档校验）。"""
    md = operator_catalog_markdown(max_summary_chars=40, tier="full")
    chip_line = next(l for l in md.splitlines() if "CHIP_PEAK_SHARPNESS(" in l)
    assert len(chip_line) < 200  # 原始行 176+，截断后明显变短
    assert "nbins=64" in chip_line  # 签名完整保留
    # 逐行校验：签名之后的摘要部分不超过 40 chars（含省略号）
    for ln in md.splitlines():
        if not ln.startswith("- `") or "— " not in ln:
            continue
        summary = ln.split("— ", 1)[1]
        assert len(summary) <= 40, f"{ln[:80]} -> {len(summary)}"


def test_catalog_markdown_include_basic_restores_flat():
    """include_basic=True 恢复基础件逐个渲染（逃生门）。"""
    md = operator_catalog_markdown(include_basic=True)
    assert "- `ADD(" in md
    assert "- `LOG(" in md
    assert "基础四则/比较/初等函数" not in md


def test_catalog_markdown_all_operators_accounted():
    """分组 + 折叠必须覆盖注册表全部算子，无遗漏（裸名时序主力不折叠）。"""
    from alphaagent.dsl.registry import build_operator_namespace

    ns = build_operator_namespace()

    def rendered_as_item(md: str, name: str) -> bool:
        """算子是否被渲染成独立列表行（而非在折叠说明/其他名字的子串中出现）。"""
        return any(l.startswith(f"- `{name}(") for l in md.splitlines())

    md_basic = operator_catalog_markdown(include_basic=True, tier="full")
    for name in ns:
        assert rendered_as_item(md_basic, name), f"{name} missing from catalog"
    md_folded = operator_catalog_markdown(tier="full")
    for name in ns:
        if name in {
            "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "GT", "LT", "GE", "LE", "EQ", "NE",
            "AND", "OR", "MAXIMUM", "MINIMUM", "MAX", "MIN", "ABS", "SIGN", "LOG",
            "EXP", "POW", "SQRT", "INV", "NEG", "CAST", "FILLNA",
        }:
            assert not rendered_as_item(md_folded, name), f"{name} should be folded"
        else:
            assert rendered_as_item(md_folded, name), f"{name} missing from grouped catalog"


def test_catalog_basic_time_series_operators_not_folded():
    """DELAY/DELTA/EMA/SMA/WMA 是高频时序主力，必须独立渲染（不得进折叠行）。"""
    md = operator_catalog_markdown()
    for name in ("DELAY", "DELTA", "EMA", "SMA", "WMA"):
        assert any(l.startswith(f"- `{name}(") for l in md.splitlines()), name
    assert "时序基础" in md
