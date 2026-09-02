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
