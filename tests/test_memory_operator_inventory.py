# -*- coding: utf-8 -*-
"""记忆层算子清单真源回归（2026-10-18 修复）。

背景：`memory/expressions.py` 曾只用硬编码 `_KNOWN_OPERATORS` 识别算子，清单外
一切算子被静默丢弃 —— `_COLD_OPS` 22/22 与 `CS_BUCKET`/`CS_GROUP_RANK`/`NEG`/
`SIGNAL_BLEND` 从未进入 `operator_list_json`（实测 16324 条含冷门算子的表达式、
0 条正确落库），且 `CHIP_PEAK_LOC` 撞 `chip_*` 变量前缀规则被误判成变量 →
算子分布软引导 / 按算子检索记忆 / 算子×面台账对冷门算子结构性失明。

本文件锁定四件事：
1. 算子识别真源 = DSL 注册表 ∪ 静态回落白名单；
2. 冷门算子被识别为算子、且不再被误判为变量（含窗口参数关联）；
3. 防漂移守卫：DSL 注册表里的任何算子都必须被记忆层识别；
4. 注册表不可用时优雅回落静态白名单（不报错、不静默变空）。
"""

from __future__ import annotations

import time

from alphaagent.dsl.registry import build_operator_namespace
from alphaagent.factor.mining.memory import expressions as E
from alphaagent.factor.mining.memory.retrieval import (
    RetrievalMixin,
    _COLD_OPS,
    _SHELL_OPS,
)

COLD_EXPR = (
    "signal = CHIP_PEAK_LOC($adj_close, $adj_low, $adj_high, $volume, 120, $float_cap)\n"
    "sig = CS_ZSCORE(TS_MEAN(signal, 40))\n"
    "r = CS_NEUTRALIZE(sig, CS_BUCKET(LOG($float_cap), 10))\n"
    "CS_NEUTRALIZE(r, $industry_sw_l1)"
)


def _attempt(op: str, window: int) -> dict:
    expr = (
        f"signal = {op}($adj_close, $adj_low, $adj_high, $volume, {window}, $float_cap)\n"
        f"sig = CS_ZSCORE(TS_MEAN(signal, {window}))\n"
        "r = CS_NEUTRALIZE(sig, CS_BUCKET(LOG($float_cap), 10))\n"
        "CS_NEUTRALIZE(r, $industry_sw_l1)"
    )
    return {"expression": expr, "ok": True}


# ── 1. 真源与覆盖 ──


def test_cold_operators_all_recognized():
    """22 个冷门算子必须全部被识别（修复前为 0/22）。"""
    names = E._operator_names()
    missing = sorted({o.lower() for o in _COLD_OPS} - names)
    assert not missing, f"冷门算子未被识别: {missing}"


def test_shell_operators_all_recognized():
    """套壳算子同样不能缺（CS_BUCKET/CS_GROUP_RANK/NEG/SIGNAL_BLEND 修复前缺失）。"""
    names = E._operator_names()
    missing = sorted({o.lower() for o in _SHELL_OPS} - names)
    assert not missing, f"套壳算子未被识别: {missing}"


def test_registry_operators_are_recognized_guard():
    """防漂移守卫：DSL 注册表新增算子后，记忆层必须自动跟上，无需改本文件。"""
    registry = {str(n).lower() for n in build_operator_namespace()}
    assert registry, "注册表为空，守卫失效"
    assert registry <= E._operator_names()


# ── 2. 提取语义 ──


def test_cold_operator_extracted_not_variable():
    st = E._parse_expression_structure(COLD_EXPR)
    assert "chip_peak_loc" in st["operators"]
    assert "chip_peak_loc" not in st["variables"], "冷门算子被误判为变量（chip_* 前缀撞名）"
    assert "adj_close" in st["variables"] and "volume" in st["variables"]
    assert "cs_zscore" in st["operators"] and "ts_mean" in st["operators"]
    assert "cs_bucket" in st["operators"], "套壳算子 CS_BUCKET 缺失"


def test_cold_operator_window_params_captured():
    st = E._parse_expression_structure(COLD_EXPR)
    assert st["window_params"].get("chip_peak_loc") == [120]
    assert st["window_params"].get("ts_mean") == [40]


def test_expression_features_feeds_operator_list():
    """ingestion 落库路径（expression_features → operator_list_json）必须带冷门算子。"""
    feats = E.expression_features(COLD_EXPR)
    assert "chip_peak_loc" in feats["operators"]
    assert "chip_peak_loc" not in feats["variables"]
    assert E.expression_ops(COLD_EXPR) == feats["operators"]


def test_crowd_operator_not_treated_as_variable():
    """CROWD_* 与 crowd_ 变量前缀同型，同样不能落进 variables。"""
    expr = "sig = CS_ZSCORE(CROWD_SHARE($volume, 10))\nCS_NEUTRALIZE(sig, $industry_sw_l1)"
    st = E._parse_expression_structure(expr)
    assert "crowd_share" in st["operators"]
    assert "crowd_share" not in st["variables"]


# ── 2b. `$` 前缀消歧（算子名与字段名同型）──


def test_dollar_prefixed_name_is_field_not_operator():
    """`$chip_peak_loc` 是筹码面**字段**，不是 CHIP_PEAK_LOC 算子。

    `chip_*`/`crowd_*` 在本仓既作筹码/拥挤面字段（facets.py 按 `chip_` 前缀识面），
    又作算子名（CHIP_PEAK_LOC / CROWD_SHARE）；唯一判据是 `$` 前缀。
    """
    st = E._parse_expression_structure("RANK(TS_MEAN($chip_peak_loc, 20))")
    assert "chip_peak_loc" not in st["operators"]
    assert "chip_peak_loc" in st["variables"]
    assert "chip_peak_loc" not in st["window_params"], "$ 前缀字段不得被当算子关联窗口"


def test_bare_and_dollar_forms_coexist():
    """同一表达式里裸算子与 `$` 同名字段各归其位。"""
    expr = "sig = CS_ZSCORE(CHIP_PEAK_LOC($adj_close, $volume, 20, $float_cap))\n" \
           "base = TS_MEAN($chip_peak_loc, 20)\nCS_NEUTRALIZE(sig, $industry_sw_l1)"
    st = E._parse_expression_structure(expr)
    assert "chip_peak_loc" in st["operators"]
    assert "chip_peak_loc" in st["variables"]


def test_dollar_field_swap_keeps_edit_motif():
    """端到端：字段替换 motif 仍判 feature_swap_chip（修复期的回归点）。"""
    from alphaagent.factor.mining.memory.expressions import extract_edit_motif

    parent = "RANK(TS_MEAN($close, 20))"
    child = "RANK(TS_MEAN($chip_peak_loc, 20))"
    assert extract_edit_motif(parent, child) == "feature_swap_chip"


def test_dollar_field_suffix_does_not_leak_operator():
    """边界必须锚在标识符起点之外：仅写 `(?<!\\$)` 时 `$raw_rank` 会在内部匹配出算子 `rank`。

    （OCR review medium 发现：负向后顾只拦 `$` 紧跟的那一个起始位置。）
    """
    st = E._parse_expression_structure("sig = TS_MEAN($raw_rank, 20)")
    assert st["operators"] == ["ts_mean"]
    assert "rank" not in st["operators"]

    st2 = E._parse_expression_structure("sig = TS_MEAN($my_volume_rank, 20)\nRANK(sig)")
    assert st2["operators"] == ["ts_mean", "rank"], "只有独立位置的 RANK 才算算子"


def test_real_operator_at_boundaries_still_detected():
    """收紧边界后，正常写法（行首/括号/逗号/空格后）不得漏识别。"""
    expr = ("sig = CS_ZSCORE(TS_MEAN(CHIP_PEAK_LOC($adj_close, $volume, 20, $float_cap), 40))\n"
            "CS_NEUTRALIZE(sig, $industry_sw_l1)")
    ops = E._parse_expression_structure(expr)["operators"]
    assert set(ops) >= {"chip_peak_loc", "ts_mean", "cs_zscore", "cs_neutralize"}


def test_no_spurious_operator_from_plain_variables():
    """普通字段（无 $ 冲突）不得凭空产出算子。"""
    st = E._parse_expression_structure("sig = TS_MEAN($adj_close, 20)\nCS_ZSCORE(sig)")
    assert st["operators"] == ["ts_mean", "cs_zscore"]


def test_window_params_no_digit_leak_from_dollar_fields():
    """`$adv20` 这类带数字后缀的字段，其尾部数字不得被当成独立常量关联到算子。

    （OCR review 第二轮 medium 发现：lookbehind 只罩标识符分支时，数字分支仍会从
    字段名内部起配 → `TS_MEAN($adv20, 5)` 产出 `{"ts_mean": [20, 5]}`。）
    """
    st = E._parse_expression_structure("sig = TS_MEAN($adv20, 5)")
    assert st["window_params"] == {"ts_mean": [5]}

    # 真实冷门表达式的末行 `CS_NEUTRALIZE(r, $industry_sw_l1)` 不得给壳算子记入 [1]
    cold = E._parse_expression_structure(COLD_EXPR)
    assert "cs_neutralize" not in cold["window_params"]
    assert cold["window_params"].get("chip_peak_loc") == [120]

    # 正常窗口字面量仍须被关联（收紧边界不得误伤）
    st3 = E._parse_expression_structure("sig = TS_MEAN($adj_close, 20)\nEMA(sig, 12)")
    assert st3["window_params"] == {"ts_mean": [20], "ema": [12]}


def test_freq_suffix_digit_not_leaked_into_window_params():
    """`$close@60m` 的 `60` 不得被当成窗口常量（OCR review 第三轮 medium）。

    `@freq` 是 DSL 一等混频语法（`$field@5m/@1d/@1w`，见 dsl/eval.py 头部与
    dsl/core/parser.py），与 `$adv20` 尾部数字同类：分隔符不同、泄漏机理相同。
    """
    st = E._parse_expression_structure("sig = TS_MEAN($volume@60m, 20)")
    assert st["window_params"] == {"ts_mean": [20]}

    mixed = (
        "x = TS_MEAN($open@5m, 5)\n"
        "y = TS_MEAN($close@1d, 20)\n"
        "RANK(SUBTRACT(y, x))"
    )
    st2 = E._parse_expression_structure(mixed)
    assert st2["window_params"] == {"ts_mean": [5, 20]}
    assert "rank" in st2["operators"] and "subtract" in st2["operators"]


def test_local_variable_shadowing_operator_name_not_counted():
    """与算子同名的中间变量不得被计成算子（OCR review 第三轮 low）。

    注册表算子含 delta/delay/scale/pow/exp/inv/cast/neg 等常见变量名；不要求调用形态
    时，`delta = ...` 与随后的裸引用 `RANK(delta)` 都会被计成算子并把它挤出 variables。
    """
    expr = "delta = SUBTRACT(TS_MEAN($close, 5), TS_MEAN($close, 20))\nRANK(delta)"
    st = E._parse_expression_structure(expr)
    assert "delta" not in st["operators"]
    assert st["operators"] == ["subtract", "ts_mean", "ts_mean", "rank"]
    # 裸引用也不得进入窗口关联队列
    assert set(st["window_params"]) == {"ts_mean"}


# ── 3. 回落语义 ──


def test_static_fallback_when_registry_unavailable(monkeypatch):
    """注册表导入/调用失败 → 回落静态白名单，不抛异常（退化为旧行为而非崩掉）。"""
    import alphaagent.dsl.registry as reg

    def _boom(*_a, **_k):
        raise ImportError("simulated registry failure")

    monkeypatch.setattr(reg, "build_operator_namespace", _boom)
    monkeypatch.setattr(E, "_OPERATOR_NAMES", None)
    monkeypatch.setattr(E, "_OPERATOR_NAMES_DEGRADED_AT", None)
    names = E._operator_names()
    assert "ts_mean" in names
    assert "cs_neutralize" in names
    # 兜底即旧行为：冷门算子识别不到，但绝不报错
    assert "chip_peak_loc" not in names


def test_extension_modules_requested_like_eval(monkeypatch):
    """与 `dsl/eval.py` 同源：构造命名空间时必须请求环境声明的扩展算子。

    （OCR review low 发现：无参 `build_operator_namespace()` 只含内置算子，
    扩展算子会在记忆层复现同一类"结构性失明"。）
    """
    import alphaagent.dsl.registry as reg

    seen: dict = {}

    def fake_build(**kwargs):
        seen.update(kwargs)
        return {"TS_MEAN": object(), "MY_EXT_OP": object()}

    monkeypatch.setattr(reg, "build_operator_namespace", fake_build)
    monkeypatch.setattr(reg, "resolve_extension_modules", lambda *_a, **_k: ["my.ext"])
    monkeypatch.setattr(E, "_OPERATOR_NAMES", None)
    monkeypatch.setattr(E, "_OPERATOR_NAMES_DEGRADED_AT", None)

    names = E._operator_names()
    assert seen.get("extension_modules") == ["my.ext"], "必须像 eval.py 一样请求扩展算子"
    assert "my_ext_op" in names


def test_extension_failure_degrades_to_builtin(monkeypatch):
    """扩展算子加载失败 → 只退到"仅内置"，不得整体退到静态白名单。"""
    import alphaagent.dsl.registry as reg

    def fake_build(**kwargs):
        if kwargs.get("extension_modules"):
            raise RuntimeError("bad extension module")
        return {"TS_MEAN": object(), "CHIP_PEAK_LOC": object()}

    monkeypatch.setattr(reg, "build_operator_namespace", fake_build)
    monkeypatch.setattr(reg, "resolve_extension_modules", lambda *_a, **_k: ["bad.ext"])
    monkeypatch.setattr(E, "_OPERATOR_NAMES", None)
    monkeypatch.setattr(E, "_OPERATOR_NAMES_DEGRADED_AT", None)

    names = E._operator_names()
    assert "chip_peak_loc" in names, "内置算子仍须可用"


def test_degraded_result_retry_cooldown(monkeypatch):
    """降级结果不永久固化，但按冷却窗口节流重试。

    第二轮 low：瞬时失败若被固化进 `_OPERATOR_NAMES`，进程会一直停在降级态
    （症状从"静默失明"变成"一次 warning 后永久失明"）。
    第三轮 low：反过来，完全不缓存会让持续性故障时**每条表达式**都重新
    `exec_module` 加载扩展，热路径开销可观。故取冷却窗口中庸。
    """
    import alphaagent.dsl.registry as reg

    state = {"fail": True, "calls": 0}

    def flaky_build(**_kwargs):
        state["calls"] += 1
        if state["fail"]:
            raise RuntimeError("registry failure")
        return {"TS_MEAN": object(), "CHIP_PEAK_LOC": object()}

    monkeypatch.setattr(reg, "build_operator_namespace", flaky_build)
    monkeypatch.setattr(reg, "resolve_extension_modules", lambda *_a, **_k: [])
    monkeypatch.setattr(E, "_OPERATOR_NAMES", None)
    monkeypatch.setattr(E, "_OPERATOR_NAMES_DEGRADED_AT", None)
    monkeypatch.setattr(E, "OPERATOR_NAMES_RETRY_COOLDOWN_S", 60.0)

    assert "chip_peak_loc" not in E._operator_names(), "失败时应退到静态白名单"
    calls_after_failure = state["calls"]

    for _ in range(5):
        E._operator_names()
    assert state["calls"] == calls_after_failure, "冷却窗口内不得重复加载注册表/扩展"

    # 冷却过期 → 重试注册表并自动恢复（不重启进程）
    state["fail"] = False
    monkeypatch.setattr(E, "_OPERATOR_NAMES_DEGRADED_AT", time.monotonic() - 61.0)
    assert "chip_peak_loc" in E._operator_names(), "冷却过期后必须重试注册表并恢复"
    assert "chip_peak_loc" in E._operator_names(), "恢复后的成功结果应永久缓存"


def test_success_result_cached_without_retry(monkeypatch):
    """① 成功的集合永久缓存：后续调用不得再触碰注册表。"""
    import alphaagent.dsl.registry as reg

    state = {"calls": 0}

    def counting_build(**_kwargs):
        state["calls"] += 1
        return {"TS_MEAN": object()}

    monkeypatch.setattr(reg, "build_operator_namespace", counting_build)
    monkeypatch.setattr(reg, "resolve_extension_modules", lambda *_a, **_k: [])
    monkeypatch.setattr(E, "_OPERATOR_NAMES", None)
    monkeypatch.setattr(E, "_OPERATOR_NAMES_DEGRADED_AT", None)

    E._operator_names()
    calls = state["calls"]
    for _ in range(5):
        E._operator_names()
    assert state["calls"] == calls
    assert E._operator_names() is E._operator_names()


# ── 4. 端到端症状回归 ──


def test_operator_diversity_block_sees_cold_operators():
    """修复前：20 次尝试全用冷门算子，统计器仍读到冷门使用 0 次、
    并把刚用过的算子当「从未用过」推荐回来。修复后必须能看见。"""
    attempts = [_attempt("CHIP_PEAK_LOC", 20 + i) for i in range(12)]
    attempts += [_attempt("CROWD_SHARE", 10 + i) for i in range(8)]

    block = RetrievalMixin._operator_diversity_block(None, attempts)

    assert block, "扎堆（Top3 100%）时应当注入软引导"
    # 分布里必须出现冷门算子及其 [冷门] 标签
    assert "CHIP_PEAK_LOC" in block and "[冷门]" in block
    assert "CROWD_SHARE" in block
    # 用过的算子不得再出现在「低频/未用」推荐清单里
    assert "CHIP_PEAK_LOC（" not in block
    assert "CROWD_SHARE（" not in block
    # 真正没用过的冷门算子仍应被推荐
    unused = sorted(set(_COLD_OPS) - {"CHIP_PEAK_LOC", "CROWD_SHARE"})
    assert any(f"{op}（从未用过）" in block for op in unused)
