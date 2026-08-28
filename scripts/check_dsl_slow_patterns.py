"""DSL 算子慢模式静态拦截（AST 扫描，不执行代码）。

拦截两类历史上造成评估超时的写法（见 2026-08 慢算子优化）：

- **pandas 逐品种循环**：``for _, sub in _gb_instrument(...)``（或 groupby(level=...)）
  循环体内调用 ``reindex`` / ``get_indexer``——每品种一次全表对齐，品种数上万时
  累积数十秒开销，且内核只能单核。应改用 ``ops_kit.instrument_group_order`` 稳定归组
  + ``accel`` 边界并行内核（见 WICK_EFFICIENCY / CHIP_* 的现有写法）。
- **纯 Python 逐 bar 循环**：未被 ``@njit`` 装饰的函数体内出现按面板长度迭代的
  ``for i in range(...)``——全面板百万级 Python 迭代（PRICE_GAP 状态机旧病）。
  应移植为 Numba 内核或改用向量化表达。

存量命中以函数名记入 ``WHITELIST``（只减不增）；**新增命中使本脚本退出码为 1**。
用法::

    python scripts/check_dsl_slow_patterns.py            # 检查默认文件
    python scripts/check_dsl_slow_patterns.py a.py b.py  # 检查指定文件
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

DEFAULT_FILES = [Path("alphaagent/dsl/core/operators.py")]

# 存量命中（函数名 → 规则）。仅在删除对应慢模式时移除，禁止新增。
WHITELIST: dict[str, set[str]] = {
    "operators.py": {
        # R1 pandas 逐品种循环（多为 numba 缺失时的回落路径）
        "_price_gap_output",       # PRICE_GAP 回落路径
        "_chip_metric_daily",      # CHIP 回落路径
        "_chip_roll_daily",        # CHIP 回落路径
        "CHIP_WASS_DIST",          # CHIP 回落路径
        "WICK_EFFICIENCY",         # 回落路径
        "_crowd_roll_panel",       # CROWD 回落路径
        "VOLUME_CLOCK_VPIN",       # 回落路径
        "MUTUAL_INFO_LAG",         # 回落路径
        "_ts_bivariate_fixed",     # TS_CORR/COV/RANKCORR（待接边界并行层）
        "TS_TREND_RANK",           # 待接边界并行层
        "_ts_cond_roll",           # 动态窗回落路径
        "TS_EFFICIENCY_RATIO",     # 动态窗回落路径
        "KLINE_GEOMETRY",          # 动态窗路径（固定窗已优化）
        "CS_NEUTRALIZE",           # 截面逐日循环
        "CS_GROUP_RANK",
        "CS_RESIDUALIZE",
        "_bucket_cs_fallback",
        # R1 pandas 逐品种循环（多为 numba 缺失时的回落路径）
        "_ts_chan_fractal_3bar_bivariate",  # 分型回落路径
        # R2 纯 Python 逐 bar 循环
        "_ts_unary_fast",          # 修复模板自身的按品种调度循环（循环体只调 numba 内核，非逐 bar）
        "_instrument_price_gap_state",  # PRICE_GAP 回落状态机（numba 版在 accel.py）
        "_ts_agg_fixed_accel",     # 归组本身含 numpy 操作，非逐 bar 热循环（复核后豁免）
        "CS_BUCKET",               # 逐日截面 qcut（每日一次，非逐 bar）
        "rolling_beta",            # 滚动 OLS lstsq 回落（numba 优先）
        "rolling_residuals",       # 滚动 OLS lstsq 回落（numba 优先）
    },
}


def _func_name(node: ast.AST) -> str:
    return getattr(node, "name", "<lambda>")


def _call_name(node: ast.Call) -> str:
    """点链调用名的尽力还原：``df.index.get_indexer(...)`` → "get_indexer"。"""
    fn = node.func
    while isinstance(fn, ast.Attribute):
        if fn.attr in ("reindex", "get_indexer", "_gb_instrument", "groupby"):
            return fn.attr
        fn = fn.value
    if isinstance(fn, ast.Name):
        return fn.id
    return ""


def _loop_is_gb_instrument(node: ast.For) -> bool:
    it = node.iter
    if isinstance(it, ast.Call):
        return _call_name(it) in ("_gb_instrument", "groupby")
    return False


def _has_forbidden_calls(body: list[ast.stmt]) -> tuple[bool, str]:
    for stmt in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(stmt, ast.Call):
            nm = _call_name(stmt)
            if nm in ("reindex", "get_indexer"):
                return True, nm
    return False, ""


def _range_loops_without_njit(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
    """函数未被 njit 装饰，且体内存在 ``for x in range(...)`` 数值循环。"""
    for dec in fn.decorator_list:
        txt = ast.unparse(dec) if hasattr(ast, "unparse") else ""
        if "njit" in txt:
            return False
    for stmt in ast.walk(fn):
        if isinstance(stmt, ast.For) and isinstance(stmt.iter, ast.Call):
            if _call_name(stmt.iter) == "range":
                return True
    return False


def check_file(path: Path) -> list[str]:
    hits: list[str] = []
    whitelist = WHITELIST.get(path.name, set())
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    funcs: dict[str, ast.AST] = {}

    def _collect(node: ast.AST) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                funcs[_func_name(child)] = child
                _collect(child)
            else:
                _collect(child)

    _collect(tree)

    for fname, fn in funcs.items():
        # R1: gb 循环体内含 reindex / get_indexer
        for stmt in ast.walk(fn):
            if isinstance(stmt, ast.For) and _loop_is_gb_instrument(stmt):
                bad, nm = _has_forbidden_calls(stmt.body)
                if bad and fname not in whitelist:
                    hits.append(
                        f"{path}:{stmt.lineno} [R1] {fname}: 逐品种循环体内调用 {nm}() "
                        f"— 请改用 ops_kit.instrument_group_order + accel 边界并行内核"
                    )
        # R2: 未 njit 的函数含 range 逐 bar 循环
        if _range_loops_without_njit(fn) and fname not in whitelist:
            hits.append(
                f"{path}:{fn.lineno} [R2] {fname}: 非 Numba 函数内存在 range 逐元素循环 "
                f"— 请移植为 Numba 内核或向量化"
            )
    return hits


def main(argv: list[str]) -> int:
    files = [Path(a) for a in argv[1:]] or DEFAULT_FILES
    all_hits: list[str] = []
    for f in files:
        if not f.is_file():
            print(f"[skip] 文件不存在: {f}")
            continue
        all_hits.extend(check_file(f))

    n_wl = sum(len(v) for v in WHITELIST.values())
    if all_hits:
        print(f"\n发现 {len(all_hits)} 处新增慢模式（存量白名单 {n_wl} 项）：")
        for h in all_hits:
            print("  " + h)
        print("\n新增逐品种循环/纯 Python 逐 bar 循环被拦截；")
        print("修复模板见 ops_kit.instrument_group_order + accel 边界并行内核（WICK_EFFICIENCY 为范例）。")
        return 1
    print(f"DSL 慢模式检查通过（存量白名单 {n_wl} 项，无新增命中）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
