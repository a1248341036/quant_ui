"""多行因子表达式求值：调用 dsl.parser 将用户文本解析为 Python 代码串，将 $列名 解析为注入的面板列，在 function_registry 命名空间中顺序 exec 赋值行并对最终表达式 eval。

失败时抛出 MultiLineFactorEvalError，附带生成代码行号与用户源码行映射。

多周期扩展：DSL 支持 ``$field@5m`` / ``@10m`` / ``@60m`` / ``@1h`` / ``@1d`` 等（与 ``data.datasets.normalize_bar_interval``
一致）。每条赋值语句按引用列的频率分派：

- **纯辅周期行**（本行只含某单一 ``@<freq>`` 列/中间变量）-> 在该辅周期面板上直接运算；
- **多辅频无主频行**（本行引用多种辅周期列/中间变量，且不含主频列）-> 与混合行相同：各辅频先广播到
  主频索引再求值；
- **混合**（本行同时含主频列与一或多个辅周期列/中间变量，或仅主频列）-> 主频行上求值，辅周期先按
  「最近一根已完成 bar」广播到主频索引（见 ``data.resample.broadcast_timeframe_to_main_freq``）；
- 中间变量保留各自频率，被跨频引用时再按需广播。

``eval_multi_line_factor`` 可传 ``aux_panels={"5m": df5, "60m": df60}`` 预建辅表，缺省由主表
``build_timeframe_panel`` 现算。兼容旧参 ``df_60m=...`` 等价于 ``aux_panels`` 中 ``"60m"`` 一项。

窗口语义：``TS_*`` 在**纯**辅周期行上时 ``N`` 为**该频**的 bar 数；**混合**行中若将 ``TS_*(...@f, N)``
与主频列写在同一行，``@f`` 会先被广播，``N`` 会表现为主频 bar 数，需拆多行见提示词/文档。
"""
from __future__ import annotations

import re
import traceback
import warnings
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from alphaagent.dsl.core.errors import MultiLineFactorEvalError
from alphaagent.dsl.core.parser import (
    dollar_ref_to_pyname,
    parse_multi_line_expression,
    parse_symbol,
)
from alphaagent.dsl.stock.aux_cache import get_or_build_aux_panel
from alphaagent.dsl.stock.intervals import normalize_bar_interval
from alphaagent.dsl.stock.resample import broadcast_timeframe_to_main_freq
from alphaagent.dsl.registry import build_operator_namespace, resolve_extension_modules

# 股票日频默认主周期
DEFAULT_BASE_INTERVAL = "1d"


def _function_lib_namespace(
    operator_modules: Sequence[str] | None = None,
) -> Dict[str, Any]:
    return build_operator_namespace(
        extension_modules=resolve_extension_modules(operator_modules),
    )


def _dollar_columns(df: pd.DataFrame) -> List[str]:
    return ["$" + str(c) for c in df.columns]


def _dollar_columns_aux(panel: Optional[pd.DataFrame], tag: str) -> List[str]:
    if panel is None or getattr(panel, "empty", True):
        return []
    t = normalize_bar_interval(tag)
    return [f"${c}@{t}" for c in panel.columns]


# ``$name@5m`` 等，捕获频率片段并归一化（跳过字面量后扫描）
_DOLLAR_AT_REF_RE = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*@([A-Za-z0-9_]+)\b")


def collect_aux_intervals_from_expr(multi_line_expr: str) -> List[str]:
    """从 DSL 中收集 ``$x@<freq>`` 里的 ``<freq>``，归一化、保序、去重。"""
    s = _strip_string_literals(multi_line_expr)
    seen: set[str] = set()
    out: List[str] = []
    for m in _DOLLAR_AT_REF_RE.finditer(s):
        raw = m.group(1)
        try:
            tag = normalize_bar_interval(raw)
        except ValueError as exc:
            raise ValueError(
                f"表达式含不支持的辅频 @{raw!r}（股票仅支持 @1d / @1w）"
            ) from exc
        if tag not in seen:
            seen.add(tag)
            out.append(tag)
    return out


def _column_bindings(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    return {str(c): df[[c]] for c in df.columns}


_IDENT_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
_ASSIGN_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.+)$")


def _strip_string_literals(s: str) -> str:
    out = re.sub(r"'[^']*'", "", s)
    out = re.sub(r'"[^"]*"', "", out)
    return out


def _extract_identifier_refs(rhs: str, known: set) -> List[str]:
    cleaned = _strip_string_literals(rhs)
    return [t for t in _IDENT_RE.findall(cleaned) if t in known]


def _split_assignment(line: str) -> Tuple[Optional[str], str]:
    m = _ASSIGN_RE.match(line)
    if m:
        return m.group(1), m.group(2).strip()
    return None, line


def _split_generated_lines(code: str) -> List[str]:
    return [ln.strip() for ln in code.strip().split("\n") if ln.strip()]


def _nonblank_user_lines(multi_line_expr: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for i, line in enumerate(multi_line_expr.splitlines(), start=1):
        s = line.strip()
        if s:
            out.append((i, s))
    return out


def _align_user_to_generated(
    generated_line_no: int,
    n_generated: int,
    multi_line_expr: str,
) -> Tuple[Optional[int], Optional[str]]:
    user = _nonblank_user_lines(multi_line_expr)
    if not user or n_generated < 1:
        return None, None
    if len(user) != n_generated:
        return None, None
    if not (1 <= generated_line_no <= len(user)):
        return None, None
    lineno, text = user[generated_line_no - 1]
    return lineno, text


def _problem_from_exception(exc: BaseException) -> Tuple[str, str]:
    tname = type(exc).__name__
    if isinstance(exc, NameError):
        name = getattr(exc, "name", None)
        if not name:
            m = re.search(r"name ['\"](\w+)['\"] is not defined", str(exc))
            name = m.group(1) if m else None
        if name:
            return tname, f"未定义的名称 {name!r}（缺少函数/变量或列未注入）"
        return tname, str(exc) or repr(exc)
    if isinstance(exc, (TypeError, ValueError, ZeroDivisionError)):
        return tname, str(exc) or repr(exc)
    if isinstance(exc, KeyError):
        return tname, f"键不存在或列缺失: {exc!r}"
    if isinstance(exc, AssertionError):
        return tname, str(exc) or "断言失败"
    return tname, str(exc) or repr(exc)


def _infer_user_line_parse_phase(multi_line_expr: str, exc: Exception) -> Tuple[Optional[int], Optional[str]]:
    msg = str(exc)
    user_lines = _nonblank_user_lines(multi_line_expr)
    if not user_lines:
        return None, None

    m = re.search(r"解析变量\s+(\w+)\s+的表达式失败", msg)
    if m:
        var = m.group(1)
        for lineno, text in user_lines:
            if re.match(rf"^{re.escape(var)}\s*=", text):
                return lineno, text
        return None, None

    if "解析最终表达式失败" in msg or "未找到最终表达式" in msg:
        lineno, text = user_lines[-1]
        return lineno, text

    m = re.search(r"表达式`([^`]+)`解析失败", msg)
    if m:
        snippet = m.group(1).strip()
        for lineno, text in user_lines:
            if snippet in text.replace(" ", "") or snippet in text:
                return lineno, text
        for lineno, text in user_lines:
            if snippet in multi_line_expr:
                return lineno, text

    return None, None


def _build_eval_error(
    phase: str,
    exc: BaseException,
    *,
    generated_code: str,
    generated_line_no: Optional[int],
    generated_line_text: Optional[str],
    multi_line_expr: str,
) -> MultiLineFactorEvalError:
    et, problem = _problem_from_exception(exc)
    tb = traceback.format_exc()
    n_gen = len([ln for ln in generated_code.strip().split("\n") if ln.strip()])
    u_no, u_text = (None, None)
    if generated_line_no is not None and n_gen > 0:
        u_no, u_text = _align_user_to_generated(
            generated_line_no, n_gen, multi_line_expr
        )
    summary = f"{phase} 阶段失败: {problem}"
    return MultiLineFactorEvalError(
        summary,
        phase=phase,
        problem=problem,
        exception_type=et,
        generated_code=generated_code,
        generated_line_no=generated_line_no,
        generated_line_text=generated_line_text,
        user_source=multi_line_expr,
        user_line_no=u_no,
        user_line_text=u_text,
        traceback_text=tb,
    )


def compile_multi_line_factor(
    multi_line_expr: str,
    *,
    columns: Optional[Sequence[str]] = None,
    verbose: bool = False,
) -> str:
    try:
        raw = parse_multi_line_expression(multi_line_expr, verbose=verbose)
    except Exception as e:
        u_no, u_text = _infer_user_line_parse_phase(multi_line_expr, e)
        et, problem = _problem_from_exception(e)
        tb = traceback.format_exc()
        raise MultiLineFactorEvalError(
            f"parse 阶段失败: {problem}",
            phase="parse",
            problem=problem,
            exception_type=et,
            user_source=multi_line_expr,
            user_line_no=u_no,
            user_line_text=u_text,
            traceback_text=tb,
        ) from e

    if not columns:
        return raw

    try:
        return parse_symbol(raw, list(columns))
    except Exception as e:
        et, problem = _problem_from_exception(e)
        tb = traceback.format_exc()
        raise MultiLineFactorEvalError(
            f"symbol 阶段失败: {problem}",
            phase="symbol",
            problem=problem,
            exception_type=et,
            generated_code=raw,
            user_source=multi_line_expr,
            traceback_text=tb,
        ) from e


SCOPE_MAIN = "main"

# 混频下生成码里辅周期列名为 ``col__freq``；若同列出现 ``TS_*(...`` 且实参区含该形式，则 ``N`` 按主频计。
_MIXED_LINE_TS_WITH_AUX_ARG_RE = re.compile(
    r"TS_[A-Z][A-Z0-9_]*\s*\([^)]*__[A-Za-z0-9_]+"
)


def _warn_mixed_line_ts_on_broadcast_aux(*, rhs: str, has_main: bool) -> None:
    if not has_main or not _MIXED_LINE_TS_WITH_AUX_ARG_RE.search(rhs):
        return
    warnings.warn(
        "混频行：TS_*() 的实参含辅周期列（已广播到主频索引），"
        "其窗口参数 N 按主频 bar 数计，不是辅周期根数；"
        "需要「N 根辅周期」时请先在仅含 @<周期> 的一行上算好中间变量。",
        UserWarning,
        stacklevel=4,
    )


def _merge_build_aux_panels(
    df: pd.DataFrame,
    *,
    required: List[str],
    base_interval: str,
    df_60m: Optional[pd.DataFrame],
    aux_panels: Optional[Mapping[str, pd.DataFrame]],
    aux_cache: Optional[MutableMapping[tuple[int, str, str], pd.DataFrame]] = None,
) -> Dict[str, pd.DataFrame]:
    """合并用户传入的辅表与按 required 自动聚合的结果（带缓存）。"""
    aux: Dict[str, pd.DataFrame] = {}
    if aux_panels:
        for k, v in aux_panels.items():
            if v is not None and not getattr(v, "empty", True):
                aux[normalize_bar_interval(str(k))] = v
    if df_60m is not None:
        try:
            aux.setdefault("60m", df_60m)
        except ValueError:
            pass
    for tag in required:
        if tag not in aux:
            aux[tag] = get_or_build_aux_panel(
                df,
                tag,
                base_interval=base_interval,
                cache=aux_cache,
            )
    return aux


def _compile_column_tokens(df: pd.DataFrame, aux: Dict[str, pd.DataFrame]) -> List[str]:
    cols = _dollar_columns(df)
    for tag in sorted(aux.keys()):
        cols.extend(_dollar_columns_aux(aux[tag], tag))
    return cols


def eval_multi_line_factor(
    multi_line_expr: str,
    df: pd.DataFrame,
    *,
    df_60m: Optional[pd.DataFrame] = None,
    aux_panels: Optional[Mapping[str, pd.DataFrame]] = None,
    base_interval: str = DEFAULT_BASE_INTERVAL,
    columns: Optional[Sequence[str]] = None,
    operator_modules: Optional[Sequence[str]] = None,
    extra_namespace: Optional[Mapping[str, Any]] = None,
    function_lib_namespace: Optional[Mapping[str, Any]] = None,
    aux_cache: Optional[MutableMapping[tuple[int, str, str], pd.DataFrame]] = None,
    verbose: bool = False,
) -> Any:
    """对多行因子 DSL 求值（股票主频默认 1d，辅频 @1d / @1w）。"""
    required = collect_aux_intervals_from_expr(multi_line_expr)
    use_multi = bool(required) or (df_60m is not None)
    aux: Dict[str, pd.DataFrame] = {}
    # 默认启用辅表缓存，避免重复聚合 1w
    if aux_cache is None:
        aux_cache = {}
    if use_multi:
        aux = _merge_build_aux_panels(
            df,
            required=required,
            base_interval=base_interval,
            df_60m=df_60m,
            aux_panels=aux_panels,
            aux_cache=aux_cache,
        )

    if columns is not None:
        cols = list(columns)
    else:
        cols = _compile_column_tokens(df, aux) if use_multi else _dollar_columns(df)
    code = compile_multi_line_factor(
        multi_line_expr, columns=cols, verbose=verbose
    )
    lines = _split_generated_lines(code)

    if not lines:
        raise MultiLineFactorEvalError(
            "生成代码为空（解析结果无任何可执行行）",
            phase="eval",
            problem="生成代码为空",
            exception_type="ValueError",
            generated_code=code,
            user_source=multi_line_expr,
        )

    base_ns: Dict[str, Any] = {
        "__builtins__": __builtins__,
        "np": np,
        "pd": pd,
    }
    if function_lib_namespace is not None:
        base_ns.update(dict(function_lib_namespace))
    else:
        base_ns.update(_function_lib_namespace(operator_modules))
    if extra_namespace:
        base_ns.update(dict(extra_namespace))

    if not use_multi:
        return _eval_single_frequency(
            lines=lines,
            code=code,
            df=df,
            base_ns=base_ns,
            multi_line_expr=multi_line_expr,
        )
    return _eval_multi_frequency(
        lines=lines,
        code=code,
        df_main=df,
        aux=aux,
        base_ns=base_ns,
        multi_line_expr=multi_line_expr,
    )


def _eval_single_frequency(
    *,
    lines: List[str],
    code: str,
    df: pd.DataFrame,
    base_ns: Dict[str, Any],
    multi_line_expr: str,
) -> Any:
    ctx: MutableMapping[str, Any] = dict(base_ns)
    ctx.update(_column_bindings(df))

    if len(lines) == 1:
        line = lines[0]
        try:
            return eval(line, ctx, ctx)
        except Exception as e:
            raise _build_eval_error(
                "eval",
                e,
                generated_code=code,
                generated_line_no=1,
                generated_line_text=line,
                multi_line_expr=multi_line_expr,
            ) from e

    for i, line in enumerate(lines[:-1], start=1):
        try:
            exec(line, ctx, ctx)
        except Exception as e:
            raise _build_eval_error(
                "exec",
                e,
                generated_code=code,
                generated_line_no=i,
                generated_line_text=line,
                multi_line_expr=multi_line_expr,
            ) from e

    last = lines[-1]
    last_no = len(lines)
    try:
        return eval(last, ctx, ctx)
    except Exception as e:
        raise _build_eval_error(
            "eval",
            e,
            generated_code=code,
            generated_line_no=last_no,
            generated_line_text=last,
            multi_line_expr=multi_line_expr,
        ) from e


def _eval_multi_frequency(
    *,
    lines: List[str],
    code: str,
    df_main: pd.DataFrame,
    aux: Dict[str, pd.DataFrame],
    base_ns: Dict[str, Any],
    multi_line_expr: str,
) -> Any:
    """按频率作用域分派的多频求值：详见模块 docstring。"""
    bindings_main = _column_bindings(df_main)
    native: Dict[str, Dict[str, pd.DataFrame]] = {}
    broadcast: Dict[str, Dict[str, pd.DataFrame]] = {}
    for tag, panel in aux.items():
        if panel is None or getattr(panel, "empty", True):
            continue
        native[tag] = {}
        broadcast[tag] = {}
        for c in panel.columns:
            py = dollar_ref_to_pyname(f"${c}@{tag}")
            native[tag][py] = panel[[c]]
            broadcast[tag][py] = broadcast_timeframe_to_main_freq(
                panel[[c]], df_main.index, tag
            )

    panel_freq: Dict[str, str] = {}
    for n in bindings_main:
        panel_freq[n] = SCOPE_MAIN
    for tag in native:
        for py in native[tag]:
            panel_freq[py] = tag

    ivar_value: Dict[str, Any] = {}
    ivar_freq: Dict[str, str] = {}

    def _known_names() -> set:
        return set(panel_freq.keys()) | set(ivar_freq.keys())

    def _ref_freq(name: str) -> str:
        if name in panel_freq:
            return panel_freq[name]
        return ivar_freq[name]

    def _ensure_main(value: Any, src_freq: str) -> Any:
        if src_freq == SCOPE_MAIN:
            return value
        return broadcast_timeframe_to_main_freq(value, df_main.index, src_freq)

    def _eval_rhs(rhs: str, line_no: int, line_text: str, phase: str) -> Tuple[Any, str]:
        refs = _extract_identifier_refs(rhs, _known_names())
        freqs = [_ref_freq(r) for r in refs]
        has_main = any(f == SCOPE_MAIN for f in freqs)
        aux_in_refs = {f for f in freqs if f != SCOPE_MAIN}

        if not has_main and len(aux_in_refs) == 1:
            tag = next(iter(aux_in_refs))
            ctx = dict(base_ns)
            for r in refs:
                if r in native.get(tag, {}):
                    ctx[r] = native[tag][r]
                elif r in ivar_value:
                    if ivar_freq[r] != tag:
                        raise MultiLineFactorEvalError(
                            f"纯 @{tag} 作用域中引用了其它频率中间变量 {r!r}；请改为混合表达式或先升频",
                            phase=phase,
                            problem="跨频引用（辅频作用域）",
                            exception_type="ValueError",
                            generated_code=code,
                            generated_line_no=line_no,
                            generated_line_text=line_text,
                            user_source=multi_line_expr,
                        )
                    ctx[r] = ivar_value[r]
            try:
                result = eval(rhs, ctx, ctx)
            except Exception as e:
                raise _build_eval_error(
                    phase,
                    e,
                    generated_code=code,
                    generated_line_no=line_no,
                    generated_line_text=line_text,
                    multi_line_expr=multi_line_expr,
                ) from e
            return result, tag

        ctx = dict(base_ns)
        for r in refs:
            if r in bindings_main:
                ctx[r] = bindings_main[r]
            else:
                placed = False
                for t in native:
                    if r in broadcast[t]:
                        ctx[r] = broadcast[t][r]
                        placed = True
                        break
                if not placed and r in ivar_value:
                    ctx[r] = _ensure_main(ivar_value[r], ivar_freq[r])
        _warn_mixed_line_ts_on_broadcast_aux(rhs=rhs, has_main=has_main)
        try:
            result = eval(rhs, ctx, ctx)
        except Exception as e:
            raise _build_eval_error(
                phase,
                e,
                generated_code=code,
                generated_line_no=line_no,
                generated_line_text=line_text,
                multi_line_expr=multi_line_expr,
            ) from e
        return result, SCOPE_MAIN

    last_idx = len(lines) - 1
    final_value: Any = None
    final_freq: str = SCOPE_MAIN
    for i, line in enumerate(lines):
        is_last = i == last_idx
        lhs, rhs = _split_assignment(line)
        phase = "eval" if is_last else "exec"
        line_no = i + 1
        value, freq = _eval_rhs(rhs, line_no, line, phase)
        if lhs is not None and not is_last:
            ivar_value[lhs] = value
            ivar_freq[lhs] = freq
        if is_last:
            final_value = value
            final_freq = freq
            if lhs is not None:
                ivar_value[lhs] = value
                ivar_freq[lhs] = freq

    if final_freq != SCOPE_MAIN:
        return broadcast_timeframe_to_main_freq(final_value, df_main.index, final_freq)
    return final_value


def _smoke_test_dataframe() -> pd.DataFrame:
    idx = pd.MultiIndex.from_product(
        [
            pd.date_range("2020-01-01", periods=8, freq="min", name="datetime"),
            ["SYM1", "SYM2"],
        ],
        names=["datetime", "instrument"],
    )
    n = len(idx)
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "open": rng.random(n) + 1.0,
            "close": rng.random(n) + 1.0,
        },
        index=idx,
    )


def eval_factor(
    expr: str,
    panel: pd.DataFrame,
    **kwargs: Any,
) -> pd.Series:
    """对表达式求值，返回与 panel 对齐的单列 Series（股票默认日频）。"""
    kwargs.setdefault("base_interval", DEFAULT_BASE_INTERVAL)
    out = eval_multi_line_factor(expr, panel, **kwargs)
    if isinstance(out, pd.DataFrame):
        return out.iloc[:, 0]
    if isinstance(out, pd.Series):
        return out
    raise TypeError(f"因子输出须为 DataFrame/Series，得到 {type(out)!r}")


if __name__ == "__main__":
    df_test = _smoke_test_dataframe()
    expr_ok = """
    x = TS_DELTA($open@5m, 1)
    y = TS_RANK(SUBTRACT(x, $close@5m), 5)
    DIVIDE(y, ADD(1e-8, 1))
    """
    print("示例输入1: expr_ok =")
    print(expr_ok)
    compiled = compile_multi_line_factor(expr_ok, columns=_dollar_columns(df_test))
    print("编译通过，生成代码:\n", compiled)

    out = eval_multi_line_factor(expr_ok, df_test)
    print("求值 OK, shape:", getattr(out, "shape", out))

    expr_bad = """
    x = TS_DELTA($open, 1)
    y = TS_RANK(UNKNOWN_FUNC($close), 5)
    y 
    """
    print("示例输入2: expr_bad =")
    print(expr_bad)
    try:
        eval_multi_line_factor(expr_bad, df_test)
    except MultiLineFactorEvalError as e:
        print("预期失败 (结构化错误信息):\n", e)
   