"""DSL 算子清单：供 eval、挖掘 prompt 等模块共享。"""

from __future__ import annotations

import inspect

from alphaagent.dsl.registry import build_operator_namespace


def list_operator_names() -> list[str]:
    return sorted(build_operator_namespace())


def _slim_signature(fn) -> str:
    """签名瘦身：去掉类型标注（对 LLM 是纯噪音），保留参数名顺序与默认值。

    `TS_MEAN(df: 'pd.DataFrame', window: 'Window')` → `TS_MEAN(df, window)`；
    `CHIP_PEAK_LOC(..., nbins: 'int' = 64, method: 'str' = 'cyq')` → `(..., nbins=64, method='cyq')`。
    位置参数顺序与真实签名严格一致（DSL 按位置传参，顺序即语义）。
    """
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return "(...)"
    parts: list[str] = []
    for p in sig.parameters.values():
        if p.default is inspect.Parameter.empty:
            parts.append(p.name)
            continue
        d = repr(p.default) if isinstance(p.default, str) else str(p.default)
        if len(d) > 24:
            d = d[:21] + "..."
        parts.append(f"{p.name}={d}")
    return "(" + ", ".join(parts) + ")"


def operator_catalog_markdown() -> str:
    ns = build_operator_namespace()
    lines: list[str] = []
    for name in sorted(ns):
        fn = ns[name]
        doc = (inspect.getdoc(fn) or "").strip().splitlines()
        summary = doc[0].strip() if doc else ""
        line = f"- `{name}{_slim_signature(fn)}`"
        if summary:
            line += f" — {summary}"
        lines.append(line)
    return "\n".join(lines)
