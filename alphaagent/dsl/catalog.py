"""DSL 算子清单：供 eval、挖掘 prompt 等模块共享。"""

from __future__ import annotations

import inspect

from alphaagent.dsl.registry import build_operator_namespace


def list_operator_names() -> list[str]:
    return sorted(build_operator_namespace())


def operator_catalog_markdown() -> str:
    ns = build_operator_namespace()
    lines: list[str] = []
    for name in sorted(ns):
        fn = ns[name]
        try:
            sig = str(inspect.signature(fn))
        except (TypeError, ValueError):
            sig = "(...)"
        doc = (inspect.getdoc(fn) or "").strip().splitlines()
        summary = doc[0].strip() if doc else ""
        line = f"- `{name}{sig}`"
        if summary:
            line += f" — {summary}"
        lines.append(line)
    return "\n".join(lines)
