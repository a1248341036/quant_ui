"""因子评估报告文本格式化（CLI / 调试输出）。"""

from __future__ import annotations

import json
import math
from typing import Any, TextIO


def _fmt_num(value: object, *, digits: int = 4, pct: bool = False) -> str:
    if value is None:
        return "—"
    try:
        x = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(x):
        return "—"
    if pct:
        return f"{x * 100:.{max(digits - 2, 1)}f}%"
    return f"{x:.{digits}f}"


def _section(title: str, width: int = 52) -> str:
    line = "─" * max(width - len(title) - 2, 8)
    return f"\n── {title} {line}"


def _row(label: str, value: str, *, width: int = 52) -> str:
    return f"  {label:<16} {value:>{width - 18}}"


def _decile_bars(rows: list[dict[str, Any]], *, bar_width: int = 24) -> list[str]:
    if not rows:
        return ["  (无十分组数据)"]
    vals = [r.get("mean_label") for r in rows]
    finite = [float(v) for v in vals if v is not None and math.isfinite(float(v))]
    if not finite:
        return ["  (十分组 label 均为 NaN)"]
    lo, hi = min(finite), max(finite)
    span = hi - lo if hi > lo else 1.0
    lines: list[str] = []
    for row in rows:
        dec = int(row.get("decile", 0))
        ml = row.get("mean_label")
        if ml is None or not math.isfinite(float(ml)):
            bar = ""
            shown = "—"
        else:
            x = float(ml)
            shown = _fmt_num(x, digits=6)
            norm = (x - lo) / span
            n = max(1, int(round(norm * bar_width))) if x >= lo else 0
            bar = "█" * n
        lines.append(f"  D{dec:>2}  {bar:<{bar_width}}  {shown}")
    return lines


def format_factor_report_text(metrics: dict[str, Any]) -> str:
    """将 evaluate_factor 返回的 metrics 格式化为可读文本报告。"""
    lines: list[str] = [
        "",
        "═" * 52,
        "  因子评估报告",
        "═" * 52,
    ]

    start = metrics.get("eval_start")
    end = metrics.get("eval_end")
    if start or end:
        lines.append(_row("评估区间", f"{start or '…'} ~ {end or '…'}"))
    lines.append(_row("标签列", str(metrics.get("label_col", "—"))))
    lines.append(_row("有效 IC 天数", str(metrics.get("n_days", "—"))))

    lines.append(_section("截面 IC"))
    lines.append(_row("IC", _fmt_num(metrics.get("ic"))))
    lines.append(_row("ICIR", _fmt_num(metrics.get("icir"))))
    lines.append(_row("Rank IC", _fmt_num(metrics.get("rank_ic"))))
    lines.append(_row("Coverage", _fmt_num(metrics.get("coverage"), pct=True)))
    lines.append(_row("CS lag-1 ρ", _fmt_num(metrics.get("cs_pearson_autocorr"))))

    mls = metrics.get("mls_fmb")
    if isinstance(mls, dict) and mls:
        lines.append(_section("MLS / FMB"))
        key_labels = [
            ("mean_rho", "mean ρ"),
            ("mean_ls", "mean LS"),
            ("ir_ls", "IR_LS"),
            ("ir_ls_annual", "IR_LS 年化"),
            ("mls", "MLS"),
            ("nw_t_rho", "NW-t(ρ)"),
            ("nw_t_ls", "NW-t(LS)"),
            ("n_days_rho", "样本天数"),
        ]
        for key, label in key_labels:
            if key in mls:
                val = mls[key]
                if key.startswith("n_"):
                    lines.append(_row(label, str(int(val)) if val is not None else "—"))
                else:
                    lines.append(_row(label, _fmt_num(val)))

    deciles = metrics.get("decile_mean_label")
    if isinstance(deciles, list) and deciles:
        lines.append(_section("十分组 label 均值 (D1=低因子 → D10=高因子)"))
        lines.extend(_decile_bars(deciles))

    lines.append("")
    lines.append("─" * 52)
    return "\n".join(lines)


def print_factor_report(
    metrics: dict[str, Any],
    *,
    file: TextIO | None = None,
) -> None:
    print(format_factor_report_text(metrics), file=file)


def format_factor_report_json(metrics: dict[str, Any]) -> str:
    """JSON 报告（数值四舍五入）。"""

    def _clean(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items() if k != "note"}
        if isinstance(obj, list):
            return [_clean(x) for x in obj]
        if isinstance(obj, float):
            return round(obj, 6) if math.isfinite(obj) else None
        return obj

    return json.dumps(_clean(metrics), ensure_ascii=False, indent=2)
