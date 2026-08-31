"""挖掘轨迹的整洁终端打印。"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, TextIO


def _use_color(stream: TextIO) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    return bool(getattr(stream, "isatty", lambda: False)())


def ensure_utf8_stream(stream: TextIO | None) -> TextIO | None:
    """Windows 控制台默认 GBK，模型输出含 ❌ 等 emoji 时会 UnicodeEncodeError。

    对可配置的文本流统一切到 UTF-8 并降级为 replace，保证诊断打印永不中断挖掘。
    """
    try:
        if stream is not None and hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    except (ValueError, OSError, AttributeError):
        pass
    return stream


def _fmt_num(value: Any, digits: int = 4) -> str:
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return "—"


def _decile_d1_d10(deciles: Any) -> tuple[str, str]:
    if not isinstance(deciles, list):
        return "—", "—"
    by: dict[int, float | None] = {}
    for row in deciles:
        if not isinstance(row, dict):
            continue
        d = row.get("decile")
        ml = row.get("mean_label")
        if d is None:
            continue
        try:
            by[int(d)] = float(ml) if ml is not None else None
        except (TypeError, ValueError):
            by[int(d)] = None
    d1 = by.get(1)
    d10 = by.get(10) or by.get(max(by) if by else 0)
    return (_fmt_num(d1) if d1 is not None else "—", _fmt_num(d10) if d10 is not None else "—")


def _normalize_neighbor(nb: Any) -> dict[str, Any]:
    if isinstance(nb, dict):
        return {
            "factor_id": nb.get("factor_id") or "?",
            "cs_corr": nb.get("cs_corr"),
            "expr": nb.get("expr") or "",
        }
    if isinstance(nb, (list, tuple)) and len(nb) >= 2:
        return {"factor_id": str(nb[0]), "cs_corr": nb[1], "expr": ""}
    return {"factor_id": "?", "cs_corr": None, "expr": ""}


def _submit_failure_detail(result: dict[str, Any]) -> str:
    detail = result.get("error") or result.get("skipped_reason") or "failed"
    delivery = result.get("delivery_check") or {}
    reasons = delivery.get("fail_reasons")
    if reasons:
        detail = f"{detail} ({','.join(str(r) for r in reasons)})"
    return detail


def _mls_fmb_parts(mls: Any) -> list[str]:
    if not isinstance(mls, dict) or not mls:
        return []
    parts: list[str] = []
    for key, label in (
        ("mean_rho", "ρ"),
        ("nw_t_rho", "NWρ"),
        ("nw_t_ls", "NWls"),
        ("mls", "MLS"),
    ):
        if key in mls and mls[key] is not None:
            parts.append(f"{label}={_fmt_num(mls[key])}")
    return parts


def _metrics_parts(
    summ: dict[str, Any],
    rob: dict[str, Any],
    sign_check: dict[str, Any] | None,
) -> list[str]:
    d1, d10 = _decile_d1_d10(summ.get("decile_mean_label"))
    parts = [
        f"IC={_fmt_num(summ.get('ic'))}",
        f"ICIR={_fmt_num(summ.get('icir'))}",
        f"RANKIC={_fmt_num(summ.get('rank_ic'))}",
        f"cov={_fmt_num(summ.get('factor_coverage'))}",
        f"csAuto={_fmt_num(summ.get('cs_pearson_autocorr'))}",
        f"days={summ.get('n_days') if summ.get('n_days') is not None else '—'}",
        f"inst={summ.get('n_instruments') if summ.get('n_instruments') is not None else '—'}",
        f"skew={_fmt_num(summ.get('factor_skewness'))}",
        f"kurt={_fmt_num(summ.get('factor_kurtosis'))}",
        f"月数={rob.get('n_months') if rob.get('n_months') is not None else '—'}",
        f"月IC均={_fmt_num(rob.get('mean_monthly_ic'))}",
        f"月IC+={_fmt_num(rob.get('share_months_ic_positive'), 3)}",
        f"D1={d1}",
        f"D10={d10}",
    ]
    parts.extend(_mls_fmb_parts(summ.get("mls_fmb")))
    if isinstance(sign_check, dict) and sign_check.get("matches_expected_sign") is not None:
        parts.append("sign" + ("✓" if sign_check["matches_expected_sign"] else "✗"))
    return parts


class ConsolePrinter:
    """逐轮打印：轮次 → 模型文本/思考 → 因子表达式 → 关键指标。"""

    def __init__(self, *, stream: TextIO | None = None, show_thinking: bool = True) -> None:
        self.stream = stream or sys.stderr
        self.show_thinking = show_thinking
        self._color = _use_color(self.stream)

    def _print(self, *args: Any, **kwargs: Any) -> None:
        # UI runs attach a pipe; losing diagnostics must not abort research.
        try:
            print(*args, file=self.stream, flush=True, **kwargs)
        except (OSError, ValueError):
            pass

    def _ansi(self, code: str, text: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self._color else text

    def _tag(self, code: str, label: str, detail: str = "") -> None:
        line = self._ansi(code, f"[{label}]")
        if detail:
            line = f"{line} {detail}"
        self._print(line)

    def _body(self, text: str | None) -> None:
        if not text:
            return
        for raw in str(text).strip().splitlines():
            self._print(f"    {raw}")

    def _blank(self) -> None:
        self._print()

    def session_start(self, model: str, n_operators: int) -> None:
        self._tag("1;32", "挖掘启动", f"model={model}  算子={n_operators}")
        self._blank()

    def info(self, text: str) -> None:
        """通用信息输出（自检/预热等启动阶段）。"""
        self._print(text)

    def error(self, text: str) -> None:
        """错误输出（自检失败等启动阶段）。"""
        self._tag("1;31", "错误", text)

    def turn(self, turn: int) -> None:
        self._blank()
        self._tag("1;36", f"轮次 {turn}")
        self._blank()

    def assistant(self, content: str | None, reasoning: str | None) -> None:
        if self.show_thinking and reasoning:
            self._tag("35", "思考")
            self._body(reasoning)
            self._blank()
        if content and content.strip():
            self._tag("1", "助手")
            self._body(content)
            self._blank()

    def tool_result(self, name: str, arguments_raw: Any, result: dict[str, Any], elapsed: float | None) -> None:
        args = arguments_raw
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = {}
        if not isinstance(args, dict):
            args = {}
        if name == "eval_on_train_set":
            split = "train"
        elif name == "eval_on_val_set":
            split = "val"
        elif name == "evaluate_factor":
            split = str(args.get("profile_id") or "evaluate")
        else:
            split = "submit"
        factor_name = args.get("factor_name") or "expr"
        secs = f"{elapsed:.1f}s" if elapsed is not None else "—"
        if name == "submit_factor":
            self._tag("1;32", "交付", f"{factor_name}  ({secs})")
            self._body(args.get("comment"))
            self._body(args.get("multi_line_expr"))
            if not result.get("ok"):
                if result.get("candidate_stored"):
                    self._tag("1;33", "候选池", "✓ 已保存，未进入正式库")
                detail = _submit_failure_detail(result)
                self._tag("1;31", "结果", self._ansi("1;31", f"✗ {detail}"))
                sim = result.get("similarity") or {}
                for nb in sim.get("top_neighbors") or []:
                    norm = _normalize_neighbor(nb)
                    self._body(
                        f"  ~{norm['factor_id']} cs_corr={_fmt_num(norm['cs_corr'])}: {norm['expr']}"
                    )
                self._blank()
                return
            summ = result.get("metrics") or {}
            sim = result.get("similarity") or {}
            max_corr = sim.get("max_abs_corr")
            parts = [
                "stored✓",
                f"IC={_fmt_num(summ.get('ic'))}",
                f"ICIR={_fmt_num(summ.get('icir'))}",
                f"RANKIC={_fmt_num(summ.get('rank_ic'))}",
                f"cov={_fmt_num(summ.get('coverage'))}",
                f"csAuto={_fmt_num(summ.get('cs_pearson_autocorr'))}",
            ]
            if max_corr is not None:
                parts.append(f"max_cs_corr={_fmt_num(max_corr)}")
            self._tag("1;33", "结果", "  ".join(parts))
            self._blank()
            return

        self._tag("1;36", f"因子 {split}", f"{factor_name}  ({secs})")
        self._body(args.get("multi_line_expr"))

        if not result.get("ok"):
            self._tag("1;31", "指标", self._ansi("1;31", f"✗ {result.get('error_type')}: {result.get('error')}"))
            self._blank()
            return
        summ = result.get("summary") or result.get("metrics") or {}
        rob = result.get("monthly_corr_robustness") or {}
        parts = _metrics_parts(summ, rob, result.get("sign_check"))
        self._tag("1;33", "指标", "  ".join(parts))
        self._blank()

    def session_end(self, reason: str, ok: int, count: int) -> None:
        self._blank()
        self._tag("1;32", "会话结束", f"{reason}  ·  工具调用 {ok}/{count} 成功")
        self._blank()
