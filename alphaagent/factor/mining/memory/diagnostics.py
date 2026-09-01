# -*- coding: utf-8 -*-
"""证据归因工具：结论重建、失败码、失效细节和 FactorMiner 式签名。"""

from __future__ import annotations

import json
import math
import re
from datetime import datetime, timezone
from typing import Any


# ── 基础工具 ──

def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_float(value: Any) -> float | None:
    try:
        v = float(value)
        if v != v:  # NaN
            return None
        return v
    except (TypeError, ValueError):
        return None


def _parse_args(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _neg_str(s: Any) -> str:
    """Return a string that sorts *before* all normal strings, so that when
    used as a secondary sort key in ascending order the *most recent*
    (lexicographically largest) timestamp comes first."""
    return "\uffff" + str(s)


# ── 失败码 ──

def _failure_code(name: str, result: dict[str, Any], error: str, verdict: str) -> str | None:
    if not error and verdict not in {"weak", "rejected", "revise_required"}:
        return None
    text = error.lower()
    if "timeout" in text:
        return "model_timeout" if "model" in text else "eval_timeout"
    if "corr" in text or "similar" in text or "duplicate" in text:
        return "correlation_duplicate"
    if "sign" in text:
        return "sign_flip"
    if name == "submit_factor":
        if "engine_gate" in text:
            return "backtest_failed"
        return "stage_one_failed" if not result.get("candidate_stored") else "stage_two_failed"
    if name == "eval_on_train_set":
        return "train_threshold"
    if name == "eval_on_val_set":
        return "val_threshold"
    return "tool_error" if error else "metric_threshold"


# ── 结论重建 ──

def _rebuild_conclusion(name: str, result: dict[str, Any], metrics: dict[str, Any], error: str) -> tuple[str, str]:
    """从工具结果中重建 verdict 和 conclusion。

    原始 ResearchMemoryStore._classify 方法的逻辑。
    入库事实优先于 Reviewer 意见：revise 不阻断提交，已入库的 submit 即使
    携带 revise/gate 错误码也按 candidate_approved/production_approved 记账
    （与 schema._classify 同序，gate 失败的 error 文本不得掩盖入库事实）。
    """
    review = result.get("factor_review") if isinstance(result.get("factor_review"), dict) else {}
    if not review and isinstance(result.get("review"), dict):
        # submit payload 的审查意见存于 "review" 键（evaluate 工具用 "factor_review"）
        review = result.get("review")
    review_verdict = review.get("verdict")
    canonical = str(review.get("canonical_form") or "因子结构审核")
    reasons = review.get("reasons") if isinstance(review.get("reasons"), list) else []
    review_note = ""
    if review_verdict == "revise":
        first_reason = str(reasons[0]) if reasons else canonical
        review_note = f" Reviewer 意见：{canonical}（{first_reason}）。"
    if review_verdict == "reject":
        return "rejected", f"{canonical}：Reviewer 拒绝；{str(reasons[0]) if reasons else '不得重复同构表达式。'}"
    if name == "submit_factor":
        if result.get("stored"):
            return "production_approved", "已通过精筛并正式入库，应保留其经济机制并避免重复。" + review_note
        if result.get("candidate_stored"):
            return "candidate_approved", "通过海选进入候选池，尚未满足精筛条件，应针对失败项改进。" + review_note
    if review_verdict == "revise":
        return "revise_required", f"{canonical}：Reviewer 要求结构性改造后再评估。"
    if error:
        snippet = error if len(error) <= 500 else error[:497] + "..."
        return "rejected", f"{name} 被否定：{snippet}"
    if name == "submit_factor":
        return "rejected", "提交未通过，避免在未改变机制或拒绝原因的情况下重复提交。"
    ic = _safe_float(metrics.get("ic"))
    icir = _safe_float(metrics.get("icir"))
    coverage = _safe_float(metrics.get("factor_coverage", metrics.get("coverage")))
    is_val = name == "eval_on_val_set" or result.get("split") == "val"

    ic_str = f"IC={ic:+.4f}" if ic is not None else "IC=N/A"
    icir_str = f"ICIR={icir:+.3f}" if icir is not None else "ICIR=N/A"
    cov_str = f"Coverage={coverage:.2f}" if coverage is not None else ""

    if is_val and (result.get("sign_check", {}).get("matches_expected_sign") is not False) and abs(ic or 0) >= 0.015:
        return "validated", f"训练外验证通过：{ic_str} {icir_str} {cov_str}。方向一致且有可用相关性，可在相邻但不重复的机制上扩展。"
    # 2026-09-01 起海选线对齐 0.02：0.015~0.02 区间因子经盲测/换手门禁几乎全灭，
    # 不再授予 promising（正向 verdict 会驱动记忆与父本策略向其倾斜）。
    if abs(ic or 0) >= 0.02 and (icir or 0) > 0.2 and (coverage or 0) > 0.85:
        return "promising", f"训练阶段有潜力：{ic_str} {icir_str} {cov_str}。优先进行训练外验证或独立性改造。"
    return "weak", f"指标不足：{ic_str} {icir_str} {cov_str}。除非改变变量、经济机制或处理方式，否则不要机械重试。"


# ── 失效细节提取 ──

def _extract_fail_detail(name: str, result: dict[str, Any], error: str) -> str | None:
    """从工具结果中提取结构化失效细节字符串。

    格式: ``stage_one:min_abs_ic; correlation:max_corr=0.62``
    """
    parts: list[str] = []

    # delivery_check 中的 stage 失败原因
    dc = result.get("delivery_check")
    if isinstance(dc, dict):
        for stage_key in ("stage_one", "stage_two"):
            stage = dc.get(stage_key)
            if isinstance(stage, dict):
                reasons = stage.get("fail_reasons")
                if isinstance(reasons, list):
                    for r in reasons:
                        parts.append(f"{stage_key}:{r}")

    # similarity 中的相关性信息
    sim = result.get("similarity")
    if isinstance(sim, dict):
        max_corr = sim.get("max_abs_corr")
        if max_corr is not None:
            parts.append(f"correlation:max_corr={max_corr}")

    # engine_gate 失败
    eg = result.get("engine_gate")
    if isinstance(eg, dict) and not eg.get("passed", True):
        parts.append("engine_gate:failed")

    # error 文本
    if error:
        parts.append(f"error:{error[:200]}")

    return "; ".join(parts) if parts else None


# ── FactorMiner 式签名 ──

# 成功模式签名：从结论/表达式特征中提取的模式标记
_SUCCESS_SIGNATURES: dict[str, re.Pattern[str]] = {
    "momentum": re.compile(r"momentum|pctchange|ma_w|ma_dev", re.IGNORECASE),
    "reversal": re.compile(r"reversal|neg_ts", re.IGNORECASE),
    "gap_overnight": re.compile(r"gap|overnight|open_close", re.IGNORECASE),
    "vwap": re.compile(r"vwap", re.IGNORECASE),
    "volume": re.compile(r"volume|amount|turnover", re.IGNORECASE),
    "volatility": re.compile(r"std|var|vol_", re.IGNORECASE),
    "fundamental": re.compile(r"funda_|roe|roa|growth|quality", re.IGNORECASE),
}

# 禁忌方向签名
_FORBIDDEN_SIGNATURES: dict[str, re.Pattern[str]] = {
    "dead_end_low_ic": re.compile(r"IC.*低于.*0\.0[01]", re.IGNORECASE),
    "dead_end_saturation": re.compile(r"饱和|saturat", re.IGNORECASE),
    "correlation_duplicate": re.compile(r"相关.*0\.[5-9]|corr.*dup", re.IGNORECASE),
    "sign_flip": re.compile(r"方向.*不一致|sign.*flip", re.IGNORECASE),
}


def _match_signature(text: str, signatures: dict[str, re.Pattern[str]]) -> str | None:
    """在文本中匹配签名，返回第一个命中的签名 key。"""
    if not text:
        return None
    for key, pattern in signatures.items():
        if pattern.search(text):
            return key
    return None

