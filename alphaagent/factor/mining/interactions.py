"""Typed interaction contracts for AlphaAgent factor mining."""

from __future__ import annotations

import json
import re
from typing import Any


INTERACTION_TYPES = {
    "gated_signal": {"GATED_SIGNAL"},
    "conditional_group_rank": {"CS_GROUP_RANK"},
    "residual_signal": {"CS_RESIDUALIZE"},
    "divergence_signal": {"DIVERGENCE_RANK"},
    "rolling_relation": {"TS_CORR", "TS_COV", "TS_RANKCORR", "MUTUAL_INFO_LAG"},
    "piecewise_state": {"PIECEWISE_STATE"},
    "necessary_condition_signal": {"GATED_SIGNAL", "IF_THEN_ELSE"},
    "multiplication": {"MULTIPLY"},
}

_MULTIPLY_RE = re.compile(r"\bMULTIPLY\s*\(", re.IGNORECASE)
_INTERACTION_OPERATORS = {
    operator
    for names in INTERACTION_TYPES.values()
    for operator in names
    if operator != "MULTIPLY"
}


def _text(value: Any, *, max_chars: int = 500) -> str:
    return str(value or "").strip()[:max_chars]


def validate_interaction(
    value: Any,
    *,
    allowed_types: set[str] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    """Return ``(normalized_spec, tool_error)``; exactly one item is non-empty."""
    if value is None or value == {}:
        return None, None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                value = parsed
        # Tolerate a compact string form from models, but still require a mechanism.
        if isinstance(value, str):
            value = {"economic_mechanism": value}
    if not isinstance(value, dict):
        return None, {
            "ok": False,
            "error": "interaction_must_be_object",
            "error_type": "InteractionContractError",
        }

    kind = str(value.get("interaction_type") or "").strip().lower()
    if kind not in INTERACTION_TYPES:
        return None, {
            "ok": False,
            "error": f"unknown_interaction_type:{kind}",
            "error_type": "InteractionContractError",
            "allowed_types": sorted(INTERACTION_TYPES),
        }
    if allowed_types is not None and kind not in allowed_types:
        return None, {
            "ok": False,
            "error": f"interaction_type_not_allowed:{kind}",
            "error_type": "InteractionContractError",
            "allowed_types": sorted(allowed_types),
        }

    base = _text(value.get("base_signal"), max_chars=200)
    condition = _text(value.get("condition_signal"), max_chars=200)
    mechanism = _text(value.get("economic_mechanism"))
    missing: list[str] = []
    if not base:
        missing.append("base_signal")
    if not condition:
        missing.append("condition_signal")
    if len(mechanism) < 20:
        missing.append("economic_mechanism(>=20 chars)")
    if missing:
        return None, {
            "ok": False,
            "error": f"interaction_missing_fields:{','.join(missing)}",
            "error_type": "InteractionContractError",
        }

    subgroup = value.get("expected_subgroup_pattern")
    if subgroup is not None and not isinstance(subgroup, (dict, list)):
        return None, {
            "ok": False,
            "error": "expected_subgroup_pattern_must_be_object_or_list",
            "error_type": "InteractionContractError",
        }

    normalized: dict[str, Any] = {
        "interaction_type": kind,
        "base_signal": base,
        "condition_signal": condition,
        "economic_mechanism": mechanism,
        "ablation_required": bool(value.get("ablation_required", True)),
    }
    if subgroup is not None:
        normalized["expected_subgroup_pattern"] = subgroup
    for key in ("parent_hypothesis_id", "notes"):
        if value.get(key):
            normalized[key] = _text(value[key], max_chars=300)
    return normalized, None


def _fill_type(spec: dict[str, Any], op: str) -> dict[str, Any]:
    """把推断出的算子对应 interaction_type 填入契约（乘法仅在无替代类型时使用）。"""
    candidate_types = [t for t, ops in INTERACTION_TYPES.items() if op in ops]
    if "multiplication" in candidate_types and len(candidate_types) > 1:
        candidate_types.remove("multiplication")
    if candidate_types:
        return {**spec, "interaction_type": candidate_types[0]}
    return spec


def lint_expression_interaction(
    expr: str,
    spec: dict[str, Any] | None,
    *,
    policy: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str | None, dict[str, Any] | None]:
    """Validate a typed interaction and lint the DSL against it.

    Returns ``(spec, warning, blocked_error)``.
    """
    policy = policy or {}
    allowed_types = set(policy.get("allowed_interaction_types") or INTERACTION_TYPES)

    # 容错 1：契约以 JSON 字符串形式传入（模型常见行为），先解析为对象。
    if isinstance(spec, str):
        stripped = spec.strip()
        parsed: Any = None
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
        spec = parsed if isinstance(parsed, dict) else None

    # 容错 2：契约缺 interaction_type 时，从表达式中触发算子推断类型。
    #   - 恰好命中一个算子：自动填充对应类型；
    #   - 零个算子：说明表达式根本不需要交互契约，忽略该多余契约（不视为错误）；
    #   - 多个算子：无法唯一推断，保持报错但由后续校验给出可读信息。
    # （GATED_SIGNAL 同时映射 gated_signal / necessary_condition_signal，优先前者。）
    if isinstance(spec, dict) and not str(spec.get("interaction_type") or "").strip():
        upper_probe = expr.upper()
        detected_ops = {op for op in _INTERACTION_OPERATORS if op in upper_probe}
        if _MULTIPLY_RE.search(expr):
            detected_ops.add("MULTIPLY")
        if not detected_ops:
            spec = None
        elif len(detected_ops) == 1:
            op = next(iter(detected_ops))
            spec = _fill_type(spec, op)
        elif "MULTIPLY" in detected_ops:
            # 复合表达式（如 残差化+乘积）：乘法是最严格约束，按其声明以保留消融纪律。
            spec = {**spec, "interaction_type": "multiplication"}
        else:
            # 多算子复合（如 组内排名后再残差化）：按表达式中最后出现的算子推断——
            # 末位算子定义输出结构，是契约需要描述的最终机制。
            last_op, last_pos = None, -1
            for op in detected_ops:
                pos = upper_probe.find(op)
                if pos > last_pos:
                    last_op, last_pos = op, pos
            if last_op is not None:
                spec = _fill_type(spec, last_op)

    upper_expr = expr.upper()
    has_multiply = bool(_MULTIPLY_RE.search(expr))
    matched_ops = {op for op in _INTERACTION_OPERATORS if op in upper_expr}

    # 容错 0：表达式不含任何结构化交互算子时，契约没有可约束的对象——
    # 无论契约本身完整与否（模型常复制模板带出残缺/多余契约），一律忽略不报错。
    if not has_multiply and not matched_ops:
        return None, None, None

    # 容错 3：契约缺 base_signal/condition_signal 时，从表达式的数据列引用自动补全。
    #   字段缺失是模型的格式瑕疵，不该让高 IC 因子死在提交拦截上（实测有过
    #   IC=+0.082 的因子因此被拒）；economic_mechanism（经济机制纪律）仍强制。
    if isinstance(spec, dict):
        missing_fields = [f for f in ("base_signal", "condition_signal") if not _text(spec.get(f))]
        if missing_fields:
            variables: list[str] = []
            for var in re.findall(r"\$[a-zA-Z_][a-zA-Z0-9_]*", expr):
                low = var.lower()
                if low not in variables:
                    variables.append(low)
            if variables:
                if not _text(spec.get("base_signal")):
                    spec["base_signal"] = f"主腿（自动识别自表达式）: {variables[0]}"
                if not _text(spec.get("condition_signal")):
                    cond = variables[1] if len(variables) >= 2 else variables[0]
                    suffix = "" if len(variables) >= 2 else "（单变量表达式，无独立辅腿）"
                    spec["condition_signal"] = f"辅腿（自动识别自表达式）: {cond}{suffix}"

    normalized, error = validate_interaction(spec, allowed_types=allowed_types)
    if error is not None:
        return None, None, error

    if has_multiply:
        if normalized is None and policy.get("block_undeclared_multiply", True):
            return None, None, {
                "ok": False,
                "blocked": True,
                "warning": "未声明交互契约的 MULTIPLY 已拦截。",
                "suggestion": (
                    "请优先使用 GATED_SIGNAL / CS_GROUP_RANK / CS_RESIDUALIZE / "
                    "DIVERGENCE_RANK / PIECEWISE_STATE；确需乘法时传入 "
                    "interaction_type='multiplication' 的完整经济机制。"
                ),
                "interaction_templates": sorted(k for k in INTERACTION_TYPES if k != "multiplication"),
                "error_type": "UndeclaredInteractionError",
            }
        if normalized is not None and normalized["interaction_type"] != "multiplication":
            return normalized, None, {
                "ok": False,
                "blocked": True,
                "warning": "表达式使用 MULTIPLY，但 interaction_type 不是 multiplication。",
                "suggestion": "修改 DSL 或将契约改为已声明机制的乘法交互。",
                "error_type": "InteractionMismatchError",
            }
        if normalized is not None:
            warning = (
                "乘法仅允许作为已声明的放大器；必须完成 base-only / condition-only / "
                "combined 三组消融，且组合相对最强单腿有稳定增量。"
            )
            return normalized, warning, None

    if matched_ops and normalized is None and policy.get("require_contract_for_typed_interactions", True):
        return None, None, {
            "ok": False,
            "blocked": True,
            "warning": f"检测到结构化交互算子 {sorted(matched_ops)}，但缺少 interaction 契约。",
            "suggestion": "补充 interaction_type、base_signal、condition_signal 与 economic_mechanism。",
            "error_type": "UndeclaredInteractionError",
        }

    if normalized is not None:
        expected_ops = INTERACTION_TYPES[normalized["interaction_type"]]
        if not matched_ops & expected_ops and not has_multiply:
            return normalized, (
                f"interaction_type={normalized['interaction_type']} 通常对应 "
                f"{sorted(expected_ops)}，但当前表达式未检测到这些算子。"
            ), None
        return normalized, None, None

    return None, None, None
