# -*- coding: utf-8 -*-
"""预测-对账与门控消融的共享构建器。

A) prediction_check：把 LLM 评估前提交的可证伪预测（预期十分位形态/强侧/IC 符号）
   与实际 decile_mean_label 自动对账，注入「预期 X 实际 Y」。
E) ablation_check：门控/条件结构自动对比 base-only vs full，量化条件化对基信号的
   增量或破坏（实证案例：年线门控把 20d 反转 IC 从 +0.039 变 -0.005）。

两者都只做纯 Python 统计，不依赖 pandas/numpy，便于单测。
"""
from __future__ import annotations

import re
from typing import Any

# 门控/条件类算子：出现即触发消融流程（表达式大写匹配）
GATING_OP_RE = re.compile(r"\b(GATED_SIGNAL|IF_THEN_ELSE|PIECEWISE_STATE|CS_GROUP_RANK)\s*\(")

_SHAPE_CN = {
    "monotonic_increasing": "单调递增（高因子端最强）",
    "monotonic_decreasing": "单调递减（低因子端最强）",
    "inverted_u": "倒U型（中间组最强）",
    "u_shape": "U型（两端强、中间弱）",
    "spike_at_extreme": "极端组尖峰",
    "irregular": "不规则",
}

_SIDE_CN = {
    "high_factor": "高因子端(D8-D10)",
    "low_factor": "低因子端(D1-D3)",
    "middle": "中间组(D4-D7)",
}

_PREDICTION_REQUIRED_KEYS = ("expected_shape", "expected_strong_side", "expected_sign")
_VALID_SHAPES = frozenset(_SHAPE_CN)
_VALID_SIDES = frozenset(_SIDE_CN)


# ── 十分位形态分类 ──

def _ranks(values: list[float]) -> list[float]:
    """平均秩（并列取均值），1 起。"""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def _spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    rx, ry = _ranks(xs), _ranks(ys)
    n = len(xs)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    if dx == 0 or dy == 0:
        return 0.0
    return num / (dx * dy)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def classify_decile_shape(decile_rows: list[dict[str, Any]] | None, *, n_deciles: int = 10) -> dict[str, Any] | None:
    """从 decile_mean_label 行分类形态。

    返回 {"shape","strong_side","spearman","d1","d10","peak_decile","means"}；
    有效 decile < 5 或 mean_label 全缺时返回 None。
    """
    if not decile_rows:
        return None
    pairs = sorted(
        ((int(r.get("decile")), r.get("mean_label")) for r in decile_rows),
        key=lambda p: p[0],
    )
    deciles = [d for d, v in pairs if v is not None]
    values = [float(v) for _, v in pairs if v is not None]
    if len(values) < 5:
        return None
    # 分箱完整性：离散值扎堆的因子（如 VPIN 类，每天截面仅 10~45 个唯一值）
    # 等频十分位会塌缩成 2~9 组，此时 d1/d10 与"高/低端"都不代表真实的
    # 极端组，形态分类（单调/倒U/U 型）是伪影。有效组数不足请求组数 →
    # 形态不可信，交上层标 unverifiable。
    if len(deciles) < n_deciles:
        return {
            "shape": "incomplete_binning",
            "strong_side": None,
            "spearman": None,
            "d1": round(values[0], 6),
            "d10": round(values[-1], 6),
            "peak_decile": None,
            "means": [round(v, 6) for v in values],
            "incomplete": True,
            "n_bins_used": len(deciles),
            "n_deciles": n_deciles,
        }
    rho = _spearman([float(d) for d in deciles], values)
    peak_d = deciles[values.index(max(values))]
    trough_d = deciles[values.index(min(values))]
    max_v, min_v = max(values), min(values)
    end_max = max(values[0], values[-1])
    end_min = min(values[0], values[-1])
    mid = values[3:-3] if len(values) >= 7 else values
    mid_max, mid_min = max(mid), min(mid)
    span = max_v - min_v

    # 形态判定先于单调性：峰/谷在中间且两端明显回落 → 倒U/U型。
    # （真实数据常见"整体爬升但尾部回落"的中间峰形态，spearman 会被前段爬升抬高，
    #   单靠 rho 阈值会把倒U误判成单调——峰位置约束优先。）
    if 4 <= peak_d <= 7 and mid_max > end_max and span > 0 and (end_max - min_v) < 0.6 * span:
        shape = "inverted_u"
    elif 4 <= trough_d <= 7 and mid_min < end_min and span > 0 and (max_v - end_min) < 0.6 * span:
        shape = "u_shape"
    elif rho >= 0.85:
        shape = "monotonic_increasing"
    elif rho <= -0.85:
        shape = "monotonic_decreasing"
    elif peak_d in (1, 10):
        shape = "spike_at_extreme"
    else:
        shape = "irregular"

    low_avg = _mean(values[:3])
    mid_avg = _mean(values[3:-3]) if len(values) >= 7 else _mean(values)
    high_avg = _mean(values[-3:])
    best = max(high_avg, low_avg, mid_avg)
    if best == high_avg:
        strong_side = "high_factor"
    elif best == low_avg:
        strong_side = "low_factor"
    else:
        strong_side = "middle"

    return {
        "shape": shape,
        "strong_side": strong_side,
        "spearman": round(rho, 3),
        "d1": round(values[0], 6),
        "d10": round(values[-1], 6),
        "peak_decile": peak_d,
        "means": [round(v, 6) for v in values],
    }


# ── A: 预测对账 ──

# 别名归一（2026-09-05 实测：GLM 系会传 "D10" 之类分位组词汇而非端位枚举，
# 一次 run 7 次调用全部被拒——评估上下文满屏 D1~D10，强求切换词汇体系是
# 逆 LLM 天性的；语义对但词汇错的输入做确定性归一，真正非法的才拒。）
_SIDE_ALIASES = {
    "high_factor": "high_factor", "high": "high_factor", "top": "high_factor",
    "d8": "high_factor", "d9": "high_factor", "d10": "high_factor",
    "q8": "high_factor", "q9": "high_factor", "q10": "high_factor",
    "low_factor": "low_factor", "low": "low_factor", "bottom": "low_factor",
    "d1": "low_factor", "d2": "low_factor", "d3": "low_factor",
    "q1": "low_factor", "q2": "low_factor", "q3": "low_factor",
    "middle": "middle", "mid": "middle",
    "d4": "middle", "d5": "middle", "d6": "middle", "d7": "middle",
    "q4": "middle", "q5": "middle", "q6": "middle", "q7": "middle",
}

_SHAPE_SEP_RE = re.compile(r"[\s\-]+")

_SHAPE_EXTRA_ALIASES = {
    "ushape": "u_shape",
    "invertedu": "inverted_u",
    "monotonicincreasing": "monotonic_increasing",
    "monotonicdecreasing": "monotonic_decreasing",
    "spike": "spike_at_extreme",
    "spikeatextreme": "spike_at_extreme",
}

# ── 散文→枚举归一（2026-09-06 实测：并行 batch 里 8/11 条 expected_shape
# 写成整句经济机制（"供给收缩(D10)最优，供给扩张(D1)最差"），别名归一救不了
# 自由文本；并行工具调用错误反馈要下一轮才到，一轮全灭。确定性关键词推断：
# 梯度词全局计票 + 端点评级词挂靠前方最近端点——把"语义对但格式错"的散文
# 映射到枚举；歧义（两端票数平/无方向词）仍拒绝并回显合法值。）

_HIGH_SIDE_RE = re.compile(r"D\s*10|D\s*9|D\s*8|高(?:因子)?(?:端|侧)|顶部|高端|最高组", re.IGNORECASE)
_LOW_SIDE_RE = re.compile(r"D\s*1(?!\d)|D\s*2(?!\d)|D\s*3(?!\d)|低(?:因子)?(?:端|侧)|底部|低端|最低组", re.IGNORECASE)
_MIDDLE_RE = re.compile(r"中间|中部|中段|居中")
_INVERTED_U_RE = re.compile(r"倒\s*U|inverted[_\s]*u", re.IGNORECASE)
_SU_SHAPE_RE = re.compile(r"(?<!倒)\bU\s*型|u[_\s]shape", re.IGNORECASE)
_SPIKE_RE = re.compile(r"尖峰|极端组|spike", re.IGNORECASE)
_IRREGULAR_RE = re.compile(r"不规则|无规律|irregular", re.IGNORECASE)
# 梯度词：描述 D1→D10 轴向（递增=high 端强），全局计票、不与端点配对
_GRAD_POS_RE = re.compile(r"递增|单调增|越高|越强|越多|更强|越大越好")
_GRAD_NEG_RE = re.compile(r"递减|单调减|越低|越弱|越少|更差|越小越好")
# 端点评级词：评级其前方最近的端点（中文语序 "D1(说明)最优"——评级词跟在端点后）
_RATE_POS_RE = re.compile(r"最优|最强|最高|最好|看多|做多")
_RATE_NEG_RE = re.compile(r"最差|最弱|最劣|越差|看空|做空")


def _prose_direction(text: str) -> tuple[bool, bool]:
    """散文里哪一端被说成强 → (high_strong, low_strong)。

    梯度词全局计票（"D1→D10 递增" → high 端强）；评级词挂靠前方最近端点
    （"D10 最差" → 对端 low 强）；两端票数打平视为歧义（返回 False, False，
    由上层拒绝并回显合法值）。
    """
    t = str(text)
    tokens: list[tuple[int, str]] = []
    for m in _HIGH_SIDE_RE.finditer(t):
        tokens.append((m.start(), "high"))
    for m in _LOW_SIDE_RE.finditer(t):
        tokens.append((m.start(), "low"))
    tokens.sort(key=lambda tk: tk[0])  # 按位置排序：评级词挂靠"前方最近"端点依赖此序
    high = low = 0
    if _GRAD_POS_RE.search(t):
        high += 1
    if _GRAD_NEG_RE.search(t):
        low += 1
    for m in _RATE_POS_RE.finditer(t):
        prev = [tk for tk in tokens if tk[0] < m.start()]
        if prev:
            if prev[-1][1] == "high":
                high += 1
            else:
                low += 1
        else:
            high += 1  # 无端点可挂：按全局正向处理（如 "收益最高"）
    for m in _RATE_NEG_RE.finditer(t):
        prev = [tk for tk in tokens if tk[0] < m.start()]
        if prev:
            if prev[-1][1] == "high":
                low += 1
            else:
                high += 1
        else:
            low += 1
    return high > low, low > high


def _prose_shape(value: str) -> str | None:
    t = str(value)
    if _INVERTED_U_RE.search(t):
        return "inverted_u"
    if _SU_SHAPE_RE.search(t):
        return "u_shape"
    if _SPIKE_RE.search(t):
        return "spike_at_extreme"
    if _IRREGULAR_RE.search(t):
        return "irregular"
    high_strong, low_strong = _prose_direction(t)
    if high_strong and not low_strong:
        return "monotonic_increasing"
    if low_strong and not high_strong:
        return "monotonic_decreasing"
    return None


def _prose_side(value: str) -> str | None:
    t = str(value)
    if _MIDDLE_RE.search(t) and not (_HIGH_SIDE_RE.search(t) or _LOW_SIDE_RE.search(t)):
        return "middle"
    high_strong, low_strong = _prose_direction(t)
    if high_strong and not low_strong:
        return "high_factor"
    if low_strong and not high_strong:
        return "low_factor"
    return None


def _prose_sign(value: str) -> int | None:
    t = str(value)
    has_pos = any(r.search(t) for r in (_GRAD_POS_RE, _RATE_POS_RE))
    has_neg = any(r.search(t) for r in (_GRAD_NEG_RE, _RATE_NEG_RE))
    if has_pos and not has_neg:
        return 1
    if has_neg and not has_pos:
        return -1
    return None


def _canon_shape(value: Any) -> str | None:
    if value is None:
        return None
    key = _SHAPE_SEP_RE.sub("_", str(value).strip().lower())
    if key in _VALID_SHAPES:
        return key
    aliased = _SHAPE_EXTRA_ALIASES.get(key)
    if aliased:
        return aliased
    # 散文回退：只有当原文不是合法枚举/别名时才做关键词推断
    return _prose_shape(str(value))


def _canon_side(value: Any) -> str | None:
    if value is None:
        return None
    canon = _SIDE_ALIASES.get(str(value).strip().lower())
    if canon:
        return canon
    return _prose_side(str(value))


def _canon_sign(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        if value == 1:
            return 1
        if value == -1:
            return -1
        return None
    s = str(value).strip().lower()
    if s in ("1", "+1", "positive", "pos", "+", "long"):
        return 1
    if s in ("-1", "negative", "neg", "-", "short"):
        return -1
    return _prose_sign(str(value))


def normalize_prediction(prediction: Any) -> dict[str, Any] | None:
    """校验并规范化 prediction（含别名归一）；不合法返回 None。"""
    if not isinstance(prediction, dict):
        return None
    shape = _canon_shape(prediction.get("expected_shape"))
    side = _canon_side(prediction.get("expected_strong_side"))
    sign = _canon_sign(prediction.get("expected_sign"))
    if shape is None or side is None or sign is None:
        return None
    return {
        "expected_shape": shape,
        "expected_strong_side": side,
        "expected_sign": int(sign),
        "falsifier": str(prediction.get("falsifier") or "").strip() or None,
    }


def describe_prediction_issues(prediction: Any) -> str | None:
    """逐字段描述 prediction 的非法原因（供工具错误信息回显实际值）。

    normalize 成功（含别名纠正）时不会走到本函数；这里只处理真正非法的输入，
    告诉 LLM "哪个字段、收到了什么、合法值是什么"——省掉盲猜重试。
    """
    if not isinstance(prediction, dict):
        return f"prediction 必须是对象（收到 {type(prediction).__name__}）"
    issues: list[str] = []
    shape_raw = prediction.get("expected_shape")
    if _canon_shape(shape_raw) is None:
        issues.append(
            f"expected_shape 收到 {shape_raw!r}，合法值: "
            "monotonic_increasing|monotonic_decreasing|inverted_u|u_shape|spike_at_extreme|irregular"
        )
    side_raw = prediction.get("expected_strong_side")
    if _canon_side(side_raw) is None:
        issues.append(
            f"expected_strong_side 收到 {side_raw!r}，合法值: high_factor|low_factor|middle"
            "（分位组别名 D8-D10→high_factor、D1-D3→low_factor、D4-D7→middle 也可）"
        )
    sign_raw = prediction.get("expected_sign")
    if _canon_sign(sign_raw) is None:
        issues.append(f"expected_sign 收到 {sign_raw!r}，合法值: 1|-1")
    if not issues:
        return None
    return "；".join(issues)


def build_prediction_check(
    prediction: Any,
    *,
    ic: Any = None,
    decile_rows: list[dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """对账：预期形态/强侧/符号 vs 实际十分位。无 prediction 时返回 None。"""
    pred = normalize_prediction(prediction)
    if pred is None:
        return None
    expected_shape = pred["expected_shape"]
    expected_side = pred["expected_strong_side"]
    actual = classify_decile_shape(decile_rows)
    if actual is None:
        return {
            "verdict": "unverifiable",
            "expected": pred,
            "actual": None,
            "message": "十分位数据不足，预测无法对账（检查 coverage 或 label 分布）。",
        }
    # 分箱塌缩（低离散度因子，如 VPIN：每天截面唯一值 < 组数）→ 形态是伪影，
    # 强侧/形状/IC 符号对账都不可信，标 unverifiable 而非 partial/contradicted。
    if actual.get("incomplete"):
        return {
            "verdict": "unverifiable",
            "expected": pred,
            "actual": {
                "shape": actual["shape"],
                "incomplete_binning": True,
                "n_bins_used": actual["n_bins_used"],
                "n_deciles": actual["n_deciles"],
                "d1": actual["d1"],
                "d10": actual["d10"],
                "means": actual["means"],
            },
            "message": (
                f"十分位分箱塌缩（仅 {actual['n_bins_used']}/{actual['n_deciles']} 组有效，"
                f"截面离散度不足）：D1={actual['d1']:.6f} D10={actual['d10']:.6f}，"
                "形态/强侧判断不可信。请改用 IC/ICIR 口径评估，或对因子做连续化变换。"
            ),
        }

    side_match = actual["strong_side"] == expected_side
    shape_match = actual["shape"] == expected_shape
    ic_val = None
    try:
        ic_val = float(ic) if ic is not None else None
    except (TypeError, ValueError):
        ic_val = None
    sign_mismatch = (
        ic_val is not None
        and abs(ic_val) >= 0.003
        and (ic_val > 0) != (pred["expected_sign"] > 0)
    )

    if not side_match:
        verdict = "contradicted"
    elif sign_mismatch:
        verdict = "contradicted"
    elif not shape_match:
        verdict = "partial"
    else:
        verdict = "confirmed"

    parts: list[str] = []
    if verdict == "confirmed":
        parts.append(f"预测对账通过：预期{_SHAPE_CN[expected_shape]}/{_SIDE_CN[expected_side]}，实际一致（spearman={actual['spearman']:.2f}）。")
    elif verdict == "partial":
        parts.append(
            f"预测部分命中：强侧一致（实际{_SIDE_CN[actual['strong_side']]}）但形态不符——"
            f"预期{_SHAPE_CN[expected_shape]}，实际{_SHAPE_CN[actual['shape']]}。"
        )
    else:
        parts.append(
            f"预测被证伪：预期{_SHAPE_CN[expected_shape]}/{_SIDE_CN[expected_side]}，"
            f"实际{_SHAPE_CN[actual['shape']]}/{_SIDE_CN[actual['strong_side']]}。"
        )
        if pred["falsifier"]:
            parts.append(f"证伪条件触发：{pred['falsifier']}")
        if actual["strong_side"] != "high_factor" and expected_side == "high_factor":
            parts.append(
                f"注意：alpha 不在高因子端（D10 均值 {actual['d10']:.4%} vs D1 均值 {actual['d1']:.4%}，"
                "20d label 口径），纯多头持仓最高组难以变现该 IC——先确认多头端可交易性再迭代参数。"
            )
    if sign_mismatch:
        parts.append(f"IC 符号与预期相反（实际 {ic_val:+.4f}）。")

    return {
        "verdict": verdict,
        "expected": pred,
        "actual": {
            "shape": actual["shape"],
            "strong_side": actual["strong_side"],
            "spearman": actual["spearman"],
            "d1": actual["d1"],
            "d10": actual["d10"],
            "peak_decile": actual["peak_decile"],
        },
        "ic": round(ic_val, 6) if ic_val is not None and abs(ic_val) < 10 else ic_val,
        "message": " ".join(parts),
    }


# ── E: 门控消融 ──

def build_ablation_check(
    base_cs: dict[str, Any],
    full_cs: dict[str, Any],
    *,
    base_expr: str = "",
) -> dict[str, Any]:
    """base-only vs full 的增量对比。输入为两份 cross_sectional_core 指标。"""
    base_ic = _safe(base_cs.get("ic"))
    full_ic = _safe(full_cs.get("ic"))
    base_icir = _safe(base_cs.get("icir"))
    full_icir = _safe(full_cs.get("icir"))
    out: dict[str, Any] = {
        "base_expr": base_expr[:200],
        "base_ic": base_ic,
        "full_ic": full_ic,
        "delta_ic": round(full_ic - base_ic, 6) if (base_ic is not None and full_ic is not None) else None,
        "base_icir": base_icir,
        "full_icir": full_icir,
    }
    if base_ic is None or full_ic is None:
        out["verdict"] = "unverifiable"
        out["message"] = "base-only 或 full 的 IC 缺失，无法量化条件化增量。"
        return out
    if (full_ic > 0) != (base_ic > 0) and abs(base_ic) >= 0.003 and abs(full_ic) >= 0.003:
        verdict = "conditioning_flipped_signal"
        message = (
            f"条件化翻转了信号方向：base-only IC={base_ic:+.4f} → 门控后 IC={full_ic:+.4f}。"
            "机制假设（条件状态放大/过滤基信号）不成立，条件与基信号在截面上存在结构性冲突。"
        )
    elif abs(full_ic) < 0.6 * abs(base_ic):
        verdict = "conditioning_destroyed_value"
        message = (
            f"条件化摧毁了基信号：base-only IC={base_ic:+.4f} → 门控后 IC={full_ic:+.4f}"
            f"（衰减 {1 - abs(full_ic) / abs(base_ic):.0%}）。"
            "门控截掉的样本里含主要 alpha——考虑去掉门控、或反转条件方向。"
        )
    elif abs(full_ic) > 1.15 * abs(base_ic):
        verdict = "conditioning_added_value"
        message = (
            f"条件化带来增量：base-only IC={base_ic:+.4f} → 门控后 IC={full_ic:+.4f}。"
            "状态过滤有效，机制假设得到数据支持。"
        )
    else:
        verdict = "neutral"
        message = (
            f"条件化增量有限：base-only IC={base_ic:+.4f} → 门控后 IC={full_ic:+.4f}。"
            "门控没有引入机制性增量，多为正交性装饰。"
        )
    out["verdict"] = verdict
    out["message"] = message
    return out


def _safe(value: Any) -> float | None:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if v == v else None  # NaN 过滤
