# -*- coding: utf-8 -*-
"""模块 05 · turnover_budget：换手预算与 label×调仓频率匹配（S3）。

背景：Gemini 基线 run 实测 19/28 提交因换手率被拒（头号杀手），根因是
**1d 快信号配 weekly 调仓**——日频翻转信号按周度换仓，组合日单边换手必然
超过硬门。S6（1d label + weekly 错配）曾被认定为"0 入库"根因之一。

本模块给模型三样**可执行**的预算信息：
1. 当前 run 的调仓频率（engine_gate.freq）与换手硬门（按 freq 分档，真源
   ``DeliveryCriteria.turnover_gate_limit``，与 delivery_checker 同口径）；
2. label 持有期 × 调仓频率的匹配表（哪些组合是结构性超门）；
3. 降换手的结构手段（不是"降低换手率"这种空话）。

与 data_calibration 的 label 说明互补：那边讲口径，这边讲预算与匹配。
"""

from alphaagent.factor.mining.delivery_criteria import DeliveryCriteria

NAME = "turnover_budget"
TITLE = "换手预算与调仓频率匹配"
ORDER = 45
REQUIRED = False
SEP_BEFORE = "\n\n---\n\n"

# 调仓频率 → 换仓周期（交易日）
_FREQ_PERIOD_DAYS = {"daily": 1, "weekly": 5, "monthly": 21}

# label 持有天数区间 → 建议匹配的调仓频率（structural match）
def _label_hold_days(label_col: str) -> int | None:
    if not label_col.startswith("label_") or "d_" not in label_col:
        return None
    try:
        return int(label_col.split("_")[1].replace("d", ""))
    except ValueError:
        return None


def _match_verdict(hold_days: int | None, freq: str) -> tuple[str, str]:
    """返回 (verdict, 说明)。"""
    period = _FREQ_PERIOD_DAYS.get(freq, 5)
    if hold_days is None:
        return "?", "持有期未知，按实际换手裁决"
    if freq == "daily":
        return "ok", "daily 调仓持有 1 日，短周期信号可直接用"
    if hold_days >= period:
        return "ok", f"持有 {hold_days} 日 ≥ {freq} 调仓周期 {period} 日，结构匹配"
    if hold_days * 2 >= period:
        return "warn", f"持有 {hold_days} 日 < {freq} 调仓周期 {period} 日——信号半衰期不够，换手偏高，建议叠信号积分/长窗"
    return "mismatch", (
        f"持有 {hold_days} 日 ≪ {freq} 调仓周期 {period} 日——纯 {hold_days}d 快信号按 {freq} 调仓"
        "换手必然超门，属结构性错配。必须构造中慢结构（背离/筹码峰距离/资金积累/"
        "长窗均值）使信号兼具 1d 灵敏度与周期延续性，或改 label。"
    )


def render(ctx) -> str:  # noqa: ANN001
    crit = DeliveryCriteria.from_spec(getattr(ctx, "research_spec", None))
    freq = str(getattr(crit.engine_gate, "freq", "daily") or "daily").lower()
    gate = crit.turnover_gate_limit
    hold = _label_hold_days(str(getattr(ctx, "label_col", "")))
    verdict, note = _match_verdict(hold, freq)

    lines = [
        "### 换手预算与调仓频率匹配",
        f"- 本 run 调仓频率：**{freq}**；换手硬门：`avg_daily_side_turnover <= {gate}`"
        "（按频率分档，与交付门槛同源）。",
        f"- label 持有期 × 调仓匹配判定：**{verdict.upper()}** —— {note}",
        "",
        "| label 持有期 | daily | weekly | monthly |",
        "|---|---|---|---|",
        "| ≤2d 快信号 | ✅ 直接可用 | ⚠️ 半衰期不够，需叠信号积分 | ❌ 结构性超门 |",
        "| 3~10d | ✅ | ✅ 匹配 | ⚠️ 需叠长窗 |",
        "| ≥10d 慢信号 | ✅ | ✅ | ✅ |",
        "",
        "**降换手手段（按优先级，均已在算子库实现）**：",
        "- `SIGNAL_BLEND(signal, λ)` 信号积分：把当日信号与昨日融合，直接压日频换手；",
        "- 长窗均值 `TS_MEAN(x, N)` / `WMA` / `EMA`：N 越大换手越低，但别堆末位长窗"
        "（同质化熔断会拦）；",
        "- 慢信息源：资金/筹码/股东/事件类列天然低换手，做截面排序后再配门控；",
        "- 结构选择：背离、筹码峰距离、资金积累等「慢变量」比价格动量换手低一个量级。",
        "",
        "**红线**：1d label 配 weekly/monthly 调仓且不叠慢结构 → 换手必超门，"
        "不要在超门后靠微调参数救（同质化熔断也会拦堆平滑）。",
    ]
    return "\n".join(lines)
