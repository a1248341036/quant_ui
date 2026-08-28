"""交付门槛阶段对象：把 submit() 的两阶段统计判定提炼为可组合 Stage。

设计目标：
- 每个 Stage 是**纯判定**：输入证据（metrics/similarity），输出
  ``StageResult(passed, fail_reasons, extra)``，不碰 session/registry 副作用；
- 门槛数值全部来自 ``DeliveryCriteria``（research_spec 注入的单一来源），
  不再散落硬编码默认值；
- ``DeliveryChecker`` 编排阶段链，供 ``submit()`` 复用判定逻辑，保持
  promotion_status / skipped_reason 语义不变。

原散落在 submit.py 的判定函数（_stage_one_stats_reasons /
_stage_one_turnover_reasons / _stage_one_val_retention_reasons /
_check_stage_one / _check_stage_two）收敛为本模块的 Stage，行为逐位等价。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from alphaagent.factor.mining.delivery_criteria import DeliveryCriteria


@dataclass(frozen=True)
class StageResult:
    """单个阶段的判定结果。"""

    passed: bool
    fail_reasons: list[str] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class DeliveryStage(Protocol):
    """阶段协议：接受证据字典，返回判定结果。"""

    name: str

    def run(self, evidence: dict[str, Any]) -> StageResult:
        ...


def _abs_ic(v: Any) -> float | None:
    try:
        return abs(float(v))
    except (TypeError, ValueError):
        return None


class StageOneStats:
    """候选池统计门槛：IC/ICIR/coverage（train-only 窗口）。

    准入看 train 窗口，防止 val 衰减被混合窗口稀释。
    """

    name = "stage_one_stats"

    def __init__(self, criteria: DeliveryCriteria) -> None:
        self.c = criteria.candidate

    def run(self, evidence: dict[str, Any]) -> StageResult:
        metrics = evidence.get("metrics") or {}
        reasons: list[str] = []

        ic = _abs_ic(metrics.get("ic"))
        if ic is None or ic < self.c.min_abs_ic:
            reasons.append("ic")
        icir = metrics.get("icir")
        try:
            icir_v = abs(float(icir))
        except (TypeError, ValueError):
            icir_v = None
        if icir_v is None or icir_v <= self.c.min_icir:
            reasons.append("icir")
        cov = metrics.get("coverage") or metrics.get("factor_coverage")
        try:
            cov_v = float(cov)
        except (TypeError, ValueError):
            cov_v = None
        if cov_v is None or cov_v <= self.c.min_coverage:
            reasons.append("coverage")

        return StageResult(passed=len(reasons) == 0, fail_reasons=reasons)


class StageOneTurnover:
    """候选池换手可行性门槛：截面 lag-1 Pearson 自相关下限。

    低于阈值的因子截面排名日度剧变，换手吃掉 alpha，不可交付。
    """

    name = "stage_one_turnover"

    def __init__(self, criteria: DeliveryCriteria) -> None:
        self.c = criteria.candidate

    def run(self, evidence: dict[str, Any]) -> StageResult:
        min_ac = float(self.c.min_cs_autocorr)
        if min_ac <= 0:
            return StageResult(passed=True, fail_reasons=[])
        ac = evidence.get("metrics") or {}
        ac_v = ac.get("cs_pearson_autocorr")
        try:
            ac_float = float(ac_v)
        except (TypeError, ValueError):
            ac_float = None
        if ac_float is None or not np.isfinite(ac_float) or ac_float < min_ac:
            return StageResult(passed=False, fail_reasons=["cs_autocorr"])
        return StageResult(passed=True, fail_reasons=[])


class StageOneValRetention:
    """样本外保留比门槛：|val_ic|/|train_ic| ≥ 阈值且方向不反转。

    val 窗口无数据时跳过（train-only 会话）。
    阈值由调用方注入（候选池用 candidate.min_val_ic_retention=0.5；
    正式库精筛用 production.min_val_ic_retention=0.60——两阶段阈值本就不同）。
    """

    name = "stage_one_val_retention"

    def __init__(self, min_val_ic_retention: float) -> None:
        self.min_ratio = float(min_val_ic_retention)

    def run(self, evidence: dict[str, Any]) -> StageResult:
        train_metrics = evidence.get("train_metrics") or {}
        val_metrics = evidence.get("val_metrics") or {}

        n_days_val = val_metrics.get("n_days") or val_metrics.get("n_instruments")
        if n_days_val is not None and int(n_days_val) == 0:
            return StageResult(passed=True, fail_reasons=[])
        t_ic = train_metrics.get("ic")
        v_ic = val_metrics.get("ic")
        if t_ic is None or v_ic is None:
            return StageResult(passed=False, fail_reasons=["val_ic_missing"])
        t, v = float(t_ic), float(v_ic)
        if not np.isfinite(t) or not np.isfinite(v):
            return StageResult(passed=False, fail_reasons=["val_ic_missing"])
        if t * v < 0:
            return StageResult(passed=False, fail_reasons=["val_sign_flip"])
        if abs(t) > 1e-12 and abs(v) / abs(t) < self.min_ratio:
            return StageResult(passed=False, fail_reasons=["val_retention"])
        return StageResult(passed=True, fail_reasons=[])


class StageOneCorrelation:
    """候选池相似度门槛：与已有因子的最大截面相关低于阈值。

    相似度在 stage_one 统一检查（统计门槛通过后、写候选池前）。
    """

    name = "stage_one_correlation"

    def __init__(self, criteria: DeliveryCriteria) -> None:
        self.c = criteria.candidate

    def run(self, evidence: dict[str, Any]) -> StageResult:
        similarity = evidence.get("similarity") or {}
        corr = similarity.get("max_abs_corr", 0.0)
        try:
            corr_v = float(corr)
        except (TypeError, ValueError):
            corr_v = None
        if corr_v is None or corr_v >= float(self.c.max_abs_corr):
            return StageResult(passed=False, fail_reasons=["max_cs_corr"])
        return StageResult(passed=True, fail_reasons=[])


class StageTwoStats:
    """正式库精筛统计门槛：双窗口（train+val）口径，2026-08 重构。

    - train 窗口：|IC| 与 ICIR 各自达标；
    - val 窗口：绝对水平 + 相对 train 的保留比（方向反转拦截）；
    - 截尾 IC 衰减取 train 窗口值；
    - TopN 可交易性（超额/Sharpe/回撤/尾部稳定）由 engine_gate 用完整回测
      引擎净值裁决，此处不做代理指标模拟。
    """

    name = "stage_two_stats"

    def __init__(self, criteria: DeliveryCriteria) -> None:
        self.p = criteria.production

    def run(self, evidence: dict[str, Any]) -> StageResult:
        train_metrics = evidence.get("train_metrics") or {}
        val_metrics = evidence.get("val_metrics") or {}
        similarity = evidence.get("similarity") or {}
        reasons: list[str] = []

        ic = _abs_ic(train_metrics.get("ic"))
        if ic is None or ic < float(self.p.min_train_abs_ic):
            reasons.append("train_ic")
        icir = train_metrics.get("icir")
        try:
            icir_v = abs(float(icir))
        except (TypeError, ValueError):
            icir_v = None
        if icir_v is None or icir_v <= float(self.p.min_train_icir):
            reasons.append("train_icir")

        v_ic = _abs_ic(val_metrics.get("ic"))
        if v_ic is None or v_ic < float(self.p.min_val_abs_ic):
            reasons.append("val_ic")

        # 正式库精筛用 production.min_val_ic_retention（0.60），
        # 与候选池的 candidate 阈值（0.50）不同，须显式注入。
        retention = StageOneValRetention(self.p.min_val_ic_retention)
        reasons += retention.run({
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }).fail_reasons

        # val 多头端毛值超额（方向自适应十分组，复利年化）：
        # IC 为正 ≠ 多头组合为正——alpha 可能全在空头端/中段排名（2026-08 审计发现的盲区）。
        thr_vle = self.p.min_val_long_excess
        if thr_vle is not None:
            vle = val_metrics.get("val_long_excess")
            try:
                vle_v = float(vle)
            except (TypeError, ValueError):
                vle_v = None
            if vle_v is None or not np.isfinite(vle_v) or vle_v <= float(thr_vle):
                reasons.append("val_long_excess")

        winsor_decay = train_metrics.get("winsorized_abs_ic_decay")
        try:
            wd_v = float(winsor_decay)
        except (TypeError, ValueError):
            wd_v = None
        if wd_v is None or wd_v > float(self.p.max_winsorized_abs_ic_decay):
            reasons.append("winsorized_abs_ic_decay")

        corr = similarity.get("max_abs_corr", 0.0)
        try:
            corr_v = float(corr)
        except (TypeError, ValueError):
            corr_v = None
        if corr_v is None or corr_v >= float(self.p.max_abs_corr):
            reasons.append("max_cs_corr")

        return StageResult(passed=len(reasons) == 0, fail_reasons=reasons)


class DeliveryChecker:
    """编排两阶段统计门槛判定，供 submit() 复用。

    只做纯判定：传入证据字典，返回各阶段 StageResult。promotion_status /
    skipped_reason / registry 副作用仍由 submit() 按原顺序编排，payload 语义不变。
    """

    def __init__(self, criteria: DeliveryCriteria) -> None:
        self.criteria = criteria
        self.stage_one = [
            StageOneStats(criteria),
            StageOneTurnover(criteria),
        ]
        self._stage_two_stats = StageTwoStats(criteria)

    def stage_one_stats(self, metrics_train: dict[str, Any]) -> StageResult:
        """海选统计 + 换手可行性（train-only 口径）。"""
        reasons: list[str] = []
        for stage in self.stage_one:
            reasons.extend(stage.run({"metrics": metrics_train}).fail_reasons)
        return StageResult(passed=len(reasons) == 0, fail_reasons=reasons)

    def stage_one_val_retention(
        self,
        metrics_train: dict[str, Any],
        val_metrics: dict[str, Any],
    ) -> StageResult:
        return StageOneValRetention(self.criteria.candidate.min_val_ic_retention).run({
            "train_metrics": metrics_train,
            "val_metrics": val_metrics,
        })

    def stage_one_correlation(self, similarity: dict[str, Any] | None) -> StageResult:
        return StageOneCorrelation(self.criteria).run({"similarity": similarity})

    def stage_two(
        self,
        metrics_train: dict[str, Any],
        val_metrics: dict[str, Any],
        similarity: dict[str, Any] | None,
    ) -> StageResult:
        return self._stage_two_stats.run({
            "train_metrics": metrics_train,
            "val_metrics": val_metrics,
            "similarity": similarity,
        })
