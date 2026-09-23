"""LLM 评估与交付工具：train/val 评估 + submit_factor 入库。"""
from __future__ import annotations

from typing import Any

from alphaagent.factor.mining.service import StockEvalService
from alphaagent.factor.mining.submit import FactorSubmitService

from ._dispatch import _DispatchMixin
from ._analysis import _AnalysisMixin
from ._prefilter import _is_naive_signal_addition
from ._schemas import (
    TOOL_NAMES,
    _EVAL_PARAMETERS,
    _PROFILE_EVAL_PARAMETERS,
    _SCREEN_FACTORS_PARAMETERS,
    _SUBMIT_PARAMETERS,
    _VAL_PARAMETERS,
)


class FactorEvalTools(_DispatchMixin, _AnalysisMixin):
    """持有一个已建会话，向 LLM 暴露 eval_on_train_set / eval_on_val_set / submit_factor。"""

    def __init__(
        self,
        service: StockEvalService,
        session_id: str,
        *,
        submit_service: FactorSubmitService | None = None,
        screener_config: dict[str, Any] | None = None,
        memory_store: Any | None = None,
        focus_facets: tuple[str, ...] | list[str] | None = None,
        cognition_policy: dict[str, Any] | None = None,
        operator_policy: dict[str, Any] | None = None,
        homogenization_policy: dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> None:
        self.service = service
        self.session_id = session_id
        self.submit_service = submit_service
        self._screener_config_dict = screener_config or {}
        # v3-lite：研究记忆硬提醒通道（None = 关闭）；hard_block_duplicates=True 时指纹死路直接拦截
        self.memory_store = memory_store
        # 当前 run id（log_dir.name）：传给 advisory_for.current_run_id，供 run 内正向豁免判定
        self.run_id = run_id
        # 数据面聚焦硬锁定（用户勾选；空 = 未启用，dispatch 不拦截）
        self.focus_facets = tuple(focus_facets or ())
        # 认知对账开关（research_spec.cognition_policy，消融 C1/C2）；缺省全开
        from ._dispatch import _DEFAULT_COGNITION_POLICY, _DEFAULT_OPERATOR_POLICY

        merged_cog = dict(_DEFAULT_COGNITION_POLICY)
        merged_cog.update({k: v for k, v in (cognition_policy or {}).items() if v is not None})
        self.cognition_policy = merged_cog
        # 算子黑名单（research_spec.operator_policy，消融 D2）；缺省空清单
        merged_op: dict[str, Any] = {k: v for k, v in _DEFAULT_OPERATOR_POLICY.items()}
        merged_op.update({k: v for k, v in (operator_policy or {}).items() if v is not None})
        merged_op["blacklist"] = tuple(str(n).upper() for n in (merged_op.get("blacklist") or ()))
        self.operator_policy = merged_op
        # 同质化平滑变体动态熔断（research_spec.homogenization_policy）
        # enabled=True（默认）：滑动窗口内同根+同外层变换+套平滑累计 max_consecutive 次 → 拦截评估
        # 2026-09-19：从"连续同根"改为"滑动窗口内同根累计"，防 LLM 换根轮换绕过
        # 2026-09-23：同根但外层变换不同（换中性化键/换截面变换）不累计，属正交化探索；
        #   max_outer_variants 兜底防"换中性化键"无限微调同一信号根
        self.homogenization_policy: dict[str, Any] = {
            "enabled": True,
            "max_consecutive": 3,
            "window_size": 10,
            "max_outer_variants": 6,
        }
        self.homogenization_policy.update({k: v for k, v in (homogenization_policy or {}).items() if v is not None})
        # 最近评估历史（记录 fingerprint + turnover + 是否含平滑算子 + 信号算子族），供熔断器使用。
        # 上限与 window_size 联动（≥ window_size * 2，下限 20），防 window_size 超配时
        # recent_evals 静默截断窗口导致熔断器实际窗口比配置小。
        _ws = int(self.homogenization_policy.get("window_size", 10) or 10)
        self._recent_evals_cap: int = max(20, _ws * 2)
        self._recent_evals: list[dict[str, Any]] = []


__all__ = [
    "FactorEvalTools",
    "TOOL_NAMES",
    "_is_naive_signal_addition",
    "_EVAL_PARAMETERS",
    "_VAL_PARAMETERS",
    "_PROFILE_EVAL_PARAMETERS",
    "_SUBMIT_PARAMETERS",
    "_SCREEN_FACTORS_PARAMETERS",
]
