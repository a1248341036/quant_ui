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
    ) -> None:
        self.service = service
        self.session_id = session_id
        self.submit_service = submit_service
        self._screener_config_dict = screener_config or {}
        # v3-lite：研究记忆硬提醒通道（None = 关闭）；hard_block_duplicates=True 时指纹死路直接拦截
        self.memory_store = memory_store
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
