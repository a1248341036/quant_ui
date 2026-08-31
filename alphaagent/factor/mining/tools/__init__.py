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
    ) -> None:
        self.service = service
        self.session_id = session_id
        self.submit_service = submit_service
        self._screener_config_dict = screener_config or {}
        # v3-lite：研究记忆硬提醒通道（None = 关闭）；hard_block_duplicates=True 时指纹死路直接拦截
        self.memory_store = memory_store


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
