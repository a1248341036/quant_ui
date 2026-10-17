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
        cold_track_policy: dict[str, Any] | None = None,
        crowd_skeletons: dict[str, Any] | None = None,
    ) -> None:
        self.service = service
        self.session_id = session_id
        self.submit_service = submit_service
        self._screener_config_dict = screener_config or {}
        # v3-lite：研究记忆硬提醒通道（None = 关闭）；hard_block_duplicates=True 时指纹死路直接拦截
        self.memory_store = memory_store
        # 当前 run id（log_dir.name）：传给 advisory_for.current_run_id，供 run 内正向豁免判定
        self.run_id = run_id
        # 冷门赛道双轨评估（spec: alphaagent_mining_anti_convergence_spec.md M1/M2）：
        # enabled=False（默认）时 dispatch 评估路径逐字节不变（黄金基线）。
        # crowd_skeletons = run 启动时的骨架快照（build_crowd_skeletons 产物，
        # 含 signature），仅实验模式注入。
        self.cold_track_policy = dict(cold_track_policy or {})
        self.crowd_skeletons = dict(crowd_skeletons or {})
        # M2 算子冷却（spec §7.3）：冷门算子连续 cold_dead(no_alpha) 达到阈值后，
        # 冷却期内拦截同算子新候选（避免为挖而挖重复跑必死算子）。run 级状态
        # （对象生命周期），跨 run 由研究记忆 cold_dead 负证据自然继承。
        self._cold_op_dead_count: dict[str, int] = {}
        self._cold_op_cool_left: dict[str, int] = {}
        # 冷门赛道评估预算计数（spec §7.4 cold_eval_budget_ratio）：分母=三类评估工具总调用，
        # 分子=真正跑了 evaluate_cold_track 的冷门候选。超限后 _cold_track_shortcut 回落既有评估。
        self._cold_eval_count: int = 0
        self._total_eval_count: int = 0
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
        # 已自动跑过样本外验证的表达式集合（_auto_val_verify 防抖，会话级）
        self._auto_val_done: set[str] = set()


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
