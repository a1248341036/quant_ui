"""Evaluation-layer default thresholds (no mining dependency).

``profile.resolve_profiles`` 需要 evaluation_policy / delivery_policy.production 的
canonical 默认值来编译 profile rules；这些数值原本从
``alphaagent.factor.mining.research_spec.DEFAULT_RESEARCH_SPEC`` 读取，造成
evaluation → mining 的分层倒置（mining 又依赖 evaluation，形成循环）。

本模块把 evaluation 层真正消费的默认门槛下沉为常量，research_spec 与 profile
都从这里取数，保持单一真源（数值与 research_spec / delivery_criteria 默认一致）。
"""

from __future__ import annotations

# 与 research_spec.DEFAULT_RESEARCH_SPEC["evaluation_policy"] 默认值严格一致
# （2026-09-11 观察池口径：评估屏幕线与 delivery 候选门 0.020/0.28 对齐；
#  2026-09-23 val 绝对门 0.012→0.015 与 delivery 候选门对齐）。
DEFAULT_EVALUATION_POLICY: dict[str, float] = {
    "min_train_abs_ic": 0.020,
    "min_train_icir": 0.28,
    "min_train_icir_soft": 0.2,  # verdict 判定用宽松线（不跟候选门槛 0.28）
    "min_train_coverage": 0.85,
    "min_val_abs_ic": 0.015,
    "min_val_ic_retention_ratio": 0.5,
    "min_cs_autocorr": 0.18,
    # 2026-10-02 阈值收口（P5）：以下三项原先硬编码在业务逻辑里（违反"阈值收口配置中心"纪律）。
    "orthogonality_max_corr": 0.7,        # 离线正交预判阈值（进候选池硬门；原 agentscope_tools._ORTHO_MAX_CORR）
    "prediction_soft_limit": 3,           # prediction 缺失软门次数（原 _dispatch._PREDICTION_SOFT_LIMIT）
    "prediction_sign_min_abs_ic": 0.003,  # 符号对账的最小 |IC|（原 eval/prediction.py 硬编码两处）
}

# 与 delivery_criteria.ProductionCriteria 默认值严格一致（唯一真源在
# delivery_criteria；此处仅 profile 编译规则需要的两个键）。
DEFAULT_PRODUCTION_THRESHOLDS: dict[str, float] = {
    "min_train_abs_ic": 0.025,
    "min_train_icir": 0.30,
}