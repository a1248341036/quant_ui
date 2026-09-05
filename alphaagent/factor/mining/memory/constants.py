# -*- coding: utf-8 -*-
"""研究记忆共享常量：verdict 分类、AlphaMemo 校准阈值、起源权重。"""

from __future__ import annotations

# ── 数据版本 ──
# v3：cells 键升级为 (family, motif, parent_bucket)，显式/隐式父本分列 + 同桶残差基线
# v4：memory_entries 加 facets_json（数据面标签）；family 允许面对组合键（跨组融合）
DATA_VERSION = "4"

# ── Verdict 分类 ──
# near_miss（2026-09-05）：IC 达门槛 80%、ICIR/coverage 达标但未过线——
# 弱负向（不算正向证据，不触发重复提交拦截；但比 weak 多一次二次机会提示）
POSITIVE_VERDICTS = frozenset({"production_approved", "validated", "candidate_approved", "promising"})
NEGATIVE_VERDICTS = frozenset({"rejected", "revise_required", "weak", "near_miss"})

# Verdict 显示顺序（正值优先，负值在后）
VERDICT_ORDER = {
    "production_approved": 0,
    "validated": 1,
    "candidate_approved": 2,
    "promising": 3,
    "near_miss": 4,
    "revise_required": 5,
    "rejected": 6,
    "weak": 7,
}

# ── AlphaMemo 校准常量 ──
EQ7_KAPPA_DEFAULT = 8          # Eq.7 置信度正则化常数
APV_TAU_C_DEFAULT = 0.35       # APV 置信度门控阈值
APV_TAU_V_DEFAULT = 0.80       # APV 失败率否决阈值
BASELINE_HALF_LIFE_DAYS = 90   # 残差基线时间衰减半衰期

# ── 编辑先验注入门控（retrieval._edit_prior_block）──
# 硬档（硬推荐/硬否决）共用；软推荐与软否决分向设阈值，
# 否决向放宽（默认 0.3）以放行「一致失败」的避坑证据
EDIT_PRIOR_HARD_CONF_DEFAULT = 0.7
EDIT_PRIOR_RECOMMEND_CONF_DEFAULT = 0.4
EDIT_PRIOR_VETO_CONF_DEFAULT = 0.3

# ── 起源权重 ──
# explicit 父本（LLM 明确声明变异轨）权重 1.0
# implicit 父本（结构相似度自动链接）权重 0.5
PARENT_ORIGIN_WEIGHT = {"explicit": 1.0, "implicit": 0.5}

# 无效尝试（报错）入账失败观测的权重
INVALID_WEIGHT = 0.5

# Verdict 权重（用于统计聚合）
VERDICT_WEIGHT = {
    "production_approved": 1.0,
    "validated": 1.0,
    "candidate_approved": 0.8,
    "promising": 0.6,
    "near_miss": -0.2,
    "revise_required": -0.3,
    "rejected": -1.0,
    "weak": -0.5,
}
