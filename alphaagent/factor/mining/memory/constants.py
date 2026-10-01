# -*- coding: utf-8 -*-
"""研究记忆共享常量：verdict 分类、AlphaMemo 校准阈值、起源权重。"""

from __future__ import annotations

# ── 数据版本 ──
# v3：cells 键升级为 (family, motif, parent_bucket)，显式/隐式父本分列 + 同桶残差基线
# v4：memory_entries 加 facets_json（数据面标签）；family 允许面对组合键（跨组融合）
# v5：因子中台 ID——memory_factors 维表（uid 主键）+ memory_entries.factor_uid；
#     uid = identity.factor_uid(factor_name) 确定性派生，回填按 factor_name 聚类
# v6：verdict 语义修正——评估未产出（面板缺列/超时/参数错）从 rejected 分离为
#     eval_error（"没算出来 ≠ 被否定"）；存量迁移把 rejected 中带 error 的条目重分类
# v7：SSPM 编辑统计层重构——cells 按 9 大粗族与细化 motif 重聚合，提升每 cell 观测密度与信噪比
DATA_VERSION = "7"

# ── Verdict 分类 ──
# near_miss（2026-09-05）：IC 达门槛 80%、ICIR/coverage 达标但未过线——
# 弱负向（不算正向证据，不触发重复提交拦截；但比 weak 多一次二次机会提示）
# eval_error（2026-09-07）：评估未产出结果（面板缺列/超时/参数错）——
# "没算出来 ≠ 被否定"，与机制性 reject 分离；负向权重最弱（-0.2）
#
# train_passed（2026-10-01 由 "promising" 改名而来）：
#   语义 = **只过训练集海选线**（train_passed），不是质量结论。旧名 "promising"
#   与 validated/candidate_approved 并列在 POSITIVE_VERDICTS 里，被误读成"已成候选"，
#   实测在多次复盘中被当成"达标"引用（包括本会话的两次误判），故改名消除歧义。
VERDICT_TRAIN_PASS = "train_passed"
# 改名前的旧字面量：**仅用于读取历史记忆数据**（memory_entries 里已落库的行）。
# 新写入一律用 VERDICT_TRAIN_PASS；查询侧用 POSITIVE_VERDICTS_READ 兼顾两代数据。
LEGACY_VERDICT_TRAIN_PASS = "promising"
LEGACY_VERDICT_ALIASES = {LEGACY_VERDICT_TRAIN_PASS: VERDICT_TRAIN_PASS}

POSITIVE_VERDICTS = frozenset(
    {"production_approved", "validated", "candidate_approved", VERDICT_TRAIN_PASS}
)
# 查询/聚合专用：把旧字面量也算进来，保证改名不改变历史数据的统计口径。
POSITIVE_VERDICTS_READ = POSITIVE_VERDICTS | frozenset({LEGACY_VERDICT_TRAIN_PASS})
# 判定「训练过线」时的可接受字面量（新写 train_passed，历史行仍是 promising）。
TRAIN_PASS_VERDICTS = (VERDICT_TRAIN_PASS, LEGACY_VERDICT_TRAIN_PASS)
NEGATIVE_VERDICTS = frozenset({"rejected", "revise_required", "weak", "near_miss", "eval_error"})


def normalize_verdict(verdict: object) -> str:
    """把历史字面量归一到当前标识（promising → train_passed）；其它原样返回。"""
    raw = str(verdict or "")
    return LEGACY_VERDICT_ALIASES.get(raw, raw)

# Verdict 显示顺序（正值优先，负值在后）
# 注：legacy "promising" 与 "train_passed" 同序（同一判定的两代字面量），
# 保留旧键是为了让历史记忆行在排序/过滤里不掉队（改名不改变历史口径）。
VERDICT_ORDER = {
    "production_approved": 0,
    "validated": 1,
    "candidate_approved": 2,
    "train_passed": 3,
    LEGACY_VERDICT_TRAIN_PASS: 3,
    "near_miss": 4,
    "revise_required": 5,
    "rejected": 6,
    "weak": 7,
    "eval_error": 8,
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

# ── near_miss 判定阈值（单一真源）──
# schema._classify 与 tools/_dispatch._near_miss_verdict 共用，避免两处硬编码漂移。
# IC 达门槛的 80%、ICIR/coverage 达标但未过线 → near_miss（2026-09-05 记忆分析：
# technical 档 239 个 near-miss 因子直接进死档，无二次机会）
NEAR_MISS_IC_RATIO = 0.8
NEAR_MISS_ICIR_SOFT = 0.2
NEAR_MISS_COVERAGE = 0.85

# ── 起源权重 ──
# explicit 父本（LLM 明确声明变异轨）权重 1.0
# implicit 父本（结构相似度自动链接）权重 0.5
PARENT_ORIGIN_WEIGHT = {"explicit": 1.0, "implicit": 0.5}

# 无效尝试（报错）入账失败观测的权重
INVALID_WEIGHT = 0.5

# 纯技术性失败模式：参数格式、DSL 解析、面板列缺失、预测格式等纯工程报错
# 不代表量化经济机制或编辑方向失效，SSPM cell 统计时直接短路跳过，不计入失败观测
TECHNICAL_ERROR_PATTERNS = frozenset({
    "ToolArgumentsError",
    "DSL_PARSE_ERROR",
    "dsl_compile_failed",
    "panel_column_missing",
    "missing column",
    "prediction_missing",
    "prediction_invalid",
    "prediction_check_error",
    "Tool execution timed out",
    "MemoryAdvisoryBlock",
})

# Verdict 权重（用于统计聚合）
# 注：legacy "promising" 与 "train_passed" 同权（同一判定的两代字面量）——
# 缺了旧键会让历史行的证据权重从 0.6 掉到默认 0.2。
VERDICT_WEIGHT = {
    "production_approved": 1.0,
    "validated": 1.0,
    "candidate_approved": 0.8,
    "train_passed": 0.6,
    LEGACY_VERDICT_TRAIN_PASS: 0.6,
    "near_miss": -0.2,
    "revise_required": -0.3,
    "rejected": -1.0,
    "weak": -0.5,
    "eval_error": -0.2,
}
