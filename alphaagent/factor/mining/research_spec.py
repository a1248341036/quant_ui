"""Versioned, runtime research policy for AlphaAgent factor mining."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from alphaagent.factor.evaluation.defaults import DEFAULT_EVALUATION_POLICY
from alphaagent.factor.evaluation.profile import default_evaluation_profiles, resolve_profiles
from alphaagent.factor.mining.interactions import INTERACTION_TYPES
from alphaagent.core.paths import RESEARCH_SPECS_DIR
from core import trading_config
from core.research_modes import RESEARCH_MODES as _MODE_REGISTRY, get_research_mode


def _default_evaluation_profile_spec() -> dict[str, Any]:
    # Rules are compiled from evaluation_policy unless the user explicitly overrides them.
    profiles: dict[str, Any] = {}
    for profile_id, profile in default_evaluation_profiles().items():
        value = profile.as_dict()
        value.pop("rules", None)
        profiles[profile_id] = value
    return profiles


RESEARCH_MODES = tuple(_MODE_REGISTRY.keys())


# ── 用户门槛覆盖持久化（每模式一个 JSON 文件）────────────────────────
# 存储的是"相对注册表默认值的增量覆盖"（compute_spec_overrides 的 diff 结果），
# 而非整份 spec：代码默认值演进时，用户未改过的键自动跟随，改过的键保持覆盖。
# 消费方：
# - effective_research_spec(mode)  → 默认 + 覆盖（normalize 后），前端编辑/展示用
# - build_run_research_spec(spec)  → 运行口径：注册表默认 < 保存覆盖 < 显式 spec
# - core.factor_categories 不涉及门槛；全链路（CLI/Web/晋升）统一走上述两个入口。
def _spec_overrides_path(mode: str) -> Path:
    if mode not in RESEARCH_MODES:
        raise ValueError(f"research_mode_invalid:{mode}")
    return RESEARCH_SPECS_DIR / f"{mode}.json"


def load_saved_overrides(mode: str) -> dict[str, Any]:
    """读取某模式的用户门槛覆盖；无文件/损坏时返回空 dict（回落默认）。"""
    try:
        value = json.loads(_spec_overrides_path(mode).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def save_research_spec_overrides(mode: str, overrides: dict[str, Any]) -> Path:
    """保存某模式的用户门槛覆盖（默认值无需保存，丢键即回落默认）。"""
    if not isinstance(overrides, dict):
        raise ValueError("research_spec.overrides_must_be_object")
    path = _spec_overrides_path(mode)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(overrides, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return path


def reset_research_spec_overrides(mode: str) -> bool:
    """删除某模式的覆盖文件，恢复注册表默认门槛。返回是否真的有文件被删。"""
    path = _spec_overrides_path(mode)
    if not path.exists():
        return False
    path.unlink(missing_ok=True)
    return True


def compute_spec_overrides(defaults: dict[str, Any], edited: dict[str, Any]) -> dict[str, Any]:
    """默认 spec 与前端编辑后 spec 的增量 diff（递归 dict；list/标量整体替换）。

    只保留与默认值不同的键 → 持久化的"门槛文件"最小、可读且随代码默认演进。
    """
    out: dict[str, Any] = {}
    for key, value in edited.items():
        if key not in defaults:
            out[key] = value
        elif isinstance(value, dict) and isinstance(defaults[key], dict):
            sub = compute_spec_overrides(defaults[key], value)
            if sub:
                out[key] = sub
        elif value != defaults[key]:
            out[key] = value
    # 编辑版显式删掉的键 = 恢复默认，无需写回
    return out


def effective_research_spec(mode: str = "technical") -> dict[str, Any]:
    """注册表默认 + 用户保存覆盖（normalize 后的完整 spec）。

    前端编辑/展示、以及"不显式传 spec"的运行路径都以此为准；
    它不会改变 default_research_spec 的纯默认语义（测试/调用方依旧可取纯默认）。
    """
    merged = _deep_merge(default_research_spec(mode), load_saved_overrides(mode))
    return normalize_research_spec(merged)


def build_run_research_spec(explicit: dict[str, Any] | None = None) -> dict[str, Any]:
    """运行口径研究规范：注册表默认 < 保存覆盖 < 显式 spec（如前端 JSON / CLI 文件）。

    显式 spec 已含保存值时幂等（保存值再合并一次不改变结果）；
    显式 spec 缺键时从保存覆盖/默认补齐——保证任何入口都不会绕过用户改过的门槛。
    """
    explicit = dict(explicit) if isinstance(explicit, dict) else {}
    mode = str(explicit.get("research_mode") or "technical")
    base = default_research_spec(mode)
    merged = _deep_merge(_deep_merge(base, load_saved_overrides(mode)), explicit)
    return normalize_research_spec(merged)


def _default_delivery_policy() -> dict[str, Any]:
    """两阶段交付门槛默认值，唯一真源在 delivery_criteria（含设计注释）。

    运行时以 research_spec 注入为准（agentscope/run 均传 delivery_policy）；
    delivery_criteria.DeliveryCriteria.defaults() 集中定义数值与设计依据，
    prompt 渲染（tools.py / prompts.py）也从同一对象取数，杜绝硬编码漂移。
    """
    from alphaagent.factor.mining.delivery_criteria import DeliveryCriteria

    return DeliveryCriteria.defaults().to_spec_dict()


DEFAULT_RESEARCH_SPEC: dict[str, Any] = {
    "version": 1,
    "research_mode": "technical",
    # 信息性提示：该模式建议的评估 label，前端/调用方可据此设置 --label-col。
    "recommended_label_col": "label_1d_open_to_open",
    # ── label 与调仓频率**强制一致**（2026-10-04 用户定调：「调仓频率是多少，label 就应该多少」）──
    # 规则：`delivery_policy.production.engine_gate.freq` 的持有期必须等于 label 的持有期，
    # 否则 **fail-closed 拒绝启动**（不再是"刻意解耦 + 分档阈值折算"）。
    # 为什么改（2026-10-04 实测）：旧组合（1d label + weekly 交付）里
    #   · 研究口径按 1d 打分、交付口径按周频回测；
    #   · 但 stage_one 的换手硬门拿**日频信号抖动**去撞 weekly 档阈值（0.65），
    #     而 hold=label 天数=1 → 组合其实每天在换（实测 n_rebalances=1615、
    #     avg_rebalance_side_turnover == avg_daily_side_turnover）→ "周频摊薄成本"的假设不成立。
    # 取值：daily=1 / weekly=5 / monthly=20（交易日）。调大/调小 = 改口径，必须同步单测与 docs。
    # 实现方向：**freq 缺省由 label 派生**（模式未显式声明 freq 时），显式声明但与之矛盾则报错。
    # 注：本仓 min_abs_ic / min_train_icir / reproduce_min_* 等门槛是按 **1d 尺度**标定的，
    #   所以当前 1d 研究口径下对齐结果 = freq:daily；若将来要 weekly 交付，必须同时换
    #   label_5d_* 并**重标定这些门槛**（属独立改动，不可只改一侧）。
    "label_freq_consistency": {
        "enforce": True,
        "freq_hold_days": {"daily": 1, "weekly": 5, "monthly": 20},
    },
    # run 末尾兜底补交（2026-10-02）：训练过线但因 max_turns 耗尽**未提交**的因子，按 |ICIR|
    # 取前 N 个走**正常 submit 通路**补交（不绕过盲测/stage_one/stage_two/engine gate 任何门槛）。
    #   0 = 关闭。代价实测：单次 submit 约 5~10 分钟（盲测+stage_one+engine gate），
    #   故默认 3（夜间 run 结尾多花 ~15~30 分钟换回产出）。
    # 依据：2026-10-02 整夜实测两段 run 分别有 32 / 30 个训练过线因子未提交（均因 max_turns），
    #   此前只有 `unsubmitted_promising` 审计日志、**无任何补交通路**（全仓无 auto_submit 实现）。
    "auto_submit_unsubmitted_top": 3,
    "search_policy": {
        "allowed_signal_families": ["volume_price", "volatility", "chip", "momentum_reversal"],
        "forbidden_signal_families": ["pure_size"],
        # 允许单机制深度因子：强制 ≥2 个原始字段会诱导"信号A+信号B"式低级叠加，
        # 降为 1 后 Agent 可专注单一信息源的深度挖掘（多层算子链而非多字段拼凑）。
        "min_distinct_raw_fields": 1,
        "require_time_series_structure": False,
        # 2026-09-03：5 → 8（并行批量验证用户诉求 20/轮；上限 24 覆盖）。
        "max_candidates_per_round": 8,
    },
    "evaluation_policy": {
        # 2026-09-11 观察池口径：评估屏幕线与 delivery 候选门（0.020/0.28）对齐，
        # 避免"评估过线但提交即拒"的算力浪费；晋升线（0.025/0.30）在 delivery_policy。
        # 数值单一真源在 alphaagent.factor.evaluation.defaults（profile 编译规则同源）。
        **DEFAULT_EVALUATION_POLICY,
        "require_sign_consistency": True,
    },
    "evaluation_profiles": _default_evaluation_profile_spec(),
    "review_policy": {
        # 2026-10-01 用户决策：**全关**。理由（实测）：
        #   ①26/26 曾全判 revise，空转且拖慢（infra/config.py:48 的默认关闭即源于此）；
        #   ②本夜 API run 里 11 次审查中 4 次 reject 全为 novelty=low，
        #     理由是"教科书短期反转/VWAP乖离重包装"——与"新颖性不作为优化目标、
        #     只看有效性"的定调冲突，且它在 stage_one、正交门之后**硬拦候选池入库**
        #     （delivery/submit.py:890 `factor_review_reject_blocked`）；
        #   ③解析失败兜底 fail-closed 判 reject（factor_reviewer.py:293-302）。
        # 防重复仍由两道**独立**门负责，均未改动：离线正交门
        # （submit.py:852-861 `offline_orthogonality`）与 stage 内 `max_cs_corr`。
        # 需要重开时：把 enabled 改回 True（亦可 CLI 用 --no-reviewer 临时关）。
        "enabled": False,
        "review_on": ["validation", "pre_submit"],
        # 经典单调变换不再作为阻断理由（同上定调）；重开 Reviewer 时也保持只提示。
        "block_classic_transforms": False,
        # 新颖度门槛暂关（2026-08）：先积累一批统计有效的候选，再回头筛新颖性。
        # 恢复严格筛选时改回 "medium"/"high"；Reviewer 仍输出 novelty 供参考。
        "minimum_novelty": "low",
    },
    "interaction_policy": {
        # 用户约束：尽量不用乘法——默认禁用 MULTIPLY 交互（含契约形式）。
        # 确需乘法时，须在运行 ResearchSpec 中显式把 "multiplication" 加回本列表，
        # 并按 reviewer 要求提供 base-only / condition-only / combined 消融证据。
        "allowed_interaction_types": [t for t in INTERACTION_TYPES if t != "multiplication"],
        "block_undeclared_multiply": True,
        "require_contract_for_typed_interactions": True,
        # 结构算子未传契约时自动补全 + warning（2026-09-06）：单面 run 无契约
        # 教学时的高频格式瑕疵，硬拦截整轮作废的算力损失大于纪律收益
        # （与 prediction 缺失软门 7577ca7 同哲学）；机制纪律由 Reviewer 把守。
        "auto_fill_missing_contract": True,
        "require_ablation_for_multiplication": True,
    },
    "memory_policy": {
        "retrieve_limit": 12,
        "dynamic_retrieve_limit": 8,        # 每轮动态注入条数（正向配额 40%，其中约 2/3 为跨族保底父本）
        "max_expression_chars": 320,
        "include_rejected_paths": True,
        "prefer_orthogonal_to_approved": True,
        "include_expression": True,
        "enable_factor_retrieval": True,    # v2 混合检索（BM25+族亲和+多样性去重），结论已数据化
        "enable_edit_patterns": True,       # v2 (family×motif) 残差单元 + APV 否决，含统计门控
        # 组件级消融开关（2026-09-16，memory component ablation spec §二.4）：
        # 与 enable_factor_retrieval / enable_edit_patterns 并列，逐个控制注入/写入组件。
        # 默认全 True 不改行为；消融实验按 arm 关闭对应开关。
        "enable_experience_block": True,    # 经验块（成功模式/禁忌方向，core 注入）
        "enable_saturation_block": True,    # 饱和度块（拥挤警告，secondary 优先级 1）
        "enable_yield_block": True,         # 产出率块（族产出率统计，secondary 优先级 2）
        "enable_diversity_block": True,     # 多样性块（面覆盖警告，secondary 优先级 3）
        "enable_operator_diversity_block": True,  # 算子分布软引导（secondary 优先级 4）
        "enable_structure_stats_block": True,  # 结构命中率块（交互结构过线率，core 预留）
        "enable_sspm_write": True,          # SSPM 编辑统计写入（memory_cells 残差更新）
        "enable_distill": True,             # 经验蒸馏（distill_batch_experience / form_memory）
        "enable_advisory_cache": True,      # advisory 查询 LRU 缓存（False = 每次评估查库）
        # v3-lite：AlphaMemo 校准 + 硬提醒通道
        # 重复探索硬闸（2026-09-23 从 False 上调为 True）：963 次评估 46% 重复（320 次死路重复
        # 全提醒不拦，LLM 可无视）。仅对 duplicate_known_dead_end（同结构指纹负证据累计
        # ≥dead_end_min_attempts 次且全历史无正向）硬拦；duplicate_prior_result 永不硬拦
        # （exempt_from_block=True）只提醒。
        # env ALPHA_MEMORY_HARD_BLOCK_DUPLICATES=0 可临时关闭。
        "hard_block_duplicates": True,
        # 指纹死路判定阈值：同结构负证据条目数或累计尝试次数 ≥ 该值（且全历史无正向）判死路。
        # 2026-09-23 从硬编码 2 收口为配置（2 次偏敏感，默认上调为 3）。
        "dead_end_min_attempts": 3,
        # OpenViking 冷路径长期记忆（2026-09-23 新增）：run 启动语义检索注入 system prompt、
        # run 结束写回摘要到 viking://resources/alphaagent/（代码硬编码 scope 隔离）。
        # 失败静默降级为纯 SQLite，不影响挖掘。
        "enable_ov_long_term_memory": True,
        "ov_endpoint": "http://127.0.0.1:1933",  # 本地 OpenViking HTTP 端点
        "ov_inject_max_chars": 2400,          # OpenViking 注入块预算（对齐 max_inject_chars）
        "max_inject_chars": 2400,           # 注入块总预算，超限按 编辑先验>经验>多样性>证据 截断
        "apv_tau_c": 0.35,                  # APV 双门 1：置信阈值（Eq.7 置信）
        "apv_tau_v": 0.80,                  # APV 双门 2：失败 Beta 后验阈值
        # 编辑先验（cells 注入）置信阈值：硬档共用，软推荐/软否决分向
        "edit_prior_hard_conf": 0.7,        # 硬推荐（有显式成功）/硬否决（有失败）
        "edit_prior_recommend_conf": 0.4,   # 软推荐（有显式成功）
        "edit_prior_veto_conf": 0.3,        # 软否决（有失败）——低于推荐向，放行避坑证据
        # 记忆出题（AlphaMemo 口径）：每轮前 k 个评估名额由 recommend_edits 推荐
        # （父本×编辑类型，残差×置信排序），0 = 关闭。注入块只做解释，名额由推荐驱动。
        "suggest_slots": 2,
        # 记忆出题字段门控（2026-09-28）：父本表达式引用本 run 未载入字段时不推荐
        # （实测 funda_* 父本在 --no-fundamentals 下连续 8 轮白占名额）。
        "suggest_field_gate": True,
        # 总开关（消融 A1）：False = 整个记忆系统关闭（不检索注入、不 APV 否决、
        # 不编辑先验、不硬拦死路、不蒸馏），等价 research_memory_path=None 的
        # 零记忆探索；默认 True 不改行为。
        "enabled": True,
    },
    # 认知对账层开关（消融 C1/C2）：关闭后评估结果不再注入 prediction_check /
    # ablation_check，软门累计缺失拦截同步关闭；默认双开不动行为。
    "cognition_policy": {
        "prediction_check_enabled": True,
        "ablation_check_enabled": True,
    },
    # 算子黑名单（消融 D2）：出现在列表中的算子名（大写）在评估/提交前直接拦截，
    # 算子目录同步隐藏，返回明确错误引导 LLM 改用基础算子；默认空清单不动行为。
    "operator_policy": {
        "blacklist": [],
    },
    "delivery_policy": _default_delivery_policy(),
    # 提示词分阶段注入策略：
    # - phase_mode="auto" → 默认按 turn 比例自动切换 explore→deepen→deliver（节约非核心模块 token，但算子目录始终全量保留冷门算子）；
    # - phase_mode="full" → 全量注入所有模块。
    # phase_ratio 三段比例默认均分 [1/3, 1/3, 1/3]。
    "prompt_policy": {
        "phase_mode": "auto",
        "phase_ratio": [1 / 3, 1 / 3, 1 / 3],
    },
    "report_policy": {
        # 研报知识注入通道（三选一 + 关闭，唯一真源 mining/report_channels.py）：
        #   report_rag（默认，R3/R4 文献 RAG + 逐轮课题对齐）/ static_manual（R1 静态手册）
        #   / mechanism_cards（R2 机制卡检索）/ off
        "knowledge_mode": "report_rag",
        "enable_report_mechanisms": False,
        "enable_report_prior": False,
        "report_cards_path": None,
        "enable_report_rag": True,
        "report_rag_max_chars": 1600,
        "report_rag_top_k": 3,
        "enable_question_queue": True,
        "question_queue_file": None,
        # 课题字段门控（2026-09-28）：派题前按「本 run 实际载入列」判定，
        # 纯缺列题跳过并顺延、部分缺列题保留但注入缺列提示。默认值唯一真源在
        # question_queue.DEFAULT_QUESTION_FIELD_GATE（此处仅镜像，校验见 build 函数）。
        "question_field_gate": True,
        "question_field_gate_min_ratio": 0.5,
        "question_field_gate_scan_limit": 40,
        "question_field_warn": True,
        # ── 复现门槛（研报模式；**唯一真源**，判定在 tools/_dispatch._report_reproduce_judge）──
        # 判定式：复现通过 = 【底线】 ∧ 【指标】 ∧ 【形态对账】（**三项全中**）
        #   （2026-10-01 用户定调：原先「指标 **或** 形态」，改为「**且**」——既要强度达标，
        #     也要事先声明的可证伪预测成立；只按形态放行会让 |IC|≈0 的噪声母本进发散，
        #     只按强度放行会让"机制理解错了但碰巧有 IC"的因子进发散。）
        #   · reproduce_min_abs_ic / reproduce_min_icir —— 指标：|IC| 与 |ICIR| 同时达线。
        #   · reproduce_shape_requirement —— 「形态对账」要求档位（2026-10-01）：
        #       "strong_side"（默认）实际强侧 == 声明强侧 —— 经济含义强（多头端是否最强，
        #         关乎多空组合能否赚钱），且**独立于 IC**（IC 只看全体相关性，看不见
        #         "收益集中在中间组但整体相关为正"的情形）；
        #       "confirmed" 要求 prediction_check 完全一致（强侧+形态+方向三项全中）——
        #         实测与 strong_side 质量等价（|IC| 0.0179/0.206 vs 0.0180/0.210）却多拦
        #         40 条同强度母本（曲线略偏即不过），故不作默认；
        #       "off" 完全不看形态，仅按指标（最松）。
        #     注：`shape` 字符串（单调/倒U/U形…）**恒作为诊断信息**写入日志与父本体检，
        #     无论本档位取值如何。
        #   已删除：reproduce_floor_abs_ic —— 与 reproduce_min_abs_ic 同值时被后者完全
        #   包含（判据改「且」后零影响），属冗余键。
        # 注 1：IC 与 ICIR **一律按绝对值**比较（负值=方向相反、强度等价；方向由形态对账的
        #       expected_sign 与正式库 abs 门槛负责，不参与强度判定）。
        # 注 2：自 2026-10-01 起这三个阈值可从 spec 配置（此前是代码内硬编码默认值，无法调参）。
        "reproduce_min_abs_ic": 0.010,
        "reproduce_min_icir": 0.10,
        "reproduce_shape_requirement": "strong_side",
        # 复现保真度「原文锚」（2026-10-01，仅研报模式；唯一真源 question_queue）：
        # ①只派发有原文公式的课题；②复现过线前校验复现公式与原文公式的字段/算子重叠。
        # 血统门禁（2026-10-02 开）：原先**缺键→默认关**，实测后果——本夜 run 的 12 个达双门槛
        # 因子**全部是同一个记忆建议父本 `mix_*` 的同根变体**（turn=0 的 memory_suggest 给了它，
        # 模型即脱稿发散），研报复现版被绕过 → 指标被同根灌水、研报血统丢失。
        #   · reproduce_of_required：复现轮的 `parent_factor` 必须声明 `reproduce_of:<课题号>`；
        #   · diverge_parent_required：发散轮的 `parent_factor` 必须指向该课题的复现版因子。
        # 两条在 agentscope_tools 里已有硬拦截实现（⛔ 会直接拒绝评估），此处只是把开关打开。
        # 若某课题机制无效，题面/拦截文案允许模型**明确放弃课题**，不是逼它硬凑。
        "reproduce_of_required": True,
        "diverge_parent_required": True,
        # 血统**自动补全**（2026-10-03，仅研报模式）：漏传 `parent_factor`（传空）时按阶段自动
        # 补齐——复现轮补 `reproduce_of:<课题号>`、发散轮补该课题复现版因子名。
        # 为什么：门禁拦截里 13+ 次/run 是「当前传入=(空)」，即模型在**正确课题与阶段**里干活
        # 却忘填字段，白烧一轮；漏传=忘写，补全即本意。**传非空错血统仍硬拦**（不猜不覆盖）。
        # 调小后果：回到严格口径（血统必须显式声明，诊断用），代价是模型漏填时继续烧轮次。
        "parent_autofill": True,
        "require_factor_records": True,
        # 结构准入（2026-10-02，仅研报模式）：只派发**报告提出了可复现结构**的课题。
        # 结构 = gated 条件门控 / composite 多因子合成 / residual 残差中性化 / timing 择时 /
        # cross_facet 跨面融合；`none`（仅通用单因子，如 ROE/PB/动量）**不出题**。
        # 依据（26707 次历史评估实测）：单算子因子达双门槛率 1.4%、1 字段 0.1%，
        # 而含结构算子 9.1%、3+ 字段 11.2%（差 6~112 倍）——复现"一条通用单公式"必然无效。
        # 兼容：题库无 `has_reproducible_structure` 字段（旧 1200 题库）→ 本项不筛，行为不变。
        "require_report_structure": True,
        # 方法类型准入（2026-10-02）：**暂时排除机器学习/图深度类研报**。
        # 理由：这类报告的"结构"是训练出来的模型/网络（GNN、Transformer、XGBoost…），
        # 在 DSL 里无法表达——硬出题只能得到"用任意表达式近似"的**伪复现**（实测案例
        # RQ_7810a8：复现目标写"异构图 GNN+两阶段训练"，实际复现成"成交额/VWAP 溢价残差"）。
        # 归类由 LLM 产出（scripts/classify_report_methods.py 的 `method_type` 字段），可选值：
        # rule_formula / factor_test / portfolio_combo / event_driven / timing_rotation /
        # ml_model / graph_deep / other。题库无该字段（旧题库）→ 不筛（向后兼容）。
        # 想放开某类：把它从本列表移除即可（例如 "ml_model"）。
        "exclude_method_types": ["ml_model", "graph_deep"],
        # **质量标签准入**（2026-10-03）：按回测出来的"效果"过滤课题。题库字段
        # `perf_label` 由 `scripts/label_question_quality.py` 依据**已落盘日志 + 状态机**打标：
        #   ok（复现过线过）/ untried（没判定过）/ suspect（判定够多却从未过线：
        #   weak_signal 从未够到 |IC|≥0.010、not_expressible 原文锚反复不达标）/
        #   deprecated（状态机已 abandoned，永久出局）。
        # 默认排除 deprecated + suspect：**不再为已知做不出的题烧整轮预算**
        # （实测单题最多烧 49~54 次判定、占当夜判定 47%）。旧题库无该字段 → 不筛。
        "exclude_perf_labels": ["deprecated", "suspect"],
        # 复现通过后锁定该课题的发散轮数（2026-10-03 收口：此前只存在于
        # report_channels.reproduce_lock_rounds 的 .get(...,3)，从未进配置中心，违反三件套）。
        # 语义：复现过线后同一课题继续发散的轮数。调大→单题挖得更深、铺题更慢；调小→铺题快、单题浅。
        # 实测（2026-10-03 夜）：唯一产出候选的 run（fc9bdcbd505b）全程只做 1 道题（深挖），
        # 而 3~5 道快换题的 run 全部 0 候选 → 故**本次收口不改值**（保持 3），改值需按数据再定。
        "reproduce_lock_rounds": 3,
        # 连续锚失败护栏（2026-10-03）：同一课题在复现阶段**连续 N 次**原文锚未达标（off_reference）
        # → **本 run 跳过该题**（run 级、不写持久 abandoned，未来 run 仍可再试）。0=关闭。
        # 动机：实测有题连续 93 次被锚拦截（占当夜全部拦截的 69%），整轮预算烧在一道
        # "DSL 表达不了"的题上。默认 8：给忠实复现足够尝试次数，又能在明显不匹配时止损。
        "anchor_block_skip_threshold": 8,
        "reproduce_fidelity": {
            "enabled": True,
            # 弱声明兜底（2026-10-02 标定）：59% 结构题未声明 structure_ops、声明字段常只有 1 个
            # → 结构锚退化。此时用复杂度地板挡一行式（实测 1 算子达双门槛 1.4% vs 4+ 算子 8.2%）。
            "min_ops_floor": 2,
            "min_fields_floor": 2,
            "require_shared_field": 1,   # 至少 1 个「报告特有字段」（通用行情字段不算）
            "min_field_jaccard": 0.0,
            "min_op_jaccard": 0.0,
            "min_shared_ops": 0,
        },
        # R2 机制卡检索参数（knowledge_mode=mechanism_cards 时生效）
        "report_prior_max_chars": 1600,
        "report_prior_top_k": 4,
        "include_negatives": True,
    },
}


def default_research_spec(mode: str = "technical") -> dict[str, Any]:
    """按研究模式生成默认 ResearchSpec（门槛/信号族/label 来自 core.research_modes 注册表）。

    加新模式只需在 core/research_modes.RESEARCH_MODES 增加一项，本函数零改动。
    """
    spec = copy.deepcopy(DEFAULT_RESEARCH_SPEC)
    if mode == "technical":
        # 2026-10-04：主档不再"完全等于 DEFAULT"——注册表里 technical 显式声明了对齐后的
        # engine_gate（label 1d ⇒ freq=daily、白名单收成 [daily]）。这里必须应用该覆盖，
        # 否则裸默认 spec 仍带 DEFAULT 的 weekly + 三档白名单（与 label_1d 不一致）。
        _tech_gate = (get_research_mode("technical").engine_gate_overrides or {})
        spec["delivery_policy"]["production"]["engine_gate"].update(_tech_gate)
        return spec
    mode_spec = get_research_mode(mode)
    spec["research_mode"] = mode
    spec["recommended_label_col"] = mode_spec.recommended_label_col
    spec["search_policy"].update({
        "allowed_signal_families": list(mode_spec.signal_families),
        "forbidden_signal_families": list(mode_spec.forbidden_families),
        "min_distinct_raw_fields": 1,
        "require_time_series_structure": True,
    })
    if mode_spec.evaluation_overrides:
        spec["evaluation_policy"].update(mode_spec.evaluation_overrides)
    if mode_spec.candidate_overrides:
        spec["delivery_policy"]["candidate"].update(mode_spec.candidate_overrides)
    if mode_spec.production_overrides:
        spec["delivery_policy"]["production"].update(mode_spec.production_overrides)
    if mode_spec.engine_gate_overrides:
        spec["delivery_policy"]["production"]["engine_gate"].update(
            mode_spec.engine_gate_overrides
        )
    if mode_spec.blind_test_overrides:
        spec["delivery_policy"]["blind_test"].update(mode_spec.blind_test_overrides)
    # 研报模式的流程开关（复现/发散阶段、复现门禁、锁定轮数）；其他模式为空 dict。
    if getattr(mode_spec, "report_policy_overrides", None):
        spec.setdefault("report_policy", {}).update(mode_spec.report_policy_overrides)
    return spec


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _require_dict(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"research_spec.{name}_must_be_object")
    return value


def _require_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"research_spec.{name}_must_be_boolean")
    return value


def _bounded_number(value: Any, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not lower <= float(value) <= upper:
        raise ValueError(f"research_spec.{name}_must_be_between_{lower}_and_{upper}")
    return float(value)


def _string_list(value: Any, name: str) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"research_spec.{name}_must_be_string_list")
    return [item.strip() for item in value]


def label_hold_days(label_col: Any) -> int:
    """label 列名 → 名义持有期（交易日）。口径与 engine/submit 侧一致（无数字视为 1d）。"""
    digits = "".join(ch for ch in str(label_col or "") if ch.isdigit())
    return max(1, int(digits) if digits else 1)


def ensure_label_freq_consistency(spec: dict[str, Any], *, label_col: str | None = None) -> None:
    """校验 label 持有期 == engine_gate.freq 持有期（2026-10-04 用户定调，fail-closed）。

    规则：「调仓频率是多少，label 就应该多少」（daily↔1d、weekly↔5d、monthly↔20d）。
    `build_run_research_spec` 内联做一次（构建期）；API 入口在**用户显式覆盖 freq 之后**
    再调用一次——否则覆盖发生在校验之后就成了绕过通道。

    `label_col` 可传本次 run 实际使用的 label（params.label_col）；不传则用
    `spec["recommended_label_col"]`。**两者都拿不到时跳过**（无法比较，交给下游
    run 级校验），避免对"部分 spec"误报。
    """
    lfc = spec.get("label_freq_consistency")
    lfc = lfc if isinstance(lfc, dict) else {}
    if not lfc.get("enforce", True):
        return
    label_col = label_col or spec.get("recommended_label_col")
    if not label_col:
        return
    hold_map = {"daily": 1, "weekly": 5, "monthly": 20}
    raw_map = lfc.get("freq_hold_days")
    if isinstance(raw_map, dict) and raw_map:
        hold_map = {str(k).lower(): int(v) for k, v in raw_map.items()}
    eg = ((spec.get("delivery_policy") or {}).get("production") or {}).get("engine_gate") or {}
    freq = str(eg.get("freq") or "").lower()
    label_col = str(label_col)
    freq_hold = hold_map.get(freq)
    if freq_hold is None:
        raise ValueError(
            f"research_spec.label_freq_consistency.freq_unknown: engine_gate.freq={freq!r} "
            f"不在 freq_hold_days 映射内（{sorted(hold_map)}）"
        )
    label_hold = label_hold_days(label_col)
    if freq_hold != label_hold:
        inverse = {days: name for name, days in hold_map.items()}
        raise ValueError(
            "research_spec.label_freq_consistency.mismatch: "
            f"engine_gate.freq={freq}（持有 {freq_hold}d）与 recommended_label_col={label_col}"
            f"（持有 {label_hold}d）不一致。按『调仓频率是多少，label 就应该多少』必须相同："
            f"要么把 label 改成 {freq_hold}d 标签（label_freq_consistency.freq_hold_days 里有对应档；"
            "换 label 属改研究口径，须同步重标定 min_abs_ic/min_train_icir 等门槛），"
            f"要么把 freq 改成与 {label_hold}d 对应的 {inverse.get(label_hold, 'daily')}"
            "（可用 label_freq_consistency.enforce=false 临时放行）"
        )


def normalize_research_spec(value: dict[str, Any] | None) -> dict[str, Any]:
    """Merge a user policy with defaults and validate fields used by the engine."""
    if value is not None and not isinstance(value, dict):
        raise ValueError("research_spec_must_be_object")
    mode = str((value or {}).get("research_mode", "technical"))
    if mode not in RESEARCH_MODES:
        raise ValueError("research_spec.research_mode_invalid")
    spec = _deep_merge(default_research_spec(mode), value or {})
    if int(spec.get("version", 0)) != 1:
        raise ValueError("research_spec.unsupported_version")
    spec["version"] = 1
    spec["research_mode"] = mode

    search = _require_dict(spec.get("search_policy"), "search_policy")
    search["allowed_signal_families"] = _string_list(search.get("allowed_signal_families"), "search_policy.allowed_signal_families")
    search["forbidden_signal_families"] = _string_list(search.get("forbidden_signal_families"), "search_policy.forbidden_signal_families")
    search["min_distinct_raw_fields"] = int(_bounded_number(search.get("min_distinct_raw_fields"), "search_policy.min_distinct_raw_fields", 1, 10))
    search["require_time_series_structure"] = _require_bool(search.get("require_time_series_structure"), "search_policy.require_time_series_structure")
    search["max_candidates_per_round"] = int(_bounded_number(search.get("max_candidates_per_round"), "search_policy.max_candidates_per_round", 1, 24))
    spec["auto_submit_unsubmitted_top"] = int(_bounded_number(
        spec.get("auto_submit_unsubmitted_top", 3), "auto_submit_unsubmitted_top", 0, 20
    ))

    evaluation = _require_dict(spec.get("evaluation_policy"), "evaluation_policy")
    for key in ("min_train_abs_ic", "min_val_abs_ic"):
        evaluation[key] = _bounded_number(evaluation.get(key), f"evaluation_policy.{key}", 0, 1)
    for key in ("min_train_icir", "min_train_icir_soft"):
        if key in evaluation:
            evaluation[key] = _bounded_number(evaluation.get(key), f"evaluation_policy.{key}", -10, 20)
    evaluation["min_train_coverage"] = _bounded_number(evaluation.get("min_train_coverage"), "evaluation_policy.min_train_coverage", 0, 1)
    evaluation["min_val_ic_retention_ratio"] = _bounded_number(evaluation.get("min_val_ic_retention_ratio"), "evaluation_policy.min_val_ic_retention_ratio", 0, 2)
    evaluation["require_sign_consistency"] = _require_bool(evaluation.get("require_sign_consistency"), "evaluation_policy.require_sign_consistency")
    evaluation["min_cs_autocorr"] = _bounded_number(
        evaluation.get("min_cs_autocorr", 0), "evaluation_policy.min_cs_autocorr", 0, 1
    )
    # 2026-10-02 P5 收口：三项原硬编码阈值，现纳入配置中心并做范围校验
    evaluation["orthogonality_max_corr"] = _bounded_number(
        evaluation.get("orthogonality_max_corr", 0.7),
        "evaluation_policy.orthogonality_max_corr", 0, 1,
    )
    evaluation["prediction_soft_limit"] = int(_bounded_number(
        evaluation.get("prediction_soft_limit", 3),
        "evaluation_policy.prediction_soft_limit", 0, 50,
    ))
    evaluation["prediction_sign_min_abs_ic"] = _bounded_number(
        evaluation.get("prediction_sign_min_abs_ic", 0.003),
        "evaluation_policy.prediction_sign_min_abs_ic", 0, 1,
    )

    review = _require_dict(spec.get("review_policy"), "review_policy")
    review["enabled"] = _require_bool(review.get("enabled"), "review_policy.enabled")
    review["review_on"] = _string_list(review.get("review_on"), "review_policy.review_on")
    invalid_hooks = set(review["review_on"]) - {"validation", "pre_submit"}
    if invalid_hooks:
        raise ValueError("research_spec.review_policy.review_on_invalid")
    review["block_classic_transforms"] = _require_bool(review.get("block_classic_transforms"), "review_policy.block_classic_transforms")
    if review.get("minimum_novelty") not in {"low", "medium", "high"}:
        raise ValueError("research_spec.review_policy.minimum_novelty_invalid")
    if review.get("reviewer_max_tokens") is not None:
        raw_tokens = review["reviewer_max_tokens"]
        try:
            coerced = int(raw_tokens)
        except (TypeError, ValueError) as exc:
            raise ValueError("research_spec.review_policy.reviewer_max_tokens_invalid") from exc
        review["reviewer_max_tokens"] = int(
            _bounded_number(coerced, "review_policy.reviewer_max_tokens", 256, 200_000)
        )

    interaction = _require_dict(spec.get("interaction_policy"), "interaction_policy")
    allowed_types = _string_list(
        interaction.get("allowed_interaction_types"),
        "interaction_policy.allowed_interaction_types",
    )
    invalid_types = set(allowed_types) - set(INTERACTION_TYPES)
    if invalid_types:
        raise ValueError(f"interaction_policy.unknown_types:{','.join(sorted(invalid_types))}")
    interaction["allowed_interaction_types"] = allowed_types
    interaction["block_undeclared_multiply"] = _require_bool(
        interaction.get("block_undeclared_multiply"),
        "interaction_policy.block_undeclared_multiply",
    )
    interaction["require_contract_for_typed_interactions"] = _require_bool(
        interaction.get("require_contract_for_typed_interactions"),
        "interaction_policy.require_contract_for_typed_interactions",
    )
    interaction["require_ablation_for_multiplication"] = _require_bool(
        interaction.get("require_ablation_for_multiplication"),
        "interaction_policy.require_ablation_for_multiplication",
    )

    memory = _require_dict(spec.get("memory_policy"), "memory_policy")
    memory["retrieve_limit"] = int(_bounded_number(memory.get("retrieve_limit"), "memory_policy.retrieve_limit", 0, 100))
    memory["dynamic_retrieve_limit"] = int(_bounded_number(
        memory.get("dynamic_retrieve_limit"), "memory_policy.dynamic_retrieve_limit", 0, 100))
    memory["max_expression_chars"] = int(_bounded_number(
        memory.get("max_expression_chars"), "memory_policy.max_expression_chars", 0, 4000))
    memory["include_rejected_paths"] = _require_bool(memory.get("include_rejected_paths"), "memory_policy.include_rejected_paths")
    memory["prefer_orthogonal_to_approved"] = _require_bool(memory.get("prefer_orthogonal_to_approved"), "memory_policy.prefer_orthogonal_to_approved")
    memory["include_expression"] = _require_bool(memory.get("include_expression"), "memory_policy.include_expression")
    memory["enable_factor_retrieval"] = _require_bool(memory.get("enable_factor_retrieval"), "memory_policy.enable_factor_retrieval")
    memory["enable_edit_patterns"] = _require_bool(memory.get("enable_edit_patterns"), "memory_policy.enable_edit_patterns")
    # 组件级消融开关（2026-09-16，memory component ablation spec §二.4）
    for _key in (
        "enable_experience_block",
        "enable_saturation_block",
        "enable_yield_block",
        "enable_diversity_block",
        "enable_structure_stats_block",
        "enable_sspm_write",
        "enable_distill",
        "enable_advisory_cache",
    ):
        memory[_key] = _require_bool(memory.get(_key), f"memory_policy.{_key}")
    # v3-lite
    memory["hard_block_duplicates"] = _require_bool(memory.get("hard_block_duplicates"), "memory_policy.hard_block_duplicates")
    memory["dead_end_min_attempts"] = int(_bounded_number(memory.get("dead_end_min_attempts"), "memory_policy.dead_end_min_attempts", 1, 20))
    memory["max_inject_chars"] = int(_bounded_number(memory.get("max_inject_chars"), "memory_policy.max_inject_chars", 0, 20000))
    memory["apv_tau_c"] = float(_bounded_number(memory.get("apv_tau_c"), "memory_policy.apv_tau_c", 0, 1))
    memory["apv_tau_v"] = float(_bounded_number(memory.get("apv_tau_v"), "memory_policy.apv_tau_v", 0, 1))
    memory["edit_prior_hard_conf"] = float(_bounded_number(memory.get("edit_prior_hard_conf"), "memory_policy.edit_prior_hard_conf", 0, 1))
    memory["edit_prior_recommend_conf"] = float(_bounded_number(memory.get("edit_prior_recommend_conf"), "memory_policy.edit_prior_recommend_conf", 0, 1))
    memory["edit_prior_veto_conf"] = float(_bounded_number(memory.get("edit_prior_veto_conf"), "memory_policy.edit_prior_veto_conf", 0, 1))
    memory["suggest_slots"] = int(_bounded_number(memory.get("suggest_slots"), "memory_policy.suggest_slots", 0, 10))
    memory["suggest_field_gate"] = _require_bool(
        memory.get("suggest_field_gate", True), "memory_policy.suggest_field_gate"
    )
    # 总开关（消融 A1）：False = 零记忆探索（等价 research_memory_path=None）
    if memory.get("enabled") is not None:
        memory["enabled"] = _require_bool(memory.get("enabled"), "memory_policy.enabled")
    else:
        memory["enabled"] = True

    # ── cognition_policy：认知对账开关（2026-09-14 消融 spec 新增）──
    cog = spec.get("cognition_policy")
    if not isinstance(cog, dict):
        cog = {}
    cog["prediction_check_enabled"] = _require_bool(
        cog.get("prediction_check_enabled"), "cognition_policy.prediction_check_enabled"
    )
    cog["ablation_check_enabled"] = _require_bool(
        cog.get("ablation_check_enabled"), "cognition_policy.ablation_check_enabled"
    )
    spec["cognition_policy"] = cog

    # ── operator_policy：算子黑名单（2026-09-14 消融 spec 新增）──
    op_policy = spec.get("operator_policy")
    if not isinstance(op_policy, dict):
        op_policy = {}
    from alphaagent.dsl.registry import build_operator_namespace

    known_ops = set(build_operator_namespace().keys())
    blacklist = _string_list(op_policy.get("blacklist"), "operator_policy.blacklist")
    unknown_ops = {name.upper() for name in blacklist} - known_ops
    if unknown_ops:
        raise ValueError(f"research_spec.operator_policy.unknown_operators:{','.join(sorted(unknown_ops))}")
    op_policy["blacklist"] = [name.upper() for name in blacklist]
    spec["operator_policy"] = op_policy

    delivery = _require_dict(spec.get("delivery_policy"), "delivery_policy")
    # 盲测终审门槛（2026-08-29 新增）
    blind = delivery.get("blind_test")
    if not isinstance(blind, dict):
        blind = {}
    if blind.get("enabled") is not None:
        blind["enabled"] = _require_bool(blind.get("enabled"), "delivery_policy.blind_test.enabled")
    blind["min_ic_retention"] = _bounded_number(blind.get("min_ic_retention"), "delivery_policy.blind_test.min_ic_retention", 0, 2)
    # 盲测段 IC 绝对下限（2026-09-11 预筛池口径新增）
    blind["min_test_abs_ic"] = _bounded_number(blind.get("min_test_abs_ic"), "delivery_policy.blind_test.min_test_abs_ic", 0, 1)
    if blind.get("require_sign_consistency") is not None:
        blind["require_sign_consistency"] = _require_bool(
            blind.get("require_sign_consistency"), "delivery_policy.blind_test.require_sign_consistency"
        )
    delivery["blind_test"] = blind

    # Screener（regime 感知因子筛选，2026-08-29 新增）
    screener = delivery.get("screener")
    if not isinstance(screener, dict):
        screener = {}
    if screener.get("enabled") is not None:
        screener["enabled"] = _require_bool(screener.get("enabled"), "delivery_policy.screener.enabled")
    screener["lookback"] = int(_bounded_number(screener.get("lookback"), "delivery_policy.screener.lookback", 3, 60))
    screener["min_ic"] = _bounded_number(screener.get("min_ic"), "delivery_policy.screener.min_ic", 0, 1)
    screener["max_corr"] = _bounded_number(screener.get("max_corr"), "delivery_policy.screener.max_corr", 0, 1)
    if screener.get("use_family_boost") is not None:
        screener["use_family_boost"] = _require_bool(
            screener.get("use_family_boost"), "delivery_policy.screener.use_family_boost"
        )
    screener["adx_threshold"] = _bounded_number(
        screener.get("adx_threshold"), "delivery_policy.screener.adx_threshold", 0, 100)
    screener["ma_period"] = int(_bounded_number(
        screener.get("ma_period"), "delivery_policy.screener.ma_period", 10, 500))
    screener["min_cross_section"] = int(_bounded_number(
        screener.get("min_cross_section"), "delivery_policy.screener.min_cross_section", 1, 1000))
    delivery["screener"] = screener

    candidate = _require_dict(delivery.get("candidate"), "delivery_policy.candidate")
    candidate["min_abs_ic"] = _bounded_number(candidate.get("min_abs_ic"), "delivery_policy.candidate.min_abs_ic", 0, 1)
    candidate["min_icir"] = _bounded_number(candidate.get("min_icir"), "delivery_policy.candidate.min_icir", -10, 20)
    candidate["min_coverage"] = _bounded_number(candidate.get("min_coverage"), "delivery_policy.candidate.min_coverage", 0, 1)
    candidate["max_abs_corr"] = _bounded_number(candidate.get("max_abs_corr"), "delivery_policy.candidate.max_abs_corr", 0, 1)
    # 换手可行性与样本外保留比（0.18 / 0.5 由 DEFAULT_RESEARCH_SPEC 提供默认值）
    candidate["min_cs_autocorr"] = _bounded_number(candidate.get("min_cs_autocorr"), "delivery_policy.candidate.min_cs_autocorr", 0, 1)
    candidate["min_val_ic_retention"] = _bounded_number(candidate.get("min_val_ic_retention"), "delivery_policy.candidate.min_val_ic_retention", 0, 1)
    # val 端 IC 绝对下限（2026-09-11 预筛池口径新增：保留比只卡相对衰减）
    candidate["min_val_abs_ic"] = _bounded_number(candidate.get("min_val_abs_ic"), "delivery_policy.candidate.min_val_abs_ic", 0, 1)
    production = _require_dict(delivery.get("production"), "delivery_policy.production")
    # 双窗口统计门槛（2026-08 重构：混合窗口稀释 val 衰减，已弃用单口径 min_abs_ic/min_icir）
    for key in ("min_train_abs_ic", "min_val_abs_ic", "max_winsorized_abs_ic_decay", "max_abs_corr"):
        production[key] = _bounded_number(production.get(key), f"delivery_policy.production.{key}", 0, 1)
    production["min_train_icir"] = _bounded_number(production.get("min_train_icir"), "delivery_policy.production.min_train_icir", -10, 20)
    production["min_val_ic_retention"] = _bounded_number(production.get("min_val_ic_retention"), "delivery_policy.production.min_val_ic_retention", 0, 2)
    if production.get("min_val_long_excess") is not None:
        production["min_val_long_excess"] = _bounded_number(
            production.get("min_val_long_excess"), "delivery_policy.production.min_val_long_excess", -1, 1
        )
    # 兼容旧 spec 的遗留键：存在则归一化（新默认值不再生成它们）
    for key, lo, hi in (
        ("min_abs_ic", 0, 1), ("min_icir", -10, 20),
        ("min_fmb_t_stat", 0, 20), ("min_ls_t_stat", 0, 20),
        ("min_quantile_excess_return", -1, 1), ("min_quantile_sharpe", -10, 20),
        ("min_monotonicity", -1, 1), ("min_long_group_annual_excess_return", -1, 1),
    ):
        if production.get(key) is not None:
            production[key] = _bounded_number(production.get(key), f"delivery_policy.production.{key}", lo, hi)
    eg = _require_dict(production.get("engine_gate"), "delivery_policy.production.engine_gate")
    production["engine_gate"] = eg
    if eg.get("enabled") is not None:
        eg["enabled"] = _require_bool(eg.get("enabled"), "engine_gate.enabled")
    eg["selection_mode"] = str(eg.get("selection_mode", "top_pct"))
    if eg["selection_mode"] not in {"top_pct", "top_n"}:
        raise ValueError("research_spec.engine_gate.selection_mode_invalid")
    eg["selection_pct"] = _bounded_number(eg.get("selection_pct"), "engine_gate.selection_pct", 0, 0.1)
    if eg.get("top_n") is not None:
        eg["top_n"] = int(_bounded_number(eg.get("top_n"), "engine_gate.top_n", 1, 500))
    if eg.get("slippage_bps") is not None:
        eg["slippage_bps"] = _bounded_number(eg.get("slippage_bps"), "engine_gate.slippage_bps", 0, 1000)
    if eg.get("max_participation") is not None:
        eg["max_participation"] = _bounded_number(eg.get("max_participation"), "engine_gate.max_participation", 0.001, 1)
    eg["allowed_freqs"] = _string_list(eg.get("allowed_freqs"), "engine_gate.allowed_freqs")
    invalid_freqs = set(eg["allowed_freqs"]) - {"daily", "weekly", "monthly"}
    if invalid_freqs:
        raise ValueError("research_spec.engine_gate.allowed_freqs_invalid")
    if eg.get("freq") is not None:
        eg["freq"] = str(eg["freq"]).lower()
        if eg["freq"] not in eg["allowed_freqs"]:
            raise ValueError("research_spec.engine_gate.freq_not_in_allowed")
    for key in ("min_excess_annual",):
        eg[key] = _bounded_number(eg.get(key), f"engine_gate.{key}", -1, 5)
    for key in ("min_excess_sharpe", "max_drawdown", "min_daily_overlap"):
        if eg.get(key) is not None:
            eg[key] = _bounded_number(eg.get(key), f"engine_gate.{key}", 0, 10)
    if eg.get("min_invested_ratio") is not None:
        eg["min_invested_ratio"] = _bounded_number(eg.get("min_invested_ratio"), "engine_gate.min_invested_ratio", 0, 1)
    if eg.get("capital") is not None:
        eg["capital"] = _bounded_number(eg.get("capital"), "engine_gate.capital", 10_000, 1_000_000_000)
    if eg.get("min_am20_yuan") is not None:
        eg["min_am20_yuan"] = _bounded_number(eg.get("min_am20_yuan"), "engine_gate.min_am20_yuan", 0, 1_000_000_000_000)

    # ── label 与调仓频率强制一致（2026-10-04 用户定调，fail-closed）──────────────
    # 规则：engine_gate.freq 的持有期 == label 的持有期（daily↔1d、weekly↔5d、monthly↔20d）。
    # 模式**未显式声明** freq 时，由 label 派生（这样裸 spec 也自洽）；显式声明但与 label
    # 矛盾则报错并给出可执行修法（改 label 或改 freq）。
    _lfc = spec.get("label_freq_consistency")
    if not isinstance(_lfc, dict):
        _lfc = {}
    _lfc["enforce"] = _require_bool(_lfc.get("enforce", True), "label_freq_consistency.enforce")
    _raw_map = _lfc.get("freq_hold_days")
    if not isinstance(_raw_map, dict) or not _raw_map:
        _raw_map = {"daily": 1, "weekly": 5, "monthly": 20}
    _hold_map: dict[str, int] = {}
    for _freq_key, _days in _raw_map.items():
        _hold_map[str(_freq_key).lower()] = int(
            _bounded_number(_days, f"label_freq_consistency.freq_hold_days.{_freq_key}", 1, 120)
        )
    _lfc["freq_hold_days"] = _hold_map
    spec["label_freq_consistency"] = _lfc

    if _lfc["enforce"]:
        _label_hold = label_hold_days(spec.get("recommended_label_col"))
        _freq_to_label = {d: f for f, d in _hold_map.items()}
        _derived = _freq_to_label.get(_label_hold, "daily")
        _declared = str(eg.get("freq") or "").lower()
        if _declared != _derived:
            # 2026-10-04 用户定调：label 与调仓频率**强制一致**。归一化阶段以 label 为准把
            # freq 收到对应档位（daily↔1d、weekly↔5d、monthly↔20d），并把白名单收成单值，
            # 防止 LLM 用别的频率提交；用户显式覆盖 freq 的路径由 API 入口
            # `ensure_label_freq_consistency` 硬校验（覆盖发生在归一化之后）。
            eg["freq"] = _derived
            eg["allowed_freqs"] = [_derived]
    # 最终一致性校验（fail-closed；与 API 入口共用同一函数，避免口径漂移）
    ensure_label_freq_consistency(spec, label_col=spec.get("recommended_label_col"))

    # ── prompt_policy：分阶段注入策略（2026-09-12 新增）──
    pp = spec.get("prompt_policy")
    if not isinstance(pp, dict):
        pp = {}
    pp["phase_mode"] = str(pp.get("phase_mode", "auto"))
    if pp["phase_mode"] not in ("full", "auto"):
        raise ValueError("research_spec.prompt_policy.phase_mode_invalid")
    # phase_ratio：三段比例，和≈1.0，每段 >0
    raw_ratio = pp.get("phase_ratio", [1 / 3, 1 / 3, 1 / 3])
    if not isinstance(raw_ratio, list) or len(raw_ratio) != 3:
        raise ValueError("research_spec.prompt_policy.phase_ratio_must_be_3_elements")
    ratio = [float(_bounded_number(r, "prompt_policy.phase_ratio", 0, 1)) for r in raw_ratio]
    if abs(sum(ratio) - 1.0) > 0.02:
        raise ValueError("research_spec.prompt_policy.phase_ratio_must_sum_to_1")
    if any(r <= 0 for r in ratio):
        raise ValueError("research_spec.prompt_policy.phase_ratio_must_be_positive")
    pp["phase_ratio"] = ratio
    spec["prompt_policy"] = pp

    # ── report_policy：研报先验机制策略（R4 课题队列 + R3 原始文献 RAG 融合方案） ──
    rp = spec.get("report_policy")
    if not isinstance(rp, dict):
        rp = {}
    rp["enable_report_mechanisms"] = _require_bool(rp.get("enable_report_mechanisms", False), "report_policy.enable_report_mechanisms")
    rp["enable_report_prior"] = _require_bool(rp.get("enable_report_prior", False), "report_policy.enable_report_prior")
    rp["enable_report_rag"] = _require_bool(rp.get("enable_report_rag", True), "report_policy.enable_report_rag")
    rp["report_rag_max_chars"] = int(_bounded_number(rp.get("report_rag_max_chars", 1600), "report_policy.report_rag_max_chars", 200, 10000))
    rp["report_rag_top_k"] = int(_bounded_number(rp.get("report_rag_top_k", 3), "report_policy.report_rag_top_k", 1, 20))
    rp["enable_question_queue"] = _require_bool(rp.get("enable_question_queue", True), "report_policy.enable_question_queue")
    # 课题字段门控（2026-09-28）：开关 + 阈值全部收口在此，question_queue 只读取不硬编码
    rp["question_field_gate"] = _require_bool(rp.get("question_field_gate", True), "report_policy.question_field_gate")
    rp["question_field_warn"] = _require_bool(rp.get("question_field_warn", True), "report_policy.question_field_warn")
    rp["question_field_gate_min_ratio"] = float(_bounded_number(
        rp.get("question_field_gate_min_ratio", 0.5), "report_policy.question_field_gate_min_ratio", 0.0, 1.0
    ))
    rp["question_field_gate_scan_limit"] = int(_bounded_number(
        rp.get("question_field_gate_scan_limit", 40), "report_policy.question_field_gate_scan_limit", 1, 500
    ))
    # 复现保真度「原文锚」（2026-10-01，**仅研报模式**）：
    #   ① require_factor_records：只派发**有原文公式**的课题（用户决策：先用有公式的研报）
    #   ② reproduce_fidelity：复现过线前做字段/算子重叠的机械校验；锚定必须落在
    #      「报告特有字段」上（通用行情字段不算），防止脱稿自由探索顶替原文机制。
    rp["require_factor_records"] = _require_bool(
        rp.get("require_factor_records", True), "report_policy.require_factor_records"
    )
    rp["reproduce_of_required"] = _require_bool(
        rp.get("reproduce_of_required", True), "report_policy.reproduce_of_required"
    )
    rp["diverge_parent_required"] = _require_bool(
        rp.get("diverge_parent_required", True), "report_policy.diverge_parent_required"
    )
    rp["parent_autofill"] = _require_bool(
        rp.get("parent_autofill", True), "report_policy.parent_autofill"
    )
    rp["require_report_structure"] = _require_bool(
        rp.get("require_report_structure", True), "report_policy.require_report_structure"
    )
    rp["exclude_method_types"] = _string_list(
        rp.get("exclude_method_types", ["ml_model", "graph_deep"]),
        "report_policy.exclude_method_types",
    )
    rp["exclude_perf_labels"] = _string_list(
        rp.get("exclude_perf_labels", ["deprecated", "suspect"]),
        "report_policy.exclude_perf_labels",
    )
    rp["reproduce_lock_rounds"] = int(_bounded_number(
        rp.get("reproduce_lock_rounds", 3), "report_policy.reproduce_lock_rounds", 0, 12
    ))
    rp["anchor_block_skip_threshold"] = int(_bounded_number(
        rp.get("anchor_block_skip_threshold", 8),
        "report_policy.anchor_block_skip_threshold", 0, 200,
    ))
    _fid = rp.get("reproduce_fidelity")
    if not isinstance(_fid, dict):
        _fid = {}
    _fid["enabled"] = _require_bool(
        _fid.get("enabled", True), "report_policy.reproduce_fidelity.enabled"
    )
    _fid["min_ops_floor"] = int(_bounded_number(
        _fid.get("min_ops_floor", 2), "report_policy.reproduce_fidelity.min_ops_floor", 0, 20,
    ))
    _fid["min_fields_floor"] = int(_bounded_number(
        _fid.get("min_fields_floor", 2), "report_policy.reproduce_fidelity.min_fields_floor", 0, 20,
    ))
    _fid["require_shared_field"] = int(_bounded_number(
        _fid.get("require_shared_field", 1),
        "report_policy.reproduce_fidelity.require_shared_field", 0, 20,
    ))
    _fid["min_field_jaccard"] = float(_bounded_number(
        _fid.get("min_field_jaccard", 0.0),
        "report_policy.reproduce_fidelity.min_field_jaccard", 0.0, 1.0,
    ))
    _fid["min_op_jaccard"] = float(_bounded_number(
        _fid.get("min_op_jaccard", 0.0),
        "report_policy.reproduce_fidelity.min_op_jaccard", 0.0, 1.0,
    ))
    _fid["min_shared_ops"] = int(_bounded_number(
        _fid.get("min_shared_ops", 0),
        "report_policy.reproduce_fidelity.min_shared_ops", 0, 20,
    ))
    rp["reproduce_fidelity"] = _fid
    # 复现门槛（研报模式）：底线 + 指标通道。判定式见 DEFAULT_RESEARCH_SPEC 里同名字段的注释。
    rp["reproduce_min_abs_ic"] = float(_bounded_number(
        rp.get("reproduce_min_abs_ic", 0.010),
        "report_policy.reproduce_min_abs_ic", 0.0, 0.5,
    ))
    rp["reproduce_min_icir"] = float(_bounded_number(
        rp.get("reproduce_min_icir", 0.10),
        "report_policy.reproduce_min_icir", 0.0, 10.0,
    ))
    _shape_req = str(rp.get("reproduce_shape_requirement", "strong_side") or "strong_side")
    if _shape_req not in ("strong_side", "confirmed", "off"):
        raise ValueError(
            "report_policy.reproduce_shape_requirement 必须是 strong_side|confirmed|off，"
            f"收到 {_shape_req!r}"
        )
    rp["reproduce_shape_requirement"] = _shape_req
    spec["report_policy"] = rp

    profiles = resolve_profiles(spec)
    spec["evaluation_profiles"] = {profile_id: profile.as_dict() for profile_id, profile in profiles.items()}

    # 注入换手率约束：低自相关(高换手)因子不入候选池
    ac_min = float(evaluation.get("min_cs_autocorr", 0))
    if ac_min > 0:
        ac_rule = {"metric": "cross_sectional_core.cs_pearson_autocorr", "op": "gte", "value": ac_min}
        for pid in ("train_screen",):
            prof = spec["evaluation_profiles"].get(pid)
            if prof is None:
                continue
            rules = prof.setdefault("rules", [])
            if not any(r.get("metric") == ac_rule["metric"] for r in rules):
                rules.append(dict(ac_rule))

    return spec


def load_research_spec(path: Path | None) -> dict[str, Any]:
    """从文件加载 ResearchSpec（None 时取默认+保存覆盖）。

    显式文件内容作"显式 spec"与注册表默认/保存覆盖合并（build_run_research_spec），
    保证 CLI 直跑也不会绕过用户在前端保存的门槛修改。
    """
    if path is None:
        return build_run_research_spec()
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"research_spec_load_failed: {exc}") from exc
    return build_run_research_spec(value)


def research_policy_prompt(spec: dict[str, Any]) -> str:
    """A concise policy contract consumed by the primary mining agent."""
    search = spec["search_policy"]
    evaluation = spec["evaluation_policy"]
    review = spec["review_policy"]
    interaction = spec.get("interaction_policy", {})
    profiles = spec.get("evaluation_profiles", {})
    mode = str(spec.get("research_mode") or "technical")
    mode_spec = get_research_mode(mode)
    needs_funda = mode_spec.needs_fundamentals
    mode_desc = (
        f"研究模式：{mode_spec.label}。信号以 `$funda_*` 财务字段为主"
        "（PIT 日频阶跃数据，注意披露生效日与低换手），"
        f"本次评估 label 为 {spec.get('recommended_label_col', mode_spec.recommended_label_col)}。"
        if needs_funda
        else f"研究模式：{mode_spec.label}。信号以价量/波动/筹码等行情字段为主。"
    )
    phase_summary = prompt_phase_summary(spec)
    return "\n".join(
        [
            "# 本次运行的 ResearchSpec（高优先级研究约束）",
            mode_desc,
            f"允许的信号族：{', '.join(search['allowed_signal_families']) or '不限制'}。",
            f"禁止作为独立候选的信号族：{', '.join(search['forbidden_signal_families']) or '不限制'}。",
            f"每个保留候选至少使用 {search['min_distinct_raw_fields']} 个彼此独立的原始字段。",
            "候选必须包含时序结构。" if search["require_time_series_structure"] else "允许纯截面候选，但必须说明其独立经济机制。",
            f"每轮最多提出 {search['max_candidates_per_round']} 个候选。",
            f"本次允许的 EvaluationProfile：{', '.join(sorted(profiles))}。只能选择已有 profile_id，不得自定义临时参数。",
            "进入验证的最低 train 要求："
            f"abs(IC)>={evaluation['min_train_abs_ic']:.4g}，abs(ICIR)>={evaluation['min_train_icir']:.4g}，"
            f"Coverage>={evaluation['min_train_coverage']:.4g}。",
            "验证要求："
            f"abs(IC)>={evaluation['min_val_abs_ic']:.4g}，val/train abs(IC) 保留比例>="
            f"{evaluation['min_val_ic_retention_ratio']:.4g}，"
            + ("方向必须一致。" if evaluation["require_sign_consistency"] else "方向一致性仅作诊断。"),
            "换手率约束："
            f"cs_pearson_autocorr>={evaluation.get('min_cs_autocorr', 0):.4g}——低于此值的因子排名日度变化过快、"
            "实际交易成本会吃掉全部 alpha，将被自动拦截不入候选池。",
            (
                "Reviewer 策略："
                f"在 {', '.join(review['review_on'])} 阶段审查；最低新颖性={review['minimum_novelty']}；"
                + ("经典单调变换必须阻断。" if review["block_classic_transforms"] else "经典变换只提示，不自动阻断。")
                if review.get("enabled", True)
                else "Reviewer：**已关闭**（不审查、不阻断）。交付由统计门槛与正交门裁决；"
                     "结构性新颖/经典变换不作为否决理由。"
            ),
            "交互策略："
            f"允许 interaction_type={', '.join(interaction.get('allowed_interaction_types', []))}；"
            + ("未声明契约的 MULTIPLY 直接拦截。" if interaction.get("block_undeclared_multiply") else "MULTIPLY 仅提示。")
            + "所有结构化多因子交互必须传完整 interaction 契约"
            + ("（未传时自动补全占位并警告——机制描述请显式写）。" if interaction.get("auto_fill_missing_contract", True) else "，缺契约直接拦截。"),
            "交付策略："
            + (
                "通过 validation 的因子自动进入 candidate 候选池；Reviewer approve 后进入 production 正式库。"
                if review.get("enabled", True)
                else "过统计门槛的因子自动进入 candidate 候选池，再按 stage_two/engine_gate 统计门槛裁决是否进 production 正式库（无 Reviewer 环节）。"
            ),
        ]
        + ([cognition_policy_summary(spec)] if cognition_policy_summary(spec) else [])
        + ([operator_policy_summary(spec)] if operator_policy_summary(spec) else [])
        + ([phase_summary] if phase_summary else [])
    )


def cognition_policy_summary(spec: dict[str, Any]) -> str:
    """认知对账开关的可读摘要（消融 C1/C2 注入 research_policy_prompt）。

    双开默认时不注入（保持旧 prompt 不变）；任一关闭时告知 LLM
    本次运行不做对应对账，避免 LLM 按教学文本白白准备 prediction。
    """
    cog = spec.get("cognition_policy") or {}
    parts = []
    if cog.get("prediction_check_enabled") is False:
        parts.append("本次运行关闭预测对账：评估无需传 prediction，结果也不会注入 prediction_check")
    if cog.get("ablation_check_enabled") is False:
        parts.append("本次运行关闭门控自动消融：结果不会注入 ablation_check/ablation_hint")
    return "；".join(parts) + "。" if parts else ""


def operator_policy_summary(spec: dict[str, Any]) -> str:
    """算子黑名单的可读摘要（消融 D2 注入 research_policy_prompt）。

    空名单（默认）不注入，保持旧行为。
    """
    blacklist = ((spec.get("operator_policy") or {}).get("blacklist")) or []
    if not blacklist:
        return ""
    return (
        "算子约束：本次运行禁用以下算子，评估/提交前会被直接拦截："
        + ", ".join(blacklist)
        + "。请改用基础时序/截面算子表达同一机制。"
    )


def prompt_phase_summary(spec: dict[str, Any]) -> str:
    """提示词分阶段策略的可读摘要，注入 research_policy_prompt 末尾。

    phase_mode=full 时不注入（旧行为）；auto 时告知 LLM 当前阶段裁剪了哪些内容。
    """
    pp = spec.get("prompt_policy") or {}
    mode = str(pp.get("phase_mode", "auto"))
    if mode != "auto":
        return ""
    return (
        "提示词分阶段注入：探索阶段(explore)精简算子目录与部分约束模块，"
        "深耕阶段(deepen)恢复完整约束，交付阶段(deliver)全量含交付模块。"
    )
