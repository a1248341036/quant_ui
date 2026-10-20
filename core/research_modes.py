"""研究模式注册表：单一事实源，前后端/全链路共享。

**目标**：加一个研究模式（technical / fundamental / strategy / sentiment...）
只改本文件一个 dict，回测 spec、因子库目录、基本面加载、前端按钮/下拉/
提示词全部自动跟随——不再散落硬编码（否则"前端多一个风格就多个键"）。

每个消费方从这里取数：
- research_spec.default_research_spec      → 门槛/信号族/label 覆盖
- core.factor_categories                   → 候选/正式库目录
- backend.alphaagent_service               → 是否载入 funda_* 列
- /api/alphaagent/research-modes           → 前端档位选择（页头「研报」按钮 / 因子实验室
  重测下拉）。**注意：不是自动渲染**——2026-09-02「模式下拉退役」后，AgentThread 的档位
  改为按数据面自动推断（infer_research_mode），新增顶层档位必须在 store/AgentThread 里
  显式接线才会出现在页面上（report 档 2026-09-29 只进了注册表，UI 入口 09-30 才补上）。

未来若加"工具型"模式（如组合扫描），只需扩展本 dataclass 字段 + 前端一处接线。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResearchModeSpec:
    """一个研究模式的完整声明（含 UI 元数据与研究策略参数）。"""

    mode_id: str                      # 稳定标识：research_mode / category 共用
    label: str                        # 前端显示名（研究模式按钮/因子库类别标签）
    hint: str                         # 前端 hover 提示
    recommended_label_col: str        # 评估 label（研究口径；慢因子模式要求长持有期）
                                      # 与 engine_gate_overrides["freq"]（交付口径）刻意解耦，
                                      # 见本文件顶部"label 口径 vs 调仓频率"区块
    signal_families: tuple[str, ...]  # 允许的信号族
    forbidden_families: tuple[str, ...]  # 禁止作为独立候选的信号族
    needs_fundamentals: bool          # 是否必须载入 funda_* 财务列
    candidate_dir: str                # 候选因子库目录名（factorzoo 下）
    production_dir: str               # 正式因子库目录名（factorzoo 下）
    default_user_message: str         # 前端"开始研究"的默认提示词
    # 门槛覆盖：相对 DEFAULT_RESEARCH_SPEC 的片段（见 research_spec.py）
    evaluation_overrides: dict = field(default_factory=dict)      # evaluation_policy
    candidate_overrides: dict = field(default_factory=dict)       # delivery_policy.candidate
    production_overrides: dict = field(default_factory=dict)      # delivery_policy.production
    engine_gate_overrides: dict = field(default_factory=dict)     # delivery_policy.production.engine_gate
    blind_test_overrides: dict = field(default_factory=dict)      # delivery_policy.blind_test
    # 研报模式专用：report_policy 覆盖（流程开关，如"复现阶段用机制卡、发散阶段用 RAG"）。
    # 非研报模式留空 → 行为与今天完全一致。
    report_policy_overrides: dict = field(default_factory=dict)
    # 冷门赛道二轨评估覆盖（research_spec.cold_track）：默认空 → 沿用 DEFAULT_RESEARCH_SPEC
    # 里 ``cold_track.enabled=False``，即所有现有档位行为逐字节不变。
    # 只有显式声明 enabled=True 的档位（如 technical_event_monthly）才走残差化赛道。
    cold_track_overrides: dict = field(default_factory=dict)


# ── label 口径 vs 调仓频率：**强制一致**（2026-10-04 用户定调；推翻 2026-09-30 的"刻意解耦"）──
# 规则（用户原话）：「调仓频率是多少，label 就应该多少」——
#   `recommended_label_col` 的持有期必须等于 `engine_gate_overrides["freq"]` 的持有期
#   （daily↔1d、weekly↔5d、monthly↔20d），由 `research_spec.label_freq_consistency`
#   **fail-closed 校验**，不一致直接拒绝启动（模式未显式声明 freq 时由 label 派生）。
#
# 为什么推翻解耦（2026-10-04 实测）：旧组合（1d label + weekly 交付）里
#   · 研究口径按 1d 打分、交付口径按周频回测，看似各管一段；
#   · 但 stage_one 换手硬门拿"日频信号抖动"去撞 weekly 档阈值（0.65），而 hold=label 天数=1
#     → 组合其实**每天在换**（实测 `n_rebalances=1615`、`avg_rebalance_side_turnover ==
#     avg_daily_side_turnover`）→ "周频摊薄成本"的折算前提根本不成立：既不是日频事实，
#     也不是周频事实，1.48 这类超门值无法自洽解释。
#   · 代价（接受）：1d 快信号不再有"周频折让"，换手门按该档阈值（daily 0.50）执行，
#     要过门只能把信号做慢（滚动均值/EMA/慢信息源），这正是 prompt 换手红线要求的动作。
#
# ⚠ 仍然成立的历史警告：**不要单独把 label 抬到 5d/20d 去凑 freq**。min_abs_ic /
#   min_train_icir / reproduce_min_abs_ic 等门槛是按 1d 尺度标定的，换 label 会一并失真，
#   必须同批重新标定门槛（属独立改动）。因此本仓当前做法是：**保持 1d 研究口径，把 freq
#   收到 daily**；将来若要 weekly/monthly 交付，走 label_5d/label_20d 的档位（technical_weekly /
#   technical_monthly / fundamental）并重标定门槛。
#
# 旧设计（已废弃，留档避免误读）：label=研究尺子（IC/ICIR 按 label 持有期算，cost_bps=0）、
#   freq=交付尺子（engine_gate 用真实行情按该频率回测、不读 label），两把尺子的落差由
#   `min_cs_autocorr` 与 `turnover_thresholds_by_freq` 两个代理门缝住。

# ── 门槛只由 label 持有期唯一决定（2026-10-08 定调）─────────────────────
# 原则：IC/ICIR/val 门槛与"数据面"（价量 vs 基本面）**彻底无关**，只由预测窗口
# （label 持有期 1d/5d/20d）决定。同一 label 的所有模式必须共用同一套门槛。
#   · label_1d   → technical / technical_daily / fundamental_daily / report 共用 0.020/0.28/0.015
#   · label_5d   → technical_weekly / fundamental_weekly 共用 _WEEKLY_5D_OVERRIDES（0.030/0.45/0.0225）
#   · label_20d  → technical_monthly 与 fundamental 共用 _MONTHLY_20D_OVERRIDES
# **数据面正交（2026-10-08 定调）**：fundamental 是**数据面**（只决定 signal_families 与
# needs_fundamentals，不是独立"慢信号模式"），与调仓频率彻底正交——基本面数据面同样可以
# 选 daily/weekly/monthly 任一频率，门槛随频率取对应 label 档：fundamental_daily=1d 档、
# fundamental_weekly=5d 档、fundamental（monthly）=20d 档。
# 历史教训：fundamental 曾单独标定 0.035/0.45/0.021（"文献月度基本面量级"），
#   造成同为 label_20d 却门槛不同的矛盾（技术月频 0.053/0.65，基本面松 0.035/0.45）。
#   故此处抽成单一常量，两个 20d 模式引用同一来源，从结构上杜绝"同口径不同门槛"漂移。
# 20d 门槛本身沿用 technical_monthly 的双锚标定（2026-10-07 v2：文献锚① + 选择性对齐锚②，
#   该池过线率 36.6% 与主档持平，见下方 technical_monthly 注释）。
_MONTHLY_20D_OVERRIDES: dict[str, dict] = {
    "evaluation_overrides": {
        "min_train_abs_ic": 0.053,
        "min_train_icir": 0.650,
        "min_val_abs_ic": 0.0398,
        "min_val_ic_retention_ratio": 0.5,
    },
    "candidate_overrides": {
        "min_abs_ic": 0.053,
        "min_icir": 0.650,
        "min_val_abs_ic": 0.0398,
        "min_val_ic_retention": 0.5,
    },
    "production_overrides": {
        "min_train_abs_ic": 0.0663,   # 0.053 × 1.25（主档"正式库更严"倍数）
        "min_train_icir": 0.6964,     # 0.650 × 0.30/0.28
        "min_val_abs_ic": 0.0398,
        "min_val_ic_retention": 0.5,
        "min_val_long_excess": 0.0,
        "max_winsorized_abs_ic_decay": 0.10,  # 与 technical 同值（消除基本面档独有放宽）
    },
    "engine_gate_overrides": {
        "freq": "monthly",
        "min_excess_annual": 0.03,    # 与 technical 同值（消除基本面档独有放宽 0.02）
        "min_excess_sharpe": 0.5,     # 与 technical 同值（消除基本面档独有放宽 0.4）
        "allowed_freqs": ["monthly"],
    },
    "blind_test_overrides": {
        # 2026-10-09 用户定调：盲测绝对门与**本档 val 段绝对门一致**（盲测=唯一诚实样本外，
        # 与 val 同一把尺子；旧 0.010 论证依赖 ~410 独立日样本 t≈2.5，实测 IC 自相关 0.998+
        # 新息样本远少，且 0.010 仅为该档候选线 0.053 的 1/5 量纲失配）。
        "min_test_abs_ic": 0.0398,    # = 20d val 绝对门（min_val_abs_ic）
    },
}

# 基本面数据面（``facets.FACET_GROUPS`` 的"基本面组"）覆盖的信号族：
# 基本面四族 + 机构面 + 股东集中面。
# 2026-10-19 补齐后两族：``infer_research_mode`` 的 ``_SLOW_FACETS`` 分支把
# 「股东面/机构面/股东集中面」都路由到 fundamental 档，但该档 signal_families 里
# 只有 fundamental_* 四族 → 勾了机构面/股东集中面却**没有对应族可挂**，
# 因子层面表现为这两个数据面在库内 0 个因子。FACET_GROUPS 本就把它们归为基本面组，
# 补进本档是让"族"与"数据面分组"对齐，不是新增数据面。
_FUNDAMENTAL_DATA_FACE_FAMILIES: tuple[str, ...] = (
    "fundamental_quality", "fundamental_growth",
    "fundamental_value", "fundamental_revision",
    "institutional", "shareholder_concentration",
)

# 5d（label_5d）门槛唯一来源（供 technical_weekly 与 fundamental_weekly 共用）。
# 抽成常量的理由同 _MONTHLY_20D_OVERRIDES：数据面正交后，每个 label 档都会被两个
# 数据面（价量/基本面）的模式引用，从结构上杜绝"同 label 不同门槛"漂移。
_WEEKLY_5D_OVERRIDES: dict[str, dict] = {
    "evaluation_overrides": {
        "min_train_abs_ic": 0.030,    # 文献"有意义线"（锚①）且选择性对齐（锚②）
        "min_train_icir": 0.450,      # 选择性对齐；文献有效线 0.3 之上
        "min_val_abs_ic": 0.0225,     # 0.015 × (0.030/0.02)
    },
    "candidate_overrides": {
        "min_abs_ic": 0.030,
        "min_icir": 0.450,
        "min_val_abs_ic": 0.0225,
    },
    "production_overrides": {
        "min_train_abs_ic": 0.0375,   # 0.030 × 1.25（主档正式库更严倍数）
        "min_train_icir": 0.4821,     # 0.450 × 0.30/0.28
        "min_val_abs_ic": 0.0225,
    },
    "engine_gate_overrides": {
        "freq": "weekly",
        "allowed_freqs": ["weekly"],
    },
    "blind_test_overrides": {
        # 2026-10-09 用户定调：盲测绝对门与本档 val 段绝对门一致（= 5d val 线）。
        "min_test_abs_ic": 0.0225,
    },
}

# ── 注册表（唯一事实源）───────────────────────────────────────────────
RESEARCH_MODES: dict[str, ResearchModeSpec] = {
    "technical": ResearchModeSpec(
        mode_id="technical",
        label="日线技术",
        hint="价量/波动/筹码等日线技术因子，label_1d 开盘到开盘",
        recommended_label_col="label_1d_open_to_open",
        signal_families=("volume_price", "volatility", "chip", "momentum_reversal"),
        forbidden_families=("pure_size",),
        needs_fundamentals=False,
        # 2026-09-03 统一大库：候选/正式库不再按模式分目录，两模式共享
        # candidate_main/production_main（因子带 facets 标签，跨模式正交查重生效）。
        candidate_dir="candidate_main",
        production_dir="production_main",
        default_user_message=(
            "请自主挖掘A股日频价量因子，先训练集评估，再验证集检验；"
            "只有通过验证和去重门槛的因子才提交。"
        ),
        # 2026-10-04 用户定调：label 与调仓频率**强制一致**（见本文件顶部区块）。
        # 本档 label=1d ⇒ freq=daily（由 research_spec 的 label_freq_consistency 复查，不一致直接拒启动）。
        engine_gate_overrides={
            "freq": "daily",
            "allowed_freqs": ["daily"],
        },
    ),
    "fundamental": ResearchModeSpec(
        mode_id="fundamental",
        label="基本面",
        hint="funda_* 财务基本面因子（PIT），label_20d 收盘到收盘（基本面数据面·月频档）",
        recommended_label_col="label_20d_close_to_close",
        signal_families=_FUNDAMENTAL_DATA_FACE_FAMILIES,
        forbidden_families=("pure_size", "pure_price_momentum"),
        needs_fundamentals=True,
        candidate_dir="candidate_main",
        production_dir="production_main",
        default_user_message=(
            "请基于已载入的基本面字段（盈利质量、杠杆、现金流、财报科目等，"
            "PIT 日频）结合价量信息挖掘A股日频因子，先训练集评估，再验证集检验；"
            "只有通过验证和去重门槛的因子才提交。"
        ),
        # ═══ 2026-10-08 数据面正交：fundamental 是**数据面**，不是独立"慢信号模式" ═══
        # 数据面（基本面 vs 价量）只决定 signal_families 与 needs_fundamentals；调仓频率
        # 由用户单独选择（infer_research_mode 按 rebalance_freq 落到对应子档）。本档 =
        # 基本面数据面 × **monthly**（label_20d），与 technical_monthly 共用同一套 20d
        # 门槛（_MONTHLY_20D_OVERRIDES）；若用户选 weekly/daily 频率则走
        # fundamental_weekly / fundamental_daily 子档（门槛随频率取 5d/1d 档）。
        # 取代此前"基本面 = 慢信号锁 20d"的耦合（2026-09-02 infer_research_mode 把
        # 基本面/股东面直接钉死到本档）；也取代 2026-10-08 之前的独立标定
        # 0.035/0.45/0.021（"文献月度基本面量级"）——同 label 必须同门槛，见文件顶部区块。
        # deepcopy 展开：fundamental 与 technical_monthly 各自持有独立子 dict，防止
        # 共享同一可变对象（原地改某档会静默污染另一档与"唯一事实源"常量）。
        **copy.deepcopy(_MONTHLY_20D_OVERRIDES),
    ),
    "fundamental_daily": ResearchModeSpec(
        mode_id="fundamental_daily",
        label="基本面·日频",
        hint="基本面数据面（funda_* PIT）× daily 调仓：label_1d，门槛 1d 档",
        recommended_label_col="label_1d_open_to_open",
        signal_families=_FUNDAMENTAL_DATA_FACE_FAMILIES,
        forbidden_families=("pure_size", "pure_price_momentum"),
        needs_fundamentals=True,
        candidate_dir="candidate_main",
        production_dir="production_main",
        default_user_message=(
            "请基于已载入的基本面字段（盈利质量、杠杆、现金流、财报科目等，"
            "PIT 日频）结合价量信息挖掘A股日频因子，daily 调仓路径，"
            "先训练集评估，再验证集检验；只有通过验证和去重门槛的因子才提交。"
        ),
        # 数据面正交子档：基本面数据面 × daily。门槛 = 1d 档（DEFAULT_RESEARCH_SPEC
        # 默认值，与 technical/technical_daily/report 同源），engine_gate 收成 daily。
        engine_gate_overrides={
            "freq": "daily",
            "allowed_freqs": ["daily"],
        },
    ),
    "fundamental_weekly": ResearchModeSpec(
        mode_id="fundamental_weekly",
        label="基本面·周频",
        hint="基本面数据面（funda_* PIT）× weekly 调仓：label_5d，门槛 5d 档",
        recommended_label_col="label_5d_close_to_close",
        signal_families=_FUNDAMENTAL_DATA_FACE_FAMILIES,
        forbidden_families=("pure_size", "pure_price_momentum"),
        needs_fundamentals=True,
        candidate_dir="candidate_main",
        production_dir="production_main",
        default_user_message=(
            "请基于已载入的基本面字段（盈利质量、杠杆、现金流、财报科目等，"
            "PIT 日频）结合价量信息挖掘A股日频因子，weekly 调仓路径，"
            "先训练集评估，再验证集检验；只有通过验证和去重门槛的因子才提交。"
        ),
        # 数据面正交子档：基本面数据面 × weekly。门槛 = 5d 档（与 technical_weekly
        # 共用 _WEEKLY_5D_OVERRIDES，见文件顶部常量定义），engine_gate 收成 weekly。
        **_WEEKLY_5D_OVERRIDES,
    ),
    # ── 三对齐子档位（2026-09-20）──
    # label 持有期 = 调仓频率持有期；快/中/慢信号分轨，各走对应 label + freq + 门槛。
    # 2026-10-04 起**全部档位**都按该规则强制对齐（主档 technical/report 已收成 1d+daily，
    # 见本文件顶部区块）；这三个子档仍是"快/中/慢分轨"的显式入口。
    "technical_daily": ResearchModeSpec(
        mode_id="technical_daily",
        label="日线技术·日频",
        hint="快信号（ρ_f<0.6）：1d 反转/vwap 偏离/跳空，label_1d + daily 调仓",
        recommended_label_col="label_1d_open_to_open",
        signal_families=("volume_price", "volatility", "momentum_reversal"),
        forbidden_families=("pure_size",),
        needs_fundamentals=False,
        candidate_dir="candidate_main",
        production_dir="production_main",
        default_user_message=(
            "请自主挖掘A股日频快信号因子（1d 反转/跳空/vwap 偏离类），"
            "daily 调仓路径，先训练集评估，再验证集检验；"
            "只有通过验证和去重门槛的因子才提交。"
        ),
        engine_gate_overrides={
            "freq": "daily",
            "allowed_freqs": ["daily"],
        },
    ),
    "technical_weekly": ResearchModeSpec(
        mode_id="technical_weekly",
        label="日线技术·周频",
        hint="中信号（ρ_f 0.6~0.85）：量价背离/筹码/VPIN，label_5d + weekly 调仓",
        recommended_label_col="label_5d_close_to_close",
        signal_families=("volume_price", "volatility", "chip", "momentum_reversal"),
        forbidden_families=("pure_size",),
        needs_fundamentals=False,
        candidate_dir="candidate_main",
        production_dir="production_main",
        default_user_message=(
            "请自主挖掘A股日频中信号因子（量价背离/筹码/VPIN 类），"
            "weekly 调仓路径，先训练集评估，再验证集检验；"
            "只有通过验证和去重门槛的因子才提交。"
        ),
        # ── 慢档门槛标定（2026-10-07，v2：文献锚 + 选择性对齐）─────────────
        # 背景：min_abs_ic / min_icir / min_val_abs_ic 按 **label_1d 尺度**标定；慢 label 会系统性
        # 放大同一因子的 IC/ICIR，沿用 1d 线会让慢档门近乎失效（选择性失衡）。
        # **双锚**（v1 只按"选择性对齐"等比放大，绝对值会被推到文献罕见区间，故 v2 加文献锚）：
        #   锚①（绝对量级，外部基准）：日频 Rank-IC ≥0.02 有筛选价值、0.03~0.05 可用；
        #        月度 IC 0.02~0.06 属正常有效、0.05~0.10 属"很好"；ICIR 有效线 0.3（0.5 算强）；
        #        文献实测月度因子 PE TTM 0.0529/0.6995、PB 0.0557/0.4888、ROE 0.0172/0.2277。
        #   锚②（相对选择性）：慢档在同一因子池上的过线率与主档（1d）持平，避免慢档灌水。
        # 实测配对数据（候选池 42 因子 × label_1d/5d/20d，train 2020-2022，n=41）：
        #   |IC| p50 1d 0.0205 → 5d 0.0312 → 20d 0.0496；|ICIR| p50 0.2698 → 0.4351 → 0.7432
        #   该池过线率：主档选 0.02/0.28 → 36.6%；本档选 0.030/0.450 → **36.6%（持平）**；
        #   若按 v1 等比放大到 0.0355/0.504 则仅 26.8%（过严，绝对值亦超文献量级）。
        # 换算规则：candidate/evaluation 用本档 IC/ICIR 线；production 按主档"正式库更严"倍数
        #   （IC ×1.25、ICIR ×0.30/0.28）派生；min_val_abs_ic 按本档 IC 线/主档 0.02 的倍数放大。
        # 刻意不动：min_val_ic_retention（比值口径，与 label 尺度无关）、
        #   min_coverage / min_cs_autocorr / max_abs_corr（因子侧属性，与 label 无关）、
        #   engine_gate 的 min_excess_annual / min_excess_sharpe（年化口径，缺该档实测，暂继承主档）。
        # 数值统一引用 _WEEKLY_5D_OVERRIDES（5d 门槛唯一来源，见文件顶部常量定义）。
        **_WEEKLY_5D_OVERRIDES,
    ),
    # ── 研报模式（2026-09-29 新增）───────────────────────────────────
    # 目标：**先照研报机制复现**一个忠实因子，**再围绕该机制单向发散**。
    # 与 technical/fundamental 的差别只在流程，不在门槛/label：
    #   - 复现阶段：只注入机制卡（构造指南/字段/参数/期望形态），不注入泛 RAG；
    #   - 判定：形态对账（prediction_check）+ train 门；过线则锁定该课题 N 轮发散；
    #   - 发散阶段：才注入研报 RAG，一轮只改一个维度（窗长/算子/同族字段/中性化键/门控形态/交互结构）。
    # 开关全部落在 report_policy_overrides，未选该模式时这些键不存在 → 老行为不变。
    "report": ResearchModeSpec(
        mode_id="report",
        label="研报复现",
        hint="先照研报机制复现因子，再围绕机制单维发散（复现阶段喂机制卡，发散阶段用研报 RAG）",
        recommended_label_col="label_1d_open_to_open",
        # 研报机制常跨数据面：价量/波动/筹码 + 基本面族都放开（needs_fundamentals=True）
        signal_families=(
            "volume_price", "volatility", "chip", "momentum_reversal",
            "fundamental_quality", "fundamental_growth",
            "fundamental_value", "fundamental_revision",
        ),
        forbidden_families=("pure_size",),
        needs_fundamentals=True,
        candidate_dir="candidate_main",
        production_dir="production_main",
        default_user_message=(
            "研报复现模式：先按本轮课题给出的研报机制复现一个忠实因子"
            "（只允许字段同族替换与算子落地，提交时带 reproduce_of=<card_id>）；"
            "复现通过后，围绕该机制做单维发散（窗长/算子/同族字段/中性化键/门控形态/交互结构，一次一维）。"
        ),
        engine_gate_overrides={
            # 2026-10-04 用户定调：label 与调仓频率强制一致 → 本档 label=1d ⇒ freq=daily。
            # （想要 weekly 交付的研报因子，请用 label_5d 的档位；换 label 属改研究口径，
            #   须同步重标定 reproduce_min_abs_ic / min_train_* 等按 1d 尺度标定的门槛。）
            "freq": "daily",
            "allowed_freqs": ["daily"],
        },
        report_policy_overrides={
            "knowledge_mode": "mechanism_cards",
            "knowledge_mode_by_phase": {
                "reproduce": "mechanism_cards",
                "diverge": "report_rag",
            },
            "reproduce_first": True,        # 首轮必须复现（题面 + 提交门禁）
            "reproduce_of_required": True,  # 提交必须声明 reproduce_of=<card_id>
            "reproduce_lock_rounds": 3,     # 复现通过后锁定该课题的发散轮数
            # 复现判定阈值（"信号存在"档，本质是保真而非强度；强度留给发散阶段）
            "reproduce_min_abs_ic": 0.010,
            "reproduce_min_icir": 0.10,
            "reproduce_max_attempts": 2,     # 复现失败可重试一次，提高进入发散的概率
            "enable_question_queue": True,
            "report_rag_max_chars": 1600,
            "report_rag_top_k": 3,
        },
    ),
    "technical_monthly": ResearchModeSpec(
        mode_id="technical_monthly",
        label="日线技术·月频",
        hint="慢信号（ρ_f≥0.85）：长动量/慢反转，label_20d + monthly 调仓",
        recommended_label_col="label_20d_close_to_close",
        signal_families=("volume_price", "volatility", "momentum_reversal"),
        forbidden_families=("pure_size",),
        needs_fundamentals=False,
        candidate_dir="candidate_main",
        production_dir="production_main",
        default_user_message=(
            "请自主挖掘A股日频慢信号因子（长动量/慢反转类），"
            "monthly 调仓路径，先训练集评估，再验证集检验；"
            "只有通过验证和去重门槛的因子才提交。"
        ),
        # ── 慢档门槛标定（2026-10-07，v2：文献锚 + 选择性对齐）─────────────
        # 方法同 technical_weekly（见该档注释）：双锚 = ① 文献绝对量级、② 与主档选择性持平。
        # 本档选定 **0.053 / 0.650**：
        #   锚①：月度 IC 0.05~0.10 属文献"很好"区间下沿，ICIR 0.65 与经典月度因子 PE TTM
        #        （实测 0.0529 / 0.6995）同档 → 准入线不低于成熟因子水平，且未进"罕见/查泄漏"区；
        #   锚②：该池过线率 **36.6%，与主档（0.02/0.28 → 36.6%）持平**；
        #        v1 等比放大到 0.0573/0.849 仅 22%（过严，且绝对值超文献最强档）；
        #        文献锚更松档 0.040/0.60 为 58.5%（选择性偏松，会灌水）。
        # 实测分位对照：|IC| p50 1d 0.0205 → 20d 0.0496；|ICIR| p50 0.2698 → 0.7432。
        # production 仍按主档"正式库更严"倍数派生；min_val_abs_ic 按本档 IC 线倍数放大。
        # 未动的项及理由同 technical_weekly。重标定建议在慢档 run 数据积累后复核。
        # 数值统一引用 _MONTHLY_20D_OVERRIDES（20d 门槛唯一来源，见文件顶部常量定义）；
        # deepcopy 展开防共享可变状态（理由同 fundamental 档）。
        **copy.deepcopy(_MONTHLY_20D_OVERRIDES),
    ),
    # ── 事件/资金数据面 · 月频档（2026-10-19 新增）──────────────────────
    # 立项依据（2026-10-19 体检）：面板 161 列里，资金面 99.9%、机构面 96.2%、
    # 股东集中面 96.3%、股东户数 99.6%、业绩预告 90.4%、披露 96.6% 覆盖的字段
    # 早已载入，但现有 11 个档位的 signal_families 全部只覆盖「价量/波动/波动/筹码 +
    # 基本面」，没有一个族能挂这些字段 → 52 条库内因子里 12 个事件/资金族字段
    # 合计仅被引用 15 次（价量字段被引用 100+ 次），事件面/两融面/机构面/股东集中面/
    # 披露面/分红面 6 个面的在库因子数为 0。
    #
    # 与 technical_monthly 的关系：**同 label（20d）⇒ 同门槛**（沿用
    # `_MONTHLY_20D_OVERRIDES`，符合 2026-10-08「门槛只由 label 唯一决定」定调）；
    # 差别只在 signal_families（放开 7 个新族）与 ``cold_track.enabled=True``。
    #
    # needs_fundamentals=False 的理由：事件/资金/两融/股东/披露/分红列由辅助插件供给，
    # **不依赖** include_fundamentals 开关（实证：include_fundamentals=false 的面板
    # 仍含 ff_*/mgn_*/inst_*/dt_*/pred_*/rc_* 全列）。保持 False 可命中现有
    # `panel_v11_*_all` 缓存（1.7GB parquet / 5GB arrow），避免一次全量重建；
    # 若将来要在此档融合 $funda_*/$fsi_*，改成 True 并接受一次重建成本。
    "technical_event_monthly": ResearchModeSpec(
        mode_id="technical_event_monthly",
        label="事件资金·月频",
        hint="慢信号：资金流/两融/机构持仓/股东集中/限售解禁/业绩预告等事件与资金数据面，label_20d + monthly 调仓",
        recommended_label_col="label_20d_close_to_close",
        signal_families=(
            # 行情四族（作为门控/交互腿仍然需要，不禁止）
            "volume_price", "volatility", "chip", "momentum_reversal",
            # 2026-10-19 新增七族：数据早已在面板，此前无族可挂
            "capital_flow", "margin", "institutional", "shareholder_concentration",
            "corporate_event", "expectation", "disclosure_dividend",
        ),
        forbidden_families=("pure_size",),
        needs_fundamentals=False,
        candidate_dir="candidate_main",
        production_dir="production_main",
        default_user_message=(
            "请基于已载入的事件与资金数据面（主力资金流/融资融券/机构持仓分型/股东集中度/"
            "限售解禁/业绩预告与快报/披露节奏/分红）挖掘 A 股日频因子，"
            "monthly 调仓路径，先训练集评估，再验证集检验；只有通过验证和去重门槛的因子才提交。"
        ),
        **copy.deepcopy(_MONTHLY_20D_OVERRIDES),
        # 冷门赛道（残差化 + val_resid_ic 主键判定）在本档打开：新数据面因子与既有
        # 价量族载荷天然重叠部分需被剥离，才能证明独立信息。其它档位保持默认关闭。
        cold_track_overrides={"enabled": True},
    ),
}


def get_research_mode(mode: str) -> ResearchModeSpec:
    try:
        return RESEARCH_MODES[mode]
    except KeyError as exc:
        raise ValueError(f"research_mode_invalid:{mode}") from exc


def mode_ids() -> list[str]:
    return list(RESEARCH_MODES.keys())


def default_focus_facets(mode: str) -> tuple[str, ...]:
    """Web run 未显式勾选数据面时的默认面集（按需取列的依据）。

    背景（2026-10-20 实测）：Web run 过去恒载全量 158 列 → 加载 5.80GB、求值峰值
    11.01GB。而列数是唯一杠杆：19 列时 1.37GB / 2.28GB。技术档的 LLM 本来就被
    限制在量价类字段内，没必要为 100+ 个用不到的基本面/事件列付内存。

    规则：
    - 恒含 **行情组**（价量/量能/筹码/拥挤）——技术能力的底座；
    - ``needs_fundamentals=True`` 的档位（fundamental*）再加 **基本面组**；
    - 事件资金组默认**不含**：成交/覆盖度不足，且需要时用户勾选数据面即可恢复。

    返回空 tuple 表示"不裁剪"（未知档位/研报模式），调用方应回退全量。
    """
    try:
        spec = get_research_mode(mode)
    except ValueError:
        return ()
    if spec.mode_id == "report":
        return ()  # 研报复现要按研报给的字段来，不能预先裁面
    from alphaagent.factor.facets import FACET_GROUPS

    groups: list[tuple[str, ...]] = [FACET_GROUPS["行情组"]]
    if getattr(spec, "needs_fundamentals", False):
        groups.append(FACET_GROUPS["基本面组"])
    return tuple(dict.fromkeys(f for g in groups for f in g))


def ui_options() -> list[dict]:
    """前端研究模式按钮/因子库类别/保存下拉共享的选项（单一事实源）。

    返回**全部**注册档位（含 internal 标记）：三对齐子档（technical_daily/weekly/
    monthly）与数据面正交子档（fundamental_daily/weekly）标记 internal=True——
    前端下拉据此过滤（researchModeOptions 只展示顶层档位），但 switchResearchMode
    的白名单用全量列表（子档也允许切换，否则 syncInferredMode 静默失效）。存/读
    门槛文件（/research-specs/{mode}）同样以全部档位为准。
    """
    return [
        {
            "value": spec.mode_id,
            "label": spec.label,
            "hint": spec.hint,
            "recommended_label_col": spec.recommended_label_col,
            "default_user_message": spec.default_user_message,
            "needs_fundamentals": spec.needs_fundamentals,
            "internal": spec.mode_id.startswith(("technical_", "fundamental_")),
        }
        for spec in RESEARCH_MODES.values()
    ]


# ── 自动定档（2026-09-03，方案 B；2026-10-08 数据面正交化）────────────
# 前端模式下拉（日线技术/基本面）退役后，档位由「数据面多选 × 调仓频率」两个
# **正交**维度自动推断：
#   · 数据面（基本面/股东面/机构面/股东集中面 vs 价量族）只决定载不载 funda_* 列、
#     开放哪些信号族（fundamental 数据面 → fundamental_* 族；价量面 → 行情族）——
#     见模式声明里的 needs_fundamentals / signal_families；
#   · 调仓频率（daily/weekly/monthly）决定 label 与门槛（1d/5d/20d 档），
#     与数据面无关（2026-10-08 用户定调：门槛只由 label 唯一决定）。
# 组合落到具体档位：fundamental_daily / fundamental_weekly / fundamental（=monthly）/
# technical_daily / technical_weekly / technical_monthly；未显式给频率时按数据面
# 惯性取默认：基本面 → fundamental（monthly，季度数据自然月频调仓）、价量 → technical
# （label_1d）。混合勾选（价量+基本面）走对应数据面频率档：信号族与列加载按
# focus_facets 联动（见 alphaagent_service），评估门槛只看选中的频率。
# 2026-10-04 起 label 与调仓频率**强制一致**（见本文件顶部区块），模式未显式声明
# freq 时由 label 派生，不一致直接拒启动。

_SLOW_FACETS: frozenset[str] = frozenset({"基本面", "股东面", "机构面", "股东集中面"})
# 事件/资金面（2026-10-19）：这些面此前**没有对应族**，勾选后会被推断成
# technical_monthly（族=价量三族），于是"勾了资金面却挖不出资金面因子"。
# 现单独落 technical_event_monthly（同 label_20d / 同门槛，仅族与冷门赛道不同）。
# 注：与 _SLOW_FACETS 同时命中时仍优先 fundamentals 档（基本面为更强信号面），
# 这是已知折中——两者都要时请显式指定 research_mode。
_EVENT_FACETS: frozenset[str] = frozenset({
    "资金面", "两融面", "事件面", "业绩面", "披露面", "分红面",
})


def infer_research_mode(
    focus_facets: list[str] | tuple[str, ...] | None,
    rebalance_freq: str | None = None,
) -> str:
    """按「数据面 × 调仓频率」两个正交维度自动推断研究档位（纯函数，前后端共享）。

    数据面 → 信号族/载列（fundamental 数据面 vs 价量数据面）；频率 → label + 门槛。
    空/未选数据面 → 价量面（technical）。未显式传 rebalance_freq 时：
      基本面数据面 → fundamental（monthly 缺省：季度 PIT 数据自然月频调仓）；
      价量数据面 → technical（label_1d + daily 门禁）。
    显式传 rebalance_freq（daily/weekly/monthly）时按频率落对齐子档：
      基本面数据面 → fundamental_{freq}（daily/weekly；monthly 回落到 fundamental）；
      事件/资金数据面 → technical_event_monthly（仅 monthly/未指定；daily/weekly
        暂无对应档位，回落 technical_{freq} 并保留原门槛口径）；
      价量数据面 → technical_{freq}（monthly 回落到 technical_monthly）。
    label 与调仓频率**强制一致**（2026-10-04 起，见本文件顶部区块）。
    """
    facets = {str(f).strip() for f in (focus_facets or []) if str(f).strip()}
    if facets & _SLOW_FACETS:
        if rebalance_freq in ("daily", "weekly"):
            return f"fundamental_{rebalance_freq}"
        return "fundamental"
    # 事件/资金面档：**只在 monthly/未指定频率**下启用。daily/weekly 没有对应的
    # 20d 门槛以外标定，硬套会退回"同 label 不同门槛"的老问题（见文件顶部区块）。
    if (facets & _EVENT_FACETS) and rebalance_freq not in ("daily", "weekly"):
        return "technical_event_monthly"
    if rebalance_freq in ("daily", "weekly", "monthly"):
        return f"technical_{rebalance_freq}"
    return "technical"
