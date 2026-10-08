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
    # 研报模式专用：report_policy 覆盖（流程开关，如"复现阶段用机制卡、发散阶段用 RAG"）。
    # 非研报模式留空 → 行为与今天完全一致。
    report_policy_overrides: dict = field(default_factory=dict)


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
#   · label_1d   → technical / technical_daily / report 共用 0.020/0.28/0.015
#   · label_5d   → technical_weekly 用 0.030/0.45/0.0225
#   · label_20d  → technical_monthly 与 fundamental **共用** _MONTHLY_20D_OVERRIDES
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
        hint="funda_* 财务基本面因子（PIT），label_20d 收盘到收盘",
        recommended_label_col="label_20d_close_to_close",
        signal_families=(
            "fundamental_quality", "fundamental_growth",
            "fundamental_value", "fundamental_revision",
        ),
        forbidden_families=("pure_size", "pure_price_momentum"),
        needs_fundamentals=True,
        candidate_dir="candidate_main",
        production_dir="production_main",
        default_user_message=(
            "请基于已载入的基本面字段（盈利质量、杠杆、现金流、财报科目等，"
            "PIT 日频）结合价量信息挖掘A股日频因子，先训练集评估，再验证集检验；"
            "只有通过验证和去重门槛的因子才提交。"
        ),
        # ═══ 2026-10-08 门槛统一（门槛只由 label 决定，与数据面无关）═════
        # 本档与 technical_monthly 同为 label_20d，故**共用同一套 20d 门槛**
        # （_MONTHLY_20D_OVERRIDES，数值来源见文件顶部常量定义）。
        # 取代此前独立标定的 0.035/0.45/0.021（"文献月度基本面量级"），消除
        # "同为 label_20d 却门槛不同"的矛盾（技术月频 0.053/0.65 vs 基本面松 0.035/0.45）。
        # 同步对齐：val 保留比 0.65/0.70→0.50、engine_gate 年化门 0.02/0.4→0.03/0.5、
        #   max_winsorized_abs_ic_decay 0.12→0.10、allowed_freqs 收成 [monthly]。
        # 本档相对 technical_monthly 的唯一区别 = needs_fundamentals=True（载入 funda_* 列）。
        # deepcopy 展开：fundamental 与 technical_monthly 各自持有独立子 dict，防止
        # 共享同一可变对象（原地改某档会静默污染另一档与"唯一事实源"常量）。
        **copy.deepcopy(_MONTHLY_20D_OVERRIDES),
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
        evaluation_overrides={
            "min_train_abs_ic": 0.030,    # 文献"有意义线"（锚①）且选择性对齐（锚②）
            "min_train_icir": 0.450,      # 选择性对齐；文献有效线 0.3 之上
            "min_val_abs_ic": 0.0225,     # 0.015 × (0.030/0.02)
        },
        candidate_overrides={
            "min_abs_ic": 0.030,
            "min_icir": 0.450,
            "min_val_abs_ic": 0.0225,
        },
        production_overrides={
            "min_train_abs_ic": 0.0375,   # 0.030 × 1.25（主档正式库更严倍数）
            "min_train_icir": 0.4821,     # 0.450 × 0.30/0.28
            "min_val_abs_ic": 0.0225,
        },
        engine_gate_overrides={
            "freq": "weekly",
            "allowed_freqs": ["weekly"],
        },
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
}


def get_research_mode(mode: str) -> ResearchModeSpec:
    try:
        return RESEARCH_MODES[mode]
    except KeyError as exc:
        raise ValueError(f"research_mode_invalid:{mode}") from exc


def mode_ids() -> list[str]:
    return list(RESEARCH_MODES.keys())


def ui_options() -> list[dict]:
    """前端研究模式按钮/因子库类别/保存下拉共享的选项。

    返回全部顶层档位（technical/fundamental/report）；三对齐子档（technical_daily/
    weekly/monthly）是内部档位，由 infer_research_mode 依据 rebalance_freq
    自动选用，不暴露给前端下拉（避免污染 UI）。
    """
    return [
        {
            "value": spec.mode_id,
            "label": spec.label,
            "hint": spec.hint,
            "recommended_label_col": spec.recommended_label_col,
            "default_user_message": spec.default_user_message,
            "needs_fundamentals": spec.needs_fundamentals,
        }
        for spec in RESEARCH_MODES.values()
        if not spec.mode_id.startswith("technical_")
    ]


# ── 自动定档（2026-09-03，方案 B）─────────────────────────────────────
# 前端模式下拉（日线技术/基本面）退役后，档位由数据面多选自动推断：
# 勾选基本面/股东面/机构面/股东集中面（慢因子数据源）→ fundamental 档
# （label_20d + monthly 门禁；门槛与 technical_monthly 共用，2026-10-08 统一——
#  门槛只由 label 决定，与数据面无关）；其余（纯价量族或混合）→ technical
# 档（label_1d + 严门槛 + daily 门禁）。混合勾选落 technical：融合因子以
# 价量为主信号、1d 评估合理。用户显式给 rebalance_freq 时改走对应的
# technical_{freq} 子档（label 与调仓频率对齐，2026-10-04 起全局强制一致）。

_SLOW_FACETS: frozenset[str] = frozenset({"基本面", "股东面", "机构面", "股东集中面"})


def infer_research_mode(
    focus_facets: list[str] | tuple[str, ...] | None,
    rebalance_freq: str | None = None,
) -> str:
    """按数据面多选自动推断研究档位（mode_id：technical/fundamental/technical_*）。

    纯函数，前后端共享。空/未选 → technical（label_1d + daily 门禁）。
    显式传入 rebalance_freq（daily/weekly/monthly）时，价量面走对应**对齐**子档
    （technical_daily/weekly/monthly，label 持有期 == 调仓持有期）；基本面面仍走
    fundamental。2026-10-04 起 label 与调仓频率**强制一致**（见本文件顶部区块），
    主档 technical/report 也已收成 1d + daily，不再有"解耦"组合。
    """
    facets = {str(f).strip() for f in (focus_facets or []) if str(f).strip()}
    if facets & _SLOW_FACETS:
        return "fundamental"
    if rebalance_freq in ("daily", "weekly", "monthly"):
        return f"technical_{rebalance_freq}"
    return "technical"