"""研究模式注册表：单一事实源，前后端/全链路共享。

**目标**：加一个研究模式（technical / fundamental / strategy / sentiment...）
只改本文件一个 dict，回测 spec、因子库目录、基本面加载、前端按钮/下拉/
提示词全部自动跟随——不再散落硬编码（否则"前端多一个风格就多个键"）。

每个消费方从这里取数：
- research_spec.default_research_spec      → 门槛/信号族/label 覆盖
- core.factor_categories                   → 候选/正式库目录
- backend.alphaagent_service               → 是否载入 funda_* 列
- /api/alphaagent/research-modes           → 前端动态渲染 mode 按钮/下拉/提示

未来若加"工具型"模式（如组合扫描），只需扩展本 dataclass 字段，
不必改任何消费方代码。
"""

from __future__ import annotations

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


# ── label 口径 vs 调仓频率：**刻意解耦**（2026-09-30 决策，勿"顺手对齐"）──────
# 本文件里这两个字段是**两把独立的尺子**，不是同一件事的两种写法：
#
#   1) recommended_label_col —— **研究口径**（筛查打分用）
#      决定 IC / RANKIC / ICIR / decile 按多长前瞻收益计算，并按持有期去重叠
#      （label_20d → 每 20 个交易日取一点）；十分组的年化/回撤/夏普也按
#      holding_days = label 天数算，且 submit 侧一律 cost_bps=0。
#   2) engine_gate_overrides["freq"] —— **交付口径**（最终收益用）
#      engine_gate 用真实行情（T+1/涨跌停/停牌/整手/滑点/参与率/buffer）按该频率
#      调仓回测，**整个函数不读 label**；入库定生死只认这一把尺子。
#
# 因此主档（如 technical）故意配 label_1d + weekly 交付：
#   - label_1d 样本多、信噪好 → 探测"单日边际预测力"更灵敏；
#   - weekly 用来控换手成本（RANKIC 日度抖动大、但周频调仓成本可控）。
# 两把尺子的落差由两个**代理门**缝住，而不是靠把 label 改成 freq 对齐：
#   · stage_one 的 min_cs_autocorr（截面排名日度延续性，低 → 周调仓必死）；
#   · 换手门槛按 freq 分档（delivery_criteria.turnover_thresholds_by_freq）。
#
# 只有 technical_daily / technical_weekly / technical_monthly 三个子档才是真
# "三对齐"（label 持有期 == 调仓持有期）。已知代价（接受）：1d IC 稳、4~5 日部分
# 回吐的因子能过 stage_one，最终在含成本的 engine_gate 被拒，浪费部分挖掘算力；
# 研究指标与实盘收益不可互相换算，UI 必须分列展示。
#
# ⚠ 不要把主档的 recommended_label_col 改成 derive_label_from_freq(freq) 来"消除
#   不一致"：那等于换掉研究口径，而 min_abs_ic / min_train_icir 等门槛是按 label
#   尺度标定的，会一并失真。真要收成单旋钮，属独立改动 + 需重标定门槛。

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
        # 放开调仓频率白名单：默认 GATE_FREQ=weekly，但允许 LLM/用户显式覆盖
        # daily/weekly/monthly。注意本档是"研究口径 label_1d + 交付口径 weekly"的
        # **刻意解耦**组合（见本文件顶部区块），不是三对齐子档。
        engine_gate_overrides={
            "freq": "weekly",
            "allowed_freqs": ["daily", "weekly", "monthly"],
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
        # 基本面为慢因子：季频 PIT 信号弱，统计门槛放宽松，但可交易性只小幅放松。
        # 2026-09-11 观察池口径：technical 候选线回到 0.020/0.28 后与 fundamental
        # 同值——本档 override 保留作为"量纲锚"（technical 若再调整，fundamental
        # 仍锚定 0.020/0.28/0.012），仅 val 保留比（0.65）与 production 更严项是实差异。
        evaluation_overrides={
            "min_train_abs_ic": 0.020,
            "min_train_icir": 0.28,
            "min_val_abs_ic": 0.012,
            "min_val_ic_retention_ratio": 0.5,
        },
        candidate_overrides={
            "min_abs_ic": 0.020,
            "min_icir": 0.28,
            "min_val_abs_ic": 0.012,
            # 2026-08-29 审计：11 个候选 8 个 val 保留比 <65%（train→val 衰减
            # 严重），10d 持有期对季频 PIT 信号过短也是成因之一（切 label_20d）
            "min_val_ic_retention": 0.65,
        },
        production_overrides={
            "min_train_abs_ic": 0.020,   # technical 0.025 → 0.020
            "min_train_icir": 0.28,      # technical 0.30 → 0.28
            "min_val_abs_ic": 0.012,     # technical 0.015 → 0.012
            "min_val_ic_retention": 0.70,
            "min_val_long_excess": 0.0,
            "max_winsorized_abs_ic_decay": 0.12,  # technical 0.10 → 0.12
        },
        engine_gate_overrides={
            "freq": "monthly",           # technical weekly → monthly
            "min_excess_annual": 0.02,   # technical 0.03 → 0.02
            "min_excess_sharpe": 0.4,    # technical 0.5 → 0.4
            # list 而非 tuple：本 dict 会直接 update 进 research_spec，normalize
            # 的 _string_list 校验要求 list（tuple 会让所有非 technical 档
            # effective/build_run 链路 ValueError，2026-09-22 修复）
            "allowed_freqs": ["daily", "weekly", "monthly"],
        },
    ),
    # ── 三对齐子档位（2026-09-20）──
    # label 持有期 = 调仓频率持有期；快/中/慢信号分轨，各走对应 label + freq + 门槛。
    # 注意：只有这三个子档是"对齐"的，主档（如 technical）是刻意解耦（见本文件顶部区块）。
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
        engine_gate_overrides={
            "freq": "weekly",
            "allowed_freqs": ["weekly"],
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
        engine_gate_overrides={
            "freq": "monthly",
            "allowed_freqs": ["monthly"],
        },
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

    仅返回顶层档位（technical/fundamental）；三对齐子档（technical_daily/
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
# （label_20d + 松门槛 + monthly 门禁）；其余（纯价量族或混合）→ technical
# 档（label_1d + 严门槛 + weekly 门禁）。混合勾选落 technical：融合因子以
# 价量为主信号、1d 评估合理，且用户可用 rebalance_freq 显式覆盖门禁频率。

_SLOW_FACETS: frozenset[str] = frozenset({"基本面", "股东面", "机构面", "股东集中面"})


def infer_research_mode(
    focus_facets: list[str] | tuple[str, ...] | None,
    rebalance_freq: str | None = None,
) -> str:
    """按数据面多选自动推断研究档位（mode_id：technical/fundamental/technical_*）。

    纯函数，前后端共享。空/未选 → technical（原默认行为不变）。
    显式传入 rebalance_freq（daily/weekly/monthly）时，价量面走对应三对齐子档
    （technical_daily/weekly/monthly，label 持有期与调仓频率对齐）；基本面面仍走
    fundamental。主档（如 technical）与 rebalance_freq 刻意解耦，见本文件顶部区块。
    """
    facets = {str(f).strip() for f in (focus_facets or []) if str(f).strip()}
    if facets & _SLOW_FACETS:
        return "fundamental"
    if rebalance_freq in ("daily", "weekly", "monthly"):
        return f"technical_{rebalance_freq}"
    return "technical"