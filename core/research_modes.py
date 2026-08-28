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
    recommended_label_col: str        # 评估 label（慢因子模式要求长持有期）
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
        candidate_dir="candidate_technical",
        production_dir="production_technical",
        default_user_message=(
            "请自主挖掘A股日频价量因子，先训练集评估，再验证集检验；"
            "只有通过验证和去重门槛的因子才提交。"
        ),
    ),
    "fundamental": ResearchModeSpec(
        mode_id="fundamental",
        label="基本面",
        hint="funda_* 财务基本面因子（PIT），label_10d 收盘到收盘",
        recommended_label_col="label_10d_close_to_close",
        signal_families=(
            "fundamental_quality", "fundamental_growth",
            "fundamental_value", "fundamental_revision",
        ),
        forbidden_families=("pure_size", "pure_price_momentum"),
        needs_fundamentals=True,
        candidate_dir="candidate_fundamental",
        production_dir="production_fundamental",
        default_user_message=(
            "请基于已载入的基本面字段（盈利质量、杠杆、现金流、财报科目等，"
            "PIT 日频）结合价量信息挖掘A股日频因子，先训练集评估，再验证集检验；"
            "只有通过验证和去重门槛的因子才提交。"
        ),
        # 基本面为慢因子：季频 PIT 信号弱，统计门槛放宽松，但可交易性只小幅放松。
        evaluation_overrides={
            "min_train_abs_ic": 0.015,   # technical 0.02 → 0.015
            "min_train_icir": 0.22,      # technical 0.25 → 0.22
            "min_val_abs_ic": 0.008,     # technical 0.01 → 0.008
            "min_val_ic_retention_ratio": 0.5,
        },
        candidate_overrides={
            "min_abs_ic": 0.012,         # technical 0.015 → 0.012
            "min_icir": 0.20,            # technical 0.25 → 0.20
        },
        production_overrides={
            "min_train_abs_ic": 0.020,   # technical 0.025 → 0.020
            "min_train_icir": 0.28,      # technical 0.30 → 0.28
            "min_val_abs_ic": 0.012,     # technical 0.015 → 0.012
            "min_val_ic_retention": 0.60,
            "min_val_long_excess": 0.0,
            "max_winsorized_abs_ic_decay": 0.12,  # technical 0.10 → 0.12
        },
        engine_gate_overrides={
            "freq": "monthly",           # technical weekly → monthly
            "min_excess_annual": 0.02,   # technical 0.03 → 0.02
            "min_excess_sharpe": 0.4,    # technical 0.5 → 0.4
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
    """前端研究模式按钮/因子库类别/保存下拉共享的选项。"""
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
    ]