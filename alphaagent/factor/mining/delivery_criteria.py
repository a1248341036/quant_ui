"""交付两阶段门槛的单一来源（single source of truth）。

候选池（candidate）与正式库（production）的统计/可交易性门槛在此集中定义，
数值与 ``research_spec.DEFAULT_RESEARCH_SPEC["delivery_policy"]`` 默认值严格一致。

下游消费方：
- ``submit.py`` 通过 ``FactorSubmitService.criteria`` 读取判定门槛；
- ``tools.py`` / ``prompts.py`` 通过 ``DeliveryCriteria.to_prompt_text()`` 动态渲染
  LLM 提示词，不再硬编码数值，保证提示词与真实门禁永不脱节；
- ``research_spec.py`` 的默认 delivery_policy 由 ``DeliveryCriteria.defaults()``
  生成，确保唯一真源（见 default_research_spec 中的用法）。

运行时以 research_spec 注入为准：``from_spec(delivery_policy)`` 对缺失键回落
到本模块默认值；未传 research_spec 的路径（如 run.py 遗留）回落同一默认值，
从而消除旧 submit.py 内散落硬编码（如候选 corr 0.6）造成的口径漂移。
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

from core import trading_config


@dataclass(frozen=True)
class CandidateCriteria:
    """候选池（海选）统计门槛：train-only 窗口口径。

    海选 ICIR 与 train_screen 规则对齐（0.20 偏松，弱稳定因子堆积候选池）；
    换手可行性硬门槛（低于阈值的因子截面排名日度剧变，不可交付）；
    样本外保留比 = val_ic / train_ic 绝对比值下限（方向反转直接拦截）。
    """

    min_abs_ic: float = 0.015
    min_icir: float = 0.25
    min_coverage: float = 0.85
    max_abs_corr: float = 0.5
    min_cs_autocorr: float = 0.18
    min_val_ic_retention: float = 0.5


@dataclass(frozen=True)
class ProductionCriteria:
    """正式库精筛统计门槛：双窗口（train+val）口径，2026-08 重构。

    统计族双窗口各自达标（混合窗口会稀释 val 衰减）；val 多头端毛值超额
    （方向自适应十分组，复利年化）为 2026-08 审计发现的 IC 盲区补丁：
    IC 为正不代表多头端赚钱——alpha 可能全在空头端/中段排名，纯多头
    可交易组合必须单独为正。

    已移除的摆设/失真门槛（研究结论 2026-08）：
    - min_fmb_t_stat / min_ls_t_stat：t ≈ ICIR×√N，700+ 天下永不拦截；
    - min_quantile_excess_return / min_quantile_sharpe / min_monotonicity /
      min_long_group_annual_excess_return：毛值十分组口径系统性高估可交易性
      （同因子净值 weekly 净超额仅 ~4%）→ 组合可行性全部交给 engine_gate
      净值裁决。
    """

    min_train_abs_ic: float = 0.025
    min_train_icir: float = 0.30
    min_val_abs_ic: float = 0.015
    min_val_ic_retention: float = 0.60
    min_val_long_excess: float = 0.0
    max_winsorized_abs_ic_decay: float = 0.10
    max_abs_corr: float = 0.4


@dataclass(frozen=True)
class EngineGateCriteria:
    """进正式库前的旧引擎完整约束回测门禁（净值裁决，非代理指标）。

    T+1/整手/涨跌停/停牌/费率/滑点/流动性全约束，纯内存、不落盘。
    口径要点（2026-08）：alpha 的意义在超额而非绝对收益——样本含熊市时
    绝对年化/绝对夏普门会把所有因子拒之门外，故只设净值超额门 + 超额夏普门；
    回撤与尾部稳定照旧。动态百分比选股自动适配停牌/涨跌停导致的候选池缩放。
    """

    enabled: bool = True
    selection_mode: str = trading_config.SELECTION_MODE
    selection_pct: float = trading_config.GATE_SELECTION_PCT
    freq: str = trading_config.GATE_FREQ
    allowed_freqs: tuple[str, ...] = ("daily", "weekly", "monthly")
    min_excess_annual: float = trading_config.GATE_MIN_EXCESS_ANNUAL
    min_excess_sharpe: float = trading_config.GATE_MIN_EXCESS_SHARPE
    max_drawdown: float = trading_config.GATE_MAX_DRAWDOWN
    min_daily_overlap: float = trading_config.GATE_MIN_DAILY_OVERLAP
    capital: float = trading_config.GATE_CAPITAL
    min_invested_ratio: float = trading_config.GATE_MIN_INVESTED_RATIO
    min_am20_yuan: float = trading_config.GATE_MIN_AM20_YUAN


@dataclass(frozen=True)
class DeliveryCriteria:
    """两阶段交付门槛全集：候选池 + 正式库统计 + engine_gate。"""

    candidate: CandidateCriteria = CandidateCriteria()
    production: ProductionCriteria = ProductionCriteria()
    engine_gate: EngineGateCriteria = EngineGateCriteria()

    # ── 构造 ──

    @classmethod
    def defaults(cls) -> "DeliveryCriteria":
        return cls()

    @classmethod
    def from_spec(cls, spec: dict[str, Any] | None) -> "DeliveryCriteria":
        """从 research_spec（或其 delivery_policy 子字典）构造门槛。

        - 传入完整 research_spec：取 ``spec["delivery_policy"]``；
        - 传入 delivery_policy 子字典：直接使用；
        - 缺失键回落 canonical 默认值；None/空 dict 得到默认门槛。
        """
        policy = spec or {}
        if "delivery_policy" in policy and isinstance(policy.get("delivery_policy"), dict):
            policy = policy["delivery_policy"]

        cand = policy.get("candidate") or {}
        prod = policy.get("production") or {}
        eg = prod.get("engine_gate") or {}
        if not isinstance(eg, dict):
            eg = {}

        def _fill(base: Any, src: dict[str, Any]) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            for f in fields(base):
                key = f.name
                if key in src and src[key] is not None:
                    merged[key] = src[key]
                else:
                    merged[key] = getattr(base, key)
            return merged

        cand_obj = CandidateCriteria(**_fill(CandidateCriteria(), cand))
        prod_obj = ProductionCriteria(**_fill(ProductionCriteria(), prod))
        eg_raw = _fill(EngineGateCriteria(), eg)
        # allowed_freqs 内部统一为 tuple（frozen dataclass 默认），
        # 与 canonical 默认一致；对外序列化时再还原为 list。
        if isinstance(eg_raw.get("allowed_freqs"), (list, tuple)):
            eg_raw["allowed_freqs"] = tuple(eg_raw["allowed_freqs"])
        eg_obj = EngineGateCriteria(**eg_raw)
        return cls(candidate=cand_obj, production=prod_obj, engine_gate=eg_obj)

    # ── 序列化 ──

    def candidate_dict(self) -> dict[str, Any]:
        return dict((f.name, getattr(self.candidate, f.name)) for f in fields(self.candidate))

    def production_dict(self) -> dict[str, Any]:
        return dict((f.name, getattr(self.production, f.name)) for f in fields(self.production))

    def engine_gate_dict(self) -> dict[str, Any]:
        d = dict((f.name, getattr(self.engine_gate, f.name)) for f in fields(self.engine_gate))
        # allowed_freqs 以 tuple 存储（frozen dataclass 默认），对外保持 list 与
        # research_spec 历史结构一致（JSON 序列化 / 调用方迭代均期望 list）。
        d["allowed_freqs"] = list(self.engine_gate.allowed_freqs)
        return d

    def to_spec_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate_dict(),
            "production": {
                **self.production_dict(),
                "engine_gate": self.engine_gate_dict(),
            },
        }

    # ── 提示词渲染 ──

    def to_prompt_text(self) -> str:
        """渲染两阶段交付门槛的 Markdown 文本，供 LLM 系统提示词使用。

        数值全部来自本对象，杜绝提示词与真实门禁脱节。
        """
        c = self.candidate
        p = self.production
        eg = self.engine_gate

        def _pct(v, digits=1) -> str:
            try:
                return f"{float(v) * 100:.{digits}f}%"
            except (TypeError, ValueError):
                return str(v)

        stage_one = (
            "第一阶段（候选登记，train-only 窗口）："
            f"`abs(IC) >= {c.min_abs_ic}`、`ICIR > {c.min_icir}`、"
            f"`Coverage > {_pct(c.min_coverage)}`、"
            f"`cs_autocorr >= {c.min_cs_autocorr}`、"
            f"`val/train IC 保留比 >= {_pct(c.min_val_ic_retention)}` 且方向不反转、"
            f"与已有因子最大截面相关 `< {c.max_abs_corr}`；"
            "通过后写入轻量候选 registry（不物化全量因子值）。"
        )
        stage_two = (
            "第二阶段（精筛正式库，双窗口口径）："
            f"train `abs(IC) >= {p.min_train_abs_ic}`、`abs(ICIR) > {p.min_train_icir}`、"
            f"val `abs(IC) >= {p.min_val_abs_ic}`、"
            f"val/train IC 保留比 >= {_pct(p.min_val_ic_retention)}、"
            f"val 多头端年化超额 >= {_pct(p.min_val_long_excess)}、"
            f"截尾 IC 衰减 <= {_pct(p.max_winsorized_abs_ic_decay)}、"
            f"与已有因子最大截面相关 `< {p.max_abs_corr}`。"
        )
        engine = (
            f"最终还需通过 engine_gate 完整回测（调仓频率 {eg.freq}、"
            f"动态 top {_pct(eg.selection_pct)} 选股、净超额年化 >= "
            f"{_pct(eg.min_excess_annual)}、超额夏普 >= {eg.min_excess_sharpe}、"
            f"回撤 <= {_pct(eg.max_drawdown)}、仓位利用率 >= {_pct(eg.min_invested_ratio)}）。"
        )
        return (
            "`submit_factor` 会先执行 pre-submit Reviewer，再在 train-start~val-end 全区间复核。"
            + stage_one + stage_two + engine
            + " 全部通过才写正式库并返回 `stored=true`。ICIR 按原始符号判断，不取绝对值。"
        )
