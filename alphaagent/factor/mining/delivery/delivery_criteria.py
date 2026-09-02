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

    min_abs_ic: float = 0.02
    min_icir: float = 0.25
    min_coverage: float = 0.85
    max_abs_corr: float = 0.5
    min_cs_autocorr: float = 0.18
    min_val_ic_retention: float = 0.5
    # 组合可交易性预检（2026-08-29）：日单边换手 >50% 的候选在 stage_one 直接拒，
    # 不再等 stage_two/engine_gate 才拦截（历史数据：30 个候选 26 个日换手>50%，
    # 全部止步 stage_two/engine_gate，浪费大量评估算力）。
    max_avg_daily_side_turnover: float = 0.5


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
    min_val_ic_retention: float = 0.50  # 2026-08-29 从 0.60 下调：最强因子 train 太好（IC 0.046）反被 0.56<0.60 惩罚
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
    top_n: int = trading_config.GATE_TOP_N
    freq: str = trading_config.GATE_FREQ
    allowed_freqs: tuple[str, ...] = ("daily", "weekly", "monthly")
    capital: float = trading_config.GATE_CAPITAL
    slippage_bps: float = trading_config.GATE_SLIPPAGE_BPS
    max_participation: float = trading_config.GATE_MAX_PARTICIPATION
    min_am20_yuan: float = trading_config.GATE_MIN_AM20_YUAN
    min_excess_annual: float = trading_config.GATE_MIN_EXCESS_ANNUAL
    min_excess_sharpe: float = trading_config.GATE_MIN_EXCESS_SHARPE
    max_drawdown: float = trading_config.GATE_MAX_DRAWDOWN
    min_daily_overlap: float = trading_config.GATE_MIN_DAILY_OVERLAP
    min_invested_ratio: float = trading_config.GATE_MIN_INVESTED_RATIO


@dataclass(frozen=True)
class BlindTestCriteria:
    """盲测终审门槛：test 段（留出测试段）硬指标门禁。

    盲测段（test_start ~ data_latest）从未参与 train/val/engine_gate，
    是最干净的样本外验证。盲测终审在 stage_one 之前执行——
    不通过直接拒绝，不进候选池，不消耗后续相似度/回测算力。

    门槛项（2026-08-29 确立）：
    - IC 保留比 = |test_ic|/|train_ic| ≥ min_ic_retention（默认 0.50）；
    - 方向一致性：test 段 IC 方向必须与 train 段一致（sign_consistent）。

    test 段当前约 20 个月（2025-01 ~ 数据最新日），样本充足可设硬门。
    """

    enabled: bool = True
    min_ic_retention: float = 0.50
    require_sign_consistency: bool = True


@dataclass(frozen=True)
class ScreenerCriteria:
    """Screener（regime 感知因子筛选）门槛。

    对应 AlphaCrafter 论文的 Screener 智能体：从因子库中按当前市场制度
    (regime) 选出适配因子子集 + 动态权重。在 quant_ui 中用确定性计算
    替代 LLM 判断（ADX+均线检 regime → Rank IC 评分 → 族偏好 boost →
    相关性去冗余 → 权重归一化）。

    挂载位置：stage_one 统计门槛通过后、入候选池前执行。开关关闭时
    跳过，不影响现有挖掘链路行为。

    与 core/screener.py 的 ScreenerConfig 参数对齐，但由 research_spec
    统一注入，保证前端门槛编辑器/提示词/判定口径同源。
    """

    enabled: bool = False             # 默认关闭，不改变现有行为
    lookback: int = 10               # Rank IC 回看天数
    min_ic: float = 0.02             # |IC| 低于此值的因子淘汰
    max_corr: float = 0.7            # 因子间 |corr| 高于此值视为冗余
    use_family_boost: bool = True    # 是否启用因子族 regime 偏好
    adx_threshold: float = 25.0     # ADX > 此值认为有趋势
    ma_period: int = 60             # 均线周期（交易日）
    min_cross_section: int = 30      # 有效股票不足此数则跳过该天 IC


@dataclass(frozen=True)
class ParamStabilityCriteria:
    """参数邻域稳定性门槛（stage_two 附加门禁，2026-08 新增）。

    真实 alpha 不应依赖某个神奇窗口（knife-edge）：对 DSL 中 TS_* 算子第二
    参数位的整数窗口做整体 ±offset 邻域扰动，在 train 窗口重评估全部变体，
    方向调整后（adj = metric * sign(train_ic)，负 IC 稳定因子不受惩罚）要求：
    - 邻域变体 adj_ic > 0 占比 >= min_positive_fraction（默认 0.67）；
    - 最差变体 adj_icir >= min_worst_icir（默认 0.0，即邻域不反向）。
    变体评估报错按非正计入（邻域边缘失效是真实风险）；全部报错或表达式
    无整数窗口参数时跳过门禁（记录 skipped_reason，不拦截）。
    """

    enabled: bool = True
    min_positive_fraction: float = 0.67
    min_worst_icir: float = 0.0
    window_offsets: tuple[int, ...] = (-1, 1, -2, 2)
    min_window: int = 2
    max_window_values: int = 2
    max_variants: int = 6


@dataclass(frozen=True)
class DeliveryCriteria:
    """交付门槛全集：盲测终审 + 候选池 + 正式库统计 + 参数稳定性 + engine_gate + Screener。"""

    blind_test: BlindTestCriteria = BlindTestCriteria()
    candidate: CandidateCriteria = CandidateCriteria()
    production: ProductionCriteria = ProductionCriteria()
    engine_gate: EngineGateCriteria = EngineGateCriteria()
    screener: ScreenerCriteria = ScreenerCriteria()
    param_stability: ParamStabilityCriteria = ParamStabilityCriteria()

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

        blind = policy.get("blind_test") or {}
        screener = policy.get("screener") or {}
        cand = policy.get("candidate") or {}
        prod = policy.get("production") or {}
        eg = prod.get("engine_gate") or {}
        if not isinstance(eg, dict):
            eg = {}
        ps = policy.get("param_stability") or {}

        def _fill(base: Any, src: dict[str, Any]) -> dict[str, Any]:
            merged: dict[str, Any] = {}
            for f in fields(base):
                key = f.name
                if key in src and src[key] is not None:
                    merged[key] = src[key]
                else:
                    merged[key] = getattr(base, key)
            return merged

        blind_obj = BlindTestCriteria(**_fill(BlindTestCriteria(), blind))
        screener_obj = ScreenerCriteria(**_fill(ScreenerCriteria(), screener))
        cand_obj = CandidateCriteria(**_fill(CandidateCriteria(), cand))
        prod_obj = ProductionCriteria(**_fill(ProductionCriteria(), prod))
        ps_raw = _fill(ParamStabilityCriteria(), ps)
        # window_offsets 内部统一为 tuple（frozen dataclass 默认），
        # 与 allowed_freqs 同样的 list/tuple 归一化处理。
        if isinstance(ps_raw.get("window_offsets"), (list, tuple)):
            ps_raw["window_offsets"] = tuple(int(x) for x in ps_raw["window_offsets"])
        ps_obj = ParamStabilityCriteria(**ps_raw)
        eg_raw = _fill(EngineGateCriteria(), eg)
        # allowed_freqs 内部统一为 tuple（frozen dataclass 默认），
        # 与 canonical 默认一致；对外序列化时再还原为 list。
        if isinstance(eg_raw.get("allowed_freqs"), (list, tuple)):
            eg_raw["allowed_freqs"] = tuple(eg_raw["allowed_freqs"])
        eg_obj = EngineGateCriteria(**eg_raw)
        ps_raw = _fill(ParamStabilityCriteria(), ps)
        # window_offsets 同理统一为 tuple[int, ...]（JSON 侧为 list）。
        if isinstance(ps_raw.get("window_offsets"), (list, tuple)):
            ps_raw["window_offsets"] = tuple(int(x) for x in ps_raw["window_offsets"])
        ps_obj = ParamStabilityCriteria(**ps_raw)
        return cls(
            blind_test=blind_obj, screener=screener_obj,
            candidate=cand_obj, production=prod_obj, engine_gate=eg_obj,
            param_stability=ps_obj,
        )

    # ── 序列化 ──

    def blind_test_dict(self) -> dict[str, Any]:
        return dict((f.name, getattr(self.blind_test, f.name)) for f in fields(self.blind_test))

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

    def screener_dict(self) -> dict[str, Any]:
        return dict((f.name, getattr(self.screener, f.name)) for f in fields(self.screener))

    def to_spec_dict(self) -> dict[str, Any]:
        return {
            "blind_test": self.blind_test_dict(),
            "screener": self.screener_dict(),
            "candidate": self.candidate_dict(),
            "production": {
                **self.production_dict(),
                "engine_gate": self.engine_gate_dict(),
            },
        }

    # ── 提示词渲染 ──

    def to_prompt_text(self) -> str:
        """渲染交付门槛的 Markdown 文本，供 LLM 系统提示词使用。

        数值全部来自本对象，杜绝提示词与真实门禁脱节。
        """
        b = self.blind_test
        sc = self.screener
        c = self.candidate
        p = self.production
        eg = self.engine_gate

        def _pct(v, digits=1) -> str:
            try:
                return f"{float(v) * 100:.{digits}f}%"
            except (TypeError, ValueError):
                return str(v)

        blind_test = (
            "盲测终审（test 段，stage_one 之前执行）："
            f"`test/train IC 保留比 >= {_pct(b.min_ic_retention)}` 且"
            f"{'方向必须一致' if b.require_sign_consistency else '方向不限制'}；"
            "不通过直接拒绝，不进候选池。"
        ) if b.enabled else ""
        screener_text = (
            f"Screener（regime 感知筛选，stage_one 通过后执行）："
            f"按 ADX({sc.adx_threshold})+MA({sc.ma_period}) 检测市场制度，"
            f"回看 {sc.lookback} 天 Rank IC 评分，|IC| >= {sc.min_ic} 入选，"
            f"因子间 |corr| > {sc.max_corr} 去冗余，"
            f"{'启用因子族 regime 偏好' if sc.use_family_boost else '不启用族偏好'}；"
            "不通过不进候选池。"
        ) if sc.enabled else ""
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
            f"submit_factor 的 rebalance_freq 必须传 \"{eg.freq}\""
            f"（用户指定档位；可选范围 {', '.join(eg.allowed_freqs)}）。"
        )
        return (
            "`submit_factor` 会先执行 pre-submit Reviewer，再在 train-start~val-end 全区间复核。"
            + blind_test + screener_text + stage_one + stage_two + engine
            + " 全部通过才写正式库并返回 `stored=true`。ICIR 按原始符号判断，不取绝对值。"
        )
