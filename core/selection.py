"""组合候选筛选与目标权重构建。

该模块不读取行情、不执行订单，只把信号截面转换为目标持仓。

选股层以 :class:`SelectionPolicy` 为中心解耦：数量模式、绝对信号门控、
数量约束、行业分组上限、弱市降仓叠加全部收拢在策略对象里，
引擎主流程只调用 ``PortfolioBuilder.build_targets`` 一个入口。
新的选股类型优先扩展本模块，而不是往 ``run_backtest`` 加散装参数。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .portfolio import portfolio_vol


@dataclass
class SelectionPolicy:
    """组合构建策略：描述"从因子截面到目标持仓"的全部规则。

    维度与流水线：
      1. 绝对门控   ``min_score``：质量分下限。质量分 = score（买大）或
                    -score（买小），即"越大越好"的方向归一化。
                    例：brk20 买大配 min_score=0 表示只买创20日新高的票；
                    vol20 买小配 min_score=0.05 表示只买波动率<=5%的票。
                    无票过线则空仓等待——排名制不凑数。
      2. 计数模式   ``count_mode``：top_n 固定只数 / top_pct 按候选池比例。
      3. 数量约束   ``min_positions`` / ``max_positions``。
      4. 行业分组   ``industry_cap``：每行业最多 N 只（依赖 builder 的 map）。
      5. 权重       当前等权；扩展点。
      6. 弱市叠加   ``regime_adx``/``regime_scale``：市场 ADX 中位数低于
                    阈值时目标权重乘以 regime_scale。
      7. 波动率目标 ``vol_target_annual``：σ_p（持仓等权组合滚动已实现波动，
                    年化）高于目标时按 target/σ_p 降仓（scale clip 到
                    [vol_target_lo, hi]）；None=功能关，恒等。
    """

    count_mode: str = "top_n"          # top_n | top_pct
    top_n: int = 3
    pct: float = 0.10
    min_positions: int = 1
    max_positions: int | None = None
    ascending: bool = False            # 门控方向语义需要
    min_score: float | None = None
    industry_cap: int | None = None
    regime_adx: float | None = None
    regime_scale: float = 0.5
    # ── 波动率目标制仓位（vol targeting，默认 None=关，行为与旧版一致）──
    vol_target_annual: float | None = None   # 目标年化波动率；None=功能关
    vol_target_lo: float = 0.3               # 缩放系数下限
    vol_target_hi: float = 1.5               # 缩放系数上限（低波时加仓封顶）
    vol_target_window: int = 60              # σ_p 估计窗口（交易日）

    def quality(self, scores) -> np.ndarray:
        """方向归一化的质量分：越大越好（买小时取负）。"""
        s = np.asarray(scores, dtype=float)
        return -s if self.ascending else s

    def gate(self, candidates, scores) -> np.ndarray:
        """绝对阈值门控：保留质量分 >= min_score 的候选（不过线宁缺毋滥）。"""
        cand = np.asarray(candidates)
        if self.min_score is None or len(cand) == 0:
            return cand
        q = self.quality(scores)[:len(cand)]
        keep = np.isfinite(q) & (q >= float(self.min_score))
        return cand[keep]

    def scale_for_regime(self, market_adx) -> float:
        """弱势环境下的权重缩放系数（1.0 = 不减仓）。"""
        if (self.regime_adx is None or market_adx is None
                or not np.isfinite(market_adx)):
            return 1.0
        return float(self.regime_scale) if market_adx < self.regime_adx else 1.0

    def vol_scale_for(self, sigma_p) -> float:
        """波动率目标缩放系数：clip(vol_target_annual / σ_p, lo, hi)。

        σ_p > 目标 → scale<1 降仓；σ_p < 目标 → 加仓但封顶 hi。
        功能关（vol_target_annual=None）或 σ_p 无效（None/NaN/≤0——含
        零波动与窗口不足的安全退化值）→ 1.0（恒等）。
        """
        if self.vol_target_annual is None or sigma_p is None:
            return 1.0
        sigma_p = float(sigma_p)
        if not np.isfinite(sigma_p) or sigma_p <= 0.0:
            return 1.0
        return float(np.clip(self.vol_target_annual / sigma_p,
                             self.vol_target_lo, self.vol_target_hi))


@dataclass
class PortfolioBuilder:
    codes: list[str]
    industry_map: dict[str, str] | None = None
    industry_cap: int | None = None
    # ── vol targeting 数据通道（engine 路径由 prepare 注入，默认 None=不启用）──
    # close_history：收盘价宽表（全窗口）；signal_dates：调仓信号日（升序）。
    # build_targets 每次调用按顺序消费一个信号日，切片出"截至该信号日"的
    # 窗口估 σ_p——消费日期 ≤ 真实信号日，故无前视。已知局限：某信号日
    # 无有效候选时 simulate 跳过调用，该信号日由后续调用按序补位消费，
    # σ_p 截止日可能滞后若干个调仓期（只会滞后、绝不前视）；滞后到窗口
    # 不足时安全退化为 scale=1。显式传 close_wide 可绕过该通道。
    close_history: pd.DataFrame | None = None
    signal_dates: list | None = None
    _vol_cursor: int = field(default=0, init=False, repr=False, compare=False)

    def rank_select(
        self,
        candidates: np.ndarray,
        scores: np.ndarray,
        ascending: bool,
        top_n: int,
        selection_mode: str = "top_n",
        selection_pct: float = 0.10,
        min_positions: int = 1,
        max_positions: int | None = None,
        limit_count: int | None = None,
    ) -> list[int]:
        """按信号排序并应用行业约束，返回原始列索引。"""
        if len(candidates) == 0:
            return []
        order = np.argsort(scores, kind="mergesort")
        if not ascending:
            order = order[::-1]
        ordered = [int(candidates[o]) for o in order]

        if self.industry_cap and self.industry_map:
            selected: list[int] = []
            counts: dict[str, int] = {}
            for k in ordered:
                industry = self.industry_map.get(str(self.codes[k]), "?")
                if counts.get(industry, 0) >= self.industry_cap:
                    continue
                counts[industry] = counts.get(industry, 0) + 1
                selected.append(k)
                if limit_count is not None and len(selected) >= limit_count:
                    break
            ordered = selected

        count = limit_count if limit_count is not None else self.selection_count(
            len(ordered), top_n, selection_mode, selection_pct,
            min_positions, max_positions)
        return ordered[:count] if count > 0 else []

    @staticmethod
    def selection_count(
        candidate_count: int,
        top_n: int,
        selection_mode: str = "top_n",
        selection_pct: float = 0.10,
        min_positions: int = 1,
        max_positions: int | None = None,
    ) -> int:
        if candidate_count <= 0:
            return 0
        if selection_mode == "top_pct":
            pct = min(max(float(selection_pct), 0.001), 1.0)
            count = int(np.ceil(candidate_count * pct))
        else:
            count = int(top_n)
        count = max(int(min_positions), count)
        if max_positions is not None and int(max_positions) > 0:
            count = min(count, int(max_positions))
        return min(candidate_count, count)

    @staticmethod
    def equal_weights(selected: list[int], scale: float = 1.0) -> dict[int, float]:
        if not selected:
            return {}
        weight = float(scale) / len(selected)
        return {k: weight for k in selected}

    def build_targets(
        self,
        policy: SelectionPolicy,
        candidates,
        scores,
        market_adx=None,
        buffer_keep: set[int] | None = None,
        buffer_ratio: float = 0.0,
        close_wide: pd.DataFrame | None = None,
    ) -> tuple[list[int], dict[int, float]]:
        """选股流水线唯一入口：门控 → 排名计数 → 等权 → 弱市/波动率叠加。

        candidates/scores 为同序数组（scores 对应候选的因子值）。
        返回 (chosen_list, targets)；无票过门控时返回 ([], {})。

        buffer_keep/buffer_ratio（P1-1 Buffer Zone，spec §4.1）：老持仓在 top-(N+M) 内
        则保留，空缺从严格 top-N 补足。buffer_keep 为老持仓 code_idx 集合，
        buffer_ratio=0.0 或 buffer_keep=None 时退化为严格 top-N（向后兼容）。

        vol targeting（policy.vol_target_annual 非 None 时启用）：σ_p 用持仓
        等权组合的滚动已实现波动（core.portfolio.portfolio_vol）估计，
        vol_scale = clip(target/σ_p, lo, hi)，与弱市 regime_scale 相乘后整体
        clip 到 [vol_target_lo, vol_target_hi]。close_wide 为显式传入的收盘价
        窗口（须已截到信号日，供库外调用/单测使用，优先于 builder 通道）；
        缺省走 builder 的 close_history+signal_dates 通道（prepare 注入，按
        调用顺序消费信号日切片，无前视）。
        """
        cand = np.asarray(candidates)
        sc = np.asarray(scores, dtype=float)
        gated = policy.gate(cand, sc)
        if len(gated) == 0:
            return [], {}
        g_scores = sc[np.isin(cand, gated)]
        chosen = self.rank_select(
            gated, g_scores, policy.ascending,
            policy.top_n, policy.count_mode, policy.pct,
            policy.min_positions, policy.max_positions,
        )
        # ── P1-1 Buffer Zone：老持仓在 top-(N+M) 内则保留 ──
        if buffer_keep and buffer_ratio > 0.0 and chosen:
            long_n = len(chosen)
            m_buf = int(round(long_n * float(buffer_ratio)))
            if m_buf > 0:
                # 重算 top-(N+M) 全序（与 rank_select 同序：argsort + ascending）
                order = np.argsort(g_scores, kind="mergesort")
                if not policy.ascending:
                    order = order[::-1]
                ordered_all = [int(gated[o]) for o in order]
                buf_zone = set(ordered_all[:long_n + m_buf])
                # 老持仓在 top-(N+M) 内的都保留（按全序排序，高分优先）。
                # 注意：必须遍历 ordered_all 而非 chosen——chosen 是 top-N，
                # 老持仓里排名 N+1~N+M 的票在 buf_zone 内但不在 chosen，需拉回来。
                old_in_buffer = [k for k in ordered_all
                                 if k in buffer_keep and k in buf_zone]
                deficit = long_n - len(old_in_buffer)
                if deficit > 0:
                    # 从严格 top-N 补足不在 old_in_buffer 的
                    refill = [k for k in ordered_all[:long_n]
                              if k not in old_in_buffer]
                    chosen = old_in_buffer + refill[:deficit]
                else:
                    chosen = old_in_buffer[:long_n]
        targets = self.equal_weights(chosen)
        scale = policy.scale_for_regime(market_adx)
        if policy.vol_target_annual is not None:
            hist = close_wide
            if hist is None and self.close_history is not None:
                # 引擎通道：每次调用对应一个调仓信号日，消费切片"截至当日"
                # 的窗口（无前视）；空选股调用也消费，保持游标与信号日对齐。
                hist = self._consume_signal_slice()
            if targets:
                sigma = (portfolio_vol(hist, window=policy.vol_target_window,
                                       codes=[self.codes[k] for k in chosen])
                         if hist is not None else float("nan"))
                vol_scale = policy.vol_scale_for(sigma)
                if vol_scale != 1.0:
                    # vol targeting 有效（σ_p 可用）：regime × vol 相乘后整体
                    # clip 到 [vol_target_lo, vol_target_hi]。
                    # σ_p 无效时 vol_scale_for 恒返回 1.0（安全退化语义），
                    # 跳过 clip——否则弱市 regime_scale 已低于 lo 时会被 clip
                    # 强制抬仓，违背"σ_p 无效→不干预"的退化语义
                    # （OCR 2026-10-09 low finding）。
                    scale = float(np.clip(scale * vol_scale,
                                          policy.vol_target_lo,
                                          policy.vol_target_hi))
        if scale != 1.0 and targets:
            targets = {k: v * scale for k, v in targets.items()}
        return chosen, targets

    def _consume_signal_slice(self) -> pd.DataFrame | None:
        """按调用顺序消费下一个调仓信号日，返回截至该日的收盘价窗口。

        signal_dates 未注入（None/空）时返回 None——拒绝在全窗口上估 σ_p，
        避免把未来数据泄漏进早期调仓（前视）；消费位置越过末尾时钳制在
        最后一个信号日（方向安全：最多滞后，不会前视）。
        """
        if not self.signal_dates:
            return None
        pos = min(self._vol_cursor, len(self.signal_dates) - 1)
        self._vol_cursor += 1
        return self.close_history.loc[:self.signal_dates[pos]]
