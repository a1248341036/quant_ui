"""统一交易参数配置 —— 全平台唯一默认值来源。

所有回测、模拟盘、因子门禁、前端默认值均从这里取。
以后改参数只需改这一个文件。

散户口径（10万级资金）：
  - 佣金：买入万2.5 (0.00025)、卖出万12.5 (0.00125，含印花税万10+佣金万2.5)
  - 滑点：3bps
  - 参与率：10%（小资金无流动性压力）
  - 选股：top 0.4% ≈ 20只
  - 流动性下限：500万（20日均成交额）
  - 资金：10万
"""
from __future__ import annotations

# ── 核心交易参数 ──
CAPITAL: float = 100_000.0          # 初始资金（10万）
BUY_COST: float = 0.00025          # 买入佣金（万2.5）
SELL_COST: float = 0.00125         # 卖出费率（万12.5，含印花税万10+佣金万2.5）
SLIPPAGE_BPS: float = 3.0          # 固定滑点（基点）
MAX_PARTICIPATION: float = 0.10    # 单笔买入 <= 20日均成交额 × 此比例
LOT_SIZE: int = 100                # 整手股数
AMOUNT_Q: float = 0.2             # am20 成交额分位过滤
WARMUP_DAYS: int = 400            # 因子预热天数
CASH_MODE: bool = True             # 现金/整手执行模型
LIMIT_FLAGS: bool = True            # 涨跌停过滤

# ── 选股参数 ──
SELECTION_MODE: str = "top_pct"    # 动态百分比选股
SELECTION_PCT: float = 0.004       # top 0.4% ≈ 20只
TOP_N: int = 20                    # 固定模式下选 20 只
FREQ: str = "weekly"               # 默认调仓频率

# ── 流动性硬过滤 ──
MIN_AM20_YUAN: float = 5_000_000.0  # 20日均成交额下限（500万）

# ── 回测对比参数（多策略对比页面）──
COMPARE_CAPITAL: float = 100_000.0  # 对比页资金
COMPARE_TOP_N: int = 3              # 对比页每策略选 3 只

# ── ETF / 基金覆盖（当用户未显式指定时）──
ETF_BUY_COST: float = 0.0003       # ETF 买入费率（万3）
ETF_SELL_COST: float = 0.0003      # ETF 卖出费率（万3）
ETF_SPREAD_BPS: float = 2.0        # ETF 买卖价差
ETF_MIN_COMMISSION: float = 5.0    # ETF 最低佣金（元）
FUND_BUY_COST: float = 0.0015      # 场外基金申购费率
FUND_SELL_COST: float = 0.0050     # 场外基金赎回费率

# ── AlphaAgent 因子门禁专用 ──
GATE_CAPITAL: float = 100_000.0     # 门禁回测资金
GATE_SELECTION_PCT: float = 0.004  # 门禁选股百分比
GATE_TOP_N: int = 20                # 门禁固定选股数
GATE_SLIPPAGE_BPS: float = 3.0     # 门禁滑点
GATE_MAX_PARTICIPATION: float = 0.10  # 门禁参与率
GATE_MIN_AM20_YUAN: float = 5_000_000.0  # 门禁流动性下限
GATE_MIN_EXCESS_ANNUAL: float = 0.03     # 净值超额年化下限（+3%）
GATE_MIN_EXCESS_SHARPE: float = 0.5     # 超额夏普下限
GATE_MAX_DRAWDOWN: float = 0.40         # 最大回撤上限
GATE_MIN_DAILY_OVERLAP: float = 0.5     # 日换手稳定性下限
GATE_MIN_INVESTED_RATIO: float = 0.8    # 仓位利用率下限
GATE_FREQ: str = "weekly"               # 门禁默认调仓频率


def defaults() -> dict:
    """返回全量默认参数字典，供 API model 默认值、前端初始化等场景使用。"""
    return {
        "capital": CAPITAL,
        "buy_cost": BUY_COST,
        "sell_cost": SELL_COST,
        "slippage_bps": SLIPPAGE_BPS,
        "max_participation": MAX_PARTICIPATION,
        "lot_size": LOT_SIZE,
        "amount_q": AMOUNT_Q,
        "warmup_days": WARMUP_DAYS,
        "cash_mode": CASH_MODE,
        "limit_flags": LIMIT_FLAGS,
        "selection_mode": SELECTION_MODE,
        "selection_pct": SELECTION_PCT,
        "top_n": TOP_N,
        "freq": FREQ,
        "min_am20_yuan": MIN_AM20_YUAN,
        # ETF
        "etf_buy_cost": ETF_BUY_COST,
        "etf_sell_cost": ETF_SELL_COST,
        "etf_spread_bps": ETF_SPREAD_BPS,
        "etf_min_commission": ETF_MIN_COMMISSION,
        # 基金
        "fund_buy_cost": FUND_BUY_COST,
        "fund_sell_cost": FUND_SELL_COST,
        # 对比页
        "compare_capital": COMPARE_CAPITAL,
        "compare_top_n": COMPARE_TOP_N,
        # 门禁
        "gate_capital": GATE_CAPITAL,
        "gate_selection_pct": GATE_SELECTION_PCT,
        "gate_top_n": GATE_TOP_N,
        "gate_slippage_bps": GATE_SLIPPAGE_BPS,
        "gate_max_participation": GATE_MAX_PARTICIPATION,
        "gate_min_am20_yuan": GATE_MIN_AM20_YUAN,
        "gate_min_excess_annual": GATE_MIN_EXCESS_ANNUAL,
        "gate_min_excess_sharpe": GATE_MIN_EXCESS_SHARPE,
        "gate_max_drawdown": GATE_MAX_DRAWDOWN,
        "gate_min_daily_overlap": GATE_MIN_DAILY_OVERLAP,
        "gate_min_invested_ratio": GATE_MIN_INVESTED_RATIO,
        "gate_freq": GATE_FREQ,
    }


def gate_policy() -> dict:
    """返回 engine_gate 专用的 policy 字典。"""
    return {
        "enabled": True,
        "selection_mode": SELECTION_MODE,
        "selection_pct": GATE_SELECTION_PCT,
        "top_n": GATE_TOP_N,
        "capital": GATE_CAPITAL,
        "slippage_bps": GATE_SLIPPAGE_BPS,
        "max_participation": GATE_MAX_PARTICIPATION,
        "min_am20_yuan": GATE_MIN_AM20_YUAN,
        "min_excess_annual": GATE_MIN_EXCESS_ANNUAL,
        "min_excess_sharpe": GATE_MIN_EXCESS_SHARPE,
        "max_drawdown": GATE_MAX_DRAWDOWN,
        "min_daily_overlap": GATE_MIN_DAILY_OVERLAP,
        "min_invested_ratio": GATE_MIN_INVESTED_RATIO,
        "freq": GATE_FREQ,
    }
