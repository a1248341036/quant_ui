"""引擎因子元数据注册表（单一事实来源）。

统一登记 core.engine.build_factor_frames 能产生的全部因子：
量价因子、基金净值专属因子、财务因子与动态因子。消费方：

- core.composites.FACTOR_OPTIONS：组合编辑器的可组合因子清单（由本表派生）
- strategies.registry.STRATEGIES：策略引用的因子名校验（防止拼写静默失效）
- 前端 /api/backtest/factors：直接返回 options

加新因子 = 在 FACTORS 表加一行 + 在 build_factor_frames 实现计算。
不要在别处再维护一份因子清单。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class FactorSpec:
    """一个可回测因子的元数据。

    name: 引擎因子名（build_factor_frames 的 key / 策略 factor 字段）。
    label/desc: 前端展示。
    types: 适用资产类型（stock/etf/fund）。
    financial: 是否依赖财务数据宽表（PG/FINANCIAL_FACTORS）。
    composable: 是否可作为自定义组合（composite）的成员因子。
        外部注入型（pred）与组合型（composite）不算。
    """

    name: str
    label: str
    desc: str
    types: tuple[str, ...]
    financial: bool = False
    composable: bool = True


FACTORS: dict[str, FactorSpec] = {
    # ── 量价因子（股票/ETF/基金通用）──────────────────────────────
    "mom20": FactorSpec("mom20", "动量(20日)", "20 日涨幅",
                        ("stock", "etf", "fund")),
    "mom60": FactorSpec("mom60", "动量(60日)", "60 日涨幅",
                        ("stock", "etf", "fund")),
    "vol20": FactorSpec("vol20", "低波动(20日)", "20 日波动率",
                        ("stock", "etf", "fund")),
    "ma_cross5_10": FactorSpec("ma_cross5_10", "双均线 5/10",
                               "MA5 相对 MA10 乖离", ("stock", "etf", "fund")),
    "ma_cross5_20": FactorSpec("ma_cross5_20", "双均线 5/20",
                               "MA5 相对 MA20 乖离", ("stock", "etf", "fund")),
    "ma_cross10_30": FactorSpec("ma_cross10_30", "双均线 10/30",
                                "MA10 相对 MA30 乖离", ("stock", "etf", "fund")),
    "ma_cross20_30": FactorSpec("ma_cross20_30", "双均线 20/30",
                                "MA20 相对 MA30 乖离", ("stock", "etf", "fund")),
    "ma_cross20_60": FactorSpec("ma_cross20_60", "双均线 20/60",
                                "MA20 相对 MA60 乖离", ("stock", "etf", "fund")),
    "brk20": FactorSpec("brk20", "趋势突破(20日)", "收盘创20日新高强度",
                        ("stock", "etf", "fund")),
    "brk20_vol": FactorSpec("brk20_vol", "放量突破(20日)",
                            "量比≥1.5 的 20 日新高突破", ("stock", "etf")),
    # ── 股票/ETF 专属（基金面板 am20/turn20 恒 1.0 无区分度）────────
    "turn20": FactorSpec("turn20", "低换手(20日)", "20 日平均换手率",
                         ("stock", "etf")),
    "am20": FactorSpec("am20", "成交额(20日)", "20 日平均成交额（资金流）",
                       ("stock", "etf")),
    # ── 复合/动态因子（不作为组合成员）──────────────────────────
    "composite": FactorSpec("composite", "复合因子",
                            "低成交 + 低波动复合打分",
                            ("stock", "etf", "fund"),
                            composable=False),
    "pred": FactorSpec("pred", "ML 预测分数", "qweave LightGBM 预测分数",
                       ("stock",), composable=False),
    # ── 基金净值专属因子 ───────────────────────────────────────
    "mdd20": FactorSpec("mdd20", "低回撤(20日)", "滚动 20 日最大回撤",
                        ("fund",)),
    "mdd60": FactorSpec("mdd60", "低回撤(60日)", "滚动 60 日最大回撤",
                        ("fund",)),
    "sharpe20": FactorSpec("sharpe20", "高夏普(20日)", "滚动 20 日夏普比率",
                           ("fund",)),
    "sharpe60": FactorSpec("sharpe60", "高夏普(60日)", "滚动 60 日夏普比率",
                           ("fund",)),
    "sortino20": FactorSpec("sortino20", "高Sortino(20日)",
                            "滚动 20 日 Sortino 比率", ("fund",)),
    "mom_accel": FactorSpec("mom_accel", "动量加速度", "短期动量 - 长期动量",
                            ("fund",)),
    "nav_stability": FactorSpec("nav_stability", "净值稳定性",
                                "滚动 60 日净值线性拟合 R²", ("fund",)),
    # ── 财务因子（仅股票）──────────────────────────────────────
    "pb": FactorSpec("pb", "市净率PB(低)", "收盘/BPS，财务因子，需 PG 财务宽表",
                     ("stock",), financial=True),
    "ep": FactorSpec("ep", "盈利收益率EP(高)", "EPS/收盘，财务因子，需 PG 财务宽表",
                     ("stock",), financial=True),
    "roe": FactorSpec("roe", "ROE(高)", "加权 ROE，财务因子，需 PG 财务宽表",
                      ("stock",), financial=True),
    "gross_margin": FactorSpec("gross_margin", "毛利率(高)",
                               "毛利率，财务因子，需 PG 财务宽表",
                               ("stock",), financial=True),
    "rev_yoy": FactorSpec("rev_yoy", "营收同比(高)",
                          "营收同比增速，财务因子，需 PG 财务宽表",
                          ("stock",), financial=True),
    "np_yoy": FactorSpec("np_yoy", "净利同比(高)",
                         "净利同比增速，财务因子，需 PG 财务宽表",
                         ("stock",), financial=True),
}


def factor_options() -> list[dict]:
    """组合编辑器可用的因子选项（FACTOR_OPTIONS 形状）。

    只返回 composable=True 的因子，保持与旧 core.composites.FACTOR_OPTIONS
    完全一致的输出契约。
    """
    out = []
    for spec in FACTORS.values():
        if not spec.composable:
            continue
        out.append({
            "name": spec.name,
            "label": spec.label,
            "desc": spec.desc,
            "types": list(spec.types),
        })
    return out


def known_factor(name: str) -> bool:
    return name in FACTORS


def validate_factor_refs(factor_names: Iterable[str]) -> list[str]:
    """返回输入中不在注册表里的因子名（空列表 = 全部合法）。"""
    return [f for f in factor_names if not known_factor(str(f))]


def factor_spec(name: str) -> FactorSpec | None:
    return FACTORS.get(name)