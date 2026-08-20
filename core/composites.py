from __future__ import annotations

"""多因子自由组合配置的持久化与因子选项。

组合 = {因子名: 权重} + {因子名: 方向}。权重可为负（反向暴露），
方向 True 表示该因子买低（rank 翻转），False 表示买高。
"""

import json
import os
import tempfile

from .store import DATA_DIR


COMPOSITES_FILE = DATA_DIR / "composites.json"


FACTOR_OPTIONS = [
    # 量价因子（股票/ETF/基金通用）
    {"name": "mom20", "label": "动量(20日)", "desc": "20 日涨幅",
     "types": ["stock", "etf", "fund"]},
    {"name": "mom60", "label": "动量(60日)", "desc": "60 日涨幅",
     "types": ["stock", "etf", "fund"]},
    {"name": "vol20", "label": "低波动(20日)", "desc": "20 日波动率",
     "types": ["stock", "etf", "fund"]},
    {"name": "ma_cross5_10", "label": "双均线 5/10", "desc": "MA5 相对 MA10 乖离",
     "types": ["stock", "etf", "fund"]},
    {"name": "ma_cross5_20", "label": "双均线 5/20", "desc": "MA5 相对 MA20 乖离",
     "types": ["stock", "etf", "fund"]},
    {"name": "ma_cross10_30", "label": "双均线 10/30", "desc": "MA10 相对 MA30 乖离",
     "types": ["stock", "etf", "fund"]},
    {"name": "ma_cross20_60", "label": "双均线 20/60", "desc": "MA20 相对 MA60 乖离",
     "types": ["stock", "etf", "fund"]},
    # 股票/ETF 专属（基金面板 am20/turn20 恒 1.0 无区分度）
    {"name": "turn20", "label": "低换手(20日)", "desc": "20 日平均换手率",
     "types": ["stock", "etf"]},
    {"name": "am20", "label": "成交额(20日)", "desc": "20 日平均成交额（资金流）",
     "types": ["stock", "etf"]},
    # 基金净值专属因子
    {"name": "mdd20", "label": "低回撤(20日)", "desc": "滚动 20 日最大回撤",
     "types": ["fund"]},
    {"name": "mdd60", "label": "低回撤(60日)", "desc": "滚动 60 日最大回撤",
     "types": ["fund"]},
    {"name": "sharpe20", "label": "高夏普(20日)", "desc": "滚动 20 日夏普比率",
     "types": ["fund"]},
    {"name": "sharpe60", "label": "高夏普(60日)", "desc": "滚动 60 日夏普比率",
     "types": ["fund"]},
    {"name": "sortino20", "label": "高Sortino(20日)", "desc": "滚动 20 日 Sortino 比率",
     "types": ["fund"]},
    {"name": "mom_accel", "label": "动量加速度", "desc": "短期动量 - 长期动量",
     "types": ["fund"]},
    {"name": "nav_stability", "label": "净值稳定性", "desc": "滚动 60 日净值线性拟合 R²",
     "types": ["fund"]},
    # 财务因子（仅股票）
    {"name": "pb", "label": "市净率PB(低)", "desc": "收盘/BPS，财务因子，需 PG 财务宽表",
     "types": ["stock"]},
    {"name": "ep", "label": "盈利收益率EP(高)", "desc": "EPS/收盘，财务因子，需 PG 财务宽表",
     "types": ["stock"]},
    {"name": "roe", "label": "ROE(高)", "desc": "加权 ROE，财务因子，需 PG 财务宽表",
     "types": ["stock"]},
    {"name": "gross_margin", "label": "毛利率(高)", "desc": "毛利率，财务因子，需 PG 财务宽表",
     "types": ["stock"]},
    {"name": "rev_yoy", "label": "营收同比(高)", "desc": "营收同比增速，财务因子，需 PG 财务宽表",
     "types": ["stock"]},
    {"name": "np_yoy", "label": "净利同比(高)", "desc": "净利同比增速，财务因子，需 PG 财务宽表",
     "types": ["stock"]},
]


def _atomic_write(data: str) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(DATA_DIR), prefix=".composites.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(data)
        os.replace(tmp, COMPOSITES_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def load_composites() -> dict[str, dict]:
    if not COMPOSITES_FILE.exists():
        return {}
    try:
        data = json.loads(COMPOSITES_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_composite(name: str, weights: dict, directions: dict | None = None) -> dict:
    if not name.strip():
        raise ValueError("组合名称不能为空")
    if not weights:
        raise ValueError("组合至少需要一个因子")
    composites = load_composites()
    item = {
        "name": name.strip(),
        "weights": {str(k): float(v) for k, v in weights.items()},
        "directions": {str(k): bool(v) for k, v in (directions or {}).items()},
    }
    composites[item["name"]] = item
    _atomic_write(json.dumps(composites, ensure_ascii=False, indent=2))
    return item


def delete_composite(name: str) -> bool:
    composites = load_composites()
    if name not in composites:
        return False
    del composites[name]
    _atomic_write(json.dumps(composites, ensure_ascii=False, indent=2))
    return True
