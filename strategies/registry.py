from __future__ import annotations

# 策略 = 因子 + 排序方向
# ascending=True 表示买因子值最小的一批（如低波动/低换手/超跌反转）
# ascending=False 表示买因子值最大的一批（如高动量/高成交额）
STRATEGIES = {
    "低换手冷门": {"factor": "turn20", "ascending": True},
    "高成交领涨": {"factor": "am20", "ascending": False},
    "低成交冷门": {"factor": "am20", "ascending": True},
    "冷门+行业分散": {"factor": "am20", "ascending": True, "industry_cap": 1},
    "动量 20 日": {"factor": "mom20", "ascending": False},
    "动量 60 日": {"factor": "mom60", "ascending": False},
    "反转 20 日": {"factor": "mom20", "ascending": True},
    "低波动": {"factor": "vol20", "ascending": True},
    "复合因子": {"factor": "composite", "ascending": True},
}


def list_strategies() -> list[str]:
    return list(STRATEGIES.keys())


def get_strategy(name: str) -> dict:
    return STRATEGIES[name]
