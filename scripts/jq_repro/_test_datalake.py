# -*- coding: utf-8 -*-
"""数据插件体系验证: 自动发现/缓存/点时切片/门面等价/status 诊断。"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import pandas as pd  # noqa: E402

from core.event_engine.jq import datalake  # noqa: E402
import jq_data  # noqa: E402


def main() -> None:
    # 1) 自动发现
    names = datalake.available()
    print("已注册插件:", names)
    assert {"income", "index_bars", "industry_members", "stock_daily"} <= set(names)

    # 2) 加载 + 缓存(两次同一对象)
    inc = datalake.load("income")
    assert datalake.load("income") is inc, "income 缓存失效"
    print(f"income: {len(inc)} 行, ann {inc['ann_date'].min().date()}~"
          f"{inc['ann_date'].max().date()}")

    # 3) 点时切片: 行业(含 fallback), income 每股最新一期
    snap = datalake.asof("industry_members", "2025-06-17")
    assert len(snap) and snap["as_of_date"].max() <= pd.Timestamp("2025-06-17")
    dup = snap.duplicated(["code", "system"]).sum()
    assert dup == 0, f"industry asof 去重失败: {dup} 重复"
    per_code = datalake.asof("income", "2025-06-30")
    assert per_code["code"].is_unique
    print(f"industry asof: {len(snap)} 行; income asof: {len(per_code)} 只")

    # 4) 未注册插件 -> 带指引的报错
    try:
        datalake.load("dragon_tiger")
        raise SystemExit("应当抛错")
    except NotImplementedError as exc:
        print("未注册报错OK:", str(exc)[:50], "...")

    # 5) 门面等价(jq_data 委托同一缓存)
    assert jq_data.load_income() is inc
    assert jq_data.load_index_bars() is datalake.load("index_bars")
    assert len(jq_data.industry_asof("2025-06-17")) == len(snap)

    # 6) stock_daily 带参数缓存键
    raw_a, meta_a = datalake.load("stock_daily", start="2025-01-01",
                                  end="2025-03-31")
    raw_b, meta_b = datalake.load("stock_daily", start="2025-01-01",
                                  end="2025-03-31")
    assert raw_a is raw_b and meta_a is meta_b
    print(f"stock_daily: raw {raw_a.shape}, meta {meta_a.shape}")

    # 7) status 诊断表
    print(jq_data.data_status().to_string(index=False))
    print("数据插件体系验证通过")


if __name__ == "__main__":
    main()
