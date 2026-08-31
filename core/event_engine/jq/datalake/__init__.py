# -*- coding: utf-8 -*-
"""聚宽兼容层数据插件: 协议 + 注册中心 + 进程级缓存。

扩展方法(与 alphaagent 数据插件同风格): 在 plugins/ 下新增一个
非下划线开头的 .py 文件:

    from core.event_engine.jq.datalake.base import JQDataPlugin

    PLUGIN = JQDataPlugin(
        name="valuation_metrics",
        description="估值指标(pe/pb/ps, 逐日点时)",
        date_column="trade_date",
        entity_keys=("code",),
    )

    def load() -> pd.DataFrame:
        ...

即可通过 datalake.load("valuation_metrics") / datalake.asof(name, date)
使用; 未注册的数据集 load() 抛出带扩展指引的 NotImplementedError,
API 层据此给出清晰的"缺数据"报错而非裸 KeyError。
"""
from __future__ import annotations

from .base import (JQDataPlugin, QDATA, PG, CNE_CURATED, ROOT, asof,  # noqa: F401
                   available, clear_cache, discover, get, load, register,
                   status)

discover()
