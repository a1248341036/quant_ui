# -*- coding: utf-8 -*-
"""模块 07 · operator_catalog：可用算子清单（机制分组 + 摘要截断瘦身版）。"""

from alphaagent.dsl.catalog import operator_catalog_markdown

NAME = "operator_catalog"
TITLE = "可用算子目录"
ORDER = 70
REQUIRED = False
SEP_BEFORE = "\n\n---\n\n"


def render(ctx) -> str:  # noqa: ANN001
    catalog = operator_catalog_markdown() if ctx.include_operator_catalog else "（本次未注入算子清单）"
    return f"""### 可用算子

算子均为**大写**（如 `TS_MEAN`、`DELTA`）。支持位置参数，也支持关键字参数语法 `name=value`（关键字参数必须在位置参数之后）。按机制分节列出；签名省略类型标注，**参数顺序即语义**（位置传参必须严格按签名顺序）；语义自明的基础四则/比较/初等函数折叠在末行。选算子前先想机制（见「A 股市场机制与 alpha 分布」），再按节定位。

{catalog}"""
