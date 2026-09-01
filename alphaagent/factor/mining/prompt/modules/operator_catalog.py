# -*- coding: utf-8 -*-
"""模块 07 · operator_catalog：可用算子清单（瘦身签名版，可整段卸载）。"""

from alphaagent.dsl.catalog import operator_catalog_markdown

NAME = "operator_catalog"
TITLE = "可用算子目录"
ORDER = 70
REQUIRED = False
SEP_BEFORE = "\n\n---\n\n"


def render(ctx) -> str:  # noqa: ANN001
    catalog = operator_catalog_markdown() if ctx.include_operator_catalog else "（本次未注入算子清单）"
    return f"""### 可用算子

算子均为**大写**（如 `TS_MEAN`、`DELTA`）。支持位置参数，也支持关键字参数语法 `name=value`（关键字参数必须在位置参数之后）。

{catalog}"""
