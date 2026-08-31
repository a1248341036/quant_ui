# -*- coding: utf-8 -*-
"""其他函数(聚宽文档类别) + 第三方模块桩。

- jqdata/jqfactor 模块桩: `from jqdata import *` 等导入不报错;
  finance.run_query 返回空表(审计意见等表无本地数据, 相关过滤恒通过)
- write_file/read_file/send_message: 占位 no-op
"""
from __future__ import annotations

import sys
import types

import pandas as pd

from core.event_engine.jq.query import _Col


def _setup_jq_modules() -> None:
    """注入 jqdata/jqfactor 兼容模块桩(幂等)。"""
    if "jqdata" in sys.modules:
        return
    jqdata = types.ModuleType("jqdata")
    # 审计意见等 finance 表无本地数据: 返回空表(相关过滤逻辑恒通过)
    jqdata.finance = types.SimpleNamespace(
        STK_AUDIT_OPINION=types.SimpleNamespace(
            code=_Col("code"), pub_date=_Col("pub_date"),
            report_type=_Col("report_type")),
        run_query=lambda *a, **k: pd.DataFrame(
            columns=["code", "pub_date", "report_type"]))
    jqfactor = types.ModuleType("jqfactor")
    sys.modules.setdefault("jqdata", jqdata)
    sys.modules.setdefault("jqfactor", jqfactor)


def install(ns: dict, rt) -> None:
    _setup_jq_modules()

    def write_file(f, content):
        # 回测沙箱文件占位: 聚宽云文件系统本地无对应物
        rt.log.warn("[runtime] write_file 暂不支持(本地回测无云文件系统)")
        return None

    def read_file(f):
        rt.log.warn("[runtime] read_file 暂不支持(本地回测无云文件系统)")
        return ""

    def send_message(message, title=None):
        rt.log.info(f"[runtime] send_message: {title or ''}{message}")
        return None

    ns.update({
        "write_file": write_file,
        "read_file": read_file,
        "send_message": send_message,
    })
