# -*- coding: utf-8 -*-
"""其他函数(聚宽文档类别) + 第三方模块桩 + 研究笔记本兼容。

- jqdata/jqfactor 模块桩: `from jqdata import *` 等导入不报错;
  finance.run_query 返回空表(审计意见等表无本地数据, 相关过滤恒通过)
- write_file/read_file/send_message: 占位 no-op
- display: 聚宽研究环境预载的展示函数
- legacy_exec_scope(): 用户代码 exec 期间的旧版环境兼容——
  * pandas 新版拒绝 display.max_colwidth=-1(聚宽研究为旧版 pandas), 翻译为 None
  * time.clock 已在 Python 3.8 移除(聚宽研究脚本常用), 提供 perf_counter 代理
"""
from __future__ import annotations

import sys
import time as _time
import types
from contextlib import contextmanager

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


@contextmanager
def legacy_exec_scope():
    """用户代码 exec 期间的旧版研究环境兼容(临时猴补, 用后还原)。

    1) pandas.set_option: display.max_colwidth 负值(旧版=-1 不限宽)翻译为 None
    2) sys.modules['time'] 代理: 补回 time.clock(=perf_counter), 其余透传
    """
    real_pd = pd
    real_time = sys.modules["time"]

    def _compat_set_option(*args, **kwargs):
        flat = []
        prev = None
        for a in args:
            if (prev == "display.max_colwidth" and isinstance(a, int)
                    and a < 0):
                a = None                       # 旧版 -1 不限宽 -> 新版 None
            flat.append(a)
            prev = a
        return real_pd.set_option(*flat, **kwargs)

    def _clock():
        return _time.perf_counter()

    pd_shim = types.ModuleType("pandas")
    pd_shim.__dict__.update({k: v for k, v in real_pd.__dict__.items()
                             if k != "set_option"})
    pd_shim.set_option = _compat_set_option    # type: ignore[attr-defined]

    time_shim = types.ModuleType("time")
    time_shim.__dict__.update({k: v for k, v in real_time.__dict__.items()
                               if k != "clock"})
    time_shim.clock = _clock                   # type: ignore[attr-defined]

    sys.modules["pandas"] = pd_shim
    sys.modules["time"] = time_shim
    try:
        yield
    finally:
        sys.modules["pandas"] = real_pd
        sys.modules["time"] = real_time


def install(ns: dict, rt) -> None:
    _setup_jq_modules()

    def display(*args, **kwargs):
        for a in args:
            print(a if isinstance(a, str) else repr(a))
        return None

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
        "display": display,
        "write_file": write_file,
        "read_file": read_file,
        "send_message": send_message,
    })
