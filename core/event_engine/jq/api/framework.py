# -*- coding: utf-8 -*-
"""策略程序架构 / 运行时间(聚宽文档类别): 定时任务注册与排序。

run_daily/run_weekly/run_monthly 注册 (kind, func, arg, time_key, seq);
执行顺序由 runtime.run_day 按 sched_sort_key 排序:
'before_open' < 'HH:MM' < 'after_close', 同时刻保持注册顺序。
reference_security 等聚宽专有参数接受但忽略。
"""
from __future__ import annotations


def sched_sort_key(entry):
    """(kind, func, arg, time_key, seq) -> 排序键。"""
    t, seq = entry[3], entry[4]
    if t == "before_open":
        return (0, 0, seq)
    if t == "after_close":
        return (3, 0, seq)
    if t in ("open", "every_bar"):
        return (1, 9 * 60 + 30, seq)
    try:
        hh, mm = str(t).split(":")
        return (1, int(hh) * 60 + int(mm), seq)
    except Exception:
        return (1, 9 * 60 + 30, seq)


def install(ns: dict, rt) -> None:
    def _schedule(kind, func, arg, time_key):
        rt.scheduled.append((kind, func, arg, str(time_key),
                             len(rt.scheduled)))

    def run_daily(func, time="9:30", **kwargs):
        _schedule("daily", func, None, time)

    def run_weekly(func, weekday=1, time="10:00", **kwargs):
        # JQ 语义: weekday=本周第 N 个交易日(1=周内首个交易日, 负数=倒数)
        _schedule("weekly", func, int(weekday), time)

    def run_monthly(func, monthday=None, time="9:30", **kwargs):
        # monthday: None=每月首个交易日, 正N=第N个交易日, 负N=倒数第N个
        _schedule("monthly", func, monthday, time)

    def unschedule_all():
        rt.scheduled.clear()

    ns.update({
        "run_daily": run_daily,
        "run_weekly": run_weekly,
        "run_monthly": run_monthly,
        "unschedule_all": unschedule_all,
    })
