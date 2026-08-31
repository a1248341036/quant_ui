# -*- coding: utf-8 -*-
"""策略组合操作(聚宽文档类别): set_subportfolios/transfer_cash。

引擎为单账户口径: set_subportfolios 仅接受单个股票账户(id=0),
transfer_cash 无资金划转语义(占位)。
"""
from __future__ import annotations


def install(ns: dict, rt) -> None:
    def set_subportfolios(subportfolios):
        n = len(subportfolios or [])
        if n > 1:
            rt.log.warn("[runtime] 多账户(subportfolios>1)暂不支持, "
                        "仅保留 id=0 主账户")
        return None

    def transfer_cash(target_subportfolio, cash):
        rt.log.warn("[runtime] transfer_cash 单账户口径为空操作")
        return None

    ns.update({
        "set_subportfolios": set_subportfolios,
        "transfer_cash": transfer_cash,
    })
