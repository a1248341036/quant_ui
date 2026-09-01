# -*- coding: utf-8 -*-
"""聚宽策略 API 预检: AST 静态扫描, 秒级发现兼容层缺失的 API/字段。

只检查两类硬缺失(其余交给运行时):
1. 策略调用了、但兼容层命名空间不存在的"裸函数调用"
   (排除方法调用/局部变量/内置名)
2. query DSL 的 valuation/income/indicator.<字段> 是否已实现

用法:
  python scripts/jq_repro/_preflight_api.py <策略.py> [--extract]
"""
from __future__ import annotations

import ast
import builtins
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class _AnyRt:
    """万能假 runtime: install() 的立即绑定(如 rt.get_price)都给占位 lambda。"""

    def __init__(self):
        self.cost_cfg = {}
        self.scheduled = []
        self.log = SimpleNamespace(info=lambda *a: None,
                                   debug=lambda *a: None,
                                   warn=lambda *a: None,
                                   error=lambda *a: None,
                                   set_level=lambda *a: None)
        self.benchmark = None
        self.capital = 1e6
        self.context = SimpleNamespace(
            portfolio=SimpleNamespace(
                positions=SimpleNamespace(keys=lambda: [], items=lambda: [],
                                          values=lambda: []),
                cash=0.0, available_cash=0.0, total_value=0.0, returns=0.0,
                inout_cash=0.0, in_out_cash=0.0, positions_value=0.0,
                locked_cash=0.0, start_date=None),
            previous_date=None, current_dt=None, subportfolios=[],
            run_params={})

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: SimpleNamespace()


def build_reference_namespace() -> dict:
    """构建兼容层"参考命名空间"(不触发数据加载)。"""
    from core.event_engine.jq.api import (data_api, framework, misc,
                                          portfolio, settings, trading)
    from core.event_engine.jq import query as q

    ns: dict = {
        "g": SimpleNamespace(), "log": _AnyRt().log,
        "context": None,
        "datetime": __import__("datetime"),
        "timedelta": __import__("datetime").timedelta,
        "date": __import__("datetime").date,
        "time": __import__("datetime").time,
        "OrderStatus": SimpleNamespace(created=1, opened=1, held=1,
                                       canceled=1, rejected=1),
        "np": __import__("numpy"), "numpy": __import__("numpy"),
        "pd": __import__("pandas"),
    }
    dummy = _AnyRt()
    for mod in (framework, settings, trading, data_api, portfolio, misc):
        try:
            mod.install(ns, dummy)
        except Exception:
            # install 失败(需真实数据上下文)时, 以模块级函数名作为 API 面
            import inspect
            for name, _ in inspect.getmembers(mod, inspect.isfunction):
                if not name.startswith("_"):
                    ns.setdefault(name, None)
    ns["valuation"] = q.valuation
    ns["income"] = q.income
    ns["indicator"] = q.indicator
    # 聚宽内置类型(兼容层恒提供)
    ns["MarketOrderStyle"] = ns.get("MarketOrderStyle")
    ns["LimitOrderStyle"] = ns.get("LimitOrderStyle")
    ns["FixedSlippage"] = ns.get("FixedSlippage")
    ns["PriceRelatedSlippage"] = ns.get("PriceRelatedSlippage")
    ns["OrderCost"] = ns.get("OrderCost")
    ns["Order"] = None
    ns["SubPortfolio"] = None
    return ns


# 这些名字是"对象的方法/属性"而非 API(由运行时对象提供, 预检不判)
_ATTR_ONLY_ROOTS = {"portfolio", "positions", "current_data", "df",
                    "ptable", "position", "stock_df", "s", "p", "ps"}


def _collect(code: str):
    tree = ast.parse(code)
    called: set[str] = set()          # 裸函数调用名
    attrs: set[tuple[str, str]] = set()
    local: set[str] = set()           # 任意赋值/定义/导入/形参名

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            called.add(node.func.id)
        elif isinstance(node, ast.Attribute):
            root = node
            while isinstance(root, ast.Attribute):
                root = root.value
            if isinstance(root, ast.Name):
                attrs.add((root.id, node.attr))
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            local.add(node.name)
            for a in node.args.args:
                local.add(a.arg)
        elif isinstance(node, ast.Assign):
            for t in ast.walk(node.targets[0] if node.targets else node):
                if isinstance(t, ast.Name):
                    local.add(t.id)
        elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
            local.add(node.target.id)
        elif isinstance(node, ast.Import):
            for a in node.names:
                local.add((a.asname or a.name).split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                local.add(a.asname or a.name)
        elif isinstance(node, ast.arg):
            local.add(node.arg)
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            local.add(node.id)
        elif isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            local.add(node.target.id)
        elif isinstance(node, ast.ExceptHandler) and node.name:
            local.add(node.name)
    return called, attrs, local


def preflight(code: str) -> list[str]:
    """返回策略缺失的 API 名单(空 = 静态检查全部通过)。"""
    ns = build_reference_namespace()
    known = set(vars(builtins))
    called, attrs, local = _collect(code)
    known |= local | {n for n in ns}

    missing: list[str] = []
    # 1. 裸函数调用必须在兼容层/内置/局部存在
    for name in sorted(called):
        if name in known:
            continue
        if name in _ATTR_ONLY_ROOTS:
            continue
        missing.append(name)
    # 2. query 表字段(_Col 链式方法 in_/between/asc/desc 除外)
    tables = {"valuation": ns.get("valuation"),
              "income": ns.get("income"),
              "indicator": ns.get("indicator")}
    col_methods = {"in_", "between", "asc", "desc"}
    for root, attr in sorted(attrs):
        tb = tables.get(root)
        if tb is not None and attr not in col_methods \
                and not hasattr(tb, attr):
            missing.append(f"{root}.{attr} (字段不存在)")
    return missing


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print("用法: _preflight_api.py <策略.py> [--extract]")
        return 2
    code = Path(args[0]).read_text(encoding="utf-8")
    if "--extract" in args and "CODE = r'''" in code:
        code = code.split("CODE = r'''", 1)[1].rsplit("'''", 1)[0]
    missing = preflight(code)
    if missing:
        print("缺失 API/字段(秒级预检):")
        for m in missing:
            print("  -", m)
        return 1
    print("预检通过: 策略引用的 API 均已支持")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
