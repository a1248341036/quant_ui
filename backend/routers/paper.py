from __future__ import annotations

"""日级模拟盘 API。"""

import ast
from pathlib import Path

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel

from backend import services
from core.data import load_etf_panel
from core import paper as paper_core
from core.strategy_pool import resolve_strategy as resolve_pool_strategy


router = APIRouter()
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LABS_DIR = PROJECT_ROOT / "labs"


class AccountRequest(BaseModel):
    name: str
    strategy_name: str
    strategy_type: str = "factor"
    module: str | None = None
    event_strategy: str | None = None
    universe: str = "科技TMT"
    capital: float = 100000.0
    top_n: int = 3
    freq: str = "monthly"
    risk_config: dict | None = None
    start_date: str | None = None
    slippage_bps: float = 0.0
    max_participation: float = 0.0
    warmup_days: int | None = 400
    buy_cost: float = 0.0008
    sell_cost: float = 0.0013
    lot_size: int = 100
    limit_flags: bool = True


class StatusRequest(BaseModel):
    status: str = "active"


class StrategySwitchRequest(BaseModel):
    strategy_name: str
    universe: str | None = None
    top_n: int | None = None
    freq: str | None = None
    risk_config: dict | None = None


class RunRequest(BaseModel):
    account_id: int | None = None
    exec_date: str | None = None
    dry_run: bool = False


def _stock_codes_by_universe() -> dict[str, list[str]]:
    return {
        "科技TMT": services.build_codes("科技TMT", True),
        "沪深300+中证500+中证1000": services.build_codes("沪深300+中证500+中证1000", True),
    }


def _stock_panel(codes: list[str], end: str | None = None) -> pd.DataFrame:
    """股票模拟盘面板：只拉账户池代码 + 最近 800 天（覆盖 400 天因子预热）。"""
    end_ts = pd.Timestamp(end) if end else pd.Timestamp.today()
    start = (end_ts - pd.Timedelta(days=800)).date().isoformat()
    return services.load_data(start=start, end=end_ts.date().isoformat(),
                              codes=codes, need_panel=True,
                              need_heavy=False)["panel"]


def _panel_for_account(account_id: int) -> pd.DataFrame:
    acc = paper_core.get_account(account_id)
    if acc and acc.get("universe") == "ETF":
        end_ts = pd.Timestamp.today()
        start = (end_ts - pd.Timedelta(days=800)).date().isoformat()
        return load_etf_panel(start=start, end=end_ts.date().isoformat())
    universe = (acc or {}).get("universe", "科技TMT")
    codes = services.build_codes(universe, True)
    return _stock_panel(codes)


def _event_strategies_from_source(src: str) -> list[str]:
    """安全解析模块源码里的 EVENT_STRATEGIES = {...} 键名（不执行用户代码）。"""
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return []
    out: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        for t in node.targets:
            if (isinstance(t, ast.Name) and t.id == "EVENT_STRATEGIES"
                    and isinstance(node.value, ast.Dict)):
                for k in node.value.keys:
                    if isinstance(k, ast.Constant) and isinstance(k.value, str):
                        out.append(k.value)
    return out


@router.get("/paper/event-strategies")
def event_strategies():
    """列出代码实验室已保存模块中的事件策略，供创建事件账户选择。"""
    items = []
    if LABS_DIR.exists():
        for p in sorted(LABS_DIR.glob("*.py")):
            try:
                src = p.read_text(encoding="utf-8")
            except OSError:
                continue
            names = _event_strategies_from_source(src)
            if names:
                items.append({"module": str(p), "name": p.stem,
                              "strategies": names})
    return {"items": items}


@router.get("/paper/accounts")
def accounts():
    return paper_core._jsonable(paper_core.list_accounts())


@router.post("/paper/accounts")
def account_create(req: AccountRequest):
    try:
        if req.universe == "场外基金":
            return {"error": "场外基金模拟盘尚未接入，请先使用 ETF/股票池或回测验证基金策略"}
        risk = dict(req.risk_config or {})
        for key, val in (
            ("slippage_bps", req.slippage_bps),
            ("max_participation", req.max_participation),
            ("warmup_days", req.warmup_days),
            ("buy_cost", req.buy_cost),
            ("sell_cost", req.sell_cost),
            ("lot_size", req.lot_size),
            ("limit_flags", req.limit_flags),
        ):
            risk.setdefault(key, val)
        if req.strategy_type == "event":
            if not req.module:
                return {"error": "事件账户需要选择代码模块"}
            if not req.event_strategy:
                return {"error": "事件账户需要选择事件策略"}
            mod_path = Path(req.module)
            if not mod_path.is_file():
                return {"error": f"代码模块不存在: {req.module}"}
            src = mod_path.read_text(encoding="utf-8")
            names = _event_strategies_from_source(src)
            if req.event_strategy not in names:
                return {"error": f"模块中没有事件策略: {req.event_strategy}"
                                + (f"（可用: {', '.join(names)}）" if names else "")}
            acc = paper_core.create_account(
                name=req.name, strategy_name=req.event_strategy,
                factor="", ascending=False,
                universe=req.universe, capital=req.capital,
                top_n=req.top_n, freq=req.freq, risk_config=risk,
                strategy_type="event", module=str(mod_path),
                event_strategy=req.event_strategy, start_date=req.start_date,
            )
        else:
            try:
                s = resolve_pool_strategy(req.strategy_name)
            except KeyError:
                return {"error": f"未知策略: {req.strategy_name}"}
            for k in ("adx_filter", "chandelier_mult", "chandelier_period",
                      "regime_adx", "regime_scale"):
                if s.get(k) not in (None, ""):
                    risk[k] = (int(s[k]) if k == "chandelier_period"
                               else float(s[k]))
            acc = paper_core.create_account(
                name=req.name, strategy_name=req.strategy_name,
                factor=s["factor"], ascending=s["ascending"],
                universe=req.universe, capital=req.capital,
                top_n=req.top_n, freq=req.freq, risk_config=risk,
                strategy_type="factor",
            )
    except ValueError as exc:
        return {"error": str(exc)}
    except OSError as exc:
        return {"error": f"读取代码模块失败: {exc}"}
    return paper_core._jsonable(acc)


@router.patch("/paper/accounts/{account_id}")
def account_status(account_id: int, req: StatusRequest):
    try:
        acc = paper_core.set_account_status(account_id, req.status)
    except ValueError as exc:
        return {"error": str(exc)}
    return paper_core._jsonable(acc) or {"error": "账户不存在"}


@router.patch("/paper/accounts/{account_id}/strategy")
def account_switch_strategy(account_id: int, req: StrategySwitchRequest):
    """在线切换 factor 账户策略；切换后前端应再调 reset 清空旧策略历史。"""
    try:
        s = resolve_pool_strategy(req.strategy_name)
        if not s:
            return {"error": f"未知策略: {req.strategy_name}"}
        acc = paper_core.update_account_strategy(
            account_id,
            strategy_name=req.strategy_name,
            factor=s.get("factor"),
            ascending=s.get("ascending"),
            universe=req.universe,
            top_n=req.top_n,
            freq=req.freq,
            risk_config=req.risk_config,
        )
    except ValueError as exc:
        return {"error": str(exc)}
    return paper_core._jsonable(acc) or {"error": "账户不存在"}


@router.delete("/paper/accounts/{account_id}")
def account_delete(account_id: int):
    return {"ok": paper_core.delete_account(account_id)}


@router.post("/paper/accounts/{account_id}/reset")
def account_reset(account_id: int):
    return {"ok": paper_core.reset_account(account_id)}


@router.post("/paper/run")
def paper_run(req: RunRequest):
    try:
        accounts = paper_core.list_accounts()
        if req.account_id is not None:
            accounts = [a for a in accounts if a["id"] == int(req.account_id)]
        stock_ids = [a["id"] for a in accounts if a.get("universe") != "ETF"]
        etf_ids = [a["id"] for a in accounts if a.get("universe") == "ETF"]
        out_accounts: list[dict] = []
        run_date = req.exec_date
        if stock_ids:
            codes_map = _stock_codes_by_universe()
            all_codes = sorted(set().union(*codes_map.values()))
            panel = _stock_panel(all_codes, req.exec_date)
            res = paper_core.run_paper_trade(
                panel, codes_map,
                account_ids=stock_ids, exec_date=req.exec_date,
                dry_run=req.dry_run,
            )
            run_date = res.get("run_date")
            out_accounts += res.get("accounts", [])
        if etf_ids:
            end_ts = pd.Timestamp(req.exec_date) if req.exec_date else pd.Timestamp.today()
            start = (end_ts - pd.Timedelta(days=800)).date().isoformat()
            etf_panel = load_etf_panel(start=start, end=end_ts.date().isoformat())
            etf_codes = {"ETF": services.build_codes("ETF", False)}
            res = paper_core.run_paper_trade(
                etf_panel, etf_codes,
                account_ids=etf_ids, exec_date=req.exec_date,
                dry_run=req.dry_run,
            )
            run_date = res.get("run_date")
            out_accounts += res.get("accounts", [])
        if not out_accounts:
            raise ValueError("未找到可执行账户")
        res = {"run_date": run_date, "accounts": out_accounts}
    except ValueError as exc:
        return {"error": str(exc)}
    return res


@router.get("/paper/accounts/{account_id}/summary")
def paper_summary(account_id: int):
    s = paper_core.account_summary(account_id, _panel_for_account(account_id))
    if s is None:
        return {"error": "账户不存在"}
    return paper_core._jsonable(s)


@router.get("/paper/accounts/{account_id}/orders")
def paper_orders(account_id: int):
    return services.clean_records(paper_core.account_orders_with_names(account_id))


@router.get("/paper/accounts/{account_id}/trades")
def paper_trades(account_id: int):
    return services.clean_records(paper_core.account_trades_with_names(account_id))


@router.get("/paper/accounts/{account_id}/positions")
def paper_positions(account_id: int):
    rows = paper_core.enrich_positions(account_id, _panel_for_account(account_id))
    return paper_core._jsonable(rows)


@router.get("/paper/accounts/{account_id}/equity")
def paper_equity(account_id: int):
    rows = paper_core.account_equity(account_id)
    if not rows:
        return {"items": [], "summary": None}
    last = rows[-1]
    return {
        "items": services.clean_records(rows),
        "summary": {
            "latest_equity": float(last["equity"]),
            "cash": float(last["cash"]),
            "market_value": float(last["market_value"]),
            "pnl": float(last["pnl"]),
            "pnl_pct": float(last["pnl_pct"]),
        },
    }


@router.get("/paper/accounts/{account_id}/events")
def paper_events(account_id: int):
    return services.clean_records(paper_core.account_events(account_id))
