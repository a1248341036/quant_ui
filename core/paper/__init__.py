"""日级模拟盘：账户 / 订单 / 撮合 / 持仓 / 日结估值 / 风控。

原 core/paper.py（1437行）拆分为 4 个子模块：
- storage.py  — 底层读写（SQLite/JSON）、常量、工具函数
- account.py  — 账户管理 CRUD
- details.py  — 明细读取与写入
- rebalance.py — 调仓/撮合/日结主流程/事件策略/因子策略/组合展示

本 __init__.py re-export 全部公共 API，外部 import 路径不变。
"""
from __future__ import annotations

# ---------- storage.py ----------
from .storage import (
    PAPER_DIR,
    PAPER_FILE,
    DEFAULT_RISK,
    _q,
    _ex,
    _ex_id,
    _json_state,
    _json_save,
    _json_next_id,
    _acc_row,
    _jsonable,
    _load_module,
    _ensure_columns,
)

# ---------- account.py ----------
from .account import (
    create_account,
    list_accounts,
    get_account,
    set_account_status,
    update_account_strategy,
    delete_account,
    reset_account,
    _clear_account_state,
    _update_account_start,
)

# ---------- details.py ----------
from .details import (
    account_orders,
    account_trades,
    account_positions,
    account_equity,
    account_events,
    account_orders_with_names,
    account_trades_with_names,
    _name_map,
    _add_order,
    _add_trade,
    _set_position,
    _load_positions,
    _add_snapshot,
    _add_event,
    _update_account_dates,
)

# ---------- rebalance.py ----------
from .rebalance import (
    _rebalance_due,
    _compute_targets,
    _execute_rebalance,
    run_paper_trade,
    _event_bt_start,
    _replay_event_positions,
    _run_one_event,
    _run_one_factor,
    _run_one,
    _last_close,
    enrich_positions,
    account_summary,
)

__all__ = [
    # 常量
    "PAPER_DIR",
    "PAPER_FILE",
    "DEFAULT_RISK",
    # 账户管理
    "create_account",
    "list_accounts",
    "get_account",
    "set_account_status",
    "update_account_strategy",
    "delete_account",
    "reset_account",
    # 明细读取
    "account_orders",
    "account_trades",
    "account_positions",
    "account_equity",
    "account_events",
    "account_orders_with_names",
    "account_trades_with_names",
    # 调仓/日结
    "run_paper_trade",
    # 组合展示
    "enrich_positions",
    "account_summary",
    # 工具函数
    "_jsonable",
    "_load_module",
]
