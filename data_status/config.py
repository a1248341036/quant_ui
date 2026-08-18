"""Shared paths, env loading and task definitions for the data-status service."""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None


BASE_DIR = Path(__file__).resolve().parent
QUANT_UI_ROOT = BASE_DIR.parent
SCRIPTS_DIR = QUANT_UI_ROOT / "scripts"
QUANT_UI_DATA_DIR = Path(
    os.getenv("QUANT_UI_DATA_DIR", str(QUANT_UI_ROOT / "data"))
).expanduser()
PG_PARQUET_DIR = QUANT_UI_DATA_DIR / "pg_parquet"
STATE_FILE = BASE_DIR / "state.json"
TASKS_FILE = BASE_DIR / "tasks.json"
LOGS_DIR = BASE_DIR / "logs"
PYTHON = os.getenv(
    "QUANT_UI_PYTHON",
    str(Path.home() / "stock-analyzer" / "local_venv" / "bin" / "python"),
)

# Date column used to judge freshness per parquet file.
TABLE_DATE_COLUMN = {
    "stock_daily": "trade_date",
    "share_float": "ann_date",
    "fina_indicator": "ann_date",
    "income": "ann_date",
    "balancesheet": "ann_date",
    "cashflow": "ann_date",
    "dividend": "ann_date",
    "stk_surv": "surv_date",
    "forecast": "ann_date",
    "express": "ann_date",
    "namechange": "ann_date",
    "trade_cal": "cal_date",
    "report_rc": "report_date",
}

PARQUET_TABLES = list(TABLE_DATE_COLUMN) + ["stock_basic"]


def load_env() -> None:
    if load_dotenv is None:
        return
    load_dotenv(QUANT_UI_ROOT / ".env")
    load_dotenv(Path.home() / "qqbot" / ".env")


load_env()
