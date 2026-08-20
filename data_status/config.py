"""Shared paths, env loading and task definitions for the data-status service."""
from __future__ import annotations

import os
import json
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
CATALOG_FILE = BASE_DIR / "catalog.json"
PYTHON = os.getenv(
    "QUANT_UI_PYTHON",
    str(Path.home() / "stock-analyzer" / "local_venv" / "bin" / "python"),
)

def load_catalog() -> dict:
    """Load the declarative dataset/task catalog used by status and task UI."""
    with CATALOG_FILE.open(encoding="utf-8") as f:
        catalog = json.load(f)
    if not isinstance(catalog, dict):
        raise ValueError("data_status/catalog.json must contain an object")
    return catalog


CATALOG = load_catalog()
DATASETS = CATALOG.get("datasets", [])
TASK_CATALOG = CATALOG.get("tasks", [])
TABLE_DATE_COLUMN = {
    item["id"]: item["date_column"]
    for item in DATASETS
    if item.get("kind") == "parquet" and item.get("date_column")
}
PARQUET_TABLES = [item["id"] for item in DATASETS if item.get("kind") == "parquet"]


def load_env() -> None:
    if load_dotenv is None:
        return
    load_dotenv(QUANT_UI_ROOT / ".env")
    load_dotenv(Path.home() / "qqbot" / ".env")


load_env()
