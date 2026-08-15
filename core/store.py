from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd


DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PANEL_FILE = DATA_DIR / "panel.parquet"
UNIVERSE_FILE = DATA_DIR / "universe.csv"
TECH_FILE = DATA_DIR / "tech.csv"
INDEX_FILE = DATA_DIR / "index.csv"
META_FILE = DATA_DIR / "meta.json"


def ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _atomic_write_text(target: Path, text: str) -> None:
    ensure_dir()
    fd, tmp = tempfile.mkstemp(dir=str(DATA_DIR), prefix=f".{target.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, target)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_panel(panel: pd.DataFrame) -> None:
    ensure_dir()
    panel.to_parquet(PANEL_FILE, index=False)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir()
    _atomic_write_text(path, df.to_csv(index=False))


def save_meta(extra: dict | None = None) -> None:
    meta = {"last_update": datetime.now().isoformat(timespec="seconds")}
    if extra:
        meta.update(extra)
    _atomic_write_text(META_FILE, json.dumps(meta, ensure_ascii=False, indent=2))


def load_meta() -> dict:
    if not META_FILE.exists():
        return {}
    try:
        return json.loads(META_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}
