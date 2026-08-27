from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv("QUANT_UI_DATA_DIR", str(PROJECT_ROOT / "data"))).expanduser()
# 旧独立数据目录（历史迁移前位于 ~/quant_data；迁移后可直接指向 QUANT_UI_DATA_DIR）
LEGACY_DATA_DIR = Path(
    os.getenv("QUANT_UI_LEGACY_DATA_DIR", str(PROJECT_ROOT.parent.parent / "quant_data"))
).expanduser()
# 舆情数据目录（sentiment-mvp/data 统一放置在 quant_ui/data 下）
SENTIMENT_DIR = Path(
    os.getenv("QUANT_UI_SENTIMENT_DIR", str(DATA_DIR / "sentiment-mvp"))
).expanduser()
# QQ 机器人凭据/推送脚本所在目录（默认 ~/qqbot，与 .env.example 文档一致）
QQBOT_DIR = Path(
    os.getenv("QUANT_UI_QQBOT_DIR", str(Path.home() / "qqbot"))
).expanduser()
# 每日导出的 PG 快照（refresh_data.py --sync-pg 生成）
PG_PARQUET_DIR = DATA_DIR / "pg_parquet"
# CNEquity 管理的原始数据（日频按年分目录）
QUANT_DATASET_DIR = DATA_DIR / "quant_dataset"

# ── 子目录划分（stock/etf/fund/db/logs）
STOCK_DIR = DATA_DIR / "stock"
ETF_DIR = DATA_DIR / "etf"
FUND_DIR = DATA_DIR / "fund"
DB_DIR = DATA_DIR / "db"

PANEL_FILE = STOCK_DIR / "panel.parquet"
UNIVERSE_FILE = STOCK_DIR / "universe.csv"
TECH_FILE = STOCK_DIR / "tech.csv"
INDEX_FILE = STOCK_DIR / "index.parquet"  # 由 CNE step_index_bars_external 维护
META_FILE = STOCK_DIR / "meta.json"
ETF_FILE = ETF_DIR / "etf.csv"
ETF_PANEL_FILE = ETF_DIR / "etf_panel.parquet"
FUND_FILE = FUND_DIR / "fund.csv"
FUND_FEE_FILE = FUND_DIR / "fund_fee.parquet"  # 由 CNE step_fund_fees 维护
FUND_NAV_FILE = FUND_DIR / "fund_nav.parquet"
FUND_PANEL_FILE = FUND_DIR / "fund_panel.parquet"
PRED_FILE = STOCK_DIR / "pred_demo.parquet"  # qweave 研究层输出的 ML 预测分数（date/code/score）


TECH_UNIVERSE = "科技TMT"
TECH_UNIVERSE_LEGACY = "科技行业"
WIDE_UNIVERSE = "沪深300+中证500+中证1000"
FUND_UNIVERSE = "场外基金"
FUND_UNIVERSE_LEGACY = "场外科技基金"


def normalize_universe(name: str) -> str:
    """归一化旧名，兼容存量账户/回测参数。"""
    if name == TECH_UNIVERSE_LEGACY:
        return TECH_UNIVERSE
    if name == FUND_UNIVERSE_LEGACY:
        return FUND_UNIVERSE
    return name


def ensure_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STOCK_DIR.mkdir(parents=True, exist_ok=True)
    ETF_DIR.mkdir(parents=True, exist_ok=True)
    FUND_DIR.mkdir(parents=True, exist_ok=True)
    DB_DIR.mkdir(parents=True, exist_ok=True)


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
    fd, tmp = tempfile.mkstemp(dir=str(DATA_DIR), prefix=".panel.", suffix=".parquet.tmp")
    os.close(fd)
    try:
        panel.to_parquet(tmp, index=False)
        os.replace(tmp, PANEL_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise

def save_fund_panel(panel: pd.DataFrame) -> None:
    ensure_dir()
    fd, tmp = tempfile.mkstemp(dir=str(DATA_DIR), prefix=".fund_panel.", suffix=".parquet.tmp")
    os.close(fd)
    try:
        panel.to_parquet(tmp, index=False)
        os.replace(tmp, FUND_PANEL_FILE)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_csv(df: pd.DataFrame, path: Path) -> None:
    ensure_dir()
    # lineterminator="\n": pandas defaults to os.linesep on Windows, and
    # to_csv already emits "\r\n" inside the string, so _atomic_write_text
    # (text mode) would double every carriage return ("\r\r\n") and poison
    # the last column name for downstream readers.
    _atomic_write_text(path, df.to_csv(index=False, lineterminator="\n"))


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

