from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from core import trading_config
from alphaagent.factor.window_config import BT_DEFAULT_START


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LABS_DIR = PROJECT_ROOT / "labs"

router = APIRouter(prefix="/api/code", tags=["code"])


class SaveRequest(BaseModel):
    name: str
    code: str = ""
    registry: str = ""
    factors: str = ""
    engine: str = "legacy"


class QweaveRunRequest(BaseModel):
    code: str = ""
    universe: str = "沪深300+中证500+中证1000"
    start: str = BT_DEFAULT_START
    end: str = ""
    alpha_set: str = "alpha158"
    alpha_limit: int | None = 30
    horizons: list[int] = [1, 5, 10, 20]
    quantiles: int = 10
    min_cs_count: int = 30
    cost_bps: float = 8.0
    exclude_kechuang: bool = True
    run_backtest: bool = False
    score_factor: str = ""
    top_n: int = 10
    selection_mode: str = "top_n"
    selection_pct: float = 0.10
    min_positions: int = 1
    max_positions: int | None = None
    capital: float = trading_config.CAPITAL
    freq: str = "weekly"
    affordable: bool = True
    amount_q: float = trading_config.AMOUNT_Q
    warmup_days: int | None = trading_config.WARMUP_DAYS
    slippage_bps: float = trading_config.SLIPPAGE_BPS
    max_participation: float = trading_config.MAX_PARTICIPATION
    max_weight: float | None = None
    buy_cost: float = trading_config.BUY_COST
    sell_cost: float = trading_config.SELL_COST
    industry_cap: int | None = None


def _module_from_req(req) -> str:
    """代码统一来自 code 字段（qweave 单模块模式）。"""
    if req.code and req.code.strip():
        return req.code
    return ""


def _safe_name(name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff_-]+", "_", name).strip("_")
    if not cleaned:
        raise ValueError("保存名称不能为空")
    return cleaned[:60]


@router.get("/qweave/default")
def get_qweave_default():
    from backend.qweave_runner import _default_code
    return {"engine": "qweave", "code": _default_code()}


@router.get("/qweave/templates")
def list_qweave_templates():
    from backend.qweave_runner import QWEAVE_TEMPLATES
    return {"items": [{"name": name, **item} for name, item in QWEAVE_TEMPLATES.items()]}


@router.get("/qweave/template")
def get_qweave_template(name: str):
    try:
        from backend.qweave_runner import QWEAVE_TEMPLATES, template_code
        item = QWEAVE_TEMPLATES[name]
        return {"ok": True, "engine": "qweave", "name": name,
                "label": item["label"], "code": template_code(name)}
    except (KeyError, ValueError) as exc:
        return {"ok": False, "error": str(exc)}


@router.post("/qweave/parse")
def parse_qweave(req: QweaveRunRequest):
    try:
        from backend.qweave_runner import parse_code as parse_qweave_code
        return parse_qweave_code(req.code)
    except Exception as exc:
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}


@router.post("/qweave/run")
def run_qweave(req: QweaveRunRequest):
    from backend.qweave_runner import execute
    return execute(req.model_dump())


@router.get("/saved")
def list_saved():
    if not LABS_DIR.exists():
        return {"items": []}
    items = []
    for p in sorted(LABS_DIR.glob("*.json")):
        try:
            meta = json.loads(p.read_text(encoding="utf-8"))
            items.append({
                "name": meta.get("name", p.stem),
                "engine": meta.get("engine", "legacy"),
                "saved_at": meta.get("saved_at", ""),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return {"items": items}


@router.get("/saved/{name}")
def get_saved(name: str):
    safe = _safe_name(name)
    p = LABS_DIR / f"{safe}.json"
    if not p.exists():
        return {"error": f"没有找到已保存代码: {safe}"}
    meta = json.loads(p.read_text(encoding="utf-8"))
    if meta.get("code"):
        return {
            "name": safe,
            "code": meta["code"],
            "saved_at": meta.get("saved_at", ""),
            "engine": meta.get("engine", "legacy"),
        }
    return {
        "name": safe,
        "registry": meta.get("registry", ""),
        "factors": meta.get("factors", ""),
        "saved_at": meta.get("saved_at", ""),
    }


@router.post("/save")
def save_code(req: SaveRequest):
    safe = _safe_name(req.name)
    LABS_DIR.mkdir(parents=True, exist_ok=True)
    code = _module_from_req(req)
    meta = {
        "name": safe,
        "code": code,
        "registry": req.registry,
        "factors": req.factors,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "engine": req.engine,
    }
    (LABS_DIR / f"{safe}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (LABS_DIR / f"{safe}.py").write_text(code, encoding="utf-8")
    return {"ok": True, "name": safe}
