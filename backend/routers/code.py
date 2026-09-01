from __future__ import annotations

import json
import re
import threading
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from core import trading_config
from core import trading_config_store
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


class JQRunRequest(BaseModel):
    code: str
    start: str
    end: str = ""
    capital: float = trading_config.CAPITAL
    warmup_days: int = 60


class JQPreflightRequest(BaseModel):
    code: str


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


@router.get("/config")
def get_code_config():
    """代码面板生效参数(全局默认 + 面板副本覆盖)。"""
    return {"ok": True, "config": trading_config_store.effective(),
            "editable": list(trading_config_store.EDITABLE_KEYS),
            "customized": trading_config_store.OVERRIDE_FILE.exists()}


@router.post("/config")
def save_code_config(req: dict):
    """保存代码面板参数副本(仅影响代码 tab)。"""
    patch = (req or {}).get("config") or (req or {})
    cfg = trading_config_store.save_overrides(patch)
    return {"ok": True, "config": cfg,
            "customized": trading_config_store.OVERRIDE_FILE.exists()}


@router.post("/config/reset")
def reset_code_config():
    cfg = trading_config_store.reset_overrides()
    return {"ok": True, "config": cfg, "customized": False}


@router.post("/jq/preflight")
def preflight_jq_strategy(req: JQPreflightRequest):
    """聚宽策略 API 预检: AST 静态扫描, 秒级报告兼容层缺失的 API/字段。

    回测前调用, 避免"跑 90 秒后炸在缺失 API 上"的验证循环。
    """
    from core.event_engine.jq.preflight import preflight
    try:
        missing = preflight(req.code)
        return {"ok": len(missing) == 0, "missing": missing,
                "message": ("预检通过: 策略引用的 API 均已支持" if not missing
                            else f"缺失 {len(missing)} 项: {', '.join(missing)}")}
    except SyntaxError as exc:
        return {"ok": False, "missing": [],
                "message": f"代码语法错误: {exc}"}
    except Exception as exc:
        return {"ok": False, "missing": [],
                "message": f"预检失败: {type(exc).__name__}: {exc}"}


@router.post("/jq/run")
def run_jq_strategy(req: JQRunRequest):
    """聚宽模式: 用户贴聚宽风格代码(initialize + run_daily/weekly + 数据API), 直接回测。"""
    from core.event_engine.jq import run_jq_backtest
    cfg = trading_config_store.effective()
    try:
        return run_jq_backtest(
            code=req.code, start=req.start,
            end=req.end or None, capital=req.capital,
            warmup_days=req.warmup_days or int(cfg["warmup_days"]),
            buy_cost=float(cfg["buy_cost"]),
            sell_cost=float(cfg["sell_cost"]),
            slippage_bps=float(cfg["slippage_bps"]),
        )
    except NotImplementedError as exc:
        return {"ok": False, "error": f"API 未支持: {exc}"}
    except Exception as exc:
        import traceback
        return {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=20)}


# ---- 异步运行管理器: 边跑边报进度(轮询), 支持停止 ----
_JQ_RUNS: dict[str, dict] = {}
_JQ_RUNS_LOCK = threading.Lock()
_JQ_ACTIVE_PHASES = {"queued", "context", "minutes", "engine"}
_JQ_RUN_KEEP = 8


def _jq_run_worker(state: dict, req: JQRunRequest) -> None:
    from core.event_engine.jq import run_jq_backtest
    from core.event_engine.runner import BacktestAborted

    def emit(ev: dict) -> None:
        with _JQ_RUNS_LOCK:
            s = _JQ_RUNS.get(state["run_id"])
            if s is None:
                return
            if ev.get("phase"):
                s["phase"] = ev["phase"]
            if "done" in ev:
                s["done"] = ev["done"]
            if "total" in ev:
                s["total"] = ev["total"]
            if ev.get("in_window") and ev.get("date"):
                s["nav"].append({"date": ev["date"], "value": ev["nav"]})

    try:
        cfg = trading_config_store.effective()
        result = run_jq_backtest(
            code=req.code, start=req.start,
            end=req.end or None, capital=req.capital,
            warmup_days=req.warmup_days or int(cfg["warmup_days"]),
            buy_cost=float(cfg["buy_cost"]),
            sell_cost=float(cfg["sell_cost"]),
            slippage_bps=float(cfg["slippage_bps"]),
            progress=emit, cancel_event=state["cancel_event"])
        with _JQ_RUNS_LOCK:
            state["result"] = result
            state["phase"] = "done"
            state["ended_at"] = time.time()
    except BacktestAborted:
        with _JQ_RUNS_LOCK:
            state["phase"] = "cancelled"
            state["error"] = "已手动停止"
            state["ended_at"] = time.time()
    except Exception as exc:
        with _JQ_RUNS_LOCK:
            state["phase"] = "error"
            state["error"] = f"{type(exc).__name__}: {exc}"
            state["traceback"] = traceback.format_exc(limit=20)
            state["ended_at"] = time.time()


def _prune_jq_runs() -> None:
    with _JQ_RUNS_LOCK:
        finished = sorted(
            (k for k, v in _JQ_RUNS.items()
             if v["phase"] not in _JQ_ACTIVE_PHASES),
            key=lambda k: _JQ_RUNS[k].get("ended_at", 0))
        for k in finished[:-_JQ_RUN_KEEP]:
            _JQ_RUNS.pop(k, None)


@router.post("/jq/run_async")
def start_jq_run_async(req: JQRunRequest):
    """后台线程启动聚宽回测, 立即返回 run_id; 进度经 /jq/runs/{id} 轮询。"""
    with _JQ_RUNS_LOCK:
        busy = [k for k, v in _JQ_RUNS.items()
                if v["phase"] in _JQ_ACTIVE_PHASES]
    if busy:
        return {"ok": False, "error": "已有回测在运行中, 请等待完成或先停止"}
    _prune_jq_runs()
    run_id = uuid.uuid4().hex[:12]
    state = {"run_id": run_id, "phase": "queued", "done": 0, "total": 0,
             "nav": [], "result": None, "error": None, "traceback": None,
             "cancel_event": threading.Event(),
             "started_at": time.time(), "ended_at": None}
    with _JQ_RUNS_LOCK:
        _JQ_RUNS[run_id] = state
    threading.Thread(target=_jq_run_worker, args=(state, req),
                     daemon=True, name=f"jq-run-{run_id}").start()
    return {"ok": True, "run_id": run_id}


@router.get("/jq/runs/{run_id}")
def get_jq_run(run_id: str):
    with _JQ_RUNS_LOCK:
        s = _JQ_RUNS.get(run_id)
        if s is None:
            raise HTTPException(404, "run not found")
        out = {k: v for k, v in s.items() if k != "cancel_event"}
        out["nav"] = list(s["nav"])
    out["elapsed"] = round(
        (out.get("ended_at") or time.time()) - out["started_at"], 1)
    return out


@router.post("/jq/runs/{run_id}/stop")
def stop_jq_run(run_id: str):
    with _JQ_RUNS_LOCK:
        s = _JQ_RUNS.get(run_id)
        if s is None:
            raise HTTPException(404, "run not found")
        s["cancel_event"].set()
    return {"ok": True}


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
