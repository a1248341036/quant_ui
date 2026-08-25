from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend import alphaagent_service as service
from alphaagent.factor.mining.research_memory import ResearchMemoryStore
from alphaagent.factor.mining.research_spec import (
    RESEARCH_MODES,
    default_research_spec as build_default_research_spec,
    normalize_research_spec,
)
from alphaagent.factor.evaluation.plugins import available_plugins


router = APIRouter(prefix="/api/alphaagent", tags=["alphaagent"])


class StartRequest(BaseModel):
    train_start: str = "2018-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2025-12-31"
    label_col: str = "label_1d_open_to_open"
    user_message: str = Field(
        default="请自主挖掘A股日频价量因子，先训练集评估，再验证集检验；只有通过验证和去重门槛的因子才提交。",
        min_length=1,
        max_length=10000,
    )
    max_turns: int = Field(default=10, ge=1, le=50)
    max_tool_calls_per_round: int = Field(default=8, ge=1, le=32)
    max_tool_workers: int = Field(default=4, ge=1, le=16)
    max_parallel_eval: int | None = Field(default=2, ge=1, le=16)
    max_tokens: int = Field(default=8192, ge=256, le=32768)
    no_fundamentals: bool = False
    # 研究模式开关（前端切换）：优先级高于 research_spec JSON 内的 research_mode。
    research_mode: str = "technical"
    research_spec: dict[str, Any] = Field(default_factory=build_default_research_spec)


class ContinueRequest(BaseModel):
    content: str = Field(min_length=1, max_length=10000)


class RenameRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class PinRequest(BaseModel):
    pinned: bool


@router.post("/runs")
def start(req: StartRequest) -> dict[str, Any]:
    if req.research_mode not in RESEARCH_MODES:
        raise HTTPException(status_code=422, detail=f"research_mode_invalid:{req.research_mode}")
    payload = req.model_dump()
    spec = dict(payload.get("research_spec") or {})
    spec["research_mode"] = payload["research_mode"]
    payload["research_spec"] = spec
    try:
        run = service.start_run(payload)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return run.snapshot()


@router.get("/runs")
def runs(include_archived: bool = False, archived_only: bool = False) -> list[dict[str, Any]]:
    return service.list_runs(include_archived=include_archived, archived_only=archived_only)


@router.get("/research-memory")
def research_memory(limit: int = 30) -> dict[str, Any]:
    store = ResearchMemoryStore(service.RESEARCH_MEMORY_FILE)
    entries = store.recent(limit=max(1, min(limit, 100)))
    return {"path": str(service.RESEARCH_MEMORY_FILE), "statistics": store.statistics(), "entries": entries}


@router.delete("/research-memory/{entry_id}")
def delete_research_memory(entry_id: str) -> dict[str, Any]:
    store = ResearchMemoryStore(service.RESEARCH_MEMORY_FILE)
    ok = store.delete_entry(entry_id)
    if not ok:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="entry not found")
    return {"ok": True}


@router.get("/research-spec/default")
def default_research_spec(mode: str = "technical") -> dict[str, Any]:
    try:
        return normalize_research_spec(build_default_research_spec(mode))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.get("/evaluation-capabilities")
def evaluation_capabilities() -> dict[str, Any]:
    spec = normalize_research_spec(build_default_research_spec())
    return {"plugins": available_plugins(), "profiles": spec.get("evaluation_profiles", {})}


@router.get("/runs/{run_id}")
def run_detail(run_id: str) -> dict[str, Any]:
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run_not_found")
    return run.snapshot(tail=200)


@router.post("/runs/{run_id}/stop")
def stop(run_id: str) -> dict[str, Any]:
    if service.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="run_not_found")
    return {"ok": service.stop_run(run_id)}


@router.post("/runs/{run_id}/messages")
def continue_run(run_id: str, req: ContinueRequest) -> dict[str, Any]:
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run_not_found")
    if not service.queue_message(run_id, req.content):
        raise HTTPException(status_code=409, detail="run_not_active")
    return {"ok": True, "status": "queued", "run_id": run_id}


@router.post("/runs/{run_id}/continue")
def resume_run(run_id: str, req: ContinueRequest) -> dict[str, Any]:
    run = service.continue_run(run_id, req.content)
    if run is None:
        raise HTTPException(status_code=409, detail="run_not_resumable")
    if run.run_id == run_id:
        return {"ok": True, "status": "queued", "run_id": run_id}
    return run.snapshot()


@router.post("/runs/{run_id}/branch")
def branch_run(run_id: str, req: ContinueRequest) -> dict[str, Any]:
    run = service.branch_run(run_id, req.content)
    if run is None:
        raise HTTPException(status_code=404, detail="run_not_found")
    return run.snapshot()


@router.post("/runs/{run_id}/rename")
def rename_run(run_id: str, req: RenameRequest) -> dict[str, Any]:
    run = service.rename_run(run_id, req.title)
    if run is None:
        raise HTTPException(status_code=404, detail="run_not_found")
    return run.snapshot(tail=20)


@router.post("/runs/{run_id}/archive")
def archive_run(run_id: str) -> dict[str, Any]:
    run = service.archive_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="run_not_found")
    return {"ok": True, "run_id": run_id, "archived": True}


@router.post("/runs/{run_id}/pin")
def pin_run(run_id: str, req: PinRequest) -> dict[str, Any]:
    run = service.pin_run(run_id, req.pinned)
    if run is None:
        raise HTTPException(status_code=404, detail="run_not_found")
    return run.snapshot(tail=20)


@router.get("/runs/{run_id}/events")
async def events(run_id: str):
    if service.get_run(run_id) is None:
        raise HTTPException(status_code=404, detail="run_not_found")

    async def body():
        async for row in service.event_stream(run_id):
            import json
            yield f"data: {json.dumps(row, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(
        body(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


# ══════════════════════════════════════════════════════════════════════
#  单因子评估 API
# ══════════════════════════════════════════════════════════════════════

class EvalFactorRequest(BaseModel):
    multi_line_expr: str = Field(min_length=1, max_length=10000)
    factor_name: str = "expr"
    profile_id: str = "train_screen"
    train_start: str = "2018-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2025-12-31"
    label_col: str = "label_1d_open_to_open"
    include_fundamentals: bool = False
    all_profiles: bool = True


@router.post("/eval-factor")
def eval_factor(req: EvalFactorRequest) -> dict[str, Any]:
    """独立评估一个因子表达式（不依赖挖掘流程）。"""
    try:
        if req.all_profiles:
            result = service.evaluate_multi_profile(
                multi_line_expr=req.multi_line_expr,
                factor_name=req.factor_name,
                train_start=req.train_start,
                train_end=req.train_end,
                val_start=req.val_start,
                val_end=req.val_end,
                label_col=req.label_col,
                include_fundamentals=req.include_fundamentals,
            )
        else:
            result = {
                req.profile_id: service.evaluate_single_factor(
                    multi_line_expr=req.multi_line_expr,
                    factor_name=req.factor_name,
                    profile_id=req.profile_id,
                    train_start=req.train_start,
                    train_end=req.train_end,
                    val_start=req.val_start,
                    val_end=req.val_end,
                    label_col=req.label_col,
                    include_fundamentals=req.include_fundamentals,
                )
            }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"results": result}


# ══════════════════════════════════════════════════════════════════════
#  因子库管理 API
# ══════════════════════════════════════════════════════════════════════

@router.get("/factors")
def list_factors(library: str = "production") -> dict[str, Any]:
    """列出因子库中的所有因子。"""
    return service.list_factors(library=library)


@router.get("/factors/{factor_id}")
def factor_detail(factor_id: str, library: str = "production") -> dict[str, Any]:
    """获取单个因子详情。"""
    result = service.get_factor_detail(factor_id, library=library)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


class DeleteFactorRequest(BaseModel):
    library: str = "production"


@router.delete("/factors/{factor_id}")
def delete_factor(factor_id: str, library: str = "production") -> dict[str, Any]:
    """删除一个因子。"""
    result = service.delete_factor(factor_id, library=library)
    if "error" in result:
        if result["error"] == "library_not_initialized":
            raise HTTPException(status_code=404, detail=result["error"])
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ══════════════════════════════════════════════════════════════════════
#  因子实验室：保存因子 + 回测
# ══════════════════════════════════════════════════════════════════════

class SaveFactorRequest(BaseModel):
    multi_line_expr: str = Field(min_length=1, max_length=10000)
    factor_name: str = Field(min_length=1, max_length=200)
    comment: str = Field(default="", max_length=2000)
    library: str = "candidate"
    train_start: str = "2018-01-01"
    train_end: str = "2022-12-31"
    val_start: str = "2023-01-01"
    val_end: str = "2025-12-31"
    label_col: str = "label_1d_open_to_open"
    include_fundamentals: bool = False


@router.post("/factors")
def save_factor(req: SaveFactorRequest) -> dict[str, Any]:
    """保存因子表达式到因子库（候选池或正式库）。"""
    try:
        return service.save_factor(
            multi_line_expr=req.multi_line_expr,
            factor_name=req.factor_name,
            comment=req.comment,
            library=req.library,
            train_start=req.train_start,
            train_end=req.train_end,
            val_start=req.val_start,
            val_end=req.val_end,
            label_col=req.label_col,
            include_fundamentals=req.include_fundamentals,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


class BacktestFactorRequest(BaseModel):
    multi_line_expr: str = Field(min_length=1, max_length=10000)
    factor_name: str = "expr"
    start: str = "2023-01-01"
    end: str = "2025-12-31"
    top_n: int = Field(default=5, ge=1, le=100)
    freq: str = "monthly"
    capital: float = Field(default=100000.0, ge=1000.0)
    ascending: bool = False
    universe: str = "全部股票"
    exclude_kechuang: bool = False
    warmup_days: int = Field(default=400, ge=0, le=9999)


@router.post("/backtest-factor")
def backtest_factor(req: BacktestFactorRequest) -> dict[str, Any]:
    """用因子表达式在指定区间做回测。"""
    try:
        return service.backtest_factor(
            multi_line_expr=req.multi_line_expr,
            factor_name=req.factor_name,
            start=req.start,
            end=req.end,
            top_n=req.top_n,
            freq=req.freq,
        capital=req.capital,
        ascending=req.ascending,
        universe=req.universe,
        exclude_kechuang=req.exclude_kechuang,
        warmup_days=req.warmup_days,
    )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
