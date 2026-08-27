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
from alphaagent.factor.types import (
    DEFAULT_TRAIN_END,
    DEFAULT_TRAIN_START,
    DEFAULT_VAL_END,
    DEFAULT_VAL_START,
)


router = APIRouter(prefix="/api/alphaagent", tags=["alphaagent"])


class StartRequest(BaseModel):
    train_start: str = DEFAULT_TRAIN_START
    train_end: str = DEFAULT_TRAIN_END
    val_start: str = DEFAULT_VAL_START
    val_end: str = DEFAULT_VAL_END
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
    max_tokens: int = Field(default=16384, ge=256, le=32768)  # hy3 thinking 需 >8K
    population_max: int = Field(default=24, ge=0, le=36)  # 种群批量上限；0=关闭路径B
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
    # pydantic 的 default_factory 总会填充 technical 默认 spec，无法用 None 判断。
    # 改用 model_fields_set 判断用户是否显式传了 research_spec：
    #   - 未传 → 基于 research_mode 构造正确的默认 spec
    #   - 显式传了 → 保留用户 spec（仍以 research_mode 覆盖模式字段）
    if "research_spec" in req.model_fields_set:
        spec = dict(payload.get("research_spec") or {})
    else:
        spec = build_default_research_spec(req.research_mode)
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


class ArchiveRequest(BaseModel):
    archived: bool = True


@router.post("/runs/{run_id}/archive")
def archive_run(run_id: str, req: ArchiveRequest | None = None) -> dict[str, Any]:
    archived = True if req is None else bool(req.archived)
    run = service.archive_run(run_id, archived=archived)
    if run is None:
        raise HTTPException(status_code=404, detail="run_not_found")
    return {"ok": True, "run_id": run_id, "archived": archived}


# 注意：/runs/archived 必须先于 /runs/{run_id} 注册，否则会被路径参数吞掉。
@router.delete("/runs/archived")
def delete_archived_runs() -> dict[str, Any]:
    """一键删除全部已归档任务（仍在运行的跳过）。"""
    return service.delete_archived_runs()


@router.delete("/runs/{run_id}")
def delete_run(run_id: str) -> dict[str, Any]:
    result = service.delete_run(run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="run_not_found")
    if not result["deleted"]:
        raise HTTPException(status_code=409, detail=result["reason"])
    return result


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
def list_factors(library: str = "production", category: str = "technical") -> dict[str, Any]:
    """列出因子库中的所有因子。"""
    return service.list_factors(library=library, category=category)


@router.get("/factors/{factor_id}")
def factor_detail(factor_id: str, library: str = "production", category: str = "technical") -> dict[str, Any]:
    """获取单个因子详情。"""
    result = service.get_factor_detail(factor_id, library=library, category=category)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


# ══════════════════════════════════════════════════════════════════════
#  会话缓存管理 API
# ══════════════════════════════════════════════════════════════════════

@router.get("/session-cache/stats")
def get_session_cache_stats() -> dict[str, Any]:
    """获取会话缓存统计信息。
    
    用于监控内存使用情况，显示当前缓存的会话数量和参数。
    """
    return service.get_session_cache_stats()


@router.post("/session-cache/evict")
def evict_all_sessions() -> dict[str, Any]:
    """清空所有会话缓存，释放内存。
    
    通常在内存压力大或参数大幅变化时调用。
    """
    return service.evict_all_sessions()


class DeleteFactorRequest(BaseModel):
    library: str = "production"


@router.delete("/factors/{factor_id}")
def delete_factor(factor_id: str, library: str = "production", category: str = "technical") -> dict[str, Any]:
    """删除一个因子。"""
    result = service.delete_factor(factor_id, library=library, category=category)
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
    category: str = "technical"
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
            category=req.category,
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
# ══════════════════════════════════════════════════════════════════════
#  通用日志查看 API
# ══════════════════════════════════════════════════════════════════════

@router.get("/logs")
def get_logs(
    run_id: str | None = None,
    level: str | None = None,
    limit: int = 100,
    tail: bool = True,
) -> dict[str, Any]:
    """查看系统日志。
    
    支持按 run_id 过滤、按日志级别过滤、限制返回条数。
    
    Args:
        run_id: 可选，按运行 ID 过滤日志
        level: 可选，日志级别 (DEBUG, INFO, WARNING, ERROR)
        limit: 返回最大条数，默认 100，最大 1000
        tail: 是否从末尾开始读取（最新日志在前）
    
    Returns:
        日志列表，每条包含 timestamp, level, message, run_id 等字段
    """
    from backend.logging_config import get_log_file_path, parse_log_file
    
    try:
        log_file = get_log_file_path()
        if not log_file.exists():
            return {
                "log_file": str(log_file),
                "total_lines": 0,
                "logs": []
            }
        
        logs = parse_log_file(
            log_file=log_file,
            run_id=run_id,
            level=level,
            limit=min(limit, 1000),
            tail=tail,
        )
        
        return {
            "log_file": str(log_file),
            "total_lines": len(logs),
            "filters": {
                "run_id": run_id,
                "level": level,
                "limit": limit,
            },
            "logs": logs
        }
    except Exception as exc:
        from fastapi import HTTPException
        raise HTTPException(status_code=500, detail=f"Failed to read logs: {exc}") from exc


@router.get("/logs/tail")
async def tail_logs(
    run_id: str | None = None,
    level: str | None = None,
):
    """实时流式输出日志（类似 tail -f）。
    
    适用于监控正在运行的挖掘任务。
    """
    from backend.logging_config import get_log_file_path
    import asyncio
    from pathlib import Path
    
    log_file = get_log_file_path()
    
    async def log_stream():
        if not log_file.exists():
            # 文件不存在时等待创建
            while not log_file.exists():
                await asyncio.sleep(1)
            position = 0
        else:
            # 从文件末尾开始
            position = log_file.stat().st_size
        
        while True:
            try:
                await asyncio.sleep(0.5)  # 每 500ms 检查一次
                
                if not log_file.exists():
                    continue
                
                with open(log_file, "r", encoding="utf-8") as f:
                    f.seek(position)
                    new_lines = f.readlines()
                    position = f.tell()
                
                for line in new_lines:
                    # 可选：按 run_id 或 level 过滤
                    if run_id and run_id not in line:
                        continue
                    if level and level.upper() not in line:
                        continue
                    
                    yield f"data: {line.strip()}\n\n"
            
            except Exception as e:
                yield f"data: Error: {str(e)}\n\n"
    
    return StreamingResponse(
        log_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
