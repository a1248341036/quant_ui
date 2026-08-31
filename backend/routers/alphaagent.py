from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend import alphaagent_service as service
from alphaagent.factor.mining.research_memory import ResearchMemoryStore
from alphaagent.factor.mining.research_memory import (
    APV_TAU_C_DEFAULT,
    APV_TAU_V_DEFAULT,
    EDIT_PRIOR_HARD_CONF_DEFAULT,
    EDIT_PRIOR_RECOMMEND_CONF_DEFAULT,
    EDIT_PRIOR_VETO_CONF_DEFAULT,
)
from alphaagent.factor.mining.memory.calibration import _apv_gate, _eq7_confidence
from alphaagent.factor.mining.research_spec import (
    RESEARCH_MODES,
    default_research_spec as build_default_research_spec,
    effective_research_spec,
    build_run_research_spec,
    load_saved_overrides,
    save_research_spec_overrides,
    reset_research_spec_overrides,
    compute_spec_overrides,
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
    max_tool_calls_per_round: int = Field(default=12, ge=1, le=32)
    max_tool_workers: int = Field(default=8, ge=1, le=16)
    max_parallel_eval: int | None = Field(default=6, ge=1, le=16)
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
def research_memory(limit: int = 50, offset: int = 0) -> dict[str, Any]:
    store = ResearchMemoryStore(service.RESEARCH_MEMORY_FILE)
    entries, total = store.recent(limit=max(1, min(limit, 500)), offset=max(0, int(offset)))
    return {"path": str(service.RESEARCH_MEMORY_FILE), "statistics": store.statistics(), "entries": entries, "total": total}


@router.delete("/research-memory/{entry_id}")
def delete_research_memory(entry_id: str) -> dict[str, Any]:
    store = ResearchMemoryStore(service.RESEARCH_MEMORY_FILE)
    ok = store.delete_entry(entry_id)
    if not ok:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="entry not found")
    return {"ok": True}


@router.get("/research-memory/layers")
def research_memory_layers(mode: str = "technical") -> dict[str, Any]:
    """研究记忆分层明细：SSPM 编辑统计层（cells）+ 经验层（experience）。

    cells 附带 Eq.7 置信度与注入门控状态，口径与 retrieval._edit_prior_block
    （硬/软推荐、硬/软否决）和 advisory APV 双门（(family, motif) 聚合否决）
    完全一致，保证 UI 展示 = Agent 运行时行为。
    门控阈值取 effective_research_spec(mode).memory_policy，与挖掘运行时同源。
    """
    try:
        memory_policy = effective_research_spec(mode).get("memory_policy") or {}
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    store = ResearchMemoryStore(
        service.RESEARCH_MEMORY_FILE,
        apv_tau_c=float(memory_policy.get("apv_tau_c") or APV_TAU_C_DEFAULT),
        apv_tau_v=float(memory_policy.get("apv_tau_v") or APV_TAU_V_DEFAULT),
        edit_prior_hard_conf=float(memory_policy.get("edit_prior_hard_conf") or EDIT_PRIOR_HARD_CONF_DEFAULT),
        edit_prior_recommend_conf=float(memory_policy.get("edit_prior_recommend_conf") or EDIT_PRIOR_RECOMMEND_CONF_DEFAULT),
        edit_prior_veto_conf=float(memory_policy.get("edit_prior_veto_conf") or EDIT_PRIOR_VETO_CONF_DEFAULT),
    )
    cells = store.list_cells()
    experience = store.list_experience()

    # APV 硬否决按 (family, motif) 跨桶聚合，与 advisory._edit_veto_findings 同口径
    pair_agg: dict[tuple[str, str], dict[str, Any]] = {}
    for cell in cells:
        key = (cell["family"], cell["motif"])
        agg = pair_agg.setdefault(key, {"s": 0.0, "f": 0.0, "residuals": []})
        agg["s"] += cell["explicit_s"] + cell["implicit_s"]
        agg["f"] += cell["explicit_f"] + cell["implicit_f"]
        agg["residuals"].extend(cell["residuals"])
    pair_veto = {
        key: _apv_gate(
            agg["s"], agg["f"],
            _eq7_confidence(agg["residuals"]) if agg["residuals"] else 0.0,
            tau_c=store.apv_tau_c, tau_v=store.apv_tau_v,
        )[0]
        for key, agg in pair_agg.items()
    }

    hard_conf = store.edit_prior_hard_conf
    recommend_conf = store.edit_prior_recommend_conf
    veto_conf = store.edit_prior_veto_conf
    for cell in cells:
        s_w = cell["explicit_s"] + cell["implicit_s"]
        f_w = cell["explicit_f"] + cell["implicit_f"]
        n_w = s_w + f_w
        conf = _eq7_confidence(cell["residuals"]) if cell["residuals"] else 0.0
        if pair_veto.get((cell["family"], cell["motif"])):
            gate = "apv_hard_veto"
        elif s_w > 0 and conf > hard_conf:
            gate = "hard_recommend"
        elif s_w > 0 and conf > recommend_conf:
            gate = "soft_recommend"
        elif f_w > 0 and conf > hard_conf:
            gate = "hard_veto"
        elif f_w > 0 and conf > veto_conf:
            gate = "soft_veto"
        else:
            gate = "not_injected"
        cell.update({
            "weighted_s": round(s_w, 2),
            "weighted_fail": round(f_w, 2),
            "weighted_n": round(n_w, 2),
            "fail_rate": round(f_w / n_w, 4) if n_w > 0 else 0.0,
            "confidence": round(conf, 4),
            "residual_count": len(cell["residuals"]),
            "gate": gate,
        })
        del cell["residuals"]

    return {
        "path": str(service.RESEARCH_MEMORY_FILE),
        "totals": {"cells": len(cells), "experience": len(experience)},
        "thresholds": {
            "hard_conf": store.edit_prior_hard_conf,
            "recommend_conf": store.edit_prior_recommend_conf,
            "veto_conf": store.edit_prior_veto_conf,
            "apv_tau_c": store.apv_tau_c,
            "apv_tau_v": store.apv_tau_v,
        },
        "cells": cells,
        "experience": experience,
    }


@router.get("/research-modes")
def research_modes() -> dict[str, Any]:
    """研究模式注册表（UI 元数据）：前端按钮/下拉/提示/默认消息的唯一来源。

    返回 core.research_modes.ui_options()（label/hint/recommended_label_col/
    default_user_message/needs_fundamentals），前端据此动态渲染，避免
    "加一个模式改 N 处前端硬编码"。
    """
    from core.research_modes import ui_options
    return {"modes": ui_options()}


@router.get("/windows")
def windows() -> dict[str, Any]:
    """统一时间窗口默认值（train/val/test），唯一来源 window_config.py。

    前端 mounted 时拉取，用于初始化训练/验证/测试日期输入框的默认值，
    避免前端散落硬编码日期、与后端配置漂移。TEST_END 动态解析数据源最新日。
    """
    from alphaagent.factor.window_config import window_defaults
    return window_defaults()


@router.get("/research-spec/default")
def default_research_spec(mode: str = "technical") -> dict[str, Any]:
    """当前模式生效的研究规范（注册表默认 + 用户保存覆盖）。

    模式切换时前端以此为准——用户改过的门槛会在这里体现。
    """
    try:
        return effective_research_spec(mode)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


class ResearchSpecUpdate(BaseModel):
    # 前端编辑后的完整有效 spec；后端与注册表默认 diff 后只持久化增量覆盖。
    spec: dict[str, Any]


def _research_spec_payload(mode: str) -> dict[str, Any]:
    """GET/PUT/DELETE 统一载荷：defaults / overrides / effective 三视图。"""
    try:
        defaults = normalize_research_spec(build_default_research_spec(mode))
        overrides = load_saved_overrides(mode)
        effective = normalize_research_spec(_merge(overrides, defaults))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    from alphaagent.core.paths import RESEARCH_SPECS_DIR
    return {
        "mode": mode,
        "defaults": defaults,
        "overrides": overrides,
        "effective": effective,
        "path": str(RESEARCH_SPECS_DIR / f"{mode}.json"),
    }


def _merge(overrides: dict[str, Any], base: dict[str, Any]) -> dict[str, Any]:
    """浅层递归合并：overrides 覆盖 base（与 research_spec._deep_merge 同语义）。"""
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(value, merged[key])
        else:
            merged[key] = value
    return merged


@router.get("/research-specs/{mode}")
def get_research_spec(mode: str) -> dict[str, Any]:
    """读取某模式的研究规范门槛文件（默认值 / 保存覆盖 / 生效值）。"""
    if mode not in RESEARCH_MODES:
        raise HTTPException(status_code=404, detail=f"research_mode_invalid:{mode}")
    return _research_spec_payload(mode)


@router.put("/research-specs/{mode}")
def update_research_spec(mode: str, req: ResearchSpecUpdate) -> dict[str, Any]:
    """保存某模式的门槛文件：diff 出增量覆盖并持久化。

    前端看到/编辑的是 effective（完整 JSON）；这里存的是相对默认值的增量，
    代码默认演进时未改键自动跟随。
    """
    if mode not in RESEARCH_MODES:
        raise HTTPException(status_code=404, detail=f"research_mode_invalid:{mode}")
    edited = req.spec
    if not isinstance(edited, dict):
        raise HTTPException(status_code=422, detail="research_spec_must_be_object")
    edited = dict(edited)
    edited.setdefault("research_mode", mode)
    # 校验：enter effective 全流程 normalize，非法值直接 422（不落盘）
    try:
        normalized = build_run_research_spec(edited)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    defaults = normalize_research_spec(build_default_research_spec(mode))
    overrides = compute_spec_overrides(defaults, normalized)
    # evaluation_profiles 是 normalize 从 evaluation_policy 派生的规则视图，
    # 运行时会重新生成；持久化它只会造成冗余与 schema 漂移，落盘前剔除。
    overrides.pop("evaluation_profiles", None)
    save_research_spec_overrides(mode, overrides)
    return _research_spec_payload(mode)


@router.delete("/research-specs/{mode}")
def delete_research_spec(mode: str) -> dict[str, Any]:
    """删除模式门槛文件，恢复注册表默认。"""
    if mode not in RESEARCH_MODES:
        raise HTTPException(status_code=404, detail=f"research_mode_invalid:{mode}")
    reset_research_spec_overrides(mode)
    return _research_spec_payload(mode)


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
    train_start: str = DEFAULT_TRAIN_START
    train_end: str = DEFAULT_TRAIN_END
    val_start: str = DEFAULT_VAL_START
    val_end: str = DEFAULT_VAL_END
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


# ══════════════════════════════════════════════════════════════════════
#  ML 组合（stacking）训练
# ══════════════════════════════════════════════════════════════════════


class StackingTrainRequest(BaseModel):
    modes: list[str] = Field(default=["technical", "fundamental"])
    model: str = Field(default="both")            # ridge | lgbm | both
    label_days: int = Field(default=5, ge=1, le=60)
    train_months: int = Field(default=18, ge=3, le=60)
    step_months: int = Field(default=6, ge=1, le=24)
    purge_days: int = Field(default=5, ge=0, le=60)
    warmup_days: int = Field(default=250, ge=0, le=750)
    max_corr: float = Field(default=0.6, ge=0.1, le=1.0)
    mining_end: str | None = None                 # auto 缺省；YYYY-MM-DD 显式
    no_candidate: bool = False                    # 只用正式库
    no_gate: bool = False                         # 跳过 engine_gate
    isolation: str = Field(default="holdout")     # strict | holdout
    size_neutral: bool = True


@router.post("/stacking/train")
def stacking_train(req: StackingTrainRequest) -> dict[str, Any]:
    """启动一次 ML 组合训练（子进程；同一时间仅一个）。"""
    from backend import stacking_service

    result = stacking_service.start_training(req.model_dump())
    if "error" in result:
        raise HTTPException(status_code=409, detail=result["error"])
    return result


@router.get("/stacking/trainings")
def stacking_trainings(limit: int = 30) -> list[dict[str, Any]]:
    from backend import stacking_service

    return stacking_service.list_trainings(limit=limit)


@router.get("/stacking/trainings/{train_id}")
def stacking_training_detail(train_id: str, tail_lines: int = 40) -> dict[str, Any]:
    from backend import stacking_service

    return stacking_service.get_training(train_id, tail_lines=tail_lines)


@router.post("/stacking/trainings/{train_id}/stop")
def stacking_stop(train_id: str) -> dict[str, Any]:
    from backend import stacking_service

    result = stacking_service.stop_training(train_id)
    if not result.get("ok"):
        raise HTTPException(status_code=404, detail=result.get("error"))
    return result


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
    train_start: str = DEFAULT_TRAIN_START
    train_end: str = DEFAULT_TRAIN_END
    val_start: str = DEFAULT_VAL_START
    val_end: str = DEFAULT_VAL_END
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
    start: str = DEFAULT_VAL_START
    end: str = DEFAULT_VAL_END
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


# ══════════════════════════════════════════════════════════════════════
#  DSL 算子耗时监控
# ══════════════════════════════════════════════════════════════════════

@router.get("/dsl-monitor")
def dsl_operator_monitor(
    top_k: int = 20,
    since_hours: float | None = None,
) -> dict[str, Any]:
    """读取累计算子耗时监控（跨评估聚合 top 慢算子）。

    ``since_hours``：只统计最近 N 小时（None = 全部历史）。
    数据来源：``artifacts/dsl_operator_profiling.jsonl``（每次 DSL 评估自动追加）。
    """
    from alphaagent.dsl.core import monitor

    since_ts = None
    if since_hours is not None and since_hours > 0:
        import time as _t

        since_ts = _t.time() - since_hours * 3600.0
    agg = monitor.read_accumulated(top_k=top_k, since_ts=since_ts)
    return {"ok": True, "source": "artifacts/dsl_operator_profiling.jsonl", **agg}
