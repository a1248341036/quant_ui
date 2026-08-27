from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from alphaagent.factor.types import (
    DEFAULT_TRAIN_END,
    DEFAULT_TRAIN_START,
    DEFAULT_VAL_END,
    DEFAULT_VAL_START,
)
from alphaagent.factor.mining.research_spec import default_research_spec, normalize_research_spec

from alphaagent.factor.mining.research_memory import ResearchMemoryStore
from core import factor_categories

logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXECUTABLE = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON_EXECUTABLE.exists():
    PYTHON_EXECUTABLE = Path(sys.executable)
DEFAULT_PANEL = "cne://"
DEFAULT_FACTORLIB = factor_categories.production_dir("technical")  # 向后兼容
RESEARCH_MEMORY_FILE = ROOT / "artifacts" / "alphaagent" / "research_memory.db"
LOG_ROOT = ROOT / "logs" / "factor_mining" / "ui"

# ── 单因子评估服务（懒初始化） ──────────────────────────────────────
_eval_service: Any = None
_eval_service_lock = threading.Lock()
_eval_service_params: dict[str, Any] | None = None


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def bootstrap_research_memory() -> int:
    """Recover historical evaluation evidence only when the memory store is empty."""
    return ResearchMemoryStore(RESEARCH_MEMORY_FILE).backfill_from_logs(LOG_ROOT)


@dataclass
class AgentRun:
    run_id: str
    command: list[str]
    log_dir: Path
    params: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=_now)
    process: subprocess.Popen[str] | None = None
    status: str = "starting"
    error: str | None = None
    console_log: Path | None = None
    parent_run_id: str | None = None
    title: str = ""
    archived: bool = False
    pinned: bool = False

    @property
    def control_file(self) -> Path:
        return self.log_dir / "continuations.jsonl"

    @property
    def meta_file(self) -> Path:
        return self.log_dir / "run_meta.json"

    def save_meta(self) -> None:
        self.meta_file.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "created_at": self.created_at,
                    "params": self.params,
                    "parent_run_id": self.parent_run_id,
                    "title": self.title,
                    "archived": self.archived,
                    "pinned": self.pinned,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _jsonl(self) -> Path | None:
        files = sorted(self.log_dir.glob("run_*.jsonl"))
        return files[-1] if files else None

    def refresh(self) -> None:
        if self.process is None:
            return
        code = self.process.poll()
        if code is None:
            self.status = "running"
        elif code == 0:
            self.status = "completed"
        else:
            self.status = "failed"
            if not self.error:
                detail = ""
                if self.console_log and self.console_log.exists():
                    try:
                        lines = self.console_log.read_text(encoding="utf-8", errors="replace").splitlines()
                        detail = " ".join(line.strip() for line in lines[-4:] if line.strip())
                    except OSError:
                        pass
                self.error = f"AlphaAgent exited with code {code}" + (f": {detail[-1200:]}" if detail else "")

    def snapshot(self, tail: int = 80) -> dict[str, Any]:
        self.refresh()
        event_count, recent_events = scan_event_tail(self._jsonl(), tail)
        summary_path = self.log_dir / "run_summary.json"
        summary: dict[str, Any] = {}
        if summary_path.exists():
            try:
                loaded = json.loads(summary_path.read_text(encoding="utf-8"))
                summary = loaded if isinstance(loaded, dict) else {}
            except (OSError, json.JSONDecodeError):
                summary = {}
        return {
            "run_id": self.run_id,
            "status": self.status,
            "source": "api" if self.command else "cli",
            "outcome": summary.get("outcome"),
            "success": summary.get("success"),
            "failure_counts": summary.get("failure_counts", {}),
            "candidate_funnel": summary.get("candidate_funnel", {}),
            "overfit_audit": summary.get("overfit_audit", {}),
            "created_at": self.created_at,
            "parent_run_id": self.parent_run_id,
            "user_message": self.params.get("user_message", ""),
            "title": self.title,
            "archived": self.archived,
            "pinned": self.pinned,
            "research_spec": self.params.get("research_spec"),
            "event_count": event_count,
            "events": recent_events,
            "log_dir": str(self.log_dir),
            "error": self.error,
        }


_RUNS: dict[str, AgentRun] = {}
_LOCK = threading.Lock()


def _status_from_events(events: list[dict[str, Any]]) -> str:
    names = {str(event.get("event", "")) for event in events}
    if "session_error" in names:
        return "failed"
    if "session_end" in names or "run_summary" in names:
        return "completed"
    return "interrupted"


def _load_run_from_disk(run_dir: Path) -> AgentRun | None:
    if not run_dir.is_dir():
        return None
    jsonl_files = sorted(run_dir.glob("run_*.jsonl"))
    if not jsonl_files:
        return None
    events = list(read_events(jsonl_files[-1]))
    if not events:
        return None
    metadata: dict[str, Any] = {}
    meta_file = run_dir / "run_meta.json"
    if meta_file.exists():
        try:
            metadata = json.loads(meta_file.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            metadata = {}
    first_user = next((event.get("content") for event in events if event.get("event") == "user_message"), "")
    created = str(metadata.get("created_at") or events[0].get("ts") or datetime.fromtimestamp(
        run_dir.stat().st_mtime, tz=timezone.utc
    ).isoformat())
    params = metadata.get("params") if isinstance(metadata.get("params"), dict) else {}
    params.setdefault("user_message", str(first_user or ""))
    status = _status_from_events(events)
    # 外部启动（CLI/脚本）的 run 没有进程句柄：轨迹仍在推进时视为运行中，
    # 而不是误标 interrupted。15 分钟无新事件才回落到事件推导状态。
    if status == "interrupted":
        try:
            if time.time() - jsonl_files[-1].stat().st_mtime < 900:
                status = "running"
        except OSError:
            pass
    return AgentRun(
        run_id=str(metadata.get("run_id") or run_dir.name),
        command=[],
        log_dir=run_dir,
        params=params,
        created_at=created,
        status=status,
        parent_run_id=metadata.get("parent_run_id"),
        title=str(metadata.get("title") or ""),
        archived=bool(metadata.get("archived", False)),
        pinned=bool(metadata.get("pinned", False)),
    )


def hydrate_runs() -> None:
    """Restore past UI runs after a FastAPI restart from their JSONL logs.

    Dirs already tracked in memory are skipped: their in-process AgentRun stays
    authoritative, so repeated listings never re-parse old trajectories.
    """
    if not LOG_ROOT.exists():
        return
    with _LOCK:
        known_dirs = {run.log_dir for run in _RUNS.values()}
    restored: list[AgentRun] = []
    for path in LOG_ROOT.iterdir():
        if path in known_dirs:
            continue
        run = _load_run_from_disk(path)
        if run is not None:
            restored.append(run)
    with _LOCK:
        for run in restored:
            if run.run_id not in _RUNS:
                _RUNS[run.run_id] = run


def _drain_output(run: AgentRun, stream) -> None:
    path = run.log_dir / "console.log"
    run.console_log = path
    with path.open("a", encoding="utf-8") as out:
        for line in iter(stream.readline, ""):
            out.write(line)
            out.flush()
    stream.close()


def read_events(path: Path | None) -> Iterator[dict[str, Any]]:
    if path is None or not path.exists():
        return
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(row, dict):
                    yield row
    except OSError:
        return


def scan_event_tail(path: Path | None, tail: int) -> tuple[int, list[dict[str, Any]]]:
    """Count rows in an event JSONL and decode only its last *tail* rows.

    Listing every run must not JSON-decode entire histories: raw lines are
    counted cheaply and only the trailing window is parsed. Non-dict or
    undecodable trailing lines are excluded from the count, matching read_events.
    """
    if path is None or not path.exists() or tail <= 0:
        return 0, []
    recent: deque[str] = deque(maxlen=tail)
    total = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if not line.strip():
                    continue
                total += 1
                recent.append(line)
    except OSError:
        return 0, []
    events: list[dict[str, Any]] = []
    for line in recent:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            total -= 1
            continue
        if isinstance(row, dict):
            events.append(row)
        else:
            total -= 1
    return total, events


def start_run(
    params: dict[str, Any],
    *,
    parent_run_id: str | None = None,
    resume_context_file: Path | None = None,
) -> AgentRun:
    params = dict(params)
    raw_spec = params.get("research_spec")
    raw_spec = dict(raw_spec) if isinstance(raw_spec, dict) else {}
    # 表单模式开关是权威来源；spec 内缺失时回填，保证模式/label/基本面加载三者一致。
    mode = str(params.get("research_mode") or raw_spec.get("research_mode") or "technical")
    raw_spec["research_mode"] = mode
    spec = normalize_research_spec(raw_spec)
    params["research_spec"] = spec
    # 研究模式决定评估 label：基本面模式的季频数据必须配慢因子 label，
    # 调用方传了不匹配的 label 时强制覆盖并记录警告。
    recommended = spec.get("recommended_label_col") or "label_1d_open_to_open"
    caller_label = str(params.get("label_col") or "").strip()
    if mode == "fundamental" and caller_label != recommended:
        logger.warning(
            "label_col 覆盖: 基本面模式要求 %s，调用方传了 %s → 已强制覆盖",
            recommended, caller_label,
        )
        params["label_col"] = recommended
    elif not caller_label:
        params["label_col"] = recommended
    params["max_tool_calls_per_round"] = min(
        int(params["max_tool_calls_per_round"]),
        int(spec["search_policy"]["max_candidates_per_round"]),
    )
    run_id = uuid.uuid4().hex[:12]
    log_dir = LOG_ROOT / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    spec_path = log_dir / "research_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # 因子库路径按 research_mode 路由到对应类别目录
    factorlib_path = factor_categories.production_dir(mode)
    command = [
        str(PYTHON_EXECUTABLE), str(ROOT / "scripts" / "run_alphaagent.py"),
        "--panel", str(DEFAULT_PANEL),
        "--factorlib", str(factorlib_path),
        "--train-start", str(params["train_start"]),
        "--train-end", str(params["train_end"]),
        "--val-start", str(params["val_start"]),
        "--val-end", str(params["val_end"]),
        "--label-col", str(params["label_col"]),
        "--max-turns", str(params["max_turns"]),
        "--max-tool-calls-per-round", str(params["max_tool_calls_per_round"]),
        "--max-tool-workers", str(params["max_tool_workers"]),
        "--max-tokens", str(params["max_tokens"]),
        "--log-dir", str(log_dir),
        "--control-file", str(log_dir / "continuations.jsonl"),
        "--research-memory-file", str(RESEARCH_MEMORY_FILE),
        "--research-spec-file", str(spec_path),
        "--user-message", str(params["user_message"]),
    ]
    if params.get("max_parallel_eval") is not None:
        command.extend(["--max-parallel-eval", str(params["max_parallel_eval"])])
    if params.get("population_max") is not None:
        command.extend(["--population-max", str(int(params["population_max"]))])
    # 研究模式决定是否载入基本面列：技术模式省内存，基本面模式必须载入 funda_* 字段。
    wants_fundamentals = spec.get("research_mode") == "fundamental"
    if not wants_fundamentals or params.get("no_fundamentals"):
        command.append("--no-fundamentals")
    if resume_context_file is not None:
        command.extend(["--resume-context-file", str(resume_context_file)])
    env = os.environ.copy()
    env.setdefault("ALPHA_LLM_PROVIDER", "codex")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    run = AgentRun(
        run_id=run_id,
        command=command,
        log_dir=log_dir,
        params=dict(params),
        parent_run_id=parent_run_id,
        title=str(params.get("user_message", ""))[:64],
    )
    run.save_meta()
    process = subprocess.Popen(
        command,
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    run.process = process
    threading.Thread(target=_drain_output, args=(run, process.stdout), daemon=True).start()
    with _LOCK:
        _RUNS[run_id] = run
    return run


def list_runs(*, include_archived: bool = False, archived_only: bool = False) -> list[dict[str, Any]]:
    hydrate_runs()
    with _LOCK:
        runs = list(_RUNS.values())
    if archived_only:
        visible = [run for run in runs if run.archived]
    elif include_archived:
        visible = runs
    else:
        visible = [run for run in runs if not run.archived]
    return [run.snapshot(tail=20) for run in sorted(visible, key=lambda x: (x.pinned, x.created_at), reverse=True)]


def get_run(run_id: str) -> AgentRun | None:
    hydrate_runs()
    with _LOCK:
        return _RUNS.get(run_id)


def stop_run(run_id: str) -> bool:
    run = get_run(run_id)
    if run is None or run.process is None or run.process.poll() is not None:
        return False
    run.process.terminate()
    run.status = "stopping"
    return True


def queue_message(run_id: str, content: str) -> bool:
    run = get_run(run_id)
    if run is None or run.process is None or run.process.poll() is not None:
        return False
    row = {"ts": _now(), "content": content.strip()}
    with run.control_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return True


def rename_run(run_id: str, title: str) -> AgentRun | None:
    run = get_run(run_id)
    if run is None:
        return None
    run.title = title.strip()
    run.save_meta()
    return run


def archive_run(run_id: str, *, archived: bool = True) -> AgentRun | None:
    run = get_run(run_id)
    if run is None:
        return None
    run.archived = bool(archived)
    run.save_meta()
    return run


def pin_run(run_id: str, pinned: bool) -> AgentRun | None:
    run = get_run(run_id)
    if run is None:
        return None
    run.pinned = pinned
    run.save_meta()
    return run


def delete_run(run_id: str) -> dict[str, Any] | None:
    """删除一个任务：从内存移除并清掉它的日志目录。

    进程还活着时拒绝删除——先 stop 再删，避免边写边删。
    """
    run = get_run(run_id)
    if run is None:
        return None
    if run.process is not None and run.process.poll() is None:
        return {"run_id": run_id, "deleted": False, "reason": "run_still_running"}
    shutil.rmtree(run.log_dir, ignore_errors=True)
    with _LOCK:
        _RUNS.pop(run_id, None)
    return {"run_id": run_id, "deleted": True}


def delete_archived_runs() -> dict[str, Any]:
    """一键删除全部已归档任务；仍在运行的归档任务跳过。"""
    hydrate_runs()
    with _LOCK:
        targets = [run for run in _RUNS.values() if run.archived]
    deleted: list[str] = []
    skipped: list[str] = []
    for run in targets:
        outcome = delete_run(run.run_id)
        if outcome is not None and outcome.get("deleted"):
            deleted.append(outcome["run_id"])
        else:
            skipped.append(run.run_id)
    return {"ok": True, "deleted": deleted, "skipped": skipped, "count": len(deleted)}


def _run_resume_context(run: AgentRun) -> str:
    events = list(read_events(run._jsonl()))
    lines = [f"# 历史研究轨迹 · {run.run_id}"]
    for event in events:
        kind = event.get("event")
        if kind in {"user_message", "agent_thinking", "assistant_message"}:
            content = str(event.get("content", "")).strip()
            if content:
                lines.append(f"## {kind}\n{content}")
        elif kind == "assistant_tool_call":
            lines.append(
                "## 工具调用\n"
                f"{event.get('name', '')}\n{event.get('arguments_raw', '')}"
            )
        elif kind == "tool_results":
            for row in event.get("results") or []:
                result = row.get("result") if isinstance(row, dict) else {}
                summary = result.get("summary", {}) if isinstance(result, dict) else {}
                lines.append(
                    "## 评估结果\n"
                    f"工具: {row.get('name', '')}\n"
                    f"IC={summary.get('ic')} RankIC={summary.get('rank_ic')} "
                    f"ICIR={summary.get('icir')} Coverage={summary.get('factor_coverage')}\n"
                    f"状态: {result.get('error') or result.get('skipped_reason') or 'ok'}"
                )
    text = "\n\n".join(lines)
    return text


def _resume_context(run: AgentRun) -> str:
    """Include every ancestor so branching a branch retains the original research."""
    lineage: list[AgentRun] = []
    seen: set[str] = set()
    current: AgentRun | None = run
    while current is not None and current.run_id not in seen:
        lineage.append(current)
        seen.add(current.run_id)
        current = get_run(current.parent_run_id) if current.parent_run_id else None
    return "\n\n".join(_run_resume_context(item) for item in reversed(lineage))[-24000:]


def _start_from_history(parent: AgentRun, content: str) -> AgentRun:
    context_path = parent.log_dir / "resume_context.md"
    context_path.write_text(_resume_context(parent), encoding="utf-8")
    params = {**parent.params, "user_message": content}
    return start_run(params, parent_run_id=parent.run_id, resume_context_file=context_path)


def branch_run(run_id: str, content: str) -> AgentRun | None:
    parent = get_run(run_id)
    if parent is None:
        return None
    return _start_from_history(parent, content)


def continue_run(run_id: str, content: str) -> AgentRun | None:
    parent = get_run(run_id)
    if parent is None:
        return None
    parent.refresh()
    if parent.status in {"starting", "running"}:
        return parent if queue_message(run_id, content) else None
    if parent.status not in {"completed", "failed", "interrupted"}:
        return None
    return _start_from_history(parent, content)


async def event_stream(run_id: str):
    run = get_run(run_id)
    if run is None:
        yield {"event": "error", "error": "run_not_found"}
        return
    offset = 0
    idle_after_exit = 0
    yield {"event": "stream_start", "run_id": run_id, "status": run.status}
    while True:
        run.refresh()
        path = run._jsonl()
        emitted = False
        if path and path.exists():
            with path.open("r", encoding="utf-8") as handle:
                handle.seek(offset)
                lines = handle.readlines()
                offset = handle.tell()
                for line in lines:
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(row, dict):
                        emitted = True
                        yield row
        if run.status in {"completed", "failed"}:
            idle_after_exit += 1
            if not emitted and idle_after_exit >= 3:
                yield {"event": "stream_end", "status": run.status}
                return
        if not emitted:
            yield {"event": "heartbeat", "status": run.status}
        await asyncio.sleep(0.5)


# ══════════════════════════════════════════════════════════════════════
#  单因子评估（独立于挖掘流程）
# ══════════════════════════════════════════════════════════════════════

def _get_eval_service(
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    val_start: str = DEFAULT_VAL_START,
    val_end: str = DEFAULT_VAL_END,
    label_col: str = "label_1d_open_to_open",
    include_fundamentals: bool = False,
) -> Any:
    """懒初始化 StockEvalService 并复用 session。

    panel 加载很重（~11M 行），首次调用时创建 session 后复用。
    如果参数变更则重建 session。
    """
    global _eval_service, _eval_service_params

    params_key = {
        "train_start": train_start,
        "train_end": train_end,
        "val_start": val_start,
        "val_end": val_end,
        "label_col": label_col,
        "include_fundamentals": include_fundamentals,
    }

    with _eval_service_lock:
        if _eval_service is not None and _eval_service_params == params_key:
            # 参数一致，复用
            return _eval_service

        from alphaagent.factor.mining.service import StockEvalService
        from alphaagent.factor.mining.schemas import SessionCreateRequest

        svc = StockEvalService()
        req = SessionCreateRequest(
            panel_path=DEFAULT_PANEL,
            train_start=train_start,
            train_end=train_end,
            val_start=val_start,
            val_end=val_end,
            label_col=label_col,
            include_fundamentals=include_fundamentals,
        )
        resp = svc.create_session(req)
        _eval_service = svc
        _eval_service_params = params_key
        return svc


def evaluate_single_factor(
    *,
    multi_line_expr: str,
    factor_name: str = "expr",
    profile_id: str = "train_screen",
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    val_start: str = DEFAULT_VAL_START,
    val_end: str = DEFAULT_VAL_END,
    label_col: str = "label_1d_open_to_open",
    include_fundamentals: bool = False,
) -> dict[str, Any]:
    """独立评估一个因子表达式。"""
    from alphaagent.factor.mining.schemas import EvalProfileRequest

    svc = _get_eval_service(
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        label_col=label_col,
        include_fundamentals=include_fundamentals,
    )

    # 使用已有的 session（第一个）
    session_ids = list(svc.sessions._sessions.keys())
    if not session_ids:
        raise RuntimeError("eval_service_session_empty")
    session_id = session_ids[-1]

    # 如果 profile_id 包含 "train" 或 "val" 分别走不同 split
    req = EvalProfileRequest(
        session_id=session_id,
        profile_id=profile_id,
        multi_line_expr=multi_line_expr,
        factor_name=factor_name,
    )
    return svc.eval_profile(req)


def evaluate_multi_profile(
    *,
    multi_line_expr: str,
    factor_name: str = "expr",
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    val_start: str = DEFAULT_VAL_START,
    val_end: str = DEFAULT_VAL_END,
    label_col: str = "label_1d_open_to_open",
    include_fundamentals: bool = False,
) -> dict[str, Any]:
    """一次评估多个 profile（train_screen + validation + size_neutral_validation）。"""
    from alphaagent.factor.mining.schemas import EvalProfileRequest

    svc = _get_eval_service(
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        label_col=label_col,
        include_fundamentals=include_fundamentals,
    )

    session_ids = list(svc.sessions._sessions.keys())
    if not session_ids:
        raise RuntimeError("eval_service_session_empty")
    session_id = session_ids[-1]

    results: dict[str, Any] = {}
    for profile_id in ("train_screen", "validation", "size_neutral_validation"):
        req = EvalProfileRequest(
            session_id=session_id,
            profile_id=profile_id,
            multi_line_expr=multi_line_expr,
            factor_name=factor_name,
        )
        results[profile_id] = svc.eval_profile(req)

    return results


# ══════════════════════════════════════════════════════════════════════
#  因子库管理
# ══════════════════════════════════════════════════════════════════════

def _candidate_root(category: str = "technical") -> Path:
    return factor_categories.candidate_dir(category)


def _candidate_registry(category: str = "technical") -> dict[str, Any]:
    path = _candidate_root(category) / "mining_candidate_registry.json"
    if not path.is_file():
        return {}
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _candidate_expr(entry: dict[str, Any], factor_id: str, *, category: str = "technical") -> str:
    expr = str(entry.get("expr") or "")
    if expr:
        return expr
    rel = str(entry.get("expression_file") or "")
    path = ROOT / rel if rel else _candidate_root(category) / "expressions" / f"{factor_id}.dsl"
    return path.read_text(encoding="utf-8").strip() if path.exists() else ""


def _candidate_factor_view(factor_id: str, entry: dict[str, Any], *, category: str = "technical") -> dict[str, Any]:
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    fingerprint = entry.get("data_fingerprint") if isinstance(entry.get("data_fingerprint"), dict) else {}
    finite_count: int | None = None
    if metrics.get("finite_ratio") is not None and fingerprint.get("n_rows") is not None:
        finite_count = int(float(metrics["finite_ratio"]) * int(fingerprint["n_rows"]))
    review = entry.get("review") if isinstance(entry.get("review"), dict) else {}

    # ── 提取 train/val 分拆指标：优先读 submit 写入的分窗口字段，回退评估证据包 ──
    ee = entry.get("evaluation_evidence") if isinstance(entry.get("evaluation_evidence"), dict) else {}
    train_ic = _safe_float(metrics.get("train_ic"))
    val_ic = _safe_float(metrics.get("val_ic"))
    val_icir = _safe_float(metrics.get("val_icir"))
    val_retention = _safe_float(metrics.get("val_ic_retention"))
    if train_ic is None:
        for split_entry in ee.get("train", []):
            s = split_entry.get("summary") or {}
            train_ic = _safe_float(s.get("ic"))
            break
    if val_ic is None:
        for split_entry in ee.get("validation", []):
            s = split_entry.get("summary") or {}
            val_ic = _safe_float(s.get("ic"))
            val_icir = _safe_float(s.get("icir"))
            break

    # ── 组合层收益指标（quantile_portfolio 由提交/回填写入）──
    qp = metrics.get("quantile_portfolio") if isinstance(metrics.get("quantile_portfolio"), dict) else {}

    # ── label 与研究模式 ──
    ingest_cfg = entry.get("ingest_config") or {}
    label_col = str(ingest_cfg.get("label_col") or "")

    # ── comment 预览 ──
    comment_full = str(entry.get("comment") or "")
    comment_preview = comment_full[:150] + ("…" if len(comment_full) > 150 else "")

    # ── reviewer 意见摘要 ──
    reasons_list = review.get("reasons") or []
    review_summary = "; ".join(str(r) for r in reasons_list[:3]) if reasons_list else ""

    return {
        "factor_id": factor_id,
        "name": str(entry.get("name") or factor_id),
        "expr": _candidate_expr(entry, factor_id),
        "col_idx": None,
        "status": str(entry.get("review_status") or "pending_review"),
        "promotion_status": str(entry.get("promotion_status") or "pending"),
        "review_verdict": str(review.get("verdict") or ""),
        "finite_count": finite_count or 0,
        "created_at": str(entry.get("ingested_at") or ""),
        "comment_preview": comment_preview,
        "comment_full": comment_full,
        "label_col": label_col,
        "train_ic": train_ic,
        "val_ic": val_ic,
        "val_icir": val_icir,
        "val_ic_retention": val_retention,
        "annualized_return": _safe_float(qp.get("top_group_annualized_return")),
        "annualized_excess_return": _safe_float(qp.get("top_group_annualized_excess_return")),
        "sharpe": _safe_float(qp.get("top_group_sharpe")),
        "review_reasons": review_summary,
        "metrics": {
            "ic": _safe_float(metrics.get("ic")),
            "icir": _safe_float(metrics.get("icir")),
            "rank_ic": _safe_float(metrics.get("rank_ic")),
            "factor_coverage": _safe_float(metrics.get("factor_coverage", metrics.get("coverage"))),
        },
        "extra": entry,
    }

def list_factors(*, library: str = "production", category: str = "technical") -> dict[str, Any]:
    """列出因子库中的所有因子。

    library:
      - "production": 正式因子库
      - "candidate": 候选因子库
    category:
      - "technical": 日线技术因子
      - "fundamental": 基本面因子
    """
    from alphaagent.factor.zoo import FactorZoo

    prod_root = factor_categories.production_dir(category)
    cand_root = _candidate_root(category)

    root = prod_root if library == "production" else cand_root

    if library == "candidate":
        registry = _candidate_registry(category)
        factors = [_candidate_factor_view(fid, entry, category=category) for fid, entry in sorted(registry.items())]
        return {
            "library": library,
            "category": category,
            "root": str(root),
            "n_factors": len(factors),
            "factors": factors,
            "registry": registry,
        }

    try:
        # Listing must not re-hash the 82 MB canonical row index on every request.
        zoo = FactorZoo.open(root, verify_hash=False)
    except FileNotFoundError:
        return {"library": library, "category": category, "root": str(root), "factors": [], "n_factors": 0, "error": "library_not_initialized"}

    df = zoo.catalog.to_dataframe()
    factors: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        extra = row.get("extra")
        if isinstance(extra, str) and extra:
            try:
                import json as _json
                extra = _json.loads(extra)
            except (ValueError, TypeError):
                extra = {}
        elif extra is None or (isinstance(extra, float) and pd.isna(extra)):
            extra = {}

        metrics = (extra or {}).get("metrics", {})
        factors.append({
            "factor_id": str(row["factor_id"]),
            "name": str(row["name"]),
            "expr": str(row["expr"]),
            "col_idx": int(row["col_idx"]),
            "status": str(row.get("status", "")),
            "finite_count": int(row.get("finite_count", 0)),
            "created_at": str(row.get("created_at", "")),
            "metrics": {
                "ic": _safe_float(metrics.get("ic")),
                "icir": _safe_float(metrics.get("icir")),
                "rank_ic": _safe_float(metrics.get("rank_ic")),
                "factor_coverage": _safe_float(metrics.get("factor_coverage")),
            } if metrics else {},
            "extra": extra,
        })

    # 读取 registry（如有）
    registry_path = root / "mining_delivered_registry.json"
    registry: dict[str, Any] = {}
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))

    return {
        "library": library,
        "category": category,
        "root": str(root),
        "n_factors": len(factors),
        "factors": factors,
        "registry": registry,
    }


def get_factor_detail(factor_id: str, *, library: str = "production", category: str = "technical") -> dict[str, Any]:
    """获取单个因子详情。"""
    from alphaagent.factor.zoo import FactorZoo

    prod_root = factor_categories.production_dir(category)
    cand_root = _candidate_root(category)

    root = prod_root if library == "production" else cand_root

    if library == "candidate":
        entry = _candidate_registry(category).get(factor_id)
        if not isinstance(entry, dict):
            return {"error": "factor_not_found"}
        view = _candidate_factor_view(factor_id, entry, category=category)
        view["registry_entry"] = entry
        return view

    try:
        zoo = FactorZoo.open(root, verify_hash=False)
    except FileNotFoundError:
        return {"error": "library_not_initialized"}

    meta = zoo.catalog.get(factor_id)
    if meta is None:
        return {"error": "factor_not_found"}

    # 读取 registry 获取完整评估信息
    registry_path = root / "mining_delivered_registry.json"
    reg_entry: dict[str, Any] = {}
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        reg_entry = registry.get(factor_id, {})

    return {
        "factor_id": meta.factor_id,
        "name": meta.name,
        "expr": meta.expr,
        "col_idx": meta.col_idx,
        "status": meta.status.value,
        "finite_count": meta.finite_count,
        "created_at": meta.created_at,
        "extra": meta.extra,
        "registry_entry": reg_entry,
    }


def delete_factor(factor_id: str, *, library: str = "production", category: str = "technical") -> dict[str, Any]:
    """删除一个因子，并同步清除研究记忆/RAG 中的相关条目。

    进程还活着时拒绝删除——先 stop 再删，避免边写边删。
    """
    from alphaagent.factor.mining.research_memory import ResearchMemoryStore
    from core import trading_config
    from alphaagent.factor.zoo import FactorZoo

    prod_root = factor_categories.production_dir(category)
    cand_root = _candidate_root(category)

    root = prod_root if library == "production" else cand_root
    factor_names: list[str] = []
    expressions: list[str] = []

    if library == "candidate":
        registry = _candidate_registry(category)
        if factor_id not in registry:
            return {"error": "factor_not_found"}
        entry = registry.pop(factor_id)
        if str(entry.get("name") or "").strip():
            factor_names.append(str(entry["name"]).strip())
        if str(entry.get("expr") or "").strip():
            expressions.append(str(entry["expr"]))
        registry_path = root / "mining_candidate_registry.json"
        registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        rel = str(entry.get("expression_file") or "")
        dsl_path = ROOT / rel if rel else root / "expressions" / f"{factor_id}.dsl"
        if dsl_path.exists():
            dsl_path.unlink()
    else:
        try:
            zoo = FactorZoo.open(root, verify_hash=False)
        except FileNotFoundError:
            return {"error": "library_not_initialized"}

        try:
            meta = zoo.catalog.get(factor_id)
            if meta is not None:
                if str(meta.name or "").strip():
                    factor_names.append(str(meta.name).strip())
                if str(meta.expr or "").strip():
                    expressions.append(str(meta.expr))
            zoo.delete_factor(factor_id)
        except KeyError:
            return {"error": "factor_not_found"}

        # 从 registry 中删除
        registry_path = root / "mining_delivered_registry.json"
        if registry_path.is_file():
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
            removed = registry.pop(factor_id, None)
            if isinstance(removed, dict):
                if str(removed.get("name") or "").strip():
                    factor_names.append(str(removed["name"]).strip())
                if str(removed.get("expr") or "").strip():
                    expressions.append(str(removed["expr"]))
            registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    purged = ResearchMemoryStore(RESEARCH_MEMORY_FILE).purge_factor(
        factor_names=factor_names,
        expressions=expressions,
    )
    return {"ok": True, "factor_id": factor_id, "deleted": True, "memory_purged": purged}


def _safe_float(v: Any) -> float | None:
    """安全转 float，None/NaN → None。"""
    if v is None:
        return None
    try:
        f = float(v)
        import math
        return f if math.isfinite(f) else None
    except (TypeError, ValueError):
        return None


# ══════════════════════════════════════════════════════════════════════
#  因子实验室：保存因子到因子库
# ══════════════════════════════════════════════════════════════════════

def save_factor(
    *,
    multi_line_expr: str,
    factor_name: str,
    comment: str = "",
    library: str = "candidate",
    category: str = "technical",
    train_start: str = DEFAULT_TRAIN_START,
    train_end: str = DEFAULT_TRAIN_END,
    val_start: str = DEFAULT_VAL_START,
    val_end: str = DEFAULT_VAL_END,
    label_col: str = "label_1d_open_to_open",
    include_fundamentals: bool = False,
) -> dict[str, Any]:
    """将因子表达式保存到因子库（候选池或正式库）。

    复用挖掘流程的 ingest_factor + upsert_mining_registry 链路。
    category 按 research_mode 路由到对应类别的目录。
    """
    from alphaagent.factor.mining.submit import slug_factor_id
    from alphaagent.factor.ingest import ingest_factor, load_panel_for_zoo, prepare_stored_values
    from alphaagent.factor.types import IngestPolicy
    from alphaagent.factor.zoo import FactorZoo
    from alphaagent.factor.mining.registry_io import upsert_mining_registry, write_candidate_registry

    factor_id = slug_factor_id(factor_name)
    name = str(factor_name).strip() or factor_id

    # 获取 eval_service 的 session 以复用已加载的 panel
    svc = _get_eval_service(
        train_start=train_start, train_end=train_end,
        val_start=val_start, val_end=val_end,
        label_col=label_col, include_fundamentals=include_fundamentals,
    )
    session_ids = list(svc.sessions._sessions.keys())
    if not session_ids:
        raise RuntimeError("eval_service_session_empty")
    session = svc.sessions.get(session_ids[-1])
    ctx = session.ctx

    # Candidate records are registry-only; production remains a dense FactorZoo.
    if library == "production":
        zoo_root = factor_categories.production_dir(category)
        registry_path = zoo_root / "mining_delivered_registry.json"
        expr_dir = zoo_root / "expressions"
    else:
        zoo_root = _candidate_root(category)
        registry_path = zoo_root / "mining_candidate_registry.json"
        expr_dir = zoo_root / "expressions"

    try:
        zoo = FactorZoo.open(zoo_root)
    except FileNotFoundError:
        return {"ok": False, "factor_id": factor_id, "error": f"factorlib_not_initialized:{zoo_root}"}

    # 加载 panel
    panel = load_panel_for_zoo(zoo, panel_path=ctx.panel_path)

    # 构建 ingest policy
    policy = IngestPolicy.from_context(ctx, max_cs_corr=1.0, similar_top_k=3)

    prepared_values, _, _, _ = prepare_stored_values(multi_line_expr.strip(), panel, zoo, policy)
    assessment = ingest_factor(
        zoo,
        factor_id=factor_id,
        name=name,
        expr=multi_line_expr.strip(),
        panel=panel,
        policy=policy,
        stored_values=prepared_values,
        dry_run=library == "candidate",
        overwrite=True,
    )

    if library == "candidate":
        fingerprint = {
            "panel_path": str(ctx.panel_path),
            "index_hash": zoo.manifest.index_hash,
            "n_rows": int(zoo.manifest.n_rows),
        }
        reg_path, dsl_path = write_candidate_registry(
            registry_path,
            factor_id=factor_id,
            name=name,
            comment=comment or name,
            expr=multi_line_expr.strip(),
            expr_dir=expr_dir,
            repo_root=ROOT,
            policy=policy,
            metrics=assessment.metrics,
            similarity=assessment.similarity,
            source="lab_save",
            data_fingerprint=fingerprint,
        )
        return {
            "ok": True,
            "factor_id": factor_id,
            "factor_name": name,
            "library": library,
            "candidate_stored": True,
            "candidate_storage": "registry_only",
            "review_status": "pending_review",
            "stored": False,
            "metrics": assessment.metrics,
            "similarity": assessment.similarity,
            "registry_path": reg_path,
            "dsl_path": dsl_path,
        }

    if not assessment.stored and assessment.skipped_reason not in (None, "already_exists"):
        return {
            "ok": False,
            "factor_id": factor_id,
            "error": assessment.skipped_reason or "production_ingest_failed",
        }

    reg_path, dsl_path = upsert_mining_registry(
        registry_path,
        factor_id=factor_id,
        name=name,
        comment=comment or name,
        expr=multi_line_expr.strip(),
        expr_dir=expr_dir,
        repo_root=ROOT,
        policy=policy,
        metrics=assessment.metrics,
        similarity=assessment.similarity,
        ingest_status="production",
        source="lab_save",
    )

    return {
        "ok": True,
        "factor_id": factor_id,
        "factor_name": name,
        "library": library,
        "stored": True,
        "metrics": assessment.metrics,
        "similarity": assessment.similarity,
        "registry_path": reg_path,
        "dsl_path": dsl_path,
    }


# ══════════════════════════════════════════════════════════════════════
#  因子实验室：因子回测
# ══════════════════════════════════════════════════════════════════════


def _eval_factor_matrix(
    *,
    multi_line_expr: str,
    factor_name: str,
    calc_start: str,
    end: str,
) -> pd.DataFrame:
    """Evaluate a DSL expression and return a date × code score matrix."""
    import numpy as np
    from alphaagent.data.adapters.cnequity import load_panel_from_cne
    from alphaagent.dsl import eval_factor
    from alphaagent.factor.align import align_series_to_panel

    panel = load_panel_from_cne(start=calc_start, end=end, universe_mask=False)
    panel = panel.sort_index()
    if panel.empty:
        raise ValueError(f"CNE 面板在区间内无数据: {calc_start} ~ {end}")

    raw = eval_factor(multi_line_expr, panel)
    if not isinstance(raw, pd.Series):
        raise TypeError(f"factor_output_must_be_series:{type(raw)!r}")

    values = align_series_to_panel(raw, panel)
    factor = pd.Series(values, index=panel.index, name=factor_name, dtype=np.float32)
    factor_df = factor.unstack(level="instrument")
    factor_df.index.name = "date"
    factor_df.columns.name = "code"

    # AlphaAgent 使用 000001.SZ，本地回测面板使用六位代码。
    factor_df.columns = [str(code).split(".")[0] for code in factor_df.columns]
    return factor_df


def backtest_factor(
    *,
    multi_line_expr: str,
    factor_name: str = "expr",
    start: str = "2023-01-01",
    end: str = "2025-12-31",
    top_n: int = 5,
    freq: str = "monthly",
    capital: float = 100000.0,
    ascending: bool = False,
    universe: str = "全部股票",
    exclude_kechuang: bool = False,
    warmup_days: int = 400,
) -> dict[str, Any]:
    """用因子表达式驱动主回测引擎 core.engine.run_backtest。"""
    from backend import services as sv
    from core.assets import STOCK_PROFILE
    from core.engine import run_backtest

    start_ts = pd.Timestamp(start)
    calc_ts = (start_ts - pd.Timedelta(days=max(0, int(warmup_days)))
               if warmup_days > 0 else start_ts)
    calc_start = calc_ts.date().isoformat()

    codes = sv.build_codes(universe=universe, exclude_kechuang=exclude_kechuang)
    if not codes:
        raise ValueError(f"股票池为空: {universe}")

    data = sv.load_data(
        start=calc_start,
        end=end,
        codes=codes,
        need_panel=True,
        need_heavy=False,
    )
    bt_panel = data.get("panel")
    if bt_panel is None or bt_panel.empty:
        raise ValueError("本地回测面板在区间内无数据")

    factor_df = _eval_factor_matrix(
        multi_line_expr=multi_line_expr,
        factor_name="_lab_factor",
        calc_start=calc_start,
        end=end,
    )
    bt_dates = pd.DatetimeIndex(sorted(bt_panel["date"].unique()))
    bt_codes = [str(code) for code in bt_panel["code"].unique()]
    factor_df = factor_df.reindex(index=bt_dates, columns=bt_codes)

    # 因子矩阵包含预热段；run_backtest 只从 start 开始输出净值。
    res = run_backtest(
        panel=bt_panel,
        codes=codes,
        factor="pred",
        external_scores=factor_df,
        ascending=ascending,
        start=start,
        end=end,
        capital=capital,
        top_n=top_n,
        freq=freq,
        affordable=True,
        amount_q=0.2,
        warmup_days=int(warmup_days),
        cash_mode=True,
        limit_flags=True,
        lot_size=100,
        buy_cost=trading_config.BUY_COST,
        sell_cost=trading_config.SELL_COST,
        slippage_bps=trading_config.SLIPPAGE_BPS,
        max_participation=trading_config.MAX_PARTICIPATION,
        execution_profile=STOCK_PROFILE,
    )

    nm = sv.get_name_map()
    holdings = res["holdings"].copy()
    holdings["name"] = [nm.get(str(c), "") for c in holdings["code"]]

    metrics = {k: _safe_float(v) for k, v in res["metrics"].items()}
    bench_metrics = {k: _safe_float(v) for k, v in res["bench_metrics"].items()}

    nav_points = sv.series_to_points(res["nav"])
    bench_points = sv.series_to_points(res["bench"])
    dd_points = sv.series_to_points(res["drawdown"])

    return {
        "ok": True,
        "metrics": metrics,
        "bench_metrics": bench_metrics,
        "nav": nav_points,
        "bench": bench_points,
        "drawdown": dd_points,
        "holdings": holdings.to_dict(orient="records"),
        "trades": res["trades"].to_dict(orient="records"),
        "last_signal_date": str(res["last_signal_date"].date()) if res["last_signal_date"] else None,
        "config": {
            "universe": universe,
            "n_codes": len(codes),
            "start": start,
            "end": end,
            "top_n": top_n,
            "freq": freq,
            "capital": capital,
            "warmup_days": int(warmup_days),
            "cash_mode": True,
        },
    }
