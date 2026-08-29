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
from alphaagent.factor.mining.research_spec import build_run_research_spec, default_research_spec, normalize_research_spec

from alphaagent.factor.mining.research_memory import ResearchMemoryStore
from backend.logging_decorators import log_function_call
from backend.logging_config import backtest_logger
from core import factor_categories, trading_config
from backend.services import (
    SessionManager,
    FactorEvaluator,
    FactorRepository,
    FactorSubmitter,
)

logger = logging.getLogger(__name__)


ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXECUTABLE = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON_EXECUTABLE.exists():
    PYTHON_EXECUTABLE = Path(sys.executable)
DEFAULT_PANEL = "cne://"
DEFAULT_FACTORLIB = factor_categories.production_dir("technical")  # 向后兼容
RESEARCH_MEMORY_FILE = ROOT / "artifacts" / "alphaagent" / "research_memory.db"
LOG_ROOT = ROOT / "logs" / "factor_mining" / "ui"

# ── 服务管理器（使用 LRU 缓存） ──────────────────────────────────────
_session_manager: SessionManager | None = None
_factor_evaluator: FactorEvaluator | None = None
_factor_repository: FactorRepository | None = None
# 必须用 RLock：_get_factor_evaluator() 在持锁状态下会调用
# _get_session_manager()，普通 Lock 同线程二次获取会死锁（接口卡死、CPU 0）。
_service_lock = threading.RLock()


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
        jsonl = self._jsonl()

        def _terminal_from_events() -> str | None:
            # 日志已写入终态事件（session_end/session_error/run_summary）则跟随终态。
            # 终态优先于任何 running 判定：session_end 发出后进程未退出（挂住）
            # 或后端重启后 mtime 仍新鲜，都不应永远显示 running。
            if jsonl is None or not jsonl.exists():
                return None
            status = _status_from_events(_tail_events(jsonl))
            return None if status == "interrupted" else status

        # 有进程句柄：以进程退出码为准。
        if self.process is not None:
            code = self.process.poll()
            if code is None:
                # 进程活着，但若终态事件已写入且日志 120 秒无新增，
                # 视为子进程在 session_end 后挂住，跟随终态结束。
                try:
                    stale = jsonl is not None and jsonl.exists() and (
                        time.time() - jsonl.stat().st_mtime > 120
                    )
                except OSError:
                    stale = False
                if stale:
                    terminal = _terminal_from_events()
                    if terminal is not None:
                        self.status = terminal
                        return
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
            return
        # 无进程句柄（后端重启后从磁盘恢复，或 CLI 启动）：
        # 终态事件优先，其次若轨迹仍在推进（15 分钟内有新事件）视为 running。
        terminal = _terminal_from_events()
        if terminal is not None:
            self.status = terminal
            return
        if jsonl is None or not jsonl.exists():
            return
        try:
            if time.time() - jsonl.stat().st_mtime < 900:
                self.status = "running"
        except OSError:
            pass

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


def get_session_cache_stats() -> dict[str, Any]:
    """获取会话缓存统计信息。
    
    用于监控内存使用情况。
    """
    try:
        manager = _get_session_manager()
        stats = manager.get_cache_stats()
        return {
            "success": True,
            "cache_stats": stats,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def evict_all_sessions() -> dict[str, Any]:
    """清空所有会话缓存，释放内存。
    
    通常在内存压力大或参数大幅变化时调用。
    """
    try:
        manager = _get_session_manager()
        manager.evict_all()
        return {
            "success": True,
            "message": "All sessions evicted",
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
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


def _tail_events(jsonl: Path, max_bytes: int = 262144) -> list[dict[str, Any]]:
    """读取 JSONL 尾部事件（refresh 高频调用，避免全量读文件）。"""
    try:
        size = jsonl.stat().st_size
        with jsonl.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
            data = handle.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    lines = data.splitlines()
    if size > max_bytes and lines:
        lines = lines[1:]  # 首行可能是半截 JSON
    out: list[dict[str, Any]] = []
    for line in lines:
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            out.append(row)
    return out


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
    # 运行口径：注册表默认 < 用户保存覆盖（前端门槛文件）< 本次显式 spec。
    # 前端传回的有效 spec 本身已含保存值，幂等；显式 spec 缺键也从保存覆盖补齐。
    spec = build_run_research_spec(raw_spec)
    params["research_spec"] = spec
    # 研究模式决定评估 label：慢因子模式（needs_fundamentals=True）必须配慢 label，
    # 调用方传了不匹配的 label 时强制覆盖并记录警告。
    from core.research_modes import get_research_mode
    mode_spec = get_research_mode(mode)
    recommended = mode_spec.recommended_label_col
    caller_label = str(params.get("label_col") or "").strip()
    if mode_spec.needs_fundamentals and caller_label != recommended:
        logger.warning(
            "label_col 覆盖: 基本面模式要求 %s，调用方传了 %s → 已强制覆盖",
            recommended, caller_label,
        )
        params["label_col"] = recommended
    elif not caller_label:
        params["label_col"] = recommended
    params["max_tool_calls_per_round"] = min(
        int(params.get("max_tool_calls_per_round") or 8),
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
        "--max-tool-workers", str(params.get("max_tool_workers") or 8),
        "--max-tokens", str(params.get("max_tokens") or 16384),
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
    # 研究模式决定是否载入基本面列：needs_fundamentals=True 的模式必须载入
    # funda_* 字段（如基本面模式）；其余省内存。
    wants_fundamentals = bool(get_research_mode(mode).needs_fundamentals)
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
    if run is None:
        return False
    run.refresh()
    if run.status in {"completed", "failed", "stopped"}:
        return True
    if run.process is None:
        # 无进程句柄：磁盘轨迹已终态时视为已结束；否则标记为停止（进程可能已死）。
        run.status = "stopped"
        run.save_meta()
        return True
    if run.process.poll() is not None:
        run.refresh()
        return True
    run.process.terminate()
    run.status = "stopping"
    return True


def queue_message(run_id: str, content: str) -> bool:
    run = get_run(run_id)
    if run is None:
        return False
    run.refresh()
    if run.status not in {"running", "starting", "stopping"}:
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
    # 无进程句柄但磁盘轨迹仍在推进（后端重启后恢复的活跃 run）：
    # 直接删会连日志一起清掉，禁止删除，避免边写边删。
    jsonl = run._jsonl()
    if run.process is None and jsonl is not None and jsonl.exists():
        try:
            if time.time() - jsonl.stat().st_mtime < 900:
                return {"run_id": run_id, "deleted": False, "reason": "run_still_running"}
        except OSError:
            pass
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
                result = row.get("result") if isinstance(row, dict) else None
                if not isinstance(result, dict):
                    result = {}  # 键存在但为 null（如 SSE 压缩/裁剪后的历史事件）
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


_CONTINUE_DEFAULT_PARAMS: dict[str, Any] = {
    "train_start": DEFAULT_TRAIN_START,
    "train_end": DEFAULT_TRAIN_END,
    "val_start": DEFAULT_VAL_START,
    "val_end": DEFAULT_VAL_END,
    "label_col": "",  # 空 → start_run 按研究模式回填推荐 label
    "max_turns": 6,
    "max_tool_calls_per_round": 12,
    "max_tool_workers": 8,
    "max_parallel_eval": 6,
    "max_tokens": 16384,
    "population_max": 24,
    "no_fundamentals": False,
}


def _start_from_history(parent: AgentRun, content: str) -> AgentRun:
    context_path = parent.log_dir / "resume_context.md"
    context_path.write_text(_resume_context(parent), encoding="utf-8")
    # parent.params 只持久化了 user_message 等增量；continue 必须回填完整运行
    # 参数（当前默认值）+ 父 run 落盘的 research_spec（研究模式/门槛随谱系延续）。
    spec_file = parent.log_dir / "research_spec.json"
    if spec_file.exists():
        try:
            saved_spec = json.loads(spec_file.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            saved_spec = {}
    else:
        saved_spec = {}
    params = {
        **_CONTINUE_DEFAULT_PARAMS,
        "research_spec": saved_spec,
        **parent.params,
        "user_message": content,
    }
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

def _get_session_manager() -> SessionManager:
    """获取会话管理器（LRU 缓存）。
    
    全局单例，自动管理多个会话的生命周期。
    """
    global _session_manager
    with _service_lock:
        if _session_manager is None:
            # 最多缓存 3 个会话（约 6-15GB 内存）
            _session_manager = SessionManager(max_cached_sessions=3)
        return _session_manager


def _get_factor_evaluator() -> FactorEvaluator:
    """获取因子评估器。
    
    全局单例，使用 LRU 缓存管理会话。
    """
    global _factor_evaluator
    with _service_lock:
        if _factor_evaluator is None:
            session_manager = _get_session_manager()
            _factor_evaluator = FactorEvaluator(session_manager)
        return _factor_evaluator


def _get_factor_repository() -> FactorRepository:
    """获取因子仓库。
    
    全局单例。
    """
    global _factor_repository
    with _service_lock:
        if _factor_repository is None:
            _factor_repository = FactorRepository()
        return _factor_repository


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
    """独立评估一个因子表达式。

    单次请求使用一次性会话：panel 用后即释放（不缓存），避免多份全量
    panel 常驻内存。批量挖掘走子进程内复用会话，不受影响。
    """
    from alphaagent.factor.mining.schemas import (
        EvalProfileRequest,
        SessionCreateRequest,
    )
    from alphaagent.factor.mining.service import StockEvalService

    service = StockEvalService()
    create_req = SessionCreateRequest(
        panel_path=DEFAULT_PANEL,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        label_col=label_col,
        include_fundamentals=include_fundamentals,
    )
    create_resp = service.create_session(create_req)
    session_id = create_resp.session_id

    try:
        eval_req = EvalProfileRequest(
            session_id=session_id,
            profile_id=profile_id,
            multi_line_expr=multi_line_expr,
            factor_name=factor_name,
        )
        return service.eval_profile(eval_req)
    finally:
        service.release_session(session_id)


@log_function_call(logger=backtest_logger)
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
    """一次评估多个 profile（train_screen + validation + size_neutral_validation）。

    单次请求使用一次性会话：panel 用后即释放（不缓存）。
    """
    from alphaagent.factor.mining.schemas import (
        EvalProfileRequest,
        SessionCreateRequest,
    )
    from alphaagent.factor.mining.service import StockEvalService

    service = StockEvalService()
    create_req = SessionCreateRequest(
        panel_path=DEFAULT_PANEL,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        label_col=label_col,
        include_fundamentals=include_fundamentals,
    )
    create_resp = service.create_session(create_req)
    session_id = create_resp.session_id

    results: dict[str, Any] = {}
    try:
        for profile_id in ("train_screen", "validation", "size_neutral_validation"):
            eval_req = EvalProfileRequest(
                session_id=session_id,
                profile_id=profile_id,
                multi_line_expr=multi_line_expr,
                factor_name=factor_name,
            )
            results[profile_id] = service.eval_profile(eval_req)
    finally:
        service.release_session(session_id)

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
    
    使用 FactorRepository 服务，支持缓存和统一接口。

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

    # 候选库仍使用原有逻辑（包含 registry）
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

    # 正式库使用 FactorRepository
    repo = _get_factor_repository()
    factors = repo.list_factors(root, category)

    # 读取 delivered registry（submit 晋级时落盘的完整指标），合并进每个因子视图。
    # 此前只把 registry 原样附带、不合并，前端 f.train_ic/f.val_ic 全空 —— 正式库
    # 因子"指标不显示"的根因。
    registry_path = root / "mining_delivered_registry.json"
    registry: dict[str, Any] = {}
    if registry_path.is_file():
        registry = json.loads(registry_path.read_text(encoding="utf-8"))

    def _merge_production_entry(item: dict[str, Any]) -> dict[str, Any]:
        entry = (
            registry.get(str(item.get("factor_id")))
            or registry.get(str(item.get("name")))
            or {}
        )
        if not isinstance(entry, dict) or not entry:
            return item
        metrics = entry.get("ingest_metrics") if isinstance(entry.get("ingest_metrics"), dict) else {}
        if not metrics:
            metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        expr_text = entry.get("expr")
        if not expr_text:
            # delivered registry 只存 expression_file 路径，读 DSL 文件补齐
            rel = str(entry.get("expression_file") or "")
            expr_file = ROOT / rel if rel else None
            if expr_file and expr_file.is_file():
                try:
                    expr_text = expr_file.read_text(encoding="utf-8")
                except OSError:
                    expr_text = None
        merged = {
            **item,
            "expr": expr_text,
            "comment": (str(entry.get("comment") or "")[:200] or None),
            "ingest_config": entry.get("ingest_config"),
            "ingested_at": entry.get("ingested_at"),
            "metrics": metrics,
            "train_ic": metrics.get("train_ic"),
            "train_icir": metrics.get("train_icir"),
            "train_rank_ic": metrics.get("train_rank_ic"),
            "val_ic": metrics.get("val_ic"),
            "val_icir": metrics.get("val_icir"),
            "val_ic_retention": metrics.get("val_ic_retention"),
            "status": entry.get("ingest_status") or item.get("status"),
        }
        qp = metrics.get("quantile_portfolio")
        if isinstance(qp, dict):
            merged["avg_daily_side_turnover"] = qp.get("avg_daily_side_turnover")
        return merged

    factors = [_merge_production_entry(item) for item in factors]

    return {
        "library": library,
        "category": category,
        "root": str(root),
        "n_factors": len(factors),
        "factors": factors,
        "registry": registry,
    }


def get_factor_detail(factor_id: str, *, library: str = "production", category: str = "technical") -> dict[str, Any]:
    """获取单个因子详情。
    
    使用 FactorRepository 服务。
    """
    from alphaagent.factor.zoo import FactorZoo

    prod_root = factor_categories.production_dir(category)
    cand_root = _candidate_root(category)

    root = prod_root if library == "production" else cand_root

    # 候选库仍使用原有逻辑
    if library == "candidate":
        entry = _candidate_registry(category).get(factor_id)
        if not isinstance(entry, dict):
            return {"error": "factor_not_found"}
        view = _candidate_factor_view(factor_id, entry, category=category)
        view["registry_entry"] = entry
        return view

    # 正式库使用 FactorRepository + delivered registry 富化（与 list_factors 同口径）
    repo = _get_factor_repository()
    try:
        detail = repo.get_factor_detail(factor_id, root)
    except Exception as e:
        return {"error": f"factor_not_found: {str(e)}"}
    registry_path = root / "mining_delivered_registry.json"
    if registry_path.is_file():
        try:
            registry = json.loads(registry_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            registry = {}
        entry = registry.get(factor_id) or registry.get(str(detail.get("name") or "")) or {}
        if isinstance(entry, dict) and entry:
            metrics = entry.get("ingest_metrics") if isinstance(entry.get("ingest_metrics"), dict) else {}
            if not metrics:
                metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
            detail.update({
                "expr": entry.get("expr") or detail.get("expr"),
                "comment": entry.get("comment") or detail.get("comment"),
                "metrics": {**(detail.get("metrics") or {}), **metrics},
                "train_ic": metrics.get("train_ic"),
                "train_icir": metrics.get("train_icir"),
                "val_ic": metrics.get("val_ic"),
                "val_icir": metrics.get("val_icir"),
                "val_ic_retention": metrics.get("val_ic_retention"),
            })
    return detail


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

def _lab_save_review_hook(metrics: dict, similarity: dict | None) -> dict[str, any]:
    """因子实验室保存时的规则评审钩子（模拟挖掘流程的 LLM Reviewer）。
    
    基于统计门槛自动判断因子质量，返回评审意见。
    """
    ic = metrics.get("ic") or metrics.get("train_ic") or 0
    icir = metrics.get("icir") or metrics.get("train_icir") or 0
    coverage = metrics.get("factor_coverage") or metrics.get("coverage") or 0
    max_corr = similarity.get("max_abs_corr") if similarity else 0
    
    # 候选池门槛（与挖掘流程 stage_one 对齐）
    min_abs_ic = 0.015
    min_icir = 0.25
    min_coverage = 0.85
    max_abs_corr = 0.6
    
    issues = []
    
    if abs(ic) < min_abs_ic:
        issues.append(f"|IC|={abs(ic):.4f} < {min_abs_ic}")
    if abs(icir) < min_icir:
        issues.append(f"ICIR={abs(icir):.4f} < {min_icir}")
    if coverage < min_coverage:
        issues.append(f"coverage={coverage:.4f} < {min_coverage}")
    if max_corr and max_corr > max_abs_corr:
        issues.append(f"max_cs_corr={max_corr:.4f} > {max_abs_corr}")
    
    if not issues:
        return {
            "verdict": "approve",
            "conclusion": "因子达到候选池门槛，建议入库",
            "reviewer": "lab_auto_review",
        }
    elif len(issues) == 1:
        return {
            "verdict": "revise",
            "conclusion": f"因子存在小问题：{issues[0]}，建议修订后重新提交",
            "reviewer": "lab_auto_review",
            "issues": issues,
        }
    else:
        return {
            "verdict": "reject",
            "conclusion": f"因子未达门槛：{'；'.join(issues)}，建议重新设计",
            "reviewer": "lab_auto_review",
            "issues": issues,
        }


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
    from alphaagent.factor.ingest import ingest_factor, prepare_stored_values
    from alphaagent.factor.types import IngestPolicy
    from alphaagent.factor.zoo import FactorZoo
    from alphaagent.factor.mining.registry_io import upsert_mining_registry, write_candidate_registry

    factor_id = slug_factor_id(factor_name)
    name = str(factor_name).strip() or factor_id

    # 单次请求使用一次性会话：panel 用后即释放（不缓存），避免多份全量
    # panel 常驻内存占用 10GB+。
    from alphaagent.factor.mining.schemas import SessionCreateRequest
    from alphaagent.factor.mining.service import StockEvalService

    service = StockEvalService()
    create_req = SessionCreateRequest(
        panel_path=DEFAULT_PANEL,
        train_start=train_start,
        train_end=train_end,
        val_start=val_start,
        val_end=val_end,
        label_col=label_col,
        include_fundamentals=include_fundamentals,
    )
    create_resp = service.create_session(create_req)
    session_id = create_resp.session_id

    session = service.sessions.get(session_id)
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
        service.release_session(session_id)
        return {"ok": False, "factor_id": factor_id, "error": f"factorlib_not_initialized:{zoo_root}"}

    # 复用 session 中已加载的 panel（避免重复加载 cne:// 数据源）
    panel = session.panel

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

    # 合并分位组合（纯多头）夏普/年化指标到入库 metrics：compute_ingest_metrics
    # 只产出 IC/ICIR/coverage 等统计量，不包含 quantile_portfolio；
    # 若不补，因子库"多头年化/夏普"列与导出 CSV 将缺失。
    try:
        from alphaagent.factor.mining.schemas import EvalProfileRequest
        profile_req = EvalProfileRequest(
            session_id=session_id,
            profile_id="validation",
            multi_line_expr=multi_line_expr.strip(),
            factor_name=factor_id,
        )
        profile_result = service.eval_profile(profile_req)
        qp = profile_result.get("metrics", {}).get("quantile_portfolio")
        if isinstance(qp, dict):
            assessment.metrics["quantile_portfolio"] = qp
    except Exception:
        # 组合评估失败不应阻断入库（统计口径指标仍有效）
        pass

    if library == "candidate":
        # 模拟挖掘流程的评审钩子：基于统计门槛的规则评审
        metrics = assessment.metrics
        review = _lab_save_review_hook(metrics, assessment.similarity)
        review_verdict = str((review or {}).get("verdict") or "").lower()
        review_status = (
            {"approve": "approved", "revise": "revise", "reject": "rejected"}.get(review_verdict, "pending_review")
        )

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
            metrics=metrics,
            similarity=assessment.similarity,
            source="lab_save",
            data_fingerprint=fingerprint,
            evaluation_evidence={"metrics": metrics, "similarity": assessment.similarity},
        )
        
        # 记录评审结果
        if review:
            from alphaagent.factor.mining.registry_io import set_candidate_review
            set_candidate_review(
                registry_path,
                factor_id=factor_id,
                review=review,
                promotion_status=review_status,
            )

        service.release_session(session_id)
        return {
            "ok": True,
            "factor_id": factor_id,
            "factor_name": name,
            "library": library,
            "review_status": review_status,
            "review_verdict": review_verdict,
            "review": review,
            "candidate_stored": True,
            "candidate_storage": "registry_only",
            "stored": False,
            "metrics": assessment.metrics,
            "similarity": assessment.similarity,
            "registry_path": reg_path,
            "dsl_path": dsl_path,
        }

    if not assessment.stored and assessment.skipped_reason not in (None, "already_exists"):
        service.release_session(session_id)
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

    service.release_session(session_id)
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
    panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Evaluate a DSL expression and return a date × code score matrix.

    如果传入 panel（已加载的回测面板），直接复用，避免重复加载 CNE 数据源。
    """
    import time as _time
    import numpy as np
    from alphaagent.dsl import eval_factor
    from alphaagent.factor.align import align_series_to_panel
    from backend.logging_config import get_logger
    _log = get_logger("alphaagent_service")

    _t0 = _time.perf_counter()
    if panel is None:
        _log.info("[_eval_factor_matrix] panel=None, loading from CNE...")
        from alphaagent.data.adapters.cnequity import load_panel_from_cne
        panel = load_panel_from_cne(start=calc_start, end=end, universe_mask=False)
        panel = panel.sort_index()
        _log.info("[_eval_factor_matrix] CNE panel loaded: shape=%s in %.2fs",
                  panel.shape, _time.perf_counter() - _t0)
    else:
        _log.info("[_eval_factor_matrix] using provided panel: shape=%s, index type=%s",
                  panel.shape, type(panel.index).__name__)
    if panel.empty:
        raise ValueError(f"面板在区间内无数据: {calc_start} ~ {end}")

    _t1 = _time.perf_counter()
    _log.info("[_eval_factor_matrix] calling eval_factor, expr=%r", multi_line_expr[:100])
    raw = eval_factor(multi_line_expr, panel)
    _log.info("[_eval_factor_matrix] eval_factor done in %.2fs, result type=%s, len=%d",
              _time.perf_counter() - _t1, type(raw).__name__, len(raw) if hasattr(raw, '__len__') else -1)
    if not isinstance(raw, pd.Series):
        raise TypeError(f"factor_output_must_be_series:{type(raw)!r}")

    _t1 = _time.perf_counter()
    values = align_series_to_panel(raw, panel)
    factor = pd.Series(values, index=panel.index, name=factor_name, dtype=np.float32)
    factor_df = factor.unstack(level="instrument")
    factor_df.index.name = "date"
    factor_df.columns.name = "code"
    _log.info("[_eval_factor_matrix] align+unstack done in %.2fs, factor_df shape=%s",
              _time.perf_counter() - _t1, factor_df.shape)

    # AlphaAgent 使用 000001.SZ，本地回测面板使用六位代码。
    factor_df.columns = [str(code).split(".")[0] for code in factor_df.columns]
    _log.info("[_eval_factor_matrix] TOTAL done in %.2fs", _time.perf_counter() - _t0)
    return factor_df


def backtest_factor(
    *,
    multi_line_expr: str,
    factor_name: str = "expr",
    start: str = DEFAULT_VAL_START,
    end: str = DEFAULT_VAL_END,
    top_n: int = 5,
    freq: str = "monthly",
    capital: float = 100000.0,
    ascending: bool = False,
    universe: str = "全部股票",
    exclude_kechuang: bool = False,
    warmup_days: int = 400,
) -> dict[str, Any]:
    """用因子表达式驱动主回测引擎 core.engine.run_backtest。"""
    import time as _time
    from backend import services as sv
    from core.assets import STOCK_PROFILE
    from core.engine import run_backtest
    from backend.logging_config import get_logger
    _log = get_logger("alphaagent_service")

    _t0 = _time.perf_counter()
    _log.info("[backtest_factor] START expr=%r universe=%s start=%s end=%s top_n=%d",
              multi_line_expr[:80], universe, start, end, top_n)

    start_ts = pd.Timestamp(start)
    calc_ts = (start_ts - pd.Timedelta(days=max(0, int(warmup_days)))
               if warmup_days > 0 else start_ts)
    calc_start = calc_ts.date().isoformat()

    _t1 = _time.perf_counter()
    codes = sv.build_codes(universe=universe, exclude_kechuang=exclude_kechuang)
    if not codes:
        raise ValueError(f"股票池为空: {universe}")
    _log.info("[backtest_factor] build_codes done: %d codes in %.2fs", len(codes), _time.perf_counter() - _t1)

    _t1 = _time.perf_counter()
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
    _log.info("[backtest_factor] load_data done: panel shape=%s in %.2fs", bt_panel.shape, _time.perf_counter() - _t1)

    _t1 = _time.perf_counter()
    factor_df = _eval_factor_matrix(
        multi_line_expr=multi_line_expr,
        factor_name="_lab_factor",
        calc_start=calc_start,
        end=end,
        panel=None,  # 让 _eval_factor_matrix 自己加载 CNE 面板（有 MultiIndex）
    )
    _log.info("[backtest_factor] _eval_factor_matrix done: factor_df shape=%s in %.2fs", factor_df.shape, _time.perf_counter() - _t1)

    _t1 = _time.perf_counter()
    bt_dates = pd.DatetimeIndex(sorted(bt_panel["date"].unique()))
    bt_codes = [str(code) for code in bt_panel["code"].unique()]
    factor_df = factor_df.reindex(index=bt_dates, columns=bt_codes)
    _log.info("[backtest_factor] reindex done in %.2fs", _time.perf_counter() - _t1)

    # 因子矩阵包含预热段；run_backtest 只从 start 开始输出净值。
    _t1 = _time.perf_counter()
    _log.info("[backtest_factor] calling run_backtest...")
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
    _log.info("[backtest_factor] run_backtest done in %.2fs, metrics keys=%s",
              _time.perf_counter() - _t1, list(res.get("metrics", {}).keys())[:5])

    _t1 = _time.perf_counter()
    nm = sv.get_name_map()
    holdings = res["holdings"].copy()
    holdings["name"] = [nm.get(str(c), "") for c in holdings["code"]]

    metrics = {k: _safe_float(v) for k, v in res["metrics"].items()}
    bench_metrics = {k: _safe_float(v) for k, v in res["bench_metrics"].items()}

    nav_points = sv.series_to_points(res["nav"])
    bench_points = sv.series_to_points(res["bench"])
    dd_points = sv.series_to_points(res["drawdown"])

    _log.info("[backtest_factor] post-processing done in %.2fs", _time.perf_counter() - _t1)
    _log.info("[backtest_factor] TOTAL done in %.2fs", _time.perf_counter() - _t0)

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
