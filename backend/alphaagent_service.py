from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from alphaagent.factor.mining.research_spec import default_research_spec, normalize_research_spec

from alphaagent.factor.mining.research_memory import ResearchMemoryStore


ROOT = Path(__file__).resolve().parents[1]
PYTHON_EXECUTABLE = ROOT / ".venv" / "Scripts" / "python.exe"
if not PYTHON_EXECUTABLE.exists():
    PYTHON_EXECUTABLE = Path(sys.executable)
DEFAULT_PANEL = "cne://"  # 从 CNE 数据湖实时构建 panel，不再依赖预构建的 parquet
DEFAULT_FACTORLIB = ROOT / "artifacts" / "alphaagent" / "factorzoo" / "stock_1d"
RESEARCH_MEMORY_FILE = ROOT / "artifacts" / "alphaagent" / "research_memory.json"
LOG_ROOT = ROOT / "logs" / "factor_mining" / "ui"


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
        events = list(read_events(self._jsonl()))
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
            "event_count": len(events),
            "events": events[-tail:],
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
    return AgentRun(
        run_id=str(metadata.get("run_id") or run_dir.name),
        command=[],
        log_dir=run_dir,
        params=params,
        created_at=created,
        status=_status_from_events(events),
        parent_run_id=metadata.get("parent_run_id"),
        title=str(metadata.get("title") or ""),
        archived=bool(metadata.get("archived", False)),
        pinned=bool(metadata.get("pinned", False)),
    )


def hydrate_runs() -> None:
    """Restore past UI runs after a FastAPI restart from their JSONL logs."""
    if not LOG_ROOT.exists():
        return
    restored = [_load_run_from_disk(path) for path in LOG_ROOT.iterdir()]
    with _LOCK:
        for run in restored:
            if run is not None and run.run_id not in _RUNS:
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


def start_run(
    params: dict[str, Any],
    *,
    parent_run_id: str | None = None,
    resume_context_file: Path | None = None,
) -> AgentRun:
    params = dict(params)
    raw_spec = params.get("research_spec")
    if raw_spec is None:
        raw_spec = {"delivery_policy": {"allow_submit": bool(params.get("allow_submit", False))}}
    spec = normalize_research_spec(raw_spec)
    params["research_spec"] = spec
    params["allow_submit"] = bool(spec["delivery_policy"]["allow_submit"])
    params["max_tool_calls_per_round"] = min(
        int(params["max_tool_calls_per_round"]),
        int(spec["search_policy"]["max_candidates_per_round"]),
    )
    run_id = uuid.uuid4().hex[:12]
    log_dir = LOG_ROOT / run_id
    log_dir.mkdir(parents=True, exist_ok=True)
    spec_path = log_dir / "research_spec.json"
    spec_path.write_text(json.dumps(spec, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    command = [
        str(PYTHON_EXECUTABLE), str(ROOT / "scripts" / "run_alphaagent.py"),
        "--panel", str(DEFAULT_PANEL),
        "--factorlib", str(DEFAULT_FACTORLIB),
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
    if params.get("no_fundamentals", True):
        command.append("--no-fundamentals")
    if not params.get("allow_submit", False):
        command.append("--no-submit")
    if resume_context_file is not None:
        command.extend(["--resume-context-file", str(resume_context_file)])
    env = os.environ.copy()
    env.setdefault("ALPHA_LLM_PROVIDER", "codex")
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


def archive_run(run_id: str) -> AgentRun | None:
    run = get_run(run_id)
    if run is None:
        return None
    run.archived = True
    run.save_meta()
    return run


def pin_run(run_id: str, pinned: bool) -> AgentRun | None:
    run = get_run(run_id)
    if run is None:
        return None
    run.pinned = pinned
    run.save_meta()
    return run


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
