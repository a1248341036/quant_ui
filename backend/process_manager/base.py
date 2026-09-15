"""跨平台子进程管理器基类与模板方法 (Template Method & Strategy Pattern)。

统一 AgentRun, JQRun, Stacking, Overnight 等子进程的生命周期管理，
内置 Windows/POSIX 递归进程树终止策略，杜绝后台孤儿工作进程泄漏。
"""

from __future__ import annotations

import platform
import subprocess
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Iterator, TypeVar


class AlreadyRunningError(RuntimeError):
    """当前已有活跃运行且已达最大并发上限时抛出。"""


@dataclass
class BaseRunState:
    """运行状态基类。"""

    run_id: str
    command: list[str] = field(default_factory=list)
    process: subprocess.Popen | None = None
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    exit_code: int | None = None
    status: str = "pending"  # pending | running | finished | failed | stopped
    error: str | None = None


T = TypeVar("T", bound=BaseRunState)


class SubprocessManager(ABC, Generic[T]):
    """子进程管理器基类（模板方法模式）。"""

    def __init__(self, max_active: int = 1) -> None:
        self._runs: dict[str, T] = {}
        self._lock = threading.RLock()
        self._max_active = max_active

    # ── 模板方法（Template Methods）──

    def start(self, params: dict[str, Any]) -> dict[str, Any]:
        """启动新 run 的标准生命周期。"""
        with self._lock:
            self._pre_start_check()
            run = self._create_run(params)
            self._runs[run.run_id] = run
            try:
                run.command = self._build_command(run)
                run.process = self._spawn_process(run)
                run.status = "running"
                self._start_output_drain(run)
                self._save_meta(run)
            except Exception as exc:
                run.status = "failed"
                run.error = str(exc)
                run.finished_at = time.time()
                raise
            return self._to_dict(run)

    def stop(self, run_id: str) -> dict[str, Any]:
        """终止指定 run。采用跨平台进程树强制终止策略。"""
        with self._lock:
            run = self._runs.get(run_id)
            if not run or not run.process:
                return {}
            if run.process.poll() is None:
                self._kill_process_tree(run.process.pid)
                try:
                    run.process.wait(timeout=2.0)
                except Exception:
                    pass
            run.status = "stopped"
            run.finished_at = time.time()
            run.exit_code = run.process.poll()
            self._save_meta(run)
            return self._to_dict(run)

    def list_runs(self) -> list[dict[str, Any]]:
        with self._lock:
            self._refresh_active_states()
            return [self._to_dict(r) for r in self._runs.values()]

    def get_run(self, run_id: str) -> T | None:
        with self._lock:
            run = self._runs.get(run_id)
            if run:
                self._refresh_run_state(run)
            return run

    def delete_run(self, run_id: str) -> bool:
        with self._lock:
            run = self._runs.get(run_id)
            if run and run.process and run.process.poll() is None:
                self.stop(run_id)
            if run_id in self._runs:
                del self._runs[run_id]
                return True
            return False

    # ── 钩子方法（子类可扩展/必须实现）──

    @abstractmethod
    def _create_run(self, params: dict[str, Any]) -> T:
        """从参数字典构造特定类型的 Run 对象。"""

    @abstractmethod
    def _build_command(self, run: T) -> list[str]:
        """为 Run 对象生成命令行执行参数。"""

    @abstractmethod
    def _save_meta(self, run: T) -> None:
        """持久化 run 元数据。"""

    @abstractmethod
    def _drain_output(self, run: T) -> None:
        """排空输出（写日志/SSE 推流等）。"""

    @abstractmethod
    def _to_dict(self, run: T) -> dict[str, Any]:
        """序列化输出字典。"""

    def _spawn_process(self, run: T) -> subprocess.Popen:
        """创建子进程（默认使用 subprocess.Popen）。"""
        return subprocess.Popen(
            run.command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _start_output_drain(self, run: T) -> None:
        """在后台线程中排空输出。"""
        t = threading.Thread(target=self._drain_output, args=(run,), daemon=True)
        t.start()

    def _pre_start_check(self) -> None:
        """启动前并发数检查。"""
        self._refresh_active_states()
        active = sum(1 for r in self._runs.values() if r.status == "running")
        if active >= self._max_active:
            raise AlreadyRunningError(
                f"当前已有 {active} 个活跃任务，达到上限 {self._max_active}"
            )

    def _refresh_active_states(self) -> None:
        for r in self._runs.values():
            self._refresh_run_state(r)

    def _refresh_run_state(self, run: T) -> None:
        if run.process and run.status == "running":
            ret = run.process.poll()
            if ret is not None:
                run.exit_code = ret
                run.status = "finished" if ret == 0 else "failed"
                run.finished_at = time.time()

    def _kill_process_tree(self, pid: int) -> None:
        """跨平台递归杀掉进程树，防止 Windows 平台多线程 worker 孤儿泄漏。"""
        if platform.system() == "Windows":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(pid)],
                capture_output=True,
                check=False,
            )
        else:
            try:
                import psutil
                parent = psutil.Process(pid)
                for child in parent.children(recursive=True):
                    child.terminate()
                parent.terminate()
            except Exception:
                try:
                    import os, signal
                    os.kill(pid, signal.SIGTERM)
                except Exception:
                    pass
