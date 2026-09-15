import sys
import time
import pytest
from dataclasses import dataclass
from backend.process_manager import SubprocessManager, BaseRunState, AlreadyRunningError


@dataclass
class DummyRun(BaseRunState):
    payload: str = ""


class DummyProcessManager(SubprocessManager[DummyRun]):
    def _create_run(self, params: dict) -> DummyRun:
        return DummyRun(
            run_id=params.get("run_id", "test_run"),
            payload=params.get("payload", ""),
        )

    def _build_command(self, run: DummyRun) -> list[str]:
        # 跨平台睡眠进程
        return [sys.executable, "-c", "import time; time.sleep(1)"]

    def _save_meta(self, run: DummyRun) -> None:
        pass

    def _drain_output(self, run: DummyRun) -> None:
        pass

    def _to_dict(self, run: DummyRun) -> dict:
        return {
            "run_id": run.run_id,
            "status": run.status,
            "payload": run.payload,
            "exit_code": run.exit_code,
        }


def test_subprocess_lifecycle():
    """测试子进程管理器完整生命周期：start -> get -> stop -> delete"""
    mgr = DummyProcessManager(max_active=2)
    res = mgr.start({"run_id": "run_1", "payload": "hello"})

    assert res["run_id"] == "run_1"
    assert res["status"] == "running"

    run_obj = mgr.get_run("run_1")
    assert run_obj is not None
    assert run_obj.status == "running"

    stop_res = mgr.stop("run_1")
    assert stop_res["status"] == "stopped"

    deleted = mgr.delete_run("run_1")
    assert deleted is True
    assert mgr.get_run("run_1") is None


def test_max_active_rejection():
    """测试当活跃任务数达到 max_active 时，抛出 AlreadyRunningError。"""
    mgr = DummyProcessManager(max_active=1)
    mgr.start({"run_id": "run_active_1"})

    with pytest.raises(AlreadyRunningError) as exc_info:
        mgr.start({"run_id": "run_active_2"})

    assert "达到上限 1" in str(exc_info.value)
    mgr.stop("run_active_1")
