"""常驻计算工作者进程（Compute Worker）。

运行在独立子进程中，常驻内存并执行计算任务（eval_factor, engine_gate 等），
避免主进程受 GIL 阻塞或单机重复冷加载面板。
"""
from __future__ import annotations

import logging
import multiprocessing as mp
import os
import queue
import sys
import time
from typing import Any

import numpy as np

from alphaagent.compute.panel_store import WorkerPanelStore
from alphaagent.compute.task import ComputeTask, TaskResult
from alphaagent.factor.evaluation.engine import EvaluationEngine
from alphaagent.factor.evaluation.profile import default_evaluation_profiles

logger = logging.getLogger(__name__)

_POISON_PILL = "__WORKER_POISON_PILL__"


class _MockSessionCtx:
    """EvaluationEngine 所需的会话上下文壳（对齐 StockEvalContext 字段）。"""

    def __init__(self, spec: dict[str, Any]):
        from alphaagent.factor.window_config import (
            DEFAULT_TRAIN_START,
            DEFAULT_TRAIN_END,
            DEFAULT_VAL_START,
            DEFAULT_VAL_END,
            DEFAULT_TEST_START,
            resolve_test_end,
        )

        self.label_col = spec.get("label_col", "label_1d_open_to_open")
        self.panel_path = spec.get("panel_path", "cne://")
        self.train_start = spec.get("train_start", DEFAULT_TRAIN_START)
        self.train_end = spec.get("train_end", DEFAULT_TRAIN_END)
        self.val_start = spec.get("val_start", DEFAULT_VAL_START)
        self.val_end = spec.get("val_end", DEFAULT_VAL_END)
        self.test_start = spec.get("test_start", DEFAULT_TEST_START)
        self.asset_type = spec.get("asset_type", "stock")
        self.test_end = spec.get("test_end") or resolve_test_end(self.asset_type)
        self.include_fundamentals = spec.get("include_fundamentals", True)

    def split_range(self, split: str) -> tuple[str, str]:
        if split == "train":
            return self.train_start, self.train_end
        elif split == "val":
            return self.val_start, self.val_end
        elif split == "test":
            return self.test_start, self.test_end
        return self.train_start, self.val_end


class _MockSession:
    """EvaluationEngine 所需的虚拟 Session 壳（对齐 StockEvalSession 接口）。"""

    def __init__(self, p: Any, spec: dict[str, Any], store: WorkerPanelStore, skey: str):
        self.panel = p
        self.spec = spec
        self.ctx = _MockSessionCtx(spec)
        self.store = store
        self.skey = skey

    def get_split_panel(self, split: str):
        start, end = self.ctx.split_range(split)
        sliced = self.store.get_split_panel(self.skey, self.spec, start, end)
        return sliced, start, end


class ComputeWorker:
    """常驻 Worker 实例，负责执行具体任务。"""

    def __init__(
        self,
        worker_id: int,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
    ) -> None:
        self.worker_id = worker_id
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.panel_store = WorkerPanelStore(max_cached=2)
        self.evaluation_engine = EvaluationEngine(default_evaluation_profiles())

    def run(self) -> None:
        """Worker 进程主循环。"""
        logger.info("[Worker %d] 启动，pid=%d", self.worker_id, os.getpid())
        while True:
            try:
                task = self.task_queue.get()
                if task == _POISON_PILL:
                    logger.info("[Worker %d] 收到停机信号，退出", self.worker_id)
                    break

                if not isinstance(task, ComputeTask):
                    continue

                t0 = time.perf_counter()
                try:
                    result_data = self._dispatch(task)
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    self.result_queue.put(
                        TaskResult(
                            task_id=task.task_id,
                            ok=True,
                            result=result_data,
                            elapsed_ms=elapsed_ms,
                        )
                    )
                except Exception as exc:  # noqa: BLE001
                    elapsed_ms = (time.perf_counter() - t0) * 1000.0
                    logger.exception("[Worker %d] 任务 %s 执行失败: %s", self.worker_id, task.task_id, exc)
                    self.result_queue.put(
                        TaskResult(
                            task_id=task.task_id,
                            ok=False,
                            error=str(exc),
                            error_type=type(exc).__name__,
                            elapsed_ms=elapsed_ms,
                        )
                    )
            except (KeyboardInterrupt, SystemExit):
                break
            except Exception as e:
                logger.error("[Worker %d] 异常: %s", self.worker_id, e)

    def _dispatch(self, task: ComputeTask) -> dict[str, Any]:
        """根据 task_type 分发任务。"""
        if task.task_type == "ping":
            return {"pong": True, "worker_id": self.worker_id, "pid": os.getpid()}
        elif task.task_type == "eval_factor":
            return self._exec_eval_factor(task.params)
        elif task.task_type == "engine_gate":
            return self._exec_engine_gate(task.params)
        else:
            raise ValueError(f"未知 task_type: {task.task_type}")

    def _exec_eval_factor(self, params: dict[str, Any]) -> dict[str, Any]:
        """执行单因子评估（对齐 StockEvalService / EvaluationEngine）。"""
        session_key = params["session_key"]
        panel_spec = params["panel_spec"]
        multi_line_expr = params["multi_line_expr"]
        factor_name = params.get("factor_name", "expr")
        profile_id = params.get("profile_id", "train_screen")
        label_quantile_n = params.get("label_quantile_n", 0)
        include_charts = params.get("include_charts", True)
        include_detail_tables = params.get("include_detail_tables", False)

        # 获取 panel
        panel = self.panel_store.get_panel(session_key, panel_spec)

        session = _MockSession(panel, panel_spec, self.panel_store, session_key)

        return self.evaluation_engine.evaluate(
            session,
            profile_id=profile_id,
            multi_line_expr=multi_line_expr,
            factor_name=factor_name,
            label_quantile_n=label_quantile_n,
            include_detail_tables=include_detail_tables,
            include_charts=include_charts,
        )

    def _exec_engine_gate(self, params: dict[str, Any]) -> dict[str, Any]:
        """执行 engine_gate 回测认证。"""
        from alphaagent.dsl import eval_factor
        from alphaagent.factor.mining.delivery.engine_gate import run_engine_gate

        session_key = params["session_key"]
        panel_spec = params["panel_spec"]
        multi_line_expr = params["multi_line_expr"]
        val_start = params["val_start"]
        val_end = params["val_end"]
        direction = params.get("direction", 1)
        policy = params.get("policy")
        asset_type = params.get("asset_type", "stock")

        panel = self.panel_store.get_panel(session_key, panel_spec)
        out = eval_factor(multi_line_expr, panel)
        values = out.reindex(panel.index).to_numpy(dtype=np.float64)

        return run_engine_gate(
            panel,
            values,
            val_start=val_start,
            val_end=val_end,
            direction=direction,
            policy=policy,
            asset_type=asset_type,
        )


def _worker_entrypoint(worker_id: int, task_queue: mp.Queue, result_queue: mp.Queue) -> None:
    """子进程入口函数。"""
    worker = ComputeWorker(worker_id, task_queue, result_queue)
    worker.run()
