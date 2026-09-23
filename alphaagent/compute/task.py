"""跨进程序列化的计算任务与结果契约。

原则：只传可序列化的最小轻量参数（str/int/float/dict/list），
面板、因子值等大内存对象在 Worker 内部按引用获取或生成，
绝不在进程间来回 pickle 大 DataFrame。
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ComputeTask:
    """计算任务对象。"""

    task_type: str  # "eval_factor" | "engine_gate" | "ping" 等
    params: dict[str, Any]
    task_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    priority: int = 0  # 0=常规, 1=高优先 (Web 实验室), -1=低优先 (离线脚本)
    created_at: float = field(default_factory=time.time)

    def __lt__(self, other: Any) -> bool:
        if not isinstance(other, ComputeTask):
            return NotImplemented
        # 优先级高的排在前面（PriorityQueue 弹出最小元素，故 priority 大者视为更小）
        if self.priority != other.priority:
            return self.priority > other.priority
        return self.created_at < other.created_at


@dataclass
class TaskResult:
    """计算任务结果。"""

    task_id: str
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    error_type: str | None = None
    elapsed_ms: float = 0.0
