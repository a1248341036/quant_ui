# -*- coding: utf-8 -*-
"""挖掘 run 步骤日志：评估 / 提交门禁 / 审查 / 记忆每一步一行日志。

- 落盘 ``<run_log_dir>/steps.log``（一行一步，本地时间，k=v 字段，可直接 tail/grep）；
- 同时镜像到 stdout（子进程 stdout 由 backend drain 进 ``console.log``）；
- 未 setup（CLI 直跑 / 测试）时仅输出到 stdout，绝不抛错阻塞挖掘。

用法::

    from alphaagent.factor.mining.runlog import log_step, set_turn, setup_run_logger

    setup_run_logger(log_dir)          # agentscope_run 入口调用一次
    set_turn(3)                        # 每个外层轮开始时
    log_step("evaluate", "vwap_dev_20 ic=+0.0300", ic=0.03, verdict="promising")
"""
from __future__ import annotations

import logging
import sys
import threading
from pathlib import Path
from typing import Any

_LOGGER_NAME = "alphaagent.mining.step"
logger = logging.getLogger(_LOGGER_NAME)

_CURRENT_TURN: int | None = None
_CONFIGURED_DIR: str | None = None
_LOCK = threading.Lock()

_FORMAT = "%(asctime)s |%(levelname)s| %(message)s"


def setup_run_logger(log_dir: Path | str | None) -> None:
    """配置 steps.log FileHandler + stdout 镜像；幂等，换目录时重建 handler。"""
    global _CONFIGURED_DIR
    with _LOCK:
        target = str(Path(log_dir)) if log_dir is not None else None
        if target is not None and target == _CONFIGURED_DIR and logger.handlers:
            return
        # 清掉旧 handler（branch/continue 换 run 目录时重建）
        for h in list(logger.handlers):
            logger.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        logger.setLevel(logging.INFO)
        logger.propagate = False
        fmt = logging.Formatter(_FORMAT, datefmt="%Y-%m-%d %H:%M:%S")
        if target is not None:
            try:
                log_path = Path(target) / "steps.log"
                log_path.parent.mkdir(parents=True, exist_ok=True)
                fh = logging.FileHandler(log_path, encoding="utf-8")
                fh.setFormatter(fmt)
                logger.addHandler(fh)
            except Exception:
                pass
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        logger.addHandler(sh)
        _CONFIGURED_DIR = target


def set_turn(turn: int | None) -> None:
    global _CURRENT_TURN
    _CURRENT_TURN = turn


def log_step(step: str, message: str = "", *, level: int = logging.INFO, **fields: Any) -> None:
    """记录一步日志；任何异常都吞掉（日志永远不能阻断挖掘）。"""
    try:
        parts: list[str] = []
        turn = _CURRENT_TURN
        parts.append(f"turn={turn if turn is not None else '-'}")
        parts.append(str(step))
        if message:
            parts.append(str(message))
        for key, value in fields.items():
            if value is None:
                continue
            if isinstance(value, float):
                text = f"{value:.4g}"
            elif isinstance(value, (list, tuple)):
                text = "[" + ",".join(str(v) for v in value) + "]"
            else:
                text = str(value)
            if len(text) > 200:
                text = text[:197] + "..."
            parts.append(f"{key}={text}")
        logger.log(level, " | ".join(parts))
    except Exception:
        pass
