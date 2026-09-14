"""统一日志系统模块（合并自 logging_config 与 logging_decorators）。

包含：
1. 统一日志配置、文件轮转、专用 Logger 实例；
2. 函数调用日志装饰器、数据/回测操作上下文管理器；
3. 日志解析与过滤工具。
"""
from __future__ import annotations

import functools
import logging
import os
import re
import time
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable

# 日志目录
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# 日志文件路径
MAIN_LOG_FILE = LOG_DIR / "quant_ui.log"
API_LOG_FILE = LOG_DIR / "api.log"
ERROR_LOG_FILE = LOG_DIR / "error.log"

# 日志配置
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5  # 保留 5 个备份文件
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Path | None = None,
    console: bool = True,
) -> logging.Logger:
    """获取配置好的 logger 实例。"""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if log_file:
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def setup_root_logger(level: int = logging.INFO) -> None:
    """设置根日志器，捕获所有未捕获的日志。"""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    root_logger.setLevel(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_handler = RotatingFileHandler(
        MAIN_LOG_FILE,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)


class RequestContext:
    """请求上下文管理器，用于在日志中添加追踪 ID。"""

    def __init__(self, request_id: str | None = None, **kwargs: Any) -> None:
        import uuid
        self.request_id = request_id or str(uuid.uuid4())[:8]
        self.extra = kwargs

    def __enter__(self) -> RequestContext:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        pass

    def format_message(self, message: str) -> str:
        parts = [f"[{self.request_id}]"]
        for k, v in self.extra.items():
            parts.append(f"[{k}={v}]")
        parts.append(message)
        return " ".join(parts)


# 专用 logger 实例
main_logger = get_logger("quant_ui.main", log_file=MAIN_LOG_FILE)
api_logger = get_logger("quant_ui.api", log_file=API_LOG_FILE)
data_logger = get_logger("quant_ui.data", log_file=MAIN_LOG_FILE)
backtest_logger = get_logger("quant_ui.backtest", log_file=MAIN_LOG_FILE)
paper_logger = get_logger("quant_ui.paper", log_file=MAIN_LOG_FILE)
error_logger = get_logger("quant_ui.error", log_file=ERROR_LOG_FILE)


# ── 装饰器与上下文管理器 ──

def log_function_call(logger: logging.Logger | None = None, level: int = logging.INFO):
    """自动记录函数调用、耗时与异常的装饰器。"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if logger is None:
                mod = func.__module__
                if "data" in mod:
                    log = data_logger
                elif "backtest" in mod:
                    log = backtest_logger
                else:
                    log = logging.getLogger(mod)
            else:
                log = logger

            arg_str = ", ".join([
                *[repr(a) for a in args[:3]],
                *[f"{k}={v!r}" for k, v in list(kwargs.items())[:5]],
            ])
            if len(args) > 3 or len(kwargs) > 5:
                arg_str += "..."

            log.log(level, f"Calling {func.__name__}({arg_str})")
            t0 = time.time()
            try:
                result = func(*args, **kwargs)
                dur = time.time() - t0
                log.log(level, f"Completed {func.__name__} in {dur:.3f}s")
                return result
            except Exception as e:
                dur = time.time() - t0
                error_logger.error(f"Failed {func.__name__} after {dur:.3f}s: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


class LogContext:
    """日志块上下文管理器。"""

    def __init__(
        self,
        name: str,
        logger: logging.Logger | None = None,
        level: int = logging.INFO,
        **kwargs: Any,
    ) -> None:
        self.name = name
        self.logger = logger or main_logger
        self.level = level
        self.kwargs = kwargs
        self.start_time = 0.0

    def __enter__(self) -> LogContext:
        self.start_time = time.time()
        extra = " " + " ".join(f"{k}={v}" for k, v in self.kwargs.items()) if self.kwargs else ""
        self.logger.log(self.level, f"Starting {self.name}{extra}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        dur = time.time() - self.start_time
        if exc_type is None:
            self.logger.log(self.level, f"Finished {self.name} in {dur:.3f}s")
        else:
            error_logger.error(f"Failed {self.name} after {dur:.3f}s: {exc_val}", exc_info=True)


log_block = LogContext


class log_data_operation(LogContext):
    def __init__(self, operation: str, **kwargs: Any) -> None:
        super().__init__(f"data_op:{operation}", logger=data_logger, **kwargs)


class log_backtest_execution(LogContext):
    def __init__(self, strategy_id: str = "", **kwargs: Any) -> None:
        super().__init__(f"backtest:{strategy_id}", logger=backtest_logger, **kwargs)


def parse_log_file(
    log_file: Path | str,
    level: str | None = None,
    limit: int = 100,
    search: str | None = None,
) -> list[dict[str, Any]]:
    """解析日志文件，返回结构化条目列表。"""
    p = Path(log_file)
    if not p.is_file():
        return []

    entries = []
    pattern = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[([A-Z]+)\] (.*?): (.*)$")
    try:
        with p.open("r", encoding="utf-8", errors="replace") as f:
            for line in reversed(f.readlines()):
                m = pattern.match(line.strip())
                if not m:
                    continue
                ts, lvl, name, msg = m.groups()
                if level and lvl != level.upper():
                    continue
                if search and search.lower() not in msg.lower():
                    continue
                entries.append({"timestamp": ts, "level": lvl, "logger": name, "message": msg})
                if len(entries) >= limit:
                    break
    except Exception:
        return []
    return entries
