"""统一日志模块（完整保留并严格对齐原 logging_config 与 logging_decorators 的行为与 API）。

包含：
- get_logger / setup_root_logger / setup_api_logger / setup_data_logger / setup_backtest_logger
- main_logger / api_logger / data_logger / backtest_logger / paper_logger / error_logger
- RequestContext / get_log_file_path / parse_log_file
- log_function_call / log_data_loading / log_backtest_execution / log_data_operation / log_backtest_operation
- LogContext / log_block
"""

from __future__ import annotations

import functools
import logging
import os
import re
import time
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

    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )

    if log_file is None:
        log_file = MAIN_LOG_FILE
    
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 错误日志单独追加记录到 error.log（任何 logger 的 ERROR 均写入 error.log）
    if log_file != ERROR_LOG_FILE:
        error_handler = RotatingFileHandler(
            ERROR_LOG_FILE,
            maxBytes=LOG_MAX_BYTES,
            backupCount=LOG_BACKUP_COUNT,
            encoding="utf-8",
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(formatter)
        logger.addHandler(error_handler)

    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def setup_root_logger() -> logging.Logger:
    """设置根 logger（应用启动时调用）。"""
    logger = get_logger("quant_ui", level=logging.INFO)
    logger.info("Quant UI 后端服务启动")
    return logger


def setup_api_logger() -> logging.Logger:
    return get_logger("quant_ui.api", log_file=API_LOG_FILE)


def setup_data_logger() -> logging.Logger:
    return get_logger("quant_ui.data", log_file=LOG_DIR / "data.log")


def setup_backtest_logger() -> logging.Logger:
    return get_logger("quant_ui.backtest", log_file=LOG_DIR / "backtest.log")


def get_log_file_path() -> Path:
    """获取主日志文件路径。"""
    return MAIN_LOG_FILE


# 预定义的 logger 实例
main_logger = get_logger("quant_ui")
api_logger = get_logger("quant_ui.api", log_file=API_LOG_FILE)
error_logger = get_logger("quant_ui.error", log_file=ERROR_LOG_FILE, console=False)
data_logger = get_logger("quant_ui.data", log_file=LOG_DIR / "data.log")
backtest_logger = get_logger("quant_ui.backtest", log_file=LOG_DIR / "backtest.log")
paper_logger = get_logger("quant_ui.paper", log_file=MAIN_LOG_FILE)


class RequestContext:
    """请求上下文，用于在日志中注入请求信息。"""

    def __init__(
        self,
        request_id: str | None = None,
        user_id: str | None = None,
        run_id: str | None = None,
    ):
        self.request_id = request_id
        self.user_id = user_id
        self.run_id = run_id

    def to_context(self) -> dict[str, Any]:
        """转换为日志上下文字典。"""
        ctx = {}
        if self.request_id:
            ctx["request_id"] = self.request_id
        if self.user_id:
            ctx["user_id"] = self.user_id
        if self.run_id:
            ctx["run_id"] = self.run_id
        return ctx


def parse_log_file(
    log_file: Path,
    run_id: str | None = None,
    level: str | None = None,
    limit: int = 100,
    tail: bool = True,
) -> list[dict[str, str]]:
    """解析日志文件，返回结构化日志列表。"""
    log_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] ([^:]+): (.*)'
    )
    logs = []
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        if tail:
            lines = reversed(lines)
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            match = log_pattern.match(line)
            if not match:
                continue
            
            timestamp, level_name, logger_name, message = match.groups()
            
            if level and level_name.upper() != level.upper():
                continue
            
            if run_id and run_id not in message:
                continue
            
            logs.append({
                'timestamp': timestamp,
                'level': level_name,
                'logger': logger_name,
                'message': message,
            })
            
            if len(logs) >= limit:
                break
        
        if not tail:
            logs.reverse()
        
        return logs
    except Exception:
        return []


# ── 装饰器与上下文管理器 ──

def log_function_call(logger: logging.Logger | None = None, level: int = logging.INFO):
    """装饰器：自动记录函数调用和返回。"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            if logger is None:
                mod = func.__module__
                if 'data' in mod:
                    log = data_logger
                elif 'backtest' in mod:
                    log = backtest_logger
                else:
                    log = logging.getLogger(mod)
            else:
                log = logger
            
            arg_str = ', '.join([
                *[repr(arg) for arg in args[:3]],
                *[f"{k}={v!r}" for k, v in list(kwargs.items())[:5]]
            ])
            if len(args) > 3 or len(kwargs) > 5:
                arg_str += '...'
            
            log.log(level, f"Calling {func.__name__}({arg_str})")
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                if isinstance(result, dict):
                    if 'panel' in result and result['panel'] is not None:
                        log.log(level, f"{func.__name__} completed in {duration:.2f}s: {len(result['panel'])} rows")
                    elif 'rows' in result:
                        log.log(level, f"{func.__name__} completed in {duration:.2f}s: {result['rows']} rows")
                    else:
                        log.log(level, f"{func.__name__} completed in {duration:.2f}s")
                else:
                    log.log(level, f"{func.__name__} completed in {duration:.2f}s")
                return result
            except Exception as e:
                duration = time.time() - start_time
                error_logger.error(f"{func.__name__} FAILED after {duration:.2f}s: {e}", exc_info=True)
                raise
        return wrapper
    return decorator


def log_data_loading(func: Callable) -> Callable:
    """专用装饰器：记录数据加载过程。"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        params = {
            'start': kwargs.get('start', args[0] if len(args) > 0 else None),
            'end': kwargs.get('end', args[1] if len(args) > 1 else None),
            'need_panel': kwargs.get('need_panel', True),
            'codes_count': len(kwargs.get('codes', args[2] if len(args) > 2 else []) or []),
            'need_heavy': kwargs.get('need_heavy', True),
        }
        data_logger.info(
            f"Data load: start={params['start']}, end={params['end']}, "
            f"need_panel={params['need_panel']}, codes={params['codes_count']}, "
            f"heavy={params['need_heavy']}"
        )
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            if isinstance(result, dict):
                stats = []
                if result.get('panel') is not None:
                    stats.append(f"panel={len(result['panel'])} rows")
                if result.get('universe') is not None:
                    stats.append(f"universe={len(result['universe'])} codes")
                if result.get('etf_panel') is not None and len(result['etf_panel']) > 0:
                    stats.append(f"etf_panel={len(result['etf_panel'])} rows")
                if result.get('fund_panel') is not None and len(result['fund_panel']) > 0:
                    stats.append(f"fund_panel={len(result['fund_panel'])} rows")
                stats_str = ', '.join(stats) if stats else 'no data'
                data_logger.info(f"Data load completed in {duration:.2f}s: {stats_str}")
            else:
                data_logger.info(f"Data load completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            error_logger.error(f"Data load FAILED after {duration:.2f}s: {e}", exc_info=True)
            raise
    return wrapper


def log_backtest_execution(func: Callable) -> Callable:
    """专用装饰器：记录回测执行过程。"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        params = {
            'strategy': kwargs.get('strategy', args[0] if len(args) > 0 else None),
            'universe': kwargs.get('universe', args[1] if len(args) > 1 else None),
            'start': kwargs.get('start', args[2] if len(args) > 2 else None),
            'end': kwargs.get('end', args[3] if len(args) > 3 else None),
            'top_n': kwargs.get('top_n', args[4] if len(args) > 4 else 5),
            'capital': kwargs.get('capital', args[5] if len(args) > 5 else 100000),
        }
        backtest_logger.info(
            f"Backtest: strategy={params['strategy']}, universe={params['universe']}, "
            f"range=[{params['start']}, {params['end']}], top_n={params['top_n']}, "
            f"capital={params['capital']}"
        )
        try:
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            if isinstance(result, dict) and 'metrics' in result:
                metrics = result['metrics']
                backtest_logger.info(
                    f"Backtest completed in {duration:.2f}s: "
                    f"return={metrics.get('总收益')}, "
                    f"sharpe={metrics.get('夏普')}, "
                    f"max_dd={metrics.get('最大回撤')}"
                )
            else:
                backtest_logger.info(f"Backtest completed in {duration:.2f}s")
            return result
        except Exception as e:
            duration = time.time() - start_time
            error_logger.error(f"Backtest FAILED after {duration:.2f}s: {e}", exc_info=True)
            raise
    return wrapper


class LogContext:
    """日志上下文管理器，用于记录代码块的执行。"""
    
    def __init__(self, logger: logging.Logger, message: str, level: int = logging.INFO):
        self.logger = logger
        self.message = message
        self.level = level
        self.start_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        self.logger.log(self.level, f"Starting: {self.message}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = time.time() - self.start_time
        if exc_type is None:
            self.logger.log(self.level, f"Completed: {self.message} in {duration:.2f}s")
        else:
            error_logger.error(f"Failed: {self.message} after {duration:.2f}s: {exc_val}", exc_info=True)
        return False


def log_block(logger: logging.Logger | None = None, message: str | None = None, level: int = logging.INFO):
    """上下文管理器：记录代码块执行。"""
    return LogContext(logger or data_logger, message or "Unknown operation", level)


# 别名（为了向后兼容）
log_data_operation = log_data_loading
log_backtest_operation = log_backtest_execution
