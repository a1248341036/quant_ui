"""日志装饰器模块。

提供装饰器和上下文管理器，减少日志代码的侵入性。
"""

from __future__ import annotations

import functools
import time
import logging
from typing import Callable, Any

from backend.logging_config import data_logger, backtest_logger, error_logger


def log_function_call(logger: logging.Logger = None, level: int = logging.INFO):
    """
    装饰器：自动记录函数调用和返回。
    
    Args:
        logger: 使用的 logger，None 则根据模块名自动选择
        level: 日志级别
    
    使用示例:
        @log_function_call()
        def load_data(...):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 确定 logger
            if logger is None:
                module_logger = logging.getLogger(func.__module__)
                # 根据模块名选择专用 logger
                if 'data' in func.__module__:
                    log = data_logger
                elif 'backtest' in func.__module__:
                    log = backtest_logger
                else:
                    log = module_logger
            else:
                log = logger
            
            # 构建参数字符串
            arg_str = ', '.join([
                *[repr(arg) for arg in args[:3]],  # 只显示前 3 个位置参数
                *[f"{k}={v!r}" for k, v in list(kwargs.items())[:5]]  # 只显示前 5 个关键字参数
            ])
            if len(args) > 3 or len(kwargs) > 5:
                arg_str += '...'
            
            # 记录调用
            log.log(level, f"Calling {func.__name__}({arg_str})")
            
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # 记录成功返回
                if isinstance(result, dict):
                    # 对于字典结果，记录关键统计信息
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
                error_logger.error(
                    f"{func.__name__} FAILED after {duration:.2f}s: {e}",
                    exc_info=True
                )
                raise
        
        return wrapper
    return decorator


def log_data_loading(func: Callable) -> Callable:
    """
    专用装饰器：记录数据加载过程。
    
    自动记录：
    - 加载参数
    - 各数据源加载进度
    - 加载耗时
    - 数据量统计
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        
        # 提取关键参数
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
            
            # 记录结果统计
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
    """
    专用装饰器：记录回测执行过程。
    
    自动记录：
    - 回测参数
    - 数据加载
    - 回测执行
    - 性能指标
    - 总耗时
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Any:
        start_time = time.time()
        
        # 提取关键参数
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
            
            # 记录结果指标
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
        return False  # 不抑制异常


def log_block(logger: logging.Logger = None, message: str = None, level: int = logging.INFO):
    """
    上下文管理器：记录代码块执行。
    
    使用示例:
        with log_block(message="Loading data"):
            data = load_data(...)
    """
    return LogContext(logger or data_logger, message or "Unknown operation", level)


# 别名（为了向后兼容）
log_data_operation = log_data_loading
log_backtest_operation = log_backtest_execution
