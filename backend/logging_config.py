"""统一日志配置模块。

为整个后端提供统一的日志格式、文件轮转和日志级别管理。
"""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any

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
    """
    获取配置好的 logger 实例。

    Args:
        name: logger 名称（通常使用 __name__）
        level: 日志级别
        log_file: 日志文件路径，None 则使用默认文件
        console: 是否输出到控制台

    Returns:
        配置好的 logger 实例

    使用示例:
        logger = get_logger(__name__)
        logger.info("启动成功")
        logger.error("发生错误", exc_info=True)
    """
    logger = logging.getLogger(name)
    
    # 避免重复添加 handler
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    logger.propagate = False  # 避免日志重复输出到父 logger

    # 日志格式
    formatter = logging.Formatter(
        fmt=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )

    # 文件处理器（带轮转）
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

    # 错误日志单独记录
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

    # 控制台输出
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def setup_root_logger() -> logging.Logger:
    """
    设置根 logger（应用启动时调用）。

    Returns:
        根 logger 实例
    """
    logger = get_logger("quant_ui", level=logging.INFO)
    logger.info("Quant UI 后端服务启动")
    return logger


def setup_api_logger() -> logging.Logger:
    """
    设置 API 专用 logger（记录所有 HTTP 请求）。

    Returns:
        API logger 实例
    """
    return get_logger("quant_ui.api", log_file=API_LOG_FILE)


def setup_data_logger() -> logging.Logger:
    """
    设置数据源专用 logger（记录数据加载和更新）。

    Returns:
        数据源 logger 实例
    """
    return get_logger("quant_ui.data", log_file=LOG_DIR / "data.log")


def setup_backtest_logger() -> logging.Logger:
    """
    设置回测专用 logger（记录回测执行过程）。

    Returns:
        回测 logger 实例
    """
    return get_logger("quant_ui.backtest", log_file=LOG_DIR / "backtest.log")


# 预定义的 logger 实例（方便导入使用）
main_logger = get_logger("quant_ui")
api_logger = get_logger("quant_ui.api", log_file=API_LOG_FILE)
error_logger = get_logger("quant_ui.error", log_file=ERROR_LOG_FILE, console=False)
data_logger = get_logger("quant_ui.data", log_file=LOG_DIR / "data.log")
backtest_logger = get_logger("quant_ui.backtest", log_file=LOG_DIR / "backtest.log")


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


def get_logger_with_context(
    base_logger: logging.Logger,
    context: dict[str, Any] | None = None,
) -> logging.Logger:
    """
    获取带上下文的 logger（用于注入 request_id 等信息）。

    Args:
        base_logger: 基础 logger
        context: 上下文信息（request_id, user_id, run_id 等）

    Returns:
        带上下文的 logger
    """
    if not context:
        return base_logger

    # 使用 Filter 注入上下文
    class ContextFilter(logging.Filter):
        def filter(self, record: logging.LogRecord) -> bool:
            for key, value in context.items():
                setattr(record, key, value)
            return True

    base_logger.addFilter(ContextFilter())
    return base_logger


def get_log_file_path() -> Path:
    """获取主日志文件路径。"""
    return MAIN_LOG_FILE


def parse_log_file(
    log_file: Path,
    run_id: str | None = None,
    level: str | None = None,
    limit: int = 100,
    tail: bool = True,
) -> list[dict[str, str]]:
    """解析日志文件，返回结构化日志列表。
    
    Args:
        log_file: 日志文件路径
        run_id: 可选，按 run_id 过滤
        level: 可选，日志级别 (DEBUG, INFO, WARNING, ERROR)
        limit: 返回最大条数
        tail: 是否从末尾开始读取
    
    Returns:
        日志列表，每条包含 timestamp, level, message 等字段
    """
    import re
    
    # 日志格式正则表达式
    log_pattern = re.compile(
        r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] ([^:]+): (.*)'
    )
    
    logs = []
    
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # 如果需要从末尾开始，反转列表
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
            
            # 按级别过滤
            if level and level_name.upper() != level.upper():
                continue
            
            # 按 run_id 过滤（在 message 中查找）
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
        
        # 如果不是从末尾开始，需要反转回正常顺序
        if not tail:
            logs.reverse()
        
        return logs
    
    except Exception as e:
        # 返回空列表而不是抛出异常
        return []
