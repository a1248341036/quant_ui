"""统一日志系统测试。"""

from pathlib import Path

from backend.logging_config import (
    LOG_DIR,
    api_logger,
    get_log_file_path,
    get_logger,
    main_logger,
    parse_log_file,
    setup_root_logger,
)


def test_logger_creation():
    """测试 logger 创建。"""
    logger = get_logger("test_module")
    assert logger is not None
    assert len(logger.handlers) > 0
    print("✓ Logger 创建成功")


def test_log_writing():
    """测试日志写入。"""
    main_logger.info("测试信息日志")
    api_logger.info("测试 API 日志")
    
    # 检查日志文件是否存在
    assert get_log_file_path().exists()
    print("✓ 日志写入成功")


def test_log_parsing():
    """测试日志解析。"""
    # 先写入一些日志
    main_logger.info("测试日志 1")
    main_logger.warning("测试日志 2")
    main_logger.error("测试日志 3")
    
    # 解析日志
    log_file = get_log_file_path()
    logs = parse_log_file(log_file, limit=10)
    
    assert len(logs) > 0
    assert 'timestamp' in logs[0]
    assert 'level' in logs[0]
    assert 'message' in logs[0]
    print(f"✓ 日志解析成功，解析了 {len(logs)} 条日志")


def test_log_filtering():
    """测试日志过滤。"""
    log_file = get_log_file_path()
    
    # 按级别过滤
    error_logs = parse_log_file(log_file, level="ERROR", limit=10)
    for log in error_logs:
        assert log['level'] == 'ERROR'
    
    # 按 run_id 过滤
    run_id_logs = parse_log_file(log_file, run_id="test123", limit=10)
    # 如果没有匹配的，应该返回空列表
    assert isinstance(run_id_logs, list)
    
    print("✓ 日志过滤成功")


def test_log_rotation():
    """测试日志轮转配置。"""
    from backend.logging_config import LOG_MAX_BYTES, LOG_BACKUP_COUNT
    
    assert LOG_MAX_BYTES == 10 * 1024 * 1024  # 10MB
    assert LOG_BACKUP_COUNT == 5
    print("✓ 日志轮转配置正确")


def test_log_directory():
    """测试日志目录创建。"""
    assert LOG_DIR.exists()
    assert LOG_DIR.is_dir()
    print(f"✓ 日志目录存在：{LOG_DIR}")


def test_multiple_loggers():
    """测试多个 logger 实例。"""
    logger1 = get_logger("module1")
    logger2 = get_logger("module2")
    
    assert logger1 is not None
    assert logger2 is not None
    assert logger1.name == "module1"
    assert logger2.name == "module2"
    
    # 测试日志隔离
    logger1.info("Logger 1 消息")
    logger2.info("Logger 2 消息")
    
    print("✓ 多个 logger 实例工作正常")


def test_error_logger():
    """测试错误日志记录。"""
    from backend.logging_config import error_logger
    
    error_logger.error("测试错误日志")
    
    # 检查 error.log 文件
    error_log_file = LOG_DIR / "error.log"
    assert error_log_file.exists()
    
    print("✓ 错误日志记录成功")


if __name__ == "__main__":
    print("开始测试统一日志系统...\n")
    
    test_logger_creation()
    test_log_writing()
    test_log_parsing()
    test_log_filtering()
    test_log_rotation()
    test_log_directory()
    test_multiple_loggers()
    test_error_logger()
    
    print("\n✅ 所有日志测试通过！")
