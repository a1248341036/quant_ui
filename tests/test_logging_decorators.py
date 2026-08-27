"""日志装饰器测试。"""

import time
from backend.logging_decorators import (
    log_function_call,
    log_data_loading,
    log_backtest_execution,
    log_block,
)
from backend.logging_config import data_logger, backtest_logger


def test_log_function_call():
    """测试通用函数调用装饰器。"""
    
    @log_function_call()
    def test_func(x, y):
        time.sleep(0.1)
        return x + y
    
    result = test_func(1, 2)
    assert result == 3
    print("✓ log_function_call 测试通过")


def test_log_data_loading():
    """测试数据加载装饰器。"""
    
    @log_data_loading
    def mock_load_data(start, end, need_panel=True, codes=None, need_heavy=False):
        time.sleep(0.1)
        return {
            "panel": None,
            "universe": [],
            "start": start,
            "end": end,
        }
    
    result = mock_load_data(
        start="2023-01-01",
        end="2023-12-31",
        need_panel=True,
        codes=["000001", "000002"],
        need_heavy=False,
    )
    
    assert result["start"] == "2023-01-01"
    print("✓ log_data_loading 测试通过")


def test_log_backtest_execution():
    """测试回测执行装饰器。"""
    
    @log_backtest_execution
    def mock_backtest(strategy, universe, start, end, top_n=5, capital=100000):
        time.sleep(0.1)
        return {
            "metrics": {
                "总收益": 0.15,
                "夏普": 1.2,
                "最大回撤": -0.08,
            },
            "strategy": strategy,
            "universe": universe,
        }
    
    result = mock_backtest(
        strategy="低换手冷门",
        universe="全部股票",
        start="2023-01-01",
        end="2023-12-31",
        top_n=10,
        capital=100000,
    )
    
    assert result["metrics"]["总收益"] == 0.15
    print("✓ log_backtest_execution 测试通过")


def test_log_block():
    """测试日志块上下文管理器。"""
    
    with log_block(message="测试操作", logger=data_logger):
        time.sleep(0.1)
        # 模拟一些操作
    
    print("✓ log_block 测试通过")


def test_log_block_with_error():
    """测试日志块在异常时的行为。"""
    
    try:
        with log_block(message="失败的操作", logger=data_logger):
            time.sleep(0.05)
            raise ValueError("测试错误")
    except ValueError:
        pass  # 预期异常
    
    print("✓ log_block 异常处理测试通过")


def test_decorator_preserves_function_metadata():
    """测试装饰器保留原函数元数据。"""
    
    @log_function_call()
    def my_function(x, y):
        """这是我的函数文档。"""
        return x + y
    
    assert my_function.__name__ == "my_function"
    assert my_function.__doc__ == "这是我的函数文档。"
    print("✓ 装饰器保留元数据测试通过")


if __name__ == "__main__":
    print("开始测试日志装饰器...\n")
    
    test_log_function_call()
    test_log_data_loading()
    test_log_backtest_execution()
    test_log_block()
    test_log_block_with_error()
    test_decorator_preserves_function_metadata()
    
    print("\n✅ 所有日志装饰器测试通过！")
