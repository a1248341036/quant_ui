#!/usr/bin/env python3
"""
Tushare 客户端
- 从项目根目录 .env 加载 TUSHARE_TOKEN
- 返回 pro_api 对象供 data 层调用
- 网络超时 / 限流等可恢复错误自动重试（指数退避）
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Any, Callable, TypeVar

import requests
import tushare as ts
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT / ".env"

T = TypeVar("T")

RETRYABLE_NETWORK_ERRORS = (
    requests.exceptions.ConnectionError,
    requests.exceptions.Timeout,
    requests.exceptions.ChunkedEncodingError,
    TimeoutError,
)

# Tushare 限流 / 临时性 API 报错关键词
RETRYABLE_API_MARKERS = (
    "最多访问",
    "请稍后再试",
    "频繁",
    "限流",
    "timeout",
    "timed out",
    "连接",
    "繁忙",
)

_config: dict[str, float | int] = {
    "max_retries": int(os.getenv("TUSHARE_MAX_RETRIES", "5")),
    "timeout": int(os.getenv("TUSHARE_TIMEOUT", "60")),
    "retry_base_delay": float(os.getenv("TUSHARE_RETRY_BASE_DELAY", "2.0")),
    "retry_max_delay": float(os.getenv("TUSHARE_RETRY_MAX_DELAY", "120.0")),
}


def configure(
    *,
    max_retries: int | None = None,
    timeout: int | None = None,
    retry_base_delay: float | None = None,
    retry_max_delay: float | None = None,
) -> None:
    """覆盖 Tushare 重试 / 超时配置（供 CLI 或脚本调用）。"""
    if max_retries is not None:
        _config["max_retries"] = max_retries
    if timeout is not None:
        _config["timeout"] = timeout
    if retry_base_delay is not None:
        _config["retry_base_delay"] = retry_base_delay
    if retry_max_delay is not None:
        _config["retry_max_delay"] = retry_max_delay


def _read_token() -> str:
    """从 .env 或环境变量读取 token。"""
    load_dotenv(ENV_FILE)
    token = os.getenv("TUSHARE_TOKEN", "")
    return token.strip().strip('"').strip("'")


def _is_retryable(exc: Exception) -> bool:
    if isinstance(exc, RETRYABLE_NETWORK_ERRORS):
        return True
    msg = str(exc)
    return any(marker in msg for marker in RETRYABLE_API_MARKERS)


def _retry_delay(attempt: int) -> float:
    base = float(_config["retry_base_delay"])
    cap = float(_config["retry_max_delay"])
    delay = min(cap, base * (2**attempt))
    # 轻微抖动，避免并发任务同一时刻重试
    return delay + random.uniform(0, min(1.0, delay * 0.1))


def call_with_retry(
    func: Callable[..., T],
    *args: Any,
    label: str | None = None,
    **kwargs: Any,
) -> T:
    """对单次 Tushare 调用做指数退避重试。"""
    max_retries = int(_config["max_retries"])
    name = label or getattr(func, "__name__", "tushare")
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries or not _is_retryable(exc):
                raise
            delay = _retry_delay(attempt)
            print(
                f"  [retry {attempt + 1}/{max_retries}] {name}: {exc}，"
                f"{delay:.1f}s 后重试",
                flush=True,
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


class _RetryingPro:
    """包装 Tushare pro_api，为各接口调用注入重试。"""

    def __init__(self, pro: Any) -> None:
        self._pro = pro

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._pro, name)
        if not callable(attr):
            return attr

        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return call_with_retry(attr, *args, label=name, **kwargs)

        return wrapper


def get_pro():
    """初始化并返回 Tushare pro_api（带自动重试）。"""
    token = _read_token()
    if not token:
        raise ValueError(
            f"未找到 TUSHARE_TOKEN。请在 {ENV_FILE} 中配置，"
            "格式: TUSHARE_TOKEN=your_token"
        )
    timeout = int(_config["timeout"])
    pro = ts.pro_api(token, timeout=timeout)
    max_retries = int(_config["max_retries"])
    if max_retries <= 0:
        return pro
    return _RetryingPro(pro)
