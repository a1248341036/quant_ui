"""时间与时区公共工具模块。"""
from __future__ import annotations

from datetime import datetime, timezone


def utc_now_iso() -> str:
    """返回当前 UTC 时间的 ISO8601 格式字符串（包含纳秒/微秒与时区）。"""
    return datetime.now(timezone.utc).isoformat()
