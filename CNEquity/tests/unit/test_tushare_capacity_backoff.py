"""tushare 中间件限流退避的回归测试。

2026-11-06 / 11-09 / 11-10 连续三次日常同步里，``report_rc`` 都在
``[application_capacity]`` 上失败。实测：通用 1.5/3/6 s 阶梯 4 次重试总共只有
10 秒，全部打在限流窗口内；而同一请求隔 2–3 分钟单发一次就返回数据（2026-11-10
09:52 探测，主通道 daily/daily_basic/adj_factor 同时段均正常）。
"""

import pytest

from cnequity.external import tushare_fetch


def test_application_capacity_backoff_is_minutes_not_seconds(monkeypatch):
    sleeps: list[float] = []
    monkeypatch.setattr(tushare_fetch.time, "sleep", lambda s: sleeps.append(s))

    class _Pro:
        def api(self, **kwargs):
            raise Exception(
                "[application_capacity] upstream scheduling capacity temporarily unavailable"
            )

    with pytest.raises(Exception, match="application_capacity"):
        tushare_fetch._fetch_with_retry(_Pro(), "api", retries=3)

    assert 30.0 in sleeps
    assert 60.0 in sleeps


def test_generic_error_keeps_short_backoff(monkeypatch):
    """普通错误不能被拖成分钟级：只有限流类才走长退避。"""
    sleeps: list[float] = []
    monkeypatch.setattr(tushare_fetch.time, "sleep", lambda s: sleeps.append(s))

    class _Pro:
        def api(self, **kwargs):
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        tushare_fetch._fetch_with_retry(_Pro(), "api", retries=3)

    assert 30.0 not in sleeps
    assert 1.5 in sleeps
    assert 3.0 in sleeps
