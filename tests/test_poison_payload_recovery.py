# -*- coding: utf-8 -*-
"""毒 payload（请求体非法）识别与恢复回归测试。

2026-09-29 整夜实证：3 个 run 因「模型偶发产出畸形工具参数 JSON → 该轮历史被原样
重放 → 上游判 ``400 Bad Request`` → CC Switch 包装成 500 → agentscope 基类把 500
当可重试，5s 间隔重发同一 payload 共 10 次，全部失败 → run 终结」而死，单次损失
15~47 分钟并丢掉已过线因子。

本测试锁定两条不变量：
1. 这类错误必须被识别为「重发无意义的请求体错误」，从而走「重建会话 + 重试本轮」；
2. 传输类错误（断流/连接失败/429）**不得**被误判成毒 payload，原有重试行为不变。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from alphaagent.factor.mining.infra.provider_compat import (
    ProviderSafeChatModel,
    is_payload_poison_error,
)

# 2026-09-29 真实报错文本（run 01a8013ff978 / e54a43c4d4e2 / 060525a45ffd）
_REAL_500 = (
    "Error code: 500 - {'error': {'message': 'CC Switch local proxy failed while handling "
    "Codex endpoint /chat/completions. Provider: gemini 反代; model: DeepSeek-V4-Flash-0731; "
    "upstream_status: HTTP 500; cause: upstream status 400: Bad Request', "
    "'type': 'server_error', 'code': 'internal_server_error'}}"
)


def test_real_upstream_400_is_poison():
    assert is_payload_poison_error(RuntimeError(_REAL_500)) is True


def test_cause_chain_is_walked():
    inner = ValueError("upstream status 400: Bad Request")
    outer = RuntimeError("model call failed")
    outer.__cause__ = inner
    assert is_payload_poison_error(outer) is True


def test_tool_json_decode_hint_is_poison():
    msg = "When decoding your tool arguments from JSON format, a JSONDecodeError was raised"
    assert is_payload_poison_error(RuntimeError(msg)) is True


def test_transport_and_quota_are_not_poison():
    assert is_payload_poison_error(RuntimeError("RemoteProtocolError: incomplete chunked read")) is False
    assert is_payload_poison_error(RuntimeError("Error code: 429 - rate limit exceeded")) is False
    assert is_payload_poison_error(RuntimeError("502 Bad Gateway from upstream proxy")) is False


def test_transport_errors_still_retryable():
    try:
        import httpx2  # noqa: PLC0415
    except Exception:  # noqa: BLE001
        pytest.skip("httpx2 不可用（非本机 openai 传输层）")
    names = {t.__name__ for t in ProviderSafeChatModel._get_retryable_exceptions()}
    assert {"RemoteProtocolError", "ReadError", "ConnectError"} <= names


def test_run_loop_wires_poison_recovery():
    """结构守卫：恢复分支与重建辅助必须留在 run 循环里（防止被重构掉）。"""
    src = (
        Path(__file__).resolve().parents[1]
        / "alphaagent"
        / "factor"
        / "mining"
        / "agent"
        / "agentscope_run.py"
    ).read_text(encoding="utf-8")
    assert "poison_payload_recovery" in src
    assert "poison_payload_exhausted" in src
    assert "_recreate_agent" in src
    assert "is_payload_poison_error(exc)" in src
