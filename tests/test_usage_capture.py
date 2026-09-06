"""UsageBridge + MiningStreamObserver 缓存统计补齐的单测。

背景：agentscope ModelCallEndEvent 不携带 cache 字段，MiningStreamObserver
经 UsageCapturedChatModel 喂入的 bridge 取回真实 cache_input_tokens。
"""

from __future__ import annotations

from types import SimpleNamespace

from alphaagent.factor.mining.infra.cli_stream import MiningStreamObserver
from alphaagent.factor.mining.infra.usage_capture import UsageBridge


def _emit_collector():
    events: list[tuple[str, dict]] = []

    def emit(event: str, payload: dict) -> None:
        events.append((event, payload))

    return events, emit


def test_bridge_pop_is_one_shot():
    bridge = UsageBridge()
    assert bridge.pop_latest() is None
    bridge.record(SimpleNamespace(cache_input_tokens=100, cache_creation_input_tokens=5))
    first = bridge.pop_latest()
    assert first is not None and first.cache_input_tokens == 100
    assert bridge.pop_latest() is None, "pop 必须清空，防止上一次调用的 usage 串号到下一次"


def test_observer_fills_cache_fields_from_bridge():
    events, emit = _emit_collector()
    bridge = UsageBridge()
    bridge.record(SimpleNamespace(cache_input_tokens=11968, cache_creation_input_tokens=7))
    observer = MiningStreamObserver(emit=emit, turn=3, usage_bridge=bridge)

    # 模拟 agentscope 真实事件：只有 input/output，没有 cache 字段
    event = SimpleNamespace(input_tokens=12028, output_tokens=8)
    observer.on_model_call_end(event)

    assert len(events) == 1
    payload = events[0][1]
    assert payload["input_tokens"] == 12028
    assert payload["output_tokens"] == 8
    assert payload["cache_input_tokens"] == 11968
    assert payload["cache_creation_input_tokens"] == 7
    assert payload["cache_hit_rate"] == round(11968 / 12028, 4)
    assert bridge.pop_latest() is None, "observer 消费后 bridge 应已清空"


def test_observer_without_bridge_keeps_event_values():
    events, emit = _emit_collector()
    observer = MiningStreamObserver(emit=emit, turn=0)

    event = SimpleNamespace(
        input_tokens=100,
        output_tokens=10,
        cache_input_tokens=55,
        cache_creation_input_tokens=3,
    )
    observer.on_model_call_end(event)

    payload = events[0][1]
    assert payload["cache_input_tokens"] == 55
    assert payload["cache_creation_input_tokens"] == 3
    assert payload["cache_hit_rate"] == 0.55


def test_observer_zero_when_no_bridge_and_no_event_fields():
    """修复前的事实基线：无 bridge 时 cache 字段恒为 0。"""
    events, emit = _emit_collector()
    observer = MiningStreamObserver(emit=emit, turn=0)
    event = SimpleNamespace(input_tokens=100, output_tokens=10)
    observer.on_model_call_end(event)

    payload = events[0][1]
    assert payload["cache_input_tokens"] == 0
    assert payload["cache_creation_input_tokens"] == 0
    assert payload["cache_hit_rate"] == 0.0


def test_cache_hit_rate_guard_zero_input():
    """input=0 时不除零，命中率回落 0.0。"""
    from alphaagent.factor.mining.agent.agentscope_run import _cache_hit_rate

    assert _cache_hit_rate({"input_tokens": 0, "cache_input_tokens": 0}) == 0.0
    assert _cache_hit_rate({}) == 0.0
    assert _cache_hit_rate({"input_tokens": 900, "cache_input_tokens": 300}) == round(300 / 900, 4)
