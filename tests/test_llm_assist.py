"""llm_assist（ML 组合 LLM 辅助 A+C）单测：不依赖真实 LLM 网络。

覆盖方案 §6.1：
- 幻觉因子名过滤（recommended 含不存在名字 → 静默丢弃）
- 空推荐回退（<2 个 → None）
- 清单压缩长度边界（>40 截断、expr 截断）
- 候选池指纹稳定性（顺序无关 + 池变化敏感）
- C 结构校验（坏结构 → None）
- chat_json 降级链（mock 中转 400 → 降级成功；正则提取 markdown 包裹 JSON）
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from alphaagent.factor.stacking.llm_assist import (
    RECOMMENDATION_LOCK_FILE,
    _build_prompt_a,
    _build_report_view,
    _compress_entries,
    candidate_pool_fingerprint,
    llm_recommend_subset,
    llm_summarize_report,
)
from alphaagent.core import llm_provider


@dataclass
class FakeEntry:
    name: str
    expr: str = "close/open"
    facets: tuple = ("price",)
    library: str = "production_main"
    created_at: str = "2026-01-01"
    eval_end: str | None = None
    label_col: str = "ret5"


def _entry(i: int = 0, name: str | None = None) -> FakeEntry:
    return FakeEntry(name or f"fac_{i:03d}")


# ── 幻觉因子名过滤 / 空推荐回退 ──


class _StubChat:
    """把 chat_json 替换为固定返回的 stub。"""

    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def __call__(self, system, user, **kw):
        self.calls.append(kw)
        return self.payload


def test_recommend_drops_hallucinated_names(monkeypatch):
    entries = [_entry(i) for i in range(5)]
    stub = _StubChat({"recommended": ["fac_000", "ghost_999", "fac_002"], "rationale": "r"})
    monkeypatch.setattr(llm_assist_mod(), "chat_json", stub)
    rec = llm_recommend_subset(entries)
    assert rec is not None
    assert rec["recommended"] == ["fac_000", "fac_002"]  # ghost_999 被静默丢弃


def test_recommend_returns_none_when_too_few_valid(monkeypatch):
    entries = [_entry(i) for i in range(5)]
    stub = _StubChat({"recommended": ["ghost_999"], "rationale": "r"})
    monkeypatch.setattr(llm_assist_mod(), "chat_json", stub)
    assert llm_recommend_subset(entries) is None


def test_recommend_returns_none_on_llm_failure(monkeypatch):
    entries = [_entry(i) for i in range(5)]
    monkeypatch.setattr(llm_assist_mod(), "chat_json", lambda *a, **kw: None)
    assert llm_recommend_subset(entries) is None


def test_recommend_requires_at_least_two_entries():
    assert llm_recommend_subset([_entry(0)]) is None


def test_recommend_pool_fingerprint_passthrough(monkeypatch):
    entries = [_entry(i) for i in range(5)]
    stub = _StubChat({"recommended": ["fac_000", "fac_001"], "rationale": "r"})
    monkeypatch.setattr(llm_assist_mod(), "chat_json", stub)
    rec = llm_recommend_subset(entries, pool_fingerprint="fp123")
    assert rec["pool_fingerprint"] == "fp123"
    assert rec["recommended"] == ["fac_000", "fac_001"]


def llm_assist_mod():
    """monkeypatch 需要 patch llm_assist 模块命名空间里的 chat_json 引用。"""
    import alphaagent.factor.stacking.llm_assist as m

    return m


# ── 压缩边界 ──


def test_compress_caps_at_40_and_keeps_newest():
    entries = [
        FakeEntry(name=f"f_{i:03d}", created_at=f"2026-{(i % 12) + 1:02d}-01")
        for i in range(50)
    ]
    items = _compress_entries(entries, max_cap=40)
    assert len(items) == 40
    # 全部来自 50 个（子集），created_at 截断到 40 个最新的语义由调用方排序保证
    assert {it["name"] for it in items} <= {e.name for e in entries}


def test_compress_truncates_expr():
    e = FakeEntry(name="f_long", expr="x" * 500)
    items = _compress_entries([e])
    assert len(items[0]["expr"]) == 120


def test_build_prompt_a_passes_max_cap():
    entries = [_entry(i) for i in range(60)]
    prompt = _build_prompt_a(entries, max_cap=10)
    payload = json.loads(prompt.split("：\n", 1)[1])
    assert len(payload) == 10


# ── 指纹 ──


def test_fingerprint_order_insensitive():
    a = [_entry(0), _entry(1), _entry(2)]
    b = [_entry(2), _entry(0), _entry(1)]
    assert candidate_pool_fingerprint(a) == candidate_pool_fingerprint(b)


def test_fingerprint_sensitive_to_pool_change():
    a = [_entry(0), _entry(1)]
    b = [_entry(0), _entry(1), _entry(2)]
    assert candidate_pool_fingerprint(a) != candidate_pool_fingerprint(b)


def test_lock_file_is_stable_repo_path():
    # 锁定文件必须不带时间戳且在 stacking 目录下（盲测可复现前提）
    path = str(RECOMMENDATION_LOCK_FILE).replace("\\", "/")
    assert path.endswith("artifacts/alphaagent/stacking/llm_recommendation.json")
    import re as _re

    assert not _re.search(r"\d{8}_\d{6}", path), "锁定路径含时间戳，永远无法命中复用"


# ── C：报告视图与结构校验 ──


def test_report_view_includes_key_sections():
    view = _build_report_view({
        "folds": 3,
        "scheme": "ml",
        "feature_names": ["a", "b"],
        "dropped": [{"name": "x", "reason": "corr"}] * 12,
        "fold_metrics": {"ridge": [{"fold": 0, "ic_mean": 0.05, "ic_ir": 1.2, "oos_sharpe": 2.0,
                                    "oos_max_drawdown": -0.1}]},
        "feature_weights": {"a": 0.3, "b": -0.1},
        "gate": {"passed": True, "selection_pct": 0.02, "sharpe": 1.5,
                 "max_drawdown": -0.05, "turnover": 0.4},
    })
    payload = json.loads(view.split("：\n", 1)[1])
    assert payload["folds"] == 3
    assert payload["dropped_count"] == 12
    assert len(payload["dropped_top"]) == 10  # 只带前 10 个
    assert payload["feature_weights_top10"] == {"a": 0.3, "b": -0.1}
    assert payload["gate"]["passed"] is True


def test_summarize_validates_structure(monkeypatch):
    stub = _StubChat({"summary": "", "strengths": []})  # summary 为空 → None
    monkeypatch.setattr(llm_assist_mod(), "chat_json", stub)
    assert llm_summarize_report({"folds": 1}) is None


def test_summarize_passes_timeout(monkeypatch):
    stub = _StubChat({"summary": "s", "strengths": ["a"], "risks": [], "suggestions": []})
    monkeypatch.setattr(llm_assist_mod(), "chat_json", stub)
    out = llm_summarize_report({"folds": 1})
    assert out is not None and out["summary"] == "s"
    assert stub.calls[0].get("timeout") == 40  # C 族超时契约（方案 §3.3）


# ── chat_json 降级链（不依赖网络，mock client） ──
# chat_json 内部是函数级 `from openai import OpenAI`，必须 patch openai 模块本身

import openai as _openai


class _FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)

    def create(self, **kw):
        if not self._responses:
            raise AssertionError("unexpected extra call")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _FakeResp(item)


class _FakeResp:
    def __init__(self, text):
        msg = type("M", (), {"content": text})()
        choice = type("C", (), {"message": msg})()
        self.choices = [choice]


class _FakeClient:
    def __init__(self, responses):
        class _Inner:
            pass
        _i = _Inner()
        _i.chat = _Inner()
        _i.chat.completions = _FakeCompletions(responses)
        self.chat = _i.chat


def test_chat_json_fallback_extracts_markdown_json(monkeypatch):
    """降级链：JSON mode 400 → 普通补全返回 markdown 包裹 JSON → 正则提取。"""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL", "test-model")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    text = '前置噪音\n```json\n{"recommended": ["a", "b"]}\n```\n后置'
    fake = _FakeClient([
        Exception("400 json_object unsupported"),  # JSON mode 被中转拒绝
        text,
    ])
    monkeypatch.setattr(_openai, "OpenAI", lambda **kw: fake)

    out = llm_provider.chat_json("sys", "usr", max_tokens=100, retries=0)
    assert out == {"recommended": ["a", "b"]}


def test_chat_json_returns_none_on_all_failure(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("MODEL", "test-model")
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    fake = _FakeClient([
        Exception("500 server error"),
        Exception("500 server error"),
    ])
    monkeypatch.setattr(_openai, "OpenAI", lambda **kw: fake)
    assert llm_provider.chat_json("sys", "usr", max_tokens=100, retries=1) is None


def test_chat_json_requires_env(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("MODEL", raising=False)
    assert llm_provider.chat_json("s", "u") is None
