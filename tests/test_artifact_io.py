# -*- coding: utf-8 -*-
"""B-04 修复测试：产物落盘统一 writer（拒绝裸 {}）。

覆盖 docs/specs/alphaagent_known_bugs_spec.md B-04：
- 有内容原样写；空内容 + empty_reason 写显式占位；
- 空内容无 reason 抛 EmptyArtifactError（拒绝裸 {}）；
- 父目录自动创建。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from alphaagent.factor.mining.infra.artifact_io import EmptyArtifactError, write_artifact_json


def test_writes_nonempty_payload_roundtrip(tmp_path: Path):
    p = tmp_path / "art" / "out.json"
    data = {"records": [1, 2, 3], "meta": "x"}
    assert write_artifact_json(p, data) is True
    assert json.loads(p.read_text(encoding="utf-8")) == data


def test_rejects_bare_empty_without_reason(tmp_path: Path):
    p = tmp_path / "out.json"
    with pytest.raises(EmptyArtifactError):
        write_artifact_json(p, {})
    assert not p.exists()  # 拒绝时不落任何文件


def test_empty_with_reason_writes_explicit_placeholder(tmp_path: Path):
    p = tmp_path / "out.json"
    assert write_artifact_json(p, {}, empty_reason="无满足条件样本") is False
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload == {"empty": True, "reason": "无满足条件样本"}


def test_empty_list_with_reason_writes_placeholder(tmp_path: Path):
    p = tmp_path / "out.json"
    assert write_artifact_json(p, [], empty_reason="scan returned nothing") is False
    payload = json.loads(p.read_text(encoding="utf-8"))
    assert payload["empty"] is True and payload["reason"] == "scan returned nothing"


def test_parent_dirs_auto_created(tmp_path: Path):
    p = tmp_path / "a" / "b" / "c" / "out.json"
    assert write_artifact_json(p, {"ok": 1}) is True
    assert json.loads(p.read_text(encoding="utf-8")) == {"ok": 1}
