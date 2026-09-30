"""batch_mineru_parse 断点续跑污染回归（review 修）。

修复前：非零退出/超时分支不清理残留的 >MIN_MD_BYTES 错误页/半成品 md，下一轮
断点续跑仅凭“存在+字节数”就把污染产物判成 skipped 永久跳过，污染知识库。
修复后：失败分支删除残留 out_md，失败后再次运行能真正重新解析。
"""
from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "batch_mineru_parse.py"
MIN_MD_TEXT = "数据正文" * 400  # 400×4 中文字节 > MIN_MD_BYTES(1000)


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("_batch_mineru_parse", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _out_md_from_cmd(cmd) -> Path:
    """从 Popen 的 cmd 还原 out_md（-o 后的 .md 参数）。"""
    for i, tok in enumerate(cmd):
        if tok == "-o":
            return Path(cmd[i + 1])
    raise AssertionError("cmd 缺少 -o 输出参数")


class _FakeProc:
    """模拟 MinerU 子进程。首次 communicate 可选写残留并按需抛超时。"""

    def __init__(self, out_md: Path, returncode: int, communicate_timeout=None):
        self._out_md = out_md
        self._returncode = returncode
        self._communicate_timeout = communicate_timeout
        self._first = True
        self.pid = 9999

    @property
    def returncode(self) -> int:
        return self._returncode

    def communicate(self, timeout=None):
        if self._first:
            self._first = False
            if self._communicate_timeout is not None:
                self._out_md.write_text(MIN_MD_TEXT, encoding="utf-8")
                raise subprocess.TimeoutExpired("mineru", timeout=self._communicate_timeout)
            if self._returncode != 0:
                self._out_md.write_text(MIN_MD_TEXT, encoding="utf-8")
        return ("", "")


def _run(mod, tmp_path: Path, run_id: str):
    pdf = tmp_path / "sub" / "2024_report.pdf"
    pdf.parent.mkdir(parents=True, exist_ok=True)
    pdf.write_text("%PDF", encoding="utf-8")
    out_root = tmp_path / "parsed"
    rec = mod.parse_one((0, str(pdf)), tmp_path, out_root, mod.MINERU,
                        timeout=10, retries=0, run_id=run_id)
    return rec, out_root / "sub" / "2024_report.md"


def test_nonzero_exit_discards_residue_and_reparses(mod, tmp_path, monkeypatch):
    calls = {"n": 0}

    def fake_popen(cmd, **kwargs):
        calls["n"] += 1
        out_md = _out_md_from_cmd(cmd)
        if calls["n"] == 1:
            return _FakeProc(out_md, returncode=1)  # 失败：首次 communicate 写残留
        out_md.write_text(MIN_MD_TEXT, encoding="utf-8")  # 成功：写有效 md
        return _FakeProc(out_md, returncode=0)

    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)

    rec, out_md = _run(mod, tmp_path, "r1")
    assert rec["status"] == "error"
    assert not out_md.exists(), "非零退出残留 md 必须被清理"

    # 断点续跑：残留若没删会直接 skipped；清理后应真正重新解析为 ok
    rec2, out_md2 = _run(mod, tmp_path, "r2")
    assert rec2["status"] == "ok"
    assert out_md2.exists()


def test_timeout_discards_residue_and_reparses(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "_kill_tree", lambda *_: None)
    monkeypatch.setattr(mod.time, "sleep", lambda *_: None)

    calls = {"n": 0}

    def fake_popen(cmd, **kwargs):
        calls["n"] += 1
        out_md = _out_md_from_cmd(cmd)
        if calls["n"] == 1:
            return _FakeProc(out_md, returncode=0, communicate_timeout=10)  # 首次超时+残留
        out_md.write_text(MIN_MD_TEXT, encoding="utf-8")
        return _FakeProc(out_md, returncode=0)

    monkeypatch.setattr(mod.subprocess, "Popen", fake_popen)

    rec, out_md = _run(mod, tmp_path, "r3")
    assert rec["status"] == "error"
    assert not out_md.exists(), "超时残留 md 必须被清理"

    rec2, out_md2 = _run(mod, tmp_path, "r4")
    assert rec2["status"] == "ok"
    assert out_md2.exists()