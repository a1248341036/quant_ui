"""fs_guard 单元测试：保护区删除拦截、等价移动拦截、白名单放行、原子写放行。"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from alphaagent.core.fs_guard import (  # noqa: E402
    ProtectedDeleteError,
    default_protected_roots,
    install,
    install_fs_guard,
    uninstall,
)


@pytest.fixture()
def zone(tmp_path: Path) -> Path:
    """装好守卫的保护区目录；测试结束自动卸载，避免污染其他用例。"""
    z = tmp_path / "protected_zone"
    z.mkdir()
    (z / "keep.txt").write_text("data", encoding="utf-8")
    install([z])
    try:
        yield z
    finally:
        uninstall()


def test_remove_in_zone_blocked(zone: Path) -> None:
    target = zone / "keep.txt"
    with pytest.raises(ProtectedDeleteError):
        os.remove(target)
    assert target.exists()


def test_os_unlink_and_pathlib_blocked(zone: Path) -> None:
    a = zone / "a.txt"
    b = zone / "b.txt"
    a.write_text("x", encoding="utf-8")
    b.write_text("x", encoding="utf-8")
    with pytest.raises(ProtectedDeleteError):
        os.unlink(a)
    with pytest.raises(ProtectedDeleteError):
        b.unlink()
    assert a.exists() and b.exists()


def test_pathlib_rmdir_blocked(zone: Path) -> None:
    sub = zone / "sub"
    sub.mkdir()
    with pytest.raises(ProtectedDeleteError):
        sub.rmdir()
    assert sub.exists()


def test_rmtree_blocked_without_partial_delete(zone: Path) -> None:
    sub = zone / "sub"
    sub.mkdir()
    (sub / "a.txt").write_text("x", encoding="utf-8")
    (sub / "b.txt").write_text("x", encoding="utf-8")
    with pytest.raises(ProtectedDeleteError):
        shutil.rmtree(zone)
    assert (sub / "a.txt").exists()
    assert (sub / "b.txt").exists()
    assert sub.exists()


def test_delete_outside_zone_allowed(tmp_path: Path, zone: Path) -> None:
    free = tmp_path / "free.txt"
    free.write_text("x", encoding="utf-8")
    os.remove(free)
    assert not free.exists()


def test_move_out_of_zone_blocked(zone: Path, tmp_path: Path) -> None:
    dst = tmp_path / "escaped.txt"
    with pytest.raises(ProtectedDeleteError):
        os.rename(zone / "keep.txt", dst)
    assert (zone / "keep.txt").exists()
    assert not dst.exists()


def test_replace_within_zone_allowed(zone: Path) -> None:
    """库内 tmp+replace 原子写 = 写语义，必须放行。"""
    tmp = zone / ".tmp-keep"
    tmp.write_text("new", encoding="utf-8")
    os.replace(tmp, zone / "keep.txt")
    assert (zone / "keep.txt").read_text(encoding="utf-8") == "new"


def test_move_in_overwrite_allowed(zone: Path, tmp_path: Path) -> None:
    """区外文件覆盖写入区内 = 写语义，放行。"""
    outside = tmp_path / "outside.txt"
    outside.write_text("fresh", encoding="utf-8")
    os.replace(outside, zone / "keep.txt")
    assert (zone / "keep.txt").read_text(encoding="utf-8") == "fresh"


def test_file_root_protected(tmp_path: Path) -> None:
    db = tmp_path / "research_memory.db"
    db.write_text("sqlite", encoding="utf-8")
    install([db])
    try:
        with pytest.raises(ProtectedDeleteError):
            os.remove(db)
        assert db.exists()
    finally:
        uninstall()


def test_allowlisted_caller_allowed(tmp_path: Path) -> None:
    z = tmp_path / "zone2"
    z.mkdir()
    f = z / "x.txt"
    f.write_text("d", encoding="utf-8")
    install([z], [Path(__file__).name])
    try:
        os.remove(f)
        assert not f.exists()
    finally:
        uninstall()


def test_uninstall_restores_originals(tmp_path: Path) -> None:
    z = tmp_path / "zone3"
    z.mkdir()
    f = z / "x.txt"
    f.write_text("d", encoding="utf-8")
    install([z])
    uninstall()
    os.remove(f)
    assert not f.exists()


def test_double_install_safe(tmp_path: Path) -> None:
    z = tmp_path / "zone4"
    z.mkdir()
    f = z / "x.txt"
    f.write_text("d", encoding="utf-8")
    install([z])
    install([z])
    with pytest.raises(ProtectedDeleteError):
        os.remove(f)
    uninstall()
    os.remove(f)  # 双重安装不得把包装函数存成原函数，一次卸载即恢复
    assert not f.exists()


def test_install_from_env_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHA_FS_GUARD", "0")
    assert install_fs_guard() is False


def test_install_from_env_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ALPHA_FS_GUARD", "1")
    try:
        assert install_fs_guard() is True
    finally:
        uninstall()


def test_default_roots_cover_business_zones() -> None:
    roots = [str(p).replace("\\", "/") for p in default_protected_roots(Path("D:/fake_root"))]
    assert any(p.endswith("artifacts/alphaagent/factorzoo") for p in roots)
    assert any(p.endswith("artifacts/alphaagent/research_specs") for p in roots)
    assert any(p.endswith("artifacts/alphaagent/research_memory.db") for p in roots)
    assert any(p.endswith("logs/factor_mining") for p in roots)
