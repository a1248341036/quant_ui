"""AlphaAgent 挖掘子进程文件守卫：只读只写，禁止删除保护区。

原理：monkey-patch 本进程的删除原语（os.remove/os.unlink/os.rmdir/shutil.rmtree、
pathlib.Path.unlink/Path.rmdir），在删除动作落到 OS 之前检查目标是否位于保护区，
命中即抛 ProtectedDeleteError；os.rename/os.replace/shutil.move 额外拦截
"把保护区内容移出"的等价删除。白名单按调用方模块文件路径放行，
供 preflight 冒烟清理等内部合法删除使用。保护区外（缓存、tmp）不受影响。

开关：环境变量 ``ALPHA_FS_GUARD`` 默认启用，设为 "0/false/off/no" 关闭；
``ALPHA_FS_GUARD_ALLOW`` 追加白名单条目（逗号分隔，支持文件名或路径后缀）；
``ALPHA_FS_GUARD_VERBOSE=1`` 时拦截动作打印到 stderr。
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path
from typing import Callable, Iterable, Sequence

__all__ = [
    "ProtectedDeleteError",
    "default_protected_roots",
    "install",
    "install_fs_guard",
    "uninstall",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_OFF_VALUES = {"", "0", "false", "off", "no"}

# 仓库相对路径后缀；匹配时与调用方模块 __file__ 做归一化后缀比较
DEFAULT_ALLOWLIST: tuple[str, ...] = (
    "alphaagent/factor/cache.py",  # 因子值缓存 LRU 淘汰（缓存区不在保护区，兜底）
    "alphaagent/data/adapters/cnequity.py",  # panel 缓存清理（同上）
    "alphaagent/data/adapters/panel_mmap.py",  # tmp 文件清理（同上）
    "alphaagent/factor/mining/agent/preflight.py",  # 冒烟因子落库后的候选池清理（保护区内的合法删除）
)

_INSTALLED = False
_ORIGINALS: list[tuple[object, str, Callable]] = []
_STATE: dict[str, list[str]] = {"roots": [], "allow": []}


class ProtectedDeleteError(PermissionError):
    """试图删除或移出保护区路径时抛出。"""


def default_protected_roots(root: Path | str | None = None) -> list[Path]:
    """默认保护区：因子库两库（含备份）、门槛覆盖目录、研究记忆库、挖掘轨迹日志。"""
    base = Path(root) if root is not None else _REPO_ROOT
    return [
        base / "artifacts" / "alphaagent" / "factorzoo",
        base / "artifacts" / "alphaagent" / "research_specs",
        base / "artifacts" / "alphaagent" / "research_memory.db",
        base / "logs" / "factor_mining",
    ]


def _norm(path: object) -> str:
    return os.path.normcase(os.path.abspath(os.path.expanduser(str(path))))


def _is_under(path_norm: str, roots: Sequence[str]) -> bool:
    for root in roots:
        if path_norm == root or path_norm.startswith(root + os.sep):
            return True
    return False


def _allow_match(norm: str, entry: str) -> bool:
    e = os.path.normcase(entry)
    if os.path.isabs(e):
        return norm == e
    return norm.endswith(os.sep + e) or norm.endswith("/" + e)


def _caller_allowlisted() -> bool:
    allow = _STATE["allow"]
    if not allow:
        return False
    guard_self = _norm(__file__)
    frame = sys._getframe(1)
    while frame is not None:
        file = frame.f_globals.get("__file__")
        frame = frame.f_back
        if not file:
            continue
        norm = _norm(file)
        if norm == guard_self:
            continue
        if any(_allow_match(norm, entry) for entry in allow):
            return True
    return False


def _verbose() -> bool:
    return (os.getenv("ALPHA_FS_GUARD_VERBOSE") or "").strip().lower() in {"1", "true", "yes", "on"}


def _check_delete(path: object) -> None:
    if not _STATE["roots"]:
        return
    if _is_under(_norm(path), _STATE["roots"]) and not _caller_allowlisted():
        if _verbose():
            print(f"[fs_guard] blocked delete: {path}", file=sys.stderr)
        raise ProtectedDeleteError(f"fs_guard: 禁止删除保护区路径: {path}")


def _check_move(src: object, dst: object) -> None:
    if not _STATE["roots"]:
        return
    if (
        _is_under(_norm(src), _STATE["roots"])
        and not _is_under(_norm(dst), _STATE["roots"])
        and not _caller_allowlisted()
    ):
        if _verbose():
            print(f"[fs_guard] blocked move: {src} -> {dst}", file=sys.stderr)
        raise ProtectedDeleteError(f"fs_guard: 禁止将保护区内容移出: {src} -> {dst}")


def _wrap_delete(name: str, orig: Callable) -> Callable:
    def guarded(path, *args, **kwargs):
        _check_delete(path)
        return orig(path, *args, **kwargs)

    guarded.__name__ = name
    return guarded


def _wrap_move(name: str, orig: Callable) -> Callable:
    def guarded(src, dst, *args, **kwargs):
        _check_move(src, dst)
        return orig(src, dst, *args, **kwargs)

    guarded.__name__ = name
    return guarded


def _wrap_path_method(orig: Callable) -> Callable:
    def guarded(self, *args, **kwargs):
        _check_delete(self)
        return orig(self, *args, **kwargs)

    guarded.__name__ = getattr(orig, "__name__", "guarded")
    return guarded


def install(protected_roots: Iterable[Path | str], allowlist: Iterable[str] = ()) -> None:
    """激活守卫（幂等）：重复调用只更新保护区/白名单，不叠加包装。"""
    global _INSTALLED
    _STATE["roots"] = sorted({_norm(r) for r in protected_roots})
    merged: list[str] = []
    for entry in (*DEFAULT_ALLOWLIST, *allowlist):
        entry = entry.strip()
        if entry and entry not in merged:
            merged.append(entry)
    _STATE["allow"] = merged
    if _INSTALLED:
        return

    for obj, name in ((os, "remove"), (os, "unlink"), (os, "rmdir"), (shutil, "rmtree")):
        orig = getattr(obj, name)
        setattr(obj, name, _wrap_delete(name, orig))
        _ORIGINALS.append((obj, name, orig))
    for obj, name in ((os, "rename"), (os, "replace"), (shutil, "move")):
        orig = getattr(obj, name)
        setattr(obj, name, _wrap_move(name, orig))
        _ORIGINALS.append((obj, name, orig))
    for cls, name in ((Path, "unlink"), (Path, "rmdir")):
        orig = getattr(cls, name)
        setattr(cls, name, _wrap_path_method(orig))
        _ORIGINALS.append((cls, name, orig))
    _INSTALLED = True


def install_fs_guard(root: Path | str | None = None) -> bool:
    """按环境变量装配默认守卫；返回是否实际启用。"""
    flag = (os.getenv("ALPHA_FS_GUARD") or "1").strip().lower()
    if flag in _OFF_VALUES:
        return False
    extra = [
        item.strip()
        for item in (os.getenv("ALPHA_FS_GUARD_ALLOW") or "").replace(";", ",").split(",")
        if item.strip()
    ]
    install(default_protected_roots(root), extra)
    return True


def uninstall() -> None:
    """还原全部补丁（测试与回退用）。"""
    global _INSTALLED
    for obj, name, orig in reversed(_ORIGINALS):
        setattr(obj, name, orig)
    _ORIGINALS.clear()
    _STATE["roots"] = []
    _STATE["allow"] = []
    _INSTALLED = False
