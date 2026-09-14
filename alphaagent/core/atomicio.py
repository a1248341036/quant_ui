"""文件原子写入公共工具模块。"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path


def atomic_write_text(path: Path | str, content: str, encoding: str = "utf-8") -> None:
    """原子化写入文本文件（tmp + os.replace）。

    先写入同目录临时文件并 flush/fsync，然后原子替换目标文件，
    避免写入中途进程崩溃/断电造成目标文件空洞或损坏。
    """
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with open(fd, "w", encoding=encoding) as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def atomic_write_bytes(path: Path | str, content: bytes) -> None:
    """原子化写入二进制文件。"""
    p = Path(path).expanduser().resolve()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=f".{p.name}.", suffix=".tmp")
    try:
        with open(fd, "wb") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, p)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
