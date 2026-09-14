"""报告文件统一 I/O 工具模块。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def read_report_json(path: Path | str) -> dict[str, Any] | None:
    """读取 report.json / json 文件：utf-8 优先，GBK 容错回退，异常返回 None。

    解决 Windows 系统下不同工具写入产物在 UTF-8 / GBK 编码混合时
    导致的 UnicodeDecodeError 偶发崩溃。
    """
    p = Path(path).expanduser().resolve()
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        try:
            return json.loads(p.read_text(encoding="gbk"))
        except (json.JSONDecodeError, OSError):
            return None
    except (json.JSONDecodeError, OSError):
        return None
