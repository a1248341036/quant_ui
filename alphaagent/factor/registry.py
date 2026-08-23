"""因子 registry 加载。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    for _ in range(10):
        if (p / "pyproject.toml").is_file():
            return p
        if p.parent == p:
            break
        p = p.parent
    return start.resolve()


def load_registry(path: Path, *, repo_root: Path | None = None) -> dict[str, dict[str, Any]]:
    """读取 registry.json，解析 expression / expression_file。"""
    reg_path = Path(path).expanduser().resolve()
    root = repo_root or _find_repo_root(reg_path.parent)
    raw = json.loads(reg_path.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for factor_id, spec in raw.items():
        if not isinstance(spec, dict):
            raise ValueError(f"registry[{factor_id!r}] 须为 object")
        entry = dict(spec)
        if "expression" not in entry and "expression_file" not in entry:
            raise ValueError(f"registry[{factor_id!r}] 缺少 expression 或 expression_file")
        if "expression_file" in entry and "expression" not in entry:
            expr_path = Path(entry["expression_file"])
            if not expr_path.is_absolute():
                expr_path = root / expr_path
            entry["expression"] = expr_path.read_text(encoding="utf-8").strip()
        entry.setdefault("name", factor_id)
        out[str(factor_id)] = entry
    return out


def list_factor_entries(path: Path, *, repo_root: Path | None = None) -> list[tuple[str, str, str]]:
    """返回 (factor_id, name, expression) 列表。"""
    reg = load_registry(path, repo_root=repo_root)
    return [(fid, str(v["name"]), str(v["expression"])) for fid, v in reg.items()]
