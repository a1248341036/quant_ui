"""YAML 配置加载。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from alphaagent.core.paths import CONFIGS_DIR


def load_yaml(path: Path | str) -> dict[str, Any]:
    """加载 YAML 配置文件。"""
    p = Path(path)
    if not p.is_absolute():
        p = CONFIGS_DIR / p
    if not p.is_file():
        raise FileNotFoundError(f"配置文件不存在: {p}")
    with open(p, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}
