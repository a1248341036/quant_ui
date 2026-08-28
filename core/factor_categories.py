"""因子类别注册表：候选库 + 正式库按研究模式分目录。

**单一事实源**：类别定义在 core.research_modes.RESEARCH_MODES 中
（mode_id/label/candidate_dir/production_dir 均在那里），本模块只做路径
派生与兼容接口，不再维护第二份清单。

加新类别 = 在 core.research_modes.RESEARCH_MODES 加一项，本模块自动跟随。

目录结构:
  artifacts/alphaagent/factorzoo/
    candidate_technical/       ← 候选库（日线技术）
    production_technical/      ← 正式库（日线技术）
    candidate_fundamental/     ← 候选库（基本面）
    production_fundamental/    ← 正式库（基本面）
"""

from __future__ import annotations

from pathlib import Path

from alphaagent.core.paths import ARTIFACTS_DIR
from core.research_modes import RESEARCH_MODES

FACTORZOO_ROOT = ARTIFACTS_DIR / "alphaagent" / "factorzoo"

DEFAULT_CATEGORY = "technical"


def get_category_label(mode: str) -> str:
    spec = RESEARCH_MODES.get(mode)
    return spec.label if spec is not None else mode


def candidate_dir(mode: str) -> Path:
    spec = RESEARCH_MODES.get(mode)
    if spec is None:
        raise ValueError(f"unknown_factor_category: {mode}")
    return FACTORZOO_ROOT / spec.candidate_dir


def production_dir(mode: str) -> Path:
    spec = RESEARCH_MODES.get(mode)
    if spec is None:
        raise ValueError(f"unknown_factor_category: {mode}")
    return FACTORZOO_ROOT / spec.production_dir


def candidate_registry_path(mode: str) -> Path:
    return candidate_dir(mode) / "mining_candidate_registry.json"


def candidate_expr_dir(mode: str) -> Path:
    return candidate_dir(mode) / "expressions"


def production_registry_path(mode: str) -> Path:
    return production_dir(mode) / "mining_delivered_registry.json"


def production_expr_dir(mode: str) -> Path:
    return production_dir(mode) / "expressions"


def all_categories() -> list[str]:
    return list(RESEARCH_MODES.keys())


def category_choices() -> list[dict[str, str]]:
    """前端下拉选项用（从研究模式注册表派生）。"""
    return [
        {"value": mode, "label": spec.label}
        for mode, spec in RESEARCH_MODES.items()
    ]