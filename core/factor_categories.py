"""因子类别注册表：候选库 + 正式库按 research_mode 分目录。

加新类别只需在 FACTOR_CATEGORIES 中注册一行，不改任何代码逻辑。
research_mode 已在整条链路传递（ResearchSpec → start_run → FactorSubmitService），
库路径自动跟随。

目录结构:
  artifacts/alphaagent/factorzoo/
    candidate_technical/       ← 候选库（日线技术）
      mining_candidate_registry.json
      expressions/
    candidate_fundamental/     ← 候选库（基本面）
      ...
    production_technical/      ← 正式库（日线技术，FactorZoo 密集矩阵）
      manifest.json
      index/
      values/
      ...
    production_fundamental/    ← 正式库（基本面）
      ...
"""

from __future__ import annotations

from pathlib import Path

from alphaagent.core.paths import ARTIFACTS_DIR

FACTORZOO_ROOT = ARTIFACTS_DIR / "alphaagent" / "factorzoo"

FACTOR_CATEGORIES: dict[str, dict[str, str]] = {
    "technical": {
        "label": "日线技术",
        "candidate_dir": "candidate_technical",
        "production_dir": "production_technical",
    },
    "fundamental": {
        "label": "基本面",
        "candidate_dir": "candidate_fundamental",
        "production_dir": "production_fundamental",
    },
    # "sentiment": {
    #     "label": "舆情",
    #     "candidate_dir": "candidate_sentiment",
    #     "production_dir": "production_sentiment",
    # },
}

DEFAULT_CATEGORY = "technical"


def get_category_label(mode: str) -> str:
    return FACTOR_CATEGORIES.get(mode, {}).get("label", mode)


def candidate_dir(mode: str) -> Path:
    entry = FACTOR_CATEGORIES.get(mode)
    if entry is None:
        raise ValueError(f"unknown_factor_category: {mode}")
    return FACTORZOO_ROOT / entry["candidate_dir"]


def production_dir(mode: str) -> Path:
    entry = FACTOR_CATEGORIES.get(mode)
    if entry is None:
        raise ValueError(f"unknown_factor_category: {mode}")
    return FACTORZOO_ROOT / entry["production_dir"]


def candidate_registry_path(mode: str) -> Path:
    return candidate_dir(mode) / "mining_candidate_registry.json"


def candidate_expr_dir(mode: str) -> Path:
    return candidate_dir(mode) / "expressions"


def production_registry_path(mode: str) -> Path:
    return production_dir(mode) / "mining_delivered_registry.json"


def production_expr_dir(mode: str) -> Path:
    return production_dir(mode) / "expressions"


def all_categories() -> list[str]:
    return list(FACTOR_CATEGORIES.keys())


def category_choices() -> list[dict[str, str]]:
    """前端下拉选项用。"""
    return [{"value": k, "label": v["label"]} for k, v in FACTOR_CATEGORIES.items()]
