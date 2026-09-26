"""Bench 冻结配置管理。

配置文件：artifacts/alphaagent/bench/config.json
支持加载、修改与规范化哈希（config_hash）。
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any

from core.atomicio import atomic_write_text

ROOT = Path(__file__).resolve().parents[4]
BENCH_DIR = ROOT / "artifacts" / "alphaagent" / "bench"
CONFIG_FILE = BENCH_DIR / "config.json"

DEFAULT_BENCH_CONFIG: dict[str, Any] = {
    "panel": "cne://",
    "train_start": "2020-01-01",
    "train_end": "2022-12-31",
    "val_start": "2023-01-01",
    "val_end": "2024-12-31",
    "label_col": "label_1d_open_to_open",
    "model": None,  # None 时读环境变量 MODEL 或 codex 配置
    "temperature": 0.7,
    "reasoning_effort": "medium",
    "max_tokens": 16384,
    "max_turns": 5,
    "max_tool_calls_per_round": 8,
    "max_tool_workers": 8,
    "max_parallel_eval": 6,
    "min_tool_call_rounds": 3,
    "focus_facets": "",
    "research_spec_file": None,
    "user_message": "请在训练集上提出并迭代多个多行因子表达式，再于验证集上检验泛化；目标为提高 abs(IC)/RANKIC 与 ICIR，并兼顾月度稳健性。",
    "no_reviewer": False,
    "quiet": False,
    # 判定容差参数
    "relative_tolerance": 0.10,  # 10%
    "absolute_tolerance_pp": 1.5,  # 1.5pp
    "saturation_epsilon": 0.001,
}

# 影响执行效果与语义的核心字段（用于生成 config_hash）
HASH_FIELDS = [
    "panel", "train_start", "train_end", "val_start", "val_end", "label_col",
    "model", "temperature", "reasoning_effort", "max_tokens", "max_turns",
    "max_tool_calls_per_round", "max_parallel_eval", "focus_facets",
    "research_spec_file", "user_message", "no_reviewer",
]


def compute_config_hash(cfg: dict[str, Any]) -> str:
    """对关键配置字段生成 12 位确定性短哈希。"""
    sub = {k: cfg.get(k) for k in HASH_FIELDS}
    raw = json.dumps(sub, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]


def load_bench_config() -> dict[str, Any]:
    """读取已保存的配置；若不存在则自动初始化默认配置。"""
    if not CONFIG_FILE.is_file():
        cfg = copy.deepcopy(DEFAULT_BENCH_CONFIG)
        save_bench_config(cfg)
        return cfg
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        merged = copy.deepcopy(DEFAULT_BENCH_CONFIG)
        merged.update(data)
        return merged
    except Exception:
        return copy.deepcopy(DEFAULT_BENCH_CONFIG)


def save_bench_config(cfg: dict[str, Any]) -> None:
    """持久化保存配置到 config.json。"""
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(CONFIG_FILE, json.dumps(cfg, ensure_ascii=False, indent=2) + "\n")


def set_config_value(key: str, val_str: str) -> dict[str, Any]:
    """设置配置键值并自动转换类型。"""
    cfg = load_bench_config()
    if key not in DEFAULT_BENCH_CONFIG:
        raise KeyError(f"未知配置项: {key} (可选: {list(DEFAULT_BENCH_CONFIG.keys())})")

    default_v = DEFAULT_BENCH_CONFIG[key]
    parsed_v: Any = val_str
    if isinstance(default_v, bool):
        parsed_v = val_str.lower() in ("true", "1", "yes", "on")
    elif isinstance(default_v, int) and not isinstance(default_v, bool):
        parsed_v = int(val_str)
    elif isinstance(default_v, float):
        parsed_v = float(val_str)
    elif default_v is None and val_str.lower() in ("none", "null", ""):
        parsed_v = None

    cfg[key] = parsed_v
    save_bench_config(cfg)
    return cfg
