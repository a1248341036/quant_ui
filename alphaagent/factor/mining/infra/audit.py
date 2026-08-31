"""Compact, reproducible audit artifacts for AlphaAgent runs."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def canonical_hash(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def file_hash(path: Path | str, *, limit_bytes: int | None = None) -> str | None:
    if isinstance(path, str) and "://" in path:
        return None
    path = Path(path)
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        remaining = limit_bytes
        while True:
            chunk = handle.read(1024 * 1024 if remaining is None else min(1024 * 1024, remaining))
            if not chunk:
                break
            digest.update(chunk)
            if remaining is not None:
                remaining -= len(chunk)
                if remaining <= 0:
                    break
    return digest.hexdigest()


def git_revision(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def build_manifest(*, root: Path, config: Any, user_message: str, research_spec: dict[str, Any],
                   panel_path: Path, factorlib_path: Path | None, model: str) -> dict[str, Any]:
    eval_ctx = config.eval
    return {
        "schema_version": 1,
        "model": model,
        "user_message_hash": canonical_hash(user_message),
        "research_spec": research_spec,
        "research_spec_hash": canonical_hash(research_spec),
        "config": {
            "train_start": eval_ctx.train_start,
            "train_end": eval_ctx.train_end,
            "val_start": eval_ctx.val_start,
            "val_end": eval_ctx.val_end,
            "label_col": eval_ctx.label_col,
            "include_fundamentals": eval_ctx.include_fundamentals,
            "max_turns": config.max_turns,
            "max_tool_calls_per_round": config.max_tool_calls_per_round,
            "max_tool_workers": config.max_tool_workers,
            "max_parallel_eval": config.max_parallel_eval,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "enable_submit": config.enable_submit,
            "enable_reviewer": config.enable_reviewer,
        },
        "artifacts": {
            "panel_path": str(panel_path),
            "panel_sha256": file_hash(panel_path),
            "factorlib_path": str(factorlib_path) if factorlib_path else None,
            "factorlib_registry_sha256": file_hash(factorlib_path / "mining_delivered_registry.json") if factorlib_path else None,
        },
        "runtime": {"python": sys.version, "platform": platform.platform(), "git_revision": git_revision(root)},
    }
