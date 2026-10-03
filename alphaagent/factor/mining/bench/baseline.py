"""基线快照管理。

文件位置：artifacts/alphaagent/bench/baseline.json
归档位置：artifacts/alphaagent/bench/history/baseline_<ts>_<run_id>.json
"""
from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from alphaagent.core.timeutil import utc_now_iso
from core.atomicio import atomic_write_text

ROOT = Path(__file__).resolve().parents[4]
BENCH_DIR = ROOT / "artifacts" / "alphaagent" / "bench"
BASELINE_FILE = BENCH_DIR / "baseline.json"
HISTORY_DIR = BENCH_DIR / "history"
UI_ROOT = ROOT / "logs" / "factor_mining" / "ui"


def load_baseline() -> dict[str, Any] | None:
    if not BASELINE_FILE.is_file():
        return None
    try:
        return json.loads(BASELINE_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None


def set_baseline(run_id: str, note: str = "") -> dict[str, Any]:
    run_dir = UI_ROOT / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(f"未找到 run 目录: {run_dir}")

    from alphaagent.factor.mining.run_metrics import generate_scorecard
    # 确保持久化生成包含 v4 完整指标的 scorecard
    sc = generate_scorecard(run_id, run_dir)

    # 归档现有旧基线
    if BASELINE_FILE.is_file():
        HISTORY_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        old_data = load_baseline() or {}
        old_id = old_data.get("run_id", "prev")
        archive_file = HISTORY_DIR / f"baseline_{stamp}_{old_id}.json"
        try:
            shutil.copy2(BASELINE_FILE, archive_file)
        except Exception:
            pass

    from alphaagent.factor.mining.bench.config import compute_config_hash, load_bench_config
    cfg = load_bench_config()
    cfg_hash = compute_config_hash(cfg)
    # ⚠ 2026-10-03：基线必须存**run 自身身份**，否则 diff 永远判"不可直接比"。
    # scorecard 的 time_meta.config_hash 在 API/监控启动的 run 上是由 (research_mode,
    # research_spec_hash) 合成的身份（见 extended_metrics._extract_time_meta）；只有 bench
    # 自己启动的 run 才有 bench_meta.json 的 frozen hash。优先用 scorecard 里的那个。
    try:
        _sc_hash = ((sc or {}).get("time_meta") or {}).get("config_hash")
    except Exception:  # noqa: BLE001
        _sc_hash = None
    if _sc_hash:
        cfg_hash = _sc_hash

    # 从 run_meta 读取补充信息
    meta_path = run_dir / "run_meta.json"
    run_commit = None
    if meta_path.is_file():
        try:
            m = json.loads(meta_path.read_text(encoding="utf-8"))
            run_commit = m.get("commit")
        except Exception:
            pass

    baseline_payload = {
        "baseline_set_at": utc_now_iso(),
        "run_id": run_id,
        "note": note,
        "config_hash": cfg_hash,
        "commit": run_commit,
        "scorecard": sc,
    }

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    atomic_write_text(BASELINE_FILE, json.dumps(baseline_payload, ensure_ascii=False, indent=2) + "\n")
    return baseline_payload
