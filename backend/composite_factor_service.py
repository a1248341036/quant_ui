"""组合因子库（composite factors）：把 ML/简单加权组合的运行产物固化为
可复现的"组合因子"条目，供前端 ML 组合页点开查看构成/指标/复现命令。

存储选型（2026-09-09）：SQLite 单文件 ``artifacts/alphaagent/composite_factors.db``
（WAL）——条目量级小（数十级）、按名称/方案检索 + 前端详情直读都够用，
且与仓库既有 factor_index.db / research_memory.db 的单文件 SQLite 惯例一致；
组合分数矩阵本身不进库，仍存 parquet（条目 out_dir 内，回测引擎可直接消费），
库里只存路径与全部复现元数据。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "artifacts" / "alphaagent" / "composite_factors.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS composite_factors (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    scheme TEXT NOT NULL,
    created_at TEXT NOT NULL,
    train_id TEXT,
    out_dir TEXT NOT NULL,
    score_path TEXT,
    params_json TEXT,
    provenance_json TEXT,
    metrics_json TEXT,
    features_json TEXT,
    note TEXT
);
"""

# 方案 → 复现所需的方法学说明（详情页展示）
_METHOD_NOTES = {
    "ml": "ML 学习加权：Ridge/LightGBM walk-forward 逐折拟合，OOS 预测混合为组合分数。",
    "equal": "等权 1/N：每折 train 段估计各因子符号（IC 均值方向），OOS 段在场因子等权平均；不做因子挑选。",
    "icir": "ICIR 加权：每折 train 段计算各因子日度 RankIC 的均值/标准差（ICIR），按 |ICIR| 归一加权 OOS 分数；符号随 train 段 IC 方向。",
    "hrp": "HRP 加权：每折 train 段计算因子间日度 RankIC 相关矩阵 → 层次聚类（距离=√(1-ρ)）→ 树内逆方差分配权重，应用到该折 OOS；窗口/系数归一化的同骨架变体视为同构。权重只由 train 段估计，无未来信息。",
}


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute(_SCHEMA)
    return con


def _find_score_file(out_dir: Path) -> str | None:
    """out_dir 内分数 parquet 探测：优先新命名 scores.parquet，回退历史 hrp_scores.parquet。"""
    for cand in ("scores.parquet", "hrp_scores.parquet"):
        p = out_dir / cand
        if p.is_file():
            return str(p)
    return None


def _read_report_json(report_path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(report_path.read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        try:
            return json.loads(report_path.read_text(encoding="gbk"))
        except (json.JSONDecodeError, OSError):
            return None
    except (json.JSONDecodeError, OSError):
        return None


def _repro_command(report: dict[str, Any], out_dir: Path) -> str:
    """从 report 元数据重建可复现命令行（与 report/分数文件所在 out_dir 一致）。"""
    parts = [
        ".venv\\Scripts\\python.exe scripts\\train_ml_composite.py",
        "--modes technical",
        f"--scheme {report.get('scheme') or 'ml'}",
        f"--label-days {report.get('label_days', 5)}",
    ]
    if int(report.get("score_smooth") or 0) > 1:
        parts.append(f"--score-smooth {report.get('score_smooth')}")
    include = report.get("include_factors")
    if include:
        parts.append("--include-factors " + " ".join(str(n) for n in include))
    parts.append(f"--mining-end {report.get('mining_end')}")
    parts.append("--isolation holdout")
    parts.append(f'--out-dir "{out_dir}"')
    parts.append(f'--pred-out "{out_dir / "scores.parquet"}"')
    return " ".join(parts)


def save_composite_factor(out_dir: str | Path, *, name: str | None = None,
                          note: str | None = None) -> dict[str, Any]:
    """把一次组合运行的产物（report.json + 分数 parquet）固化为组合因子条目。"""
    out_dir = Path(out_dir)
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    report = _read_report_json(out_dir / "report.json")
    if report is None:
        return {"error": f"report_not_found（{out_dir / 'report.json'} 无有效 report）"}

    scheme = str(report.get("scheme") or "ml")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    cf_id = f"cf_{ts}"
    default_name = report.get("scheme_label") or scheme
    entry_name = name or f"{default_name} · OOS{str(report.get('panel_start', ''))[:7]}起"

    decay = report.get("decay_table") or []
    ratios = [float(r["decay_ratio"]) for r in decay if isinstance(r.get("decay_ratio"), (int, float))]
    gate = report.get("gate") or {}

    metrics = {
        "oos_ic_blended": report.get("oos_ic_blended"),
        "gate_passed": gate.get("passed"),
        "gate_fail_reasons": gate.get("fail_reasons") or [],
        "decay_retention_mean": round(sum(ratios) / len(ratios), 4) if ratios else None,
        "decay_table": decay,
    }
    provenance = {
        "scheme": scheme,
        "scheme_label": report.get("scheme_label") or scheme,
        "method": _METHOD_NOTES.get(scheme, ""),
        "mining_end": report.get("mining_end"),
        "time_isolation": report.get("time_isolation"),
        "panel_start": report.get("panel_start"),
        "panel_end": report.get("panel_end"),
        "label_days": report.get("label_days"),
        "score_smooth": report.get("score_smooth"),
        "folds": report.get("folds"),
        # 分数 parquet 命名演进：(scores.parquet 新 self-contained 命名) / (hrp_scores.parquet 历史命名)
        "score_path": _find_score_file(out_dir),
        "report_path": str(out_dir / "report.json"),
        "repro_command": _repro_command(report, out_dir),
        "saved_from_train_id": out_dir.name,
    }
    features = {
        "feature_names": report.get("feature_names") or [],
        "dropped": report.get("dropped") or [],
    }

    with _connect() as con:
        con.execute(
            """
            INSERT OR REPLACE INTO composite_factors
            (id, name, scheme, created_at, train_id, out_dir, score_path,
             params_json, provenance_json, metrics_json, features_json, note)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                cf_id,
                entry_name,
                scheme,
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                out_dir.name,
                str(out_dir),
                provenance["score_path"],
                None,
                json.dumps(provenance, ensure_ascii=False, indent=1),
                json.dumps(metrics, ensure_ascii=False, indent=1),
                json.dumps(features, ensure_ascii=False, indent=1),
                note,
            ),
        )
    return {"ok": True, "id": cf_id, "name": entry_name}


def list_composite_factors(limit: int = 50) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with _connect() as con:
        rows = con.execute(
            "SELECT * FROM composite_factors ORDER BY created_at DESC LIMIT ?",
            (max(1, int(limit)),),
        ).fetchall()
    for row in rows:
        metrics = json.loads(row["metrics_json"] or "{}")
        provenance = json.loads(row["provenance_json"] or "{}")
        features = json.loads(row["features_json"] or "{}")
        blended = metrics.get("oos_ic_blended") or {}
        out.append({
            "id": row["id"],
            "name": row["name"],
            "scheme": row["scheme"],
            "scheme_label": provenance.get("scheme_label") or row["scheme"],
            "created_at": row["created_at"],
            "train_id": row["train_id"],
            "oos_ic": blended.get("ic_mean"),
            "oos_ic_ir": blended.get("ic_ir"),
            "n_days": blended.get("n_days"),
            "gate_passed": metrics.get("gate_passed"),
            "feature_count": len(features.get("feature_names") or []),
            "mining_end": provenance.get("mining_end"),
            "note": row["note"],
        })
    return out


def get_composite_factor(cf_id: str) -> dict[str, Any] | None:
    with _connect() as con:
        row = con.execute(
            "SELECT * FROM composite_factors WHERE id = ?", (cf_id,)
        ).fetchone()
    if row is None:
        return None
    out = dict(row)
    for key in ("params_json", "provenance_json", "metrics_json", "features_json"):
        raw = out.pop(key, None)
        out[key.removesuffix("_json")] = json.loads(raw) if raw else None
    return out


def delete_composite_factor(cf_id: str) -> bool:
    with _connect() as con:
        cursor = con.execute("DELETE FROM composite_factors WHERE id = ?", (cf_id,))
        return cursor.rowcount > 0
