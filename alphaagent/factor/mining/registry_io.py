"""mining_delivered_registry.json 读写（submit / repair 共用）。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alphaagent.factor.types import IngestPolicy
from alphaagent.factor.zoo import FactorZoo
from alphaagent.factor.zoo.similarity import SimilarityMatrix


def load_mining_registry(path: Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def save_mining_registry(path: Path, registry: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _trim_similarity(sim: dict[str, Any] | None, top_k: int = 3) -> dict[str, Any] | None:
    if not isinstance(sim, dict) or not sim:
        return None
    out = dict(sim)
    nb = out.get("top_neighbors")
    if isinstance(nb, list) and top_k > 0:
        out["top_neighbors"] = nb[:top_k]
    return out


def upsert_mining_registry(
    registry_path: Path,
    *,
    factor_id: str,
    name: str,
    expr: str,
    expr_dir: Path,
    repo_root: Path,
    policy: IngestPolicy,
    metrics: dict[str, Any] | None,
    similarity: dict[str, Any] | None,
    comment: str = "",
    ingest_status: str = "stored",
    source: str = "submit",
    merge: bool = True,
) -> tuple[str, str]:
    """写入或合并一条 registry 记录；返回 (registry_path, dsl_path)。"""
    registry_path = Path(registry_path).expanduser().resolve()
    expr_dir = Path(expr_dir).expanduser().resolve()
    repo_root = Path(repo_root).resolve()
    expr_dir.mkdir(parents=True, exist_ok=True)

    dsl_path = expr_dir / f"{factor_id}.dsl"
    dsl_path.write_text(expr.strip() + "\n", encoding="utf-8")
    rel_expr = dsl_path.relative_to(repo_root).as_posix()

    registry = load_mining_registry(registry_path) if merge else {}
    prev = registry.get(factor_id, {}) if merge else {}

    entry: dict[str, Any] = {
        "name": name,
        "comment": comment or prev.get("comment") or name,
        "expression_file": rel_expr,
        "ingest_config": policy.ingest_config_dict(),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "ingest_metrics": metrics,
        "ingest_status": ingest_status,
    }
    sim = _trim_similarity(similarity)
    if sim:
        entry["similarity"] = sim
    elif prev.get("similarity"):
        entry["similarity"] = prev["similarity"]
    if source == "submit" and "source_runs" not in prev:
        entry["source"] = "submit"
    for key in ("source_runs", "mining_metrics"):
        if key in prev:
            entry[key] = prev[key]
    if prev.get("source") == "submit" and source != "submit":
        entry["comment"] = prev.get("comment") or entry["comment"]
        if "source" in prev:
            entry["source"] = prev["source"]

    registry[factor_id] = entry
    save_mining_registry(registry_path, registry)
    return str(registry_path), str(dsl_path)


def sync_registry_from_zoo(
    registry_path: Path,
    zoo: FactorZoo,
    *,
    expr_dir: Path,
    repo_root: Path,
    policy: IngestPolicy,
) -> dict[str, Any]:
    """将 zoo 中已有因子同步进 registry（补缺 + 刷新 ingest_metrics / similarity）。"""
    registry = load_mining_registry(registry_path)
    zoo_ids = set(zoo.catalog.list_factor_ids())
    added: list[str] = []
    refreshed: list[str] = []
    sim_mat = SimilarityMatrix(zoo.paths, zoo.manifest.max_factors) if zoo_ids else None
    for fid in sorted(zoo_ids):
        meta = zoo.catalog.get(fid)
        if meta is None:
            continue
        prev = registry.get(fid, {})
        metrics = (meta.extra or {}).get("metrics")
        if fid not in registry:
            added.append(fid)
        elif metrics and prev.get("ingest_metrics") != metrics:
            refreshed.append(fid)
        sim_info = None
        if sim_mat is not None:
            values = zoo.read_factor(fid)
            report = sim_mat.cross_sectional_neighbor_report(
                zoo,
                values,
                exclude_factor_id=fid,
                top_k=policy.similar_top_k,
            )
            sim_info = {
                "col_idx": meta.col_idx,
                "n_factors": zoo.n_factors,
                "kind": report.get("kind"),
                "max_abs_corr": report.get("max_abs_corr"),
                "top_neighbors": report.get("top_neighbors"),
            }
        upsert_mining_registry(
            registry_path,
            factor_id=fid,
            name=meta.name,
            expr=meta.expr,
            expr_dir=expr_dir,
            repo_root=repo_root,
            policy=policy,
            metrics=metrics,
            similarity=sim_info,
            comment=str(prev.get("comment") or meta.name),
            ingest_status=str(prev.get("ingest_status") or "stored"),
            source=str(prev.get("source") or "zoo_sync"),
            merge=True,
        )
    registry_after = load_mining_registry(registry_path)
    orphan = sorted(set(registry_after.keys()) - zoo_ids)
    return {
        "added": added,
        "refreshed": refreshed,
        "orphan_in_registry": orphan,
        "n_registry": len(registry_after),
        "n_zoo": len(zoo_ids),
    }
