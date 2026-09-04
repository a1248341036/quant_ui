"""mining_delivered_registry.json 读写（submit / repair 共用）。"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from alphaagent.factor.types import IngestPolicy
from alphaagent.factor.zoo import FactorZoo
from alphaagent.factor.zoo.similarity import SimilarityMatrix

_LABEL_HORIZON_RE = re.compile(r"label_(\d+)d")


def derive_freq_from_label_col(label_col: str | None) -> tuple[str | None, str | None]:
    """从 label_col 推导 (research_mode, 档位默认门禁调仓频率)。

    长持有期（≥10d）→ fundamental/monthly；短持有期 → technical/weekly。
    只作老条目兜底展示：新条目入库时已记录真实 rebalance_freq（含用户覆盖），
    读取方仅在 entry 缺字段时才调用本函数。
    """
    m = _LABEL_HORIZON_RE.search(str(label_col or ""))
    if not m:
        return None, None
    return ("fundamental", "monthly") if int(m.group(1)) >= 10 else ("technical", "weekly")


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
    interaction: dict[str, Any] | None = None,
    facets: list[str] | None = None,
    eval_label: str | None = None,
    rebalance_freq: str | None = None,
    research_mode: str | None = None,
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
    # 数据面元数据（2026-09-03）：facets/is_fusion/family/eval_label；
    # 未显式传 facets（zoo_sync 等路径）时按表达式现算，老条目自然补齐。
    from alphaagent.factor.mining.memory.expressions import (
        classify_family_ex,
        expr_facets,
    )

    facet_list = sorted(facets) if facets else sorted(expr_facets(expr))
    entry["facets"] = facet_list
    entry["is_fusion"] = len(facet_list) >= 2
    entry["family"] = classify_family_ex(name, expr)[0]
    entry["eval_label"] = eval_label or prev.get("eval_label")
    # 调仓频率/研究档位溯源（2026-09-03）：submit 路径记录真实 chosen_freq
    #（用户覆盖 > 档位默认）；zoo_sync/lab 等未传路径沿用旧值或留空由视图层推导。
    entry["rebalance_freq"] = rebalance_freq or prev.get("rebalance_freq")
    entry["research_mode"] = research_mode or prev.get("research_mode")
    sim = _trim_similarity(similarity)
    if sim:
        entry["similarity"] = sim
    elif prev.get("similarity"):
        entry["similarity"] = prev["similarity"]
    if source == "submit" and "source_runs" not in prev:
        entry["source"] = "submit"
    if interaction is not None:
        entry["interaction"] = interaction
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


def write_candidate_registry(
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
    source: str = "submit_stage_one",
    evaluation_evidence: dict[str, Any] | None = None,
    data_fingerprint: dict[str, Any] | None = None,
    interaction: dict[str, Any] | None = None,
    facets: list[str] | None = None,
    eval_label: str | None = None,
    rebalance_freq: str | None = None,
    research_mode: str | None = None,
) -> tuple[str, str]:
    """Registry-only candidate storage: evidence and DSL, never dense values."""
    registry_path = Path(registry_path).expanduser().resolve()
    expr_dir = Path(expr_dir).expanduser().resolve()
    repo_root = Path(repo_root).resolve()
    expr_dir.mkdir(parents=True, exist_ok=True)

    dsl_path = expr_dir / f"{factor_id}.dsl"
    dsl_path.write_text(expr.strip() + "\n", encoding="utf-8")

    registry = load_mining_registry(registry_path)
    prev = registry.get(factor_id, {})
    # 数据面元数据（2026-09-03）：与 upsert_mining_registry 同口径
    from alphaagent.factor.mining.memory.expressions import (
        classify_family_ex,
        expr_facets,
    )

    facet_list = sorted(facets) if facets else sorted(expr_facets(expr))
    entry: dict[str, Any] = {
        "schema_version": 2,
        "name": name,
        "comment": comment or prev.get("comment") or name,
        "expr": expr.strip(),
        "expression_file": dsl_path.relative_to(repo_root).as_posix(),
        "ingest_config": policy.ingest_config_dict(),
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics or {},
        "similarity": _trim_similarity(similarity),
        "evaluation_evidence": evaluation_evidence or prev.get("evaluation_evidence"),
        "interaction": interaction or prev.get("interaction"),
        "data_fingerprint": data_fingerprint or prev.get("data_fingerprint"),
        "review_status": str(prev.get("review_status") or "pending_review"),
        "promotion_status": "pending",
        "ingest_status": "candidate",
        "source": source,
        "facets": facet_list,
        "is_fusion": len(facet_list) >= 2,
        "family": classify_family_ex(name, expr)[0],
        "eval_label": eval_label or prev.get("eval_label"),
        "rebalance_freq": rebalance_freq or prev.get("rebalance_freq"),
        "research_mode": research_mode or prev.get("research_mode"),
    }
    for key in ("source_runs", "mining_metrics", "review", "reviewed_at"):
        if key in prev:
            entry[key] = prev[key]
    registry[factor_id] = entry
    save_mining_registry(registry_path, registry)
    return str(registry_path), str(dsl_path)


def set_candidate_review(
    registry_path: Path,
    *,
    factor_id: str,
    review: dict[str, Any],
    promotion_status: str,
) -> dict[str, Any] | None:
    """Attach reviewer output and update candidate lifecycle state."""
    registry = load_mining_registry(registry_path)
    entry = registry.get(factor_id)
    if not isinstance(entry, dict):
        return None
    verdict = str(review.get("verdict") or "pending").lower()
    entry["review"] = review
    entry["review_status"] = verdict if verdict in {"approve", "revise", "reject"} else "pending_review"
    entry["reviewed_at"] = datetime.now(timezone.utc).isoformat()
    entry["promotion_status"] = promotion_status
    save_mining_registry(registry_path, registry)
    return entry


def set_candidate_promotion(
    registry_path: Path,
    *,
    factor_id: str,
    promotion_status: str,
) -> dict[str, Any] | None:
    """Update lifecycle state when no reviewer is attached to the submit call."""
    registry = load_mining_registry(registry_path)
    entry = registry.get(factor_id)
    if not isinstance(entry, dict):
        return None
    entry["promotion_status"] = promotion_status
    if promotion_status == "promoted":
        entry["promoted_at"] = datetime.now(timezone.utc).isoformat()
    save_mining_registry(registry_path, registry)
    return entry


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
