"""因子交付物删除：factorzoo + registry + dsl 同步清理。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from alphaagent.factor.zoo import FactorZoo
from alphaagent.factor.mining.registry_io import load_mining_registry, save_mining_registry


def _resolve_expr_path(
    factor_id: str,
    *,
    registry_entry: dict[str, Any] | None,
    expr_dir: Path,
    repo_root: Path,
) -> Path | None:
    if registry_entry:
        rel = registry_entry.get("expression_file")
        if rel:
            candidate = (repo_root / str(rel)).resolve()
            if candidate.is_file():
                return candidate
    default = (expr_dir / f"{factor_id}.dsl").resolve()
    return default if default.is_file() else None


def delete_factor_delivery(
    factor_id: str,
    *,
    zoo: FactorZoo | None = None,
    registry_path: Path | None = None,
    expr_dir: Path | None = None,
    repo_root: Path | None = None,
    skip_zoo: bool = False,
    skip_registry: bool = False,
    skip_dsl: bool = False,
) -> dict[str, Any]:
    """删除因子及其交付物；至少一项存在方可成功。"""
    fid = str(factor_id).strip()
    if not fid:
        raise ValueError("factor_id 不能为空")

    reg_path = Path(registry_path).expanduser().resolve() if registry_path else None
    expr_root = Path(expr_dir).expanduser().resolve() if expr_dir else None
    root = Path(repo_root).expanduser().resolve() if repo_root else Path.cwd()

    registry = load_mining_registry(reg_path) if reg_path and not skip_registry else {}
    entry = registry.get(fid)

    in_zoo = zoo is not None and not skip_zoo and zoo.catalog.get(fid) is not None
    in_registry = bool(entry) and not skip_registry
    dsl_path = None
    if not skip_dsl and expr_root is not None:
        dsl_path = _resolve_expr_path(fid, registry_entry=entry, expr_dir=expr_root, repo_root=root)
    has_dsl = dsl_path is not None and dsl_path.is_file()

    if not in_zoo and not in_registry and not has_dsl:
        raise KeyError(f"因子不存在于 zoo/registry/dsl: {fid}")

    result: dict[str, Any] = {
        "factor_id": fid,
        "zoo_deleted": False,
        "registry_removed": False,
        "dsl_deleted": False,
        "dsl_path": None,
    }

    if in_zoo:
        assert zoo is not None
        zoo.delete_factor(fid)
        result["zoo_deleted"] = True

    if in_registry:
        assert reg_path is not None
        del registry[fid]
        save_mining_registry(reg_path, registry)
        result["registry_removed"] = True

    if has_dsl and dsl_path is not None:
        dsl_path.unlink()
        result["dsl_deleted"] = True
        result["dsl_path"] = str(dsl_path)

    return result


def delete_factors_delivery(
    factor_ids: list[str],
    *,
    zoo: FactorZoo | None = None,
    registry_path: Path | None = None,
    expr_dir: Path | None = None,
    repo_root: Path | None = None,
    skip_zoo: bool = False,
    skip_registry: bool = False,
    skip_dsl: bool = False,
    ignore_missing: bool = True,
) -> list[dict[str, Any]]:
    """批量删除；ignore_missing=True 时跳过不存在的 factor_id。"""
    out: list[dict[str, Any]] = []
    for fid in factor_ids:
        try:
            out.append(
                delete_factor_delivery(
                    fid,
                    zoo=zoo,
                    registry_path=registry_path,
                    expr_dir=expr_dir,
                    repo_root=repo_root,
                    skip_zoo=skip_zoo,
                    skip_registry=skip_registry,
                    skip_dsl=skip_dsl,
                )
            )
        except KeyError:
            if not ignore_missing:
                raise
    return out
