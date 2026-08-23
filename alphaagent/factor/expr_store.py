"""因子 DSL 源文件读写（与 factorzoo 同目录：{lib}/expressions/）。"""

from __future__ import annotations

from pathlib import Path

from alphaagent.core.paths import FACTOR_EXPR_DIR
from alphaagent.factor.zoo import FactorZoo


def factor_expr_path(factor_id: str, *, expr_dir: Path | None = None) -> Path:
    root = Path(expr_dir or FACTOR_EXPR_DIR)
    return root / f"{factor_id}.dsl"


def write_factor_expr(factor_id: str, expr: str, *, expr_dir: Path | None = None) -> Path:
    path = factor_expr_path(factor_id, expr_dir=expr_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(expr.strip() + "\n", encoding="utf-8")
    return path


def list_expr_dir_entries(expr_dir: Path | str) -> list[tuple[str, str, str]]:
    """扫描目录下 ``*.dsl``，返回 (factor_id, name, expr) 列表。"""
    root = Path(expr_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"表达式目录不存在: {root}")
    entries: list[tuple[str, str, str]] = []
    for path in sorted(root.glob("*.dsl")):
        factor_id = path.stem
        expr = path.read_text(encoding="utf-8").strip()
        if not expr:
            continue
        entries.append((factor_id, factor_id, expr))
    if not entries:
        raise ValueError(f"目录下无有效 .dsl: {root}")
    return entries


def export_zoo_expressions(
    zoo: FactorZoo,
    *,
    expr_dir: Path | None = None,
    overwrite: bool = True,
) -> list[tuple[str, Path]]:
    """从 factorzoo catalog 导出全部因子 DSL 到 {lib}/expressions/。"""
    out_dir = Path(expr_dir or zoo.paths.expressions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[tuple[str, Path]] = []
    for factor_id in zoo.catalog.list_factor_ids():
        meta = zoo.catalog.get(factor_id)
        if meta is None:
            continue
        path = factor_expr_path(factor_id, expr_dir=out_dir)
        if path.is_file() and not overwrite:
            written.append((factor_id, path))
            continue
        path = write_factor_expr(factor_id, meta.expr, expr_dir=out_dir)
        written.append((factor_id, path))
    return written
