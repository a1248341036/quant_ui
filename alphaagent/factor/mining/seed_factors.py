"""从 .dsl 文件加载初始种子因子，拼入 user 消息。"""

from __future__ import annotations

from pathlib import Path


def resolve_seed_factor_paths(paths: list[Path], repo_root: Path) -> list[Path]:
    """解析并校验种子因子路径（相对路径相对 repo_root）。"""
    resolved: list[Path] = []
    for raw in paths:
        p = raw if raw.is_absolute() else repo_root / raw
        if not p.is_file():
            raise FileNotFoundError(f"seed factor not found: {p}")
        resolved.append(p.resolve())
    return resolved


def build_user_message_with_seed_factors(
    user_message: str,
    seed_paths: list[Path],
    *,
    repo_root: Path,
) -> str:
    """将种子因子 DSL 注入 user 消息，供 LLM 在既有表达式基础上迭代优化。"""
    if not seed_paths:
        return user_message

    resolved = resolve_seed_factor_paths(seed_paths, repo_root)
    blocks: list[str] = [
        "以下 `.dsl` 文件为**初始种子因子**，请作为本轮挖掘起点：",
        "1. **首轮**先用 `eval_on_train_set` 评估各种子因子的 train 基准表现（`multi_line_expr` 与文件内容一致）；",
        "2. 再在其基础上迭代优化（调整结构、窗口、中性化、温莎化等），避免仅做同质微调；",
        "3. 达标后 `submit_factor` 时使用**新的** `factor_name`（勿覆盖种子因子 id）。",
        "",
        "## 初始种子因子",
    ]

    for path in resolved:
        expr = path.read_text(encoding="utf-8").strip()
        factor_name = path.stem
        try:
            rel = path.relative_to(repo_root.resolve())
            source = str(rel).replace("\\", "/")
        except ValueError:
            source = str(path)
        blocks.extend(
            [
                "",
                f"### {factor_name}（`{source}`）",
                "",
                "```text",
                expr,
                "```",
            ]
        )

    blocks.extend(["", "---", "", user_message.strip()])
    return "\n".join(blocks)
