# -*- coding: utf-8 -*-
"""冷门赛道客观判定与拥挤族骨架（spec: alphaagent_mining_anti_convergence_spec.md M1-M3）。

纯逻辑层、零 IO 副作用，便于单测：
- 客观 D 轨判定（新族 ∧ 信号根算子命中 _COLD_OPS）——**不信任 LLM 自标 track/parent_origin**；
- 拥挤族判定（细族/融合族 → 粗族映射落在历史饱和粗族）；
- 拥挤族骨架清单（factorzoo production/candidate registry 代表表达式，快照语义）。

拥挤族粗族白名单来自战役实证（R0-R8）：换手×反转、低波、流动性、三低
是生产赢家占用族。用户语境 to→volume_liquidity、reversal→momentum_reversal、
lowvol→volatility、liquidity→volume_liquidity、tri→volatility|volume_liquidity。
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from .constants import COLD_TRACK_VERDICT_DEAD

# 冷门赛道死亡 verdict（spec §3.3-6）：独立于既有 weak/near_miss，
# 语义 = 「客观新族但残差化后无独立信息」，带 fail_detail 三标签。
COLD_TRACK_VERDICT = COLD_TRACK_VERDICT_DEAD  # 单一真源（.constants）

# fail_detail 三标签（spec §9 R3）：M4 回填后按 data_gap 重放、underpowered 重测、
# no_alpha 不再碰。underpowered 不入永久负证据（spec §7.3 样本外纪律）。
FAIL_DETAIL_NO_ALPHA = "no_alpha"
FAIL_DETAIL_DATA_GAP = "data_gap"
FAIL_DETAIL_UNDERPOWERED = "underpowered"
FAIL_DETAILS = (FAIL_DETAIL_NO_ALPHA, FAIL_DETAIL_DATA_GAP, FAIL_DETAIL_UNDERPOWERED)
# 可重试标签：研究记忆检索注入时不作为死路负证据展示（spec §7.3）。
COLD_DEAD_RETRYABLE = frozenset({FAIL_DETAIL_DATA_GAP, FAIL_DETAIL_UNDERPOWERED})

# 拥挤族粗族白名单（classify_family_coarse 命名空间）。
CROWD_COARSE_FAMILIES = frozenset({"momentum_reversal", "volatility", "volume_liquidity"})

# 冷门赛道通过者 registry provenance（防 M1 成功者自噬骨架，spec §5.3）。
COLD_TRACK_APPROVAL_PROVENANCE = "cold_track-approval"


def crowd_coarse_of(
    family: str, *, facets: set[str] | None = None, expr: str = ""
) -> str | None:
    """细族/融合族映射到的拥挤粗族（非拥挤族返回 None）。

    与 memory/expressions.classify_family_coarse 同源映射；融合族按最慢信息面
    主导判定（fundamental/chip/interaction 均非拥挤）。供 M3 父本屏蔽统计
    被屏蔽粗族名与 M2 套壳 advisory 判定共用。
    """
    try:
        from alphaagent.dsl.core.ast import classify_family_coarse

        coarse = classify_family_coarse(str(family or ""), set(facets or []), expr)
    except Exception:  # noqa: BLE001 — 判定失败安全回落
        coarse = str(family or "")
    if coarse in CROWD_COARSE_FAMILIES:
        return coarse
    # 兜底：调用方偶传粗族名本身（如记忆条目 family 恰好为粗族键）时直接命中，
    # 避免 M3 屏蔽/套壳判定被「粗族名不可映射」漏掉。
    key = str(family or "").strip()
    if key in CROWD_COARSE_FAMILIES:
        return key
    return None


def is_crowd_family(family: str, *, facets: set[str] | None = None, expr: str = "") -> bool:
    """细族/融合族是否落在历史饱和（拥挤）粗族（crowd_coarse_of 的 bool 化）。"""
    return crowd_coarse_of(family, facets=facets, expr=expr) is not None


# __CT__


# ── 拥挤族骨架清单（spec §5.3）──────────────────────────────────────


def _expr_from_entry(entry: dict[str, Any], repo_root: Path | None) -> str:
    """registry 条目取表达式：candidate 内联 expr；production 走 expression_file。"""
    expr = str(entry.get("expr") or "").strip()
    if expr:
        return expr
    rel = str(entry.get("expression_file") or "").strip()
    if rel and repo_root is not None:
        try:
            p = Path(rel)
            if not p.is_absolute():
                p = Path(repo_root) / p
            return p.read_text(encoding="utf-8").strip()
        except (OSError, ValueError):  # noqa: BLE001
            return ""
    return ""


def _entry_ic(entry: dict[str, Any]) -> float:
    """registry 条目 train IC（ingest_metrics/metrics 下的 ic）。"""
    metrics = entry.get("ingest_metrics")
    if not isinstance(metrics, dict):
        metrics = entry.get("metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    try:
        return abs(float(metrics.get("ic")))
    except (TypeError, ValueError):
        return 0.0


def fingerprint(expr: str) -> str:
    """结构指纹（subtree_hash）：骨架去重与快照签名用。"""
    try:
        from alphaagent.dsl.core.ast import structure_fingerprint

        return str(structure_fingerprint(expr))
    except Exception:  # noqa: BLE001
        canonical = "\n".join(l.strip() for l in str(expr).splitlines() if l.strip())
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:20]


def build_crowd_skeletons(
    production_registry: Path | str | None,
    candidate_registry: Path | str | None,
    *,
    repo_root: Path | str | None = None,
    max_ctrl_families: int = 8,
    exclude_provenance: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """生成拥挤族骨架清单（run 启动时快照语义）。

    从 production + candidate registry 读全部条目，筛选拥挤粗族，每粗族保留
    |train IC| 最高者为代表（fingerprint 去重）；剔除 provenance =
    cold_track-approval 因子（防 M1 自己的成功者把骨架越滚越挤，spec §5.3）。
    返回 {"skeletons": {coarse_family: {...}}, "signature": ..., "crowd_families": [...]}。
    任一 registry 缺失/空 → 骨架为空（冷门赛道退化为绝对 IC，仍安全）；
    不抛异常（骨架是增益信息，绝不阻断评估）。
    """
    exclude = set(exclude_provenance or (COLD_TRACK_APPROVAL_PROVENANCE,))
    root = Path(repo_root).resolve() if repo_root else None
    entries: list[tuple[str, dict[str, Any]]] = []
    for reg in (production_registry, candidate_registry):
        if not reg:
            continue
        try:
            from alphaagent.factor.mining.infra.registry_io import load_mining_registry

            reg_dict = load_mining_registry(Path(reg))
        except Exception:  # noqa: BLE001
            continue
        for _fid, entry in reg_dict.items():
            if not isinstance(entry, dict):
                continue
            if str(entry.get("provenance") or "") in exclude:
                continue
            expr = _expr_from_entry(entry, root)
            if not expr:
                continue
            entries.append((expr, entry))

    by_family: dict[str, dict[str, Any]] = {}
    seen_fp: set[str] = set()
    for expr, entry in entries:
        family, facets = _classify_family_ex(str(entry.get("name") or ""), expr)
        if not is_crowd_family(family, facets=facets, expr=expr):
            continue
        try:
            from alphaagent.dsl.core.ast import classify_family_coarse

            coarse = classify_family_coarse(family, facets)
        except Exception:  # noqa: BLE001
            coarse = family
        fp = fingerprint(expr)
        if fp in seen_fp:
            continue
        ic = _entry_ic(entry)
        cur = by_family.get(coarse)
        if cur is None or ic > float(cur.get("train_ic", 0)):
            by_family[coarse] = {
                "expr": expr,
                "name": str(entry.get("name") or ""),
                "train_ic": ic,
                "family": family,
            }
            seen_fp.add(fp)

    # max_ctrl_families 上限（防共线膨胀，spec §3.4）
    limited = sorted(
        by_family.items(), key=lambda kv: float(kv[1].get("train_ic", 0)), reverse=True
    )
    skeletons = {fam: meta for fam, meta in limited[: max(1, int(max_ctrl_families))]}
    sig = hashlib.sha256(
        "\n".join(sorted(s["expr"] for s in skeletons.values())).encode("utf-8")
    ).hexdigest()[:16]
    return {
        "skeletons": skeletons,
        "signature": sig,
        "crowd_families": sorted(skeletons.keys()),
    }


def build_snapshot_for_run(
    spec: dict[str, Any] | None,
    *,
    production_registry: Path | str | None,
    candidate_registry: Path | str | None,
    repo_root: Path | str | None = None,
    max_ctrl_families: int | None = None,
) -> dict[str, Any] | None:
    """run 启动时生成一次性拥挤族骨架快照（spec §5.3 快照语义）。

    cold_track.enabled=false（默认）或 spec 缺省时返回 None（黄金基线零注入）。
    骨架缺失/失败时返回空骨架结构（冷门赛道退化为绝对残差化但仍是独立信息面），
    绝不阻断 run 启动。
    """
    if not isinstance(spec, dict):
        return None
    cold = spec.get("cold_track")
    if not isinstance(cold, dict) or not cold.get("enabled", False):
        return None
    fam_cap = (
        int(max_ctrl_families)
        if max_ctrl_families is not None
        else int(cold.get("max_ctrl_families", 8) or 8)
    )
    return build_crowd_skeletons(
        production_registry,
        candidate_registry,
        repo_root=repo_root,
        max_ctrl_families=fam_cap,
    )


def signal_root_op(expr: str) -> str:
    """表达式信号根算子（剥掉归一化/平滑外壳后的根调用）。

    与 dsl.core.ast.classify_family_ast 剥离逻辑同源（NORMALIZE_OPS +
    SMOOTHING_OPS 逐层剥壳到信号根）；返回算子大写名；信号根为裸列引用或
    解析失败时返回空串。用于条件 (2)：冷门算子必须是**信号根**才算「真用」。
    """
    if not expr or not str(expr).strip():
        return ""
    try:
        from alphaagent.dsl.core import ast as _ast
    except Exception:  # noqa: BLE001
        return ""
    try:
        node = _ast.parse_ast(str(expr))
    except Exception:  # noqa: BLE001
        return ""
    if node is None:
        return ""
    try:
        var_table = _ast._parse_var_table(str(expr)) if hasattr(_ast, "_parse_var_table") else {}

        def _resolve_var(n):
            seen: set[str] = set()
            while n is not None and n.type == "var" and n.value in var_table:
                if n.value in seen:
                    return None
                seen.add(n.value)
                n = _ast.parse_ast(var_table[n.value])
            return n

        normalize_ops = getattr(_ast, "NORMALIZE_OPS", frozenset())
        smooth_ops = getattr(_ast, "SMOOTHING_OPS", frozenset())
        while node is not None:
            node = _resolve_var(node)
            if node is None:
                break
            if node.is_call() and (node.op.upper() in normalize_ops or node.op.upper() in smooth_ops):
                node = node.args[0] if node.args else None
                continue
            break
        if node is not None and node.type == "call":
            return str(node.op).upper()
    except Exception:  # noqa: BLE001 — 剥离失败回落空
        return ""
    return ""


def _eval_top_level_cold(expr: str) -> bool:
    """条件 (2) 算子级检查：信号根命中冷门清单。

    _COLD_OPS 从 memory/retrieval 取（单一真源，spec §9 Q2：与
    _operator_diversity_block 复用同一清单）；延迟 import 避免拖全局依赖。
    """
    root = signal_root_op(expr)
    if not root:
        return False
    try:
        from alphaagent.factor.mining.memory.retrieval import _COLD_OPS
    except Exception:  # noqa: BLE001 — 按未命中处理（不进冷门赛道，保守）
        return False
    return root in _COLD_OPS


def classify_track(expr: str, factor_name: str = "") -> tuple[str, bool, bool]:
    """客观赛道判定：返回 (track, is_new_family, has_cold_root)。

    track ∈ {"cold", "crowd"}：
    - "cold"  iff 新族 ∧ 信号根 ∈ _COLD_OPS（spec §3.2）;
    - "crowd" 任一条件不满足 → 既有全链路（行为逐字节不变）。
    仅 cold_track.enabled=true 时 "cold" 才有意义；关闭时一律按 "crowd" 处理。
    """
    family, facets = _classify_family_ex(factor_name, str(expr))
    new_family = not is_crowd_family(family, facets=facets, expr=str(expr))
    has_cold_root = _eval_top_level_cold(str(expr))
    if new_family and has_cold_root:
        return "cold", new_family, True
    return "crowd", new_family, has_cold_root


def _classify_family_ex(name: str, expr: str) -> tuple[str, set[str]]:
    try:
        from alphaagent.factor.mining.memory.expressions import classify_family_ex

        return classify_family_ex(name, expr)
    except Exception:  # noqa: BLE001 — 解析失败回落 "other"
        return "other", set()


# __CT__
