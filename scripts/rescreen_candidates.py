#!/usr/bin/env python3
"""候选池重筛：按组合日换手硬门清退不可交付候选。

背景（2026-08-29）：候选池 30 个技术候选 26 个日单边换手 >50%，全部止步
stage_two/engine_gate——换手裁决太靠后，浪费算力。stage_one 已加
``max_avg_daily_side_turnover`` 硬门；本脚本把该门**回溯**应用于存量候选：

- 换手数据直接读 registry 里存好的 ``metrics.quantile_portfolio.avg_daily_side_turnover``
  （submit 时已算好，无需重算）；
- 超限者从 registry 删除 + 删除 DSL 文件 + 清理研究记忆（与 dedup_candidate_factors.py
  同一套清理机制）；promoted 候选一律不动。

用法：
  .venv\\Scripts\\python.exe scripts\\rescreen_candidates.py --dry-run
  .venv\\Scripts\\python.exe scripts\\rescreen_candidates.py --mode technical --mode fundamental --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DEFAULT_TURNOVER = 0.5


def _load_registry(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    # 兼容 {fid: entry} 与 {"candidates": {...}} 两种存法
    if isinstance(data, dict) and isinstance(data.get("candidates"), dict):
        return data["candidates"]
    return data if isinstance(data, dict) else {}


def _entry_turnover(entry: dict) -> float | None:
    m = entry.get("metrics") or {}
    qp = m.get("quantile_portfolio") or {}
    t = qp.get("avg_daily_side_turnover") if isinstance(qp, dict) else None
    try:
        return float(t) if t is not None else None
    except (TypeError, ValueError):
        return None


def rescreen_pool(lib: str, threshold: float, apply: bool) -> tuple[int, int]:
    lib_dir = ROOT / "artifacts" / "alphaagent" / "factorzoo" / lib
    registry_path = lib_dir / "mining_candidate_registry.json"
    registry = _load_registry(registry_path)
    if not registry:
        print(f"[{lib}] registry 为空或不存在，跳过")
        return 0, 0

    to_delete: dict[str, str] = {}
    for fid, entry in registry.items():
        if (entry.get("promotion_status") or "") == "promoted":
            continue  # 已晋级正库的不动
        t = _entry_turnover(entry)
        if t is None:
            continue  # 无换手数据（旧记录）不误删
        if t > threshold:
            to_delete[fid] = f"turnover={t:.1%}"

    print(f"\n[{lib}] 候选 {len(registry)} 个，超换手门（>{threshold:.0%}）：{len(to_delete)} 个")
    for fid, why in sorted(to_delete.items(), key=lambda kv: kv[1]):
        print(f"  - {fid:44s} {why}")
    kept = len(registry) - len(to_delete)
    print(f"  保留 {kept} 个")

    if not apply or not to_delete:
        return len(to_delete), kept

    purged_names = [
        str((registry.get(fid) or {}).get("name") or fid) for fid in to_delete
    ]
    for fid in to_delete:
        entry = registry.pop(fid, None)
        rel = str((entry or {}).get("expression_file") or "")
        dsl_path = ROOT / rel if rel else lib_dir / "expressions" / f"{fid}.dsl"
        if dsl_path.exists():
            dsl_path.unlink()

    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"[{lib}] registry 已更新: {registry_path}")

    # 清理研究记忆残留（与 dedup_candidate_factors.py 同机制）
    try:
        from alphaagent.factor.mining.research_memory import ResearchMemoryStore

        mem_path = ROOT / "artifacts" / "alphaagent" / "research_memory.db"
        if mem_path.exists():
            purged = ResearchMemoryStore(mem_path).purge_factor(
                factor_names=purged_names,
                expressions=[],
            )
            print(f"[{lib}] 研究记忆清理: {purged} 条", flush=True)
    except Exception as exc:  # noqa: BLE001
        print(f"[{lib}] 研究记忆清理跳过: {exc}")

    return len(to_delete), kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mode", action="append", default=None,
                    help="research mode（可多次）；缺省 technical+fundamental")
    ap.add_argument("--threshold", type=float, default=DEFAULT_TURNOVER,
                    help="日单边换手阈值（默认 0.5，与 stage_one 硬门一致）")
    ap.add_argument("--apply", action="store_true", help="执行删除（缺省 dry-run）")
    args = ap.parse_args()
    modes = args.mode or ["technical", "fundamental"]

    total_del = total_kept = 0
    for mode in modes:
        deleted, kept = rescreen_pool(f"candidate_{mode}", args.threshold, args.apply)
        total_del += deleted
        total_kept += kept

    print(f"\n{'已删除' if args.apply else '将删除（dry-run）'}: {total_del} 个 | 保留: {total_kept} 个")
    if not args.apply:
        print("确认无误后加 --apply 执行。")


if __name__ == "__main__":
    main()
