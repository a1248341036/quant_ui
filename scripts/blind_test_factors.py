"""盲测因子评估：在锁定的盲测段（默认 2025 年起）重测因子库全部因子。

时间隔离契约
------------
- 盲测段（test_start ~ test_end）**从未参与挖掘循环与入库门槛**：
  挖掘会话面板 coverage = train_start ~ val_end（2020~2024），
  LLM 迭代、stage_one/stage_two、engine_gate 全部看不到盲测段数据。
- 本脚本是对"已定稿因子"的一次性离线重测，结果不回流任何门槛；
  频繁重测会把盲测段重新"烧掉"（多重检验），请克制使用频率。

产出
----
- 控制台逐因子对比表（入库时 train |IC| vs 盲测 |IC|，保留率）
- artifacts/alphaagent/blind_test/<run_ts>/report.json
- --clean 模式自动清理失效因子（retention < 阈值）并同步研究记忆

用法
----
  python scripts/blind_test_factors.py                       # 全部库，仅出报告
  python scripts/blind_test_factors.py --libs production_technical
  python scripts/blind_test_factors.py --test-start 2025-01-01
  python scripts/blind_test_factors.py --clean               # 清理失效因子 + 记忆库
  python scripts/blind_test_factors.py --clean --retention-threshold 0.5 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from alphaagent.data.adapters.cnequity import load_panel_from_cne
from alphaagent.dsl import eval_factor
from alphaagent.factor.ingest import compute_ingest_metrics
from alphaagent.factor.types import IngestPolicy
from alphaagent.factor.zoo import DEFAULT_FACTORLIB_ROOT, FactorZoo

# DEFAULT_FACTORLIB_ROOT 指向 production_technical，取其父目录（factorzoo/）
ZOO_ROOT = Path(DEFAULT_FACTORLIB_ROOT).parent if DEFAULT_FACTORLIB_ROOT else ROOT / "artifacts/alphaagent/factorzoo"
REGISTRY_NAME = "mining_candidate_registry.json"


def _iter_libraries(root: Path, libs: list[str]) -> list[tuple[str, str, Path]]:
    """返回 (library, kind, path)。kind: production 读 zoo catalog；candidate 读 registry。"""
    out: list[tuple[str, str, Path]] = []
    for lib in libs:
        p = root / lib
        if not p.is_dir():
            print(f"[skip] {lib}: 目录不存在")
            continue
        kind = "production" if lib.startswith("production") else "candidate"
        out.append((lib, kind, p))
    return out


def _collect_factors(kind: str, path: Path) -> list[dict]:
    """收集 (factor_id, name, expr, stored_train_ic, stored_train_icir)。"""
    items: list[dict] = []
    if kind == "production":
        zoo = FactorZoo.open(path)
        for fid in zoo.catalog.list_factor_ids():
            meta = zoo.catalog.get(fid)
            if meta is None:
                continue
            metrics = meta.extra.get("metrics", {}) if isinstance(meta.extra, dict) else {}
            items.append({
                "factor_id": fid,
                "name": meta.name,
                "expr": meta.expr,
                "ref_ic": metrics.get("ic"),
                "ref_icir": metrics.get("icir"),
            })
    else:
        # candidate 库 = registry ∪ catalog（去重，registry 优先）
        seen: set[str] = set()
        reg = path / REGISTRY_NAME
        if reg.is_file():
            data = json.loads(reg.read_text(encoding="utf-8"))
            for fid, entry in data.items():
                m = entry.get("metrics") or {}
                items.append({
                    "factor_id": fid,
                    "name": entry.get("name", fid),
                    "expr": entry.get("expr", ""),
                    "ref_ic": m.get("ic"),
                    "ref_icir": m.get("icir"),
                    "ref_val_ic": m.get("val_ic"),
                })
                seen.add(fid)
        try:
            zoo = FactorZoo.open(path)
            for fid in zoo.catalog.list_factor_ids():
                if fid in seen:
                    continue
                meta = zoo.catalog.get(fid)
                if meta is None:
                    continue
                metrics = meta.extra.get("metrics", {}) if isinstance(meta.extra, dict) else {}
                items.append({
                    "factor_id": fid,
                    "name": meta.name,
                    "expr": meta.expr,
                    "ref_ic": metrics.get("ic"),
                    "ref_icir": metrics.get("icir"),
                })
                seen.add(fid)
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] {path.name} catalog 读取失败（无 meta/factors.parquet）: {exc}")
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="盲测段因子重测")
    parser.add_argument("--test-start", default="2025-01-01")
    parser.add_argument("--test-end", default=None,
                        help="盲测段右端（缺省 = 数据源最新交易日，动态解析）")
    parser.add_argument("--panel-start", default=None,
                        help="面板预热起点（默认 test_start 往前 400 个自然日，供滚动算子预热）")
    parser.add_argument("--label-col", default="label_1d_open_to_open")
    parser.add_argument("--libs", nargs="*", default=[
        "production_main", "candidate_main",
    ])
    parser.add_argument("--clean", action="store_true",
                        help="清理失效因子（retention < 阈值）并同步研究记忆")
    parser.add_argument("--retention-threshold", type=float, default=0.5,
                        help="保留比阈值，低于此值判定失效（默认 0.5）")
    parser.add_argument("--dry-run", action="store_true",
                        help="--clean 模式下只输出将删除的因子，不执行")
    args = parser.parse_args()

    if not args.test_end:
        from alphaagent.factor.window_config import resolve_test_end

        args.test_end = resolve_test_end()

    panel_start = args.panel_start or (
        pd.Timestamp(args.test_start) - pd.Timedelta(days=400)
    ).date().isoformat()

    print(f"盲测段: {args.test_start} ~ {args.test_end}（面板预热自 {panel_start}）")
    print("加载面板（含基本面/事件面列）…")
    t0 = time.perf_counter()
    panel = load_panel_from_cne(
        start=panel_start, end=args.test_end, universe_mask=False,
        include_fundamentals=True,
    )
    panel = panel.sort_index()
    dt = panel.index.get_level_values("datetime")
    in_test = (dt >= pd.Timestamp(args.test_start)) & (dt <= pd.Timestamp(args.test_end))
    test_days = panel.index[in_test].get_level_values(0).nunique()
    print(f"面板就绪: {len(panel):,} 行 × {panel.shape[1]} 列，"
          f"盲测段 {test_days} 个交易日（{time.perf_counter() - t0:.0f}s）")
    if test_days < 20:
        print("[warn] 盲测段交易日过少，结论统计效力有限")

    policy = IngestPolicy(
        train_start=args.test_start, val_end=args.test_end, label_col=args.label_col,
    )

    rows: list[dict] = []
    for lib, kind, path in _iter_libraries(ZOO_ROOT, args.libs):
        factors = _collect_factors(kind, path)
        print(f"\n== {lib}（{kind}，{len(factors)} 个因子）==")
        for i, f in enumerate(factors):
            t1 = time.perf_counter()
            try:
                raw = eval_factor(f["expr"], panel)
                if not isinstance(raw, pd.Series):
                    raise TypeError(f"输出非 Series: {type(raw).__name__}")
                values = np.asarray(raw, dtype=np.float32)
                m = compute_ingest_metrics(values, panel, policy)
                test_ic, test_icir = float(m["ic"]), float(m["icir"])
                ref_ic = float(f["ref_ic"]) if f.get("ref_ic") is not None and np.isfinite(float(f["ref_ic"])) else np.nan
                retention = abs(test_ic) / abs(ref_ic) if np.isfinite(ref_ic) and abs(ref_ic) > 1e-12 else np.nan
                row = {
                    "library": lib, "factor_id": f["factor_id"], "name": f["name"],
                    "ref_ic": f.get("ref_ic"), "ref_icir": f.get("ref_icir"),
                    "test_ic": round(test_ic, 6) if np.isfinite(test_ic) else None,
                    "test_icir": round(test_icir, 6) if np.isfinite(test_icir) else None,
                    "test_rank_ic": m.get("rank_ic"),
                    "test_coverage": round(float(m.get("coverage", np.nan)), 4),
                    "test_cs_autocorr": m.get("cs_pearson_autocorr"),
                    "test_long_excess": m.get("long_group_annual_excess_return"),
                    "winsorized_abs_ic_decay": m.get("winsorized_abs_ic_decay"),
                    "ic_retention_vs_ref": round(float(retention), 4) if np.isfinite(retention) else None,
                    "eval_ms": round((time.perf_counter() - t1) * 1000),
                }
                rows.append(row)
                print(f"  [{i + 1}/{len(factors)}] {f['name']:34} "
                      f"train|IC|={abs(ref_ic):.4f}  test|IC|={abs(test_ic):.4f}  "
                      f"保留={row['ic_retention_vs_ref']}  多头超额={row['test_long_excess']}")
            except Exception as exc:  # noqa: BLE001
                rows.append({
                    "library": lib, "factor_id": f["factor_id"], "name": f["name"],
                    "error": f"{type(exc).__name__}: {exc}",
                })
                print(f"  [{i + 1}/{len(factors)}] {f['name']:34} 评估失败: {exc}")

    out_dir = ROOT / "artifacts" / "alphaagent" / "blind_test" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "generated_at": datetime.now().isoformat(),
        "test_window": {"start": args.test_start, "end": args.test_end, "trading_days": int(test_days)},
        "label_col": args.label_col,
        "note": "盲测段未参与挖掘与任何入库门槛；结果仅作定稿因子的样本外体检。",
        "factors": rows,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )
    ok = [r for r in rows if "error" not in r]
    decayed = [r for r in ok if (r.get("ic_retention_vs_ref") or 0) < args.retention_threshold]
    print(f"\n完成: {len(ok)}/{len(rows)} 评估成功；|IC| 保留比 < {args.retention_threshold} 的因子 {len(decayed)} 个")
    print(f"报告: {out_dir / 'report.json'}")

    # ── --clean 模式：清理失效因子 + 同步研究记忆 ──
    if args.clean and decayed:
        total_deleted = _clean_decayed_factors(decayed, args.dry_run)
        if total_deleted:
            action = "将删除（dry-run）" if args.dry_run else "已删除"
            print(f"\n{action} {total_deleted} 个失效因子，研究记忆已同步清理")
        elif not args.dry_run:
            print("\n无失效因子需清理")
    elif args.clean and not decayed:
        print("\n--clean: 无失效因子，无需清理")

    return 0


def _clean_decayed_factors(decayed: list[dict], dry_run: bool) -> int:
    """清理盲测失效因子：从 registry/zoo 删除 + 同步研究记忆。

    删除路径与 alphaagent_service.delete_factor / rescreen_candidates 一致，
    确保记忆库（research_memory.db）中的残留条目被同步清除。
    """
    from alphaagent.factor.mining.research_memory import ResearchMemoryStore

    mem_path = ROOT / "artifacts" / "alphaagent" / "research_memory.db"
    mem_store = ResearchMemoryStore(mem_path) if mem_path.exists() else None

    # 按 library 分组（同库批量处理 registry）
    by_lib: dict[str, list[dict]] = {}
    for f in decayed:
        lib = f.get("library", "")
        by_lib.setdefault(lib, []).append(f)

    total = 0
    for lib, factors in by_lib.items():
        lib_dir = ROOT / "artifacts" / "alphaagent" / "factorzoo" / lib
        is_production = lib.startswith("production")
        kind = "production" if is_production else "candidate"
        factor_ids = [f["factor_id"] for f in factors]
        factor_names = [str(f.get("name") or f["factor_id"]) for f in factors]

        print(f"\n[{lib}] {'将删除' if dry_run else '删除'} {len(factor_ids)} 个失效因子:")

        if dry_run:
            for fid in factor_ids:
                print(f"  - {fid}")
            total += len(factor_ids)
            continue

        if is_production:
            # production 库：先从 zoo 删除，再从 delivered_registry 删除
            try:
                zoo = FactorZoo.open(lib_dir, verify_hash=False)
                for fid in factor_ids:
                    try:
                        zoo.delete_factor(fid)
                        print(f"  - {fid} (zoo)")
                    except KeyError:
                        print(f"  - {fid} (zoo: not found, skip)")
            except FileNotFoundError:
                print(f"  [warn] {lib} zoo 未初始化，跳过 zoo 删除")

            reg_path = lib_dir / "mining_delivered_registry.json"
            if reg_path.is_file():
                registry = json.loads(reg_path.read_text(encoding="utf-8"))
                for fid in factor_ids:
                    registry.pop(fid, None)
                reg_path.write_text(
                    json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )
        else:
            # candidate 库：从 candidate_registry 删除 + 删 DSL
            reg_path = lib_dir / REGISTRY_NAME
            registry = json.loads(reg_path.read_text(encoding="utf-8")) if reg_path.is_file() else {}
            for fid in factor_ids:
                entry = registry.pop(fid, None)
                rel = str((entry or {}).get("expression_file") or "")
                dsl_path = ROOT / rel if rel else lib_dir / "expressions" / f"{fid}.dsl"
                if dsl_path.exists():
                    dsl_path.unlink()
                print(f"  - {fid} (registry)")
            if reg_path.is_file():
                reg_path.write_text(
                    json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                )

        # 同步清理研究记忆
        if mem_store is not None:
            purged = mem_store.purge_factor(
                factor_names=factor_names,
                expressions=[],
            )
            print(f"  研究记忆清理: {purged} 条")

        total += len(factor_ids)

    return total


if __name__ == "__main__":
    raise SystemExit(main())
