# -*- coding: utf-8 -*-
"""生成 combo_score 缓存（增量轨的控制变量）。

**为什么要有缓存**：增量判定需要一个"现有因子库组合"作为控制变量，它由候选池因子
在 **train 段** 等权合成。若每次评估都现算，要重复求值 40+ 个因子（约 2-3 分钟），
增量轨就没法批量用了。这里一次性算好落盘，评估侧直接读（1 列，秒级）。

**为什么不用 `data/stock/pred_demo.parquet`**：它只有 2025-01 起（盲测段），
拿它做因子筛选就是数据窥视。

用法：
    python scripts/build_combo_score.py                  # train 段 2020-01-01 ~ 2024-12-31
    python scripts/build_combo_score.py --force          # 覆盖已有缓存
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2020-01-01", help="train 段起点（避开盲测）")
    ap.add_argument("--end", default="2024-12-31", help="train 段终点（避开 2025+ 盲测段）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true", help="覆盖已有缓存")
    args = ap.parse_args()

    from alphaagent.data.adapters.cnequity import load_panel_from_cne
    from alphaagent.dsl.eval import eval_multi_line_factor, referenced_column_names
    from alphaagent.factor.mining.eval.combo_increment import (
        COMBO_CACHE, cs_zscore, load_candidate_exprs, save_combo,
    )

    out = Path(args.out) if args.out else COMBO_CACHE
    if out.is_file() and not args.force:
        print(f"[combo] 已存在 {out}（--force 覆盖）")
        return 0

    cands = load_candidate_exprs()
    if not cands:
        print("[combo] 候选池为空，无法合成")
        return 1

    # 跳过引用基本面列的因子（本机无 fundamentals 缓存，加载它会触发整份 8GB 重建）
    keep, skipped = [], 0
    for c in cands:
        try:
            refs = referenced_column_names(c["expr"])
        except Exception:  # noqa: BLE001
            skipped += 1
            continue
        if any(str(r).startswith(("funda_", "fsi_")) for r in refs):
            skipped += 1
            continue
        keep.append(c)
    print(f"[combo] 候选池 {len(cands)} 个，参与合成 {len(keep)} 个（跳过 {skipped} 个需基本面列）")

    need: set[str] = set()
    for c in keep:
        try:
            need |= set(referenced_column_names(c["expr"]))
        except Exception:  # noqa: BLE001
            pass

    t0 = time.perf_counter()
    panel = load_panel_from_cne(start=args.start, end=args.end, include_fundamentals=False,
                                keep_columns=tuple(sorted(need)))
    print(f"[combo] panel {panel.shape} load={time.perf_counter()-t0:.0f}s 窗口 {args.start}~{args.end}")

    zs = {}
    for i, c in enumerate(keep, 1):
        try:
            v = eval_multi_line_factor(c["expr"], panel)
            if isinstance(v, pd.DataFrame):
                v = v.iloc[:, 0] if v.shape[1] == 1 else v.mean(axis=1)
            s = pd.Series(v).astype("float64").reindex(panel.index)
            zs[c["name"]] = cs_zscore(s)
            print(f"  [{i:>3}/{len(keep)}] {c['name'][:36]:36s} notna={s.notna().mean()*100:5.1f}%")
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i:>3}/{len(keep)}] {c['name'][:36]:36s} ERR {type(exc).__name__}")

    if not zs:
        print("[combo] 没有因子求值成功")
        return 1
    Z = pd.DataFrame(zs)
    combo = Z.mean(axis=1, skipna=True).dropna()
    print(f"[combo] 合成 {Z.shape[1]} 个因子 → {len(combo)} 行，"
          f"覆盖 {len(combo)/len(panel)*100:.1f}%")
    p = save_combo(combo, out)
    print(f"[combo] 写入 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
