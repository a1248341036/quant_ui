"""穷举子集因子组合扫描：全部因子参与，2~K 元子集逐个评估。

与 scan_factor_combos.py（面分组加权）互补：这里不做分组、不做去冗余，
候选/正式库全部因子（表达式去重后）都参与，穷举所有 size ∈ [min_size, max_size]
的成员子集，每个子集等权合成（成员方向按 mining 窗口 IC 归一）后评估。

时间隔离契约与 scan_factor_combos.py 一致：
- 方向归一、组合枚举、排序全部只用 mining 窗口（<= mining_end）；
- OOS（mining_end 之后，通常落在盲测段）只对 mining ICIR 前 ``--top-oos``
  名的组合各读一次——穷举枚举本身已是多重检验，OOS 读取必须克制；
- 只有总赢家跑 engine_gate。
- 穷举分布即 null 分布：同一 size 内的 ICIR 分位数就是该组合的"运气分位"，
  不再另做随机子集 null。

组合分数 = 成员（rank+zscore+市值中性化后、方向归一）逐行 nanmean 等权。
组合标签期 = 成员 label_col 期数的中位数（无 label_col 回退 --label-days）。

性能：mining 窗口逐日 Spearman 用分块向量化实现（NaN 排序末位 + 有效前缀
秩），与 dataset.daily_spearman_ic 在无并列时逐位一致（tests/
test_scan_factor_subsets.py 对拍）；OOS 指标走 authoritative 的
daily_spearman_ic。

用法：
  python scripts/scan_factor_subsets.py                    # 全部 size 2..K
  python scripts/scan_factor_subsets.py --min-size 3 --top-oos 20 --no-gate
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alphaagent.data.adapters.cnequity import load_panel_from_cne  # noqa: E402
from alphaagent.factor.cache import FactorValueCache  # noqa: E402
from alphaagent.factor.stacking.dataset import (  # noqa: E402
    _to_utc_naive,
    collect_factor_entries,
    daily_spearman_ic,
    forward_return_label,
    materialize_entries,
    transform_factor_values,
)
from alphaagent.factor.stacking.groups import parse_label_days  # noqa: E402
from alphaagent.factor.window_config import resolve_test_end  # noqa: E402

MIN_COMBOS_PER_DAY = 30  # 单日有效样本下限，低于此的天跳过


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modes", nargs="+", default=["technical", "fundamental"],
                    choices=["technical", "fundamental"], help="纳入哪些因子库模式（统一大库后仅派生路径/档位）")
    ap.add_argument("--no-candidate", action="store_true", help="只用正式库因子")
    ap.add_argument("--mining-end", default="auto", help="时间隔离边界（YYYY-MM-DD 或 auto）")
    ap.add_argument("--end", default=None, help="数据截止日（默认数据源最新交易日）")
    ap.add_argument("--decay-months", type=int, default=12, help="mining 窗口长度（月）")
    ap.add_argument("--warmup-days", type=int, default=250, help="panel 起点提前量（因子窗口预热）")
    ap.add_argument("--min-size", type=int, default=2, help="最小组合成员数")
    ap.add_argument("--max-size", type=int, default=None, help="最大组合成员数（默认=因子数）")
    ap.add_argument("--max-combos", type=int, default=20000,
                    help="组合总数上限（穷举规模保护；超出则拒绝并提示缩小 size 区间）")
    ap.add_argument("--min-coverage", type=float, default=0.30, help="组合分数最小行覆盖率（mining 窗口）")
    ap.add_argument("--size-neutral/--no-size-neutral", dest="size_neutral", default=True)
    ap.add_argument("--label-days", type=int, default=5, help="成员 label_col 无法解析时的回退标签期")
    ap.add_argument("--top-oos", type=int, default=10, help="mining ICIR 前 N 名读一次 OOS 确认")
    ap.add_argument("--no-gate", action="store_true", help="跳过总赢家的 engine_gate 回测裁决")
    ap.add_argument("--chunk", type=int, default=32, help="逐日向量化评估的组合分块大小")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认 artifacts/alphaagent/stacking_scan/subsets_<时间戳>）")
    return ap.parse_args()


def resolve_mining_end(entries, args: argparse.Namespace) -> pd.Timestamp:
    """与 scan_factor_combos 同口径：auto = max(eval_end)，回退 max(created_at)。"""
    if args.mining_end != "auto":
        ts = pd.Timestamp(args.mining_end)
        return ts.tz_localize(None) if ts.tzinfo is not None else ts
    eval_ends = [c for c in (_to_utc_naive(e.eval_end) for e in entries) if c is not None]
    if eval_ends:
        print("mining_end 依据：registry eval_end（挖掘循环真实评估边界）")
        return max(eval_ends)
    created = [c for c in (_to_utc_naive(e.created_at) for e in entries) if c is not None]
    if not created:
        print("因子库无入库时间/评估边界，无法 auto 推断 mining-end，请显式传 --mining-end")
        sys.exit(1)
    print("警告：registry 无 eval_end，退回入库时间推断（偏保守）")
    return max(created)


def enumerate_subsets(k: int, min_size: int, max_size: int) -> list[tuple[int, ...]]:
    """全部 size ∈ [min_size, max_size] 的成员子集（索引元组）。"""
    max_size = min(max_size or k, k)
    if min_size < 2 or max_size < min_size:
        raise ValueError(f"非法 size 区间 [{min_size}, {max_size}]（min_size 须 >=2）")
    subsets: list[tuple[int, ...]] = []
    for size in range(min_size, max_size + 1):
        subsets.extend(itertools.combinations(range(k), size))
    return subsets


def combo_horizon(member_label_days: list[int | None], fallback: int) -> int:
    vals = [int(v) for v in member_label_days if v]
    return int(np.median(vals)) if vals else int(fallback)


def _ordinal_ranks(matrix: np.ndarray) -> np.ndarray:
    """逐列序数秩（NaN 排末位）：有效元素的秩 = 其在有效前缀中的序位（1 起）。

    np.argsort 升序把 NaN 排在末尾，因此有限元素恰好占据排序结果的前
    n_finite 位——对任意"先挖 NaN 再排名"的联合掩码语义都精确。
    """
    order = np.argsort(matrix, axis=0, kind="stable")
    ranks = np.empty(matrix.shape, dtype=np.float64)
    cols = np.arange(matrix.shape[1])[None, :]
    ranks[order, cols] = np.arange(1, matrix.shape[0] + 1, dtype=np.float64)[:, None]
    return np.where(np.isfinite(matrix), ranks, 0.0)


def chunk_daily_spearman_ic(
    scores: np.ndarray,          # [n_rows, n_combos] 组合分数（mining 窗口行，NaN=无效）
    label: np.ndarray,           # [n_rows] 前向收益标签（NaN=无效）
    day_bounds: list[tuple[int, int]],  # 逐日 (start, end) 行切片（行序按日期排序）
) -> np.ndarray:
    """分块组合的逐日 Spearman IC。

    返回 [n_days, n_combos] 逐日 IC（联合有效样本 < MIN_COMBOS_PER_DAY 的天为
    NaN）。语义与 dataset.daily_spearman_ic 精确一致（dropna 后双方在联合有效
    行上取秩再 Pearson；无并列时逐位一致，见 tests 对拍）。
    """
    n_days, n_combos = len(day_bounds), scores.shape[1]
    out = np.full((n_days, n_combos), np.nan, dtype=np.float64)
    for di, (s, e) in enumerate(day_bounds):
        sc = scores[s:e]
        lab = label[s:e]
        joint = np.isfinite(sc) & np.isfinite(lab)[:, None]
        n = joint.sum(axis=0).astype(np.float64)
        ok = n >= MIN_COMBOS_PER_DAY
        if not ok.any():
            continue
        r_x = _ordinal_ranks(np.where(joint, sc, np.nan))
        r_y = _ordinal_ranks(np.where(joint, lab[:, None], np.nan))
        n_safe = np.where(n > 0, n, 1.0)
        mx = r_x.sum(axis=0) / n_safe
        my = r_y.sum(axis=0) / n_safe
        cov = (r_x * r_y).sum(axis=0) / n_safe - mx * my
        vx = (r_x * r_x).sum(axis=0) / n_safe - mx * mx
        vy = (r_y * r_y).sum(axis=0) / n_safe - my * my
        denom = np.sqrt(np.maximum(vx, 0.0) * np.maximum(vy, 0.0))
        ic = np.where(denom > 1e-12, cov / np.where(denom > 1e-12, denom, 1.0), np.nan)
        out[di] = np.where(ok, ic, np.nan)
    return out


def summarize_ics(ics: np.ndarray) -> dict:
    vals = ics[np.isfinite(ics)]
    if len(vals) < 2 or vals.std(ddof=1) <= 1e-12:
        return {"ic_mean": None, "ic_ir": None, "n_days": int(len(vals))}
    return {"ic_mean": float(vals.mean()), "ic_ir": float(vals.mean() / vals.std(ddof=1)),
            "n_days": int(len(vals))}


def main() -> None:
    args = _parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "artifacts" / "alphaagent" / "stacking_scan" / f"subsets_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ① 因子枚举：candidate + production 全部参与（仅表达式去重），不做去冗余
    entries = collect_factor_entries(
        modes=tuple(args.modes), include_candidate=not args.no_candidate, include_production=True
    )
    print(f"因子枚举：{len(entries)} 个（表达式去重后，全部参与组合）")
    if len(entries) < 2:
        print("因子数不足（<2），无法组合。")
        sys.exit(1)
    k = len(entries)
    subsets = enumerate_subsets(k, args.min_size, args.max_size or k)
    if len(subsets) > args.max_combos:
        print(f"组合总数 {len(subsets)} 超过上限 {args.max_combos}；"
              f"请用 --max-size 收缩（如 --max-size {min(5, k)}）或调大 --max-combos")
        sys.exit(1)
    print(f"穷举组合：{len(subsets)} 个（size {args.min_size}~{min(args.max_size or k, k)}）")

    mining_end = resolve_mining_end(entries, args)
    data_latest = pd.Timestamp(resolve_test_end())
    end = pd.Timestamp(args.end) if args.end else data_latest
    if end > data_latest:
        print(f"[warn] --end={end.date()} 超过数据源最新交易日 {data_latest.date()}，收敛")
        end = data_latest
    if mining_end > end:
        print(f"[warn] mining_end={mining_end.date()} 超过数据右端 {end.date()}，收敛")
        mining_end = end
    panel_start = mining_end - pd.DateOffset(months=args.decay_months) - pd.DateOffset(days=args.warmup_days)
    print(f"mining_end={mining_end.date()}；panel 区间 [{panel_start.date()} ~ {end.date()}]")

    # ② panel + 物化 + transform（全部因子，无去冗余）
    print("加载 CNE panel …")
    panel = load_panel_from_cne(start=panel_start, end=end, include_fundamentals=True)
    print(f"panel: {panel.shape[0]} 行 × {panel.shape[1]} 列")
    dts = pd.Series(panel.index.get_level_values("datetime"))

    cache = FactorValueCache()
    materialized, dropped = materialize_entries(panel, entries, cache=cache,
                                                progress=lambda m: print(" ", m, flush=True))
    for d in dropped:
        print(f"  - 物化剔除 {d['name']} ({d['library']}): {d['reason']}")
    if len(materialized) < 2:
        print("有效因子不足（<2），无法组合。")
        sys.exit(1)
    if len(materialized) != k:
        k = len(materialized)
        subsets = enumerate_subsets(k, args.min_size, args.max_size or k)
        print(f"物化后有效因子 {k} 个，重新穷举：{len(subsets)} 个组合")
        if len(subsets) > args.max_combos:
            print(f"组合总数 {len(subsets)} 超过上限 {args.max_combos}；请用 --max-size 收缩")
            sys.exit(1)

    names = [e.name for e, _ in materialized]
    arrays = [transform_factor_values(raw, panel, size_neutral=args.size_neutral)
              for _, raw in materialized]

    # ③ 方向归一 + 每个(组合的)标签期 —— 只用 mining 窗口
    mining_mask = (dts <= mining_end).to_numpy()
    oos_mask = ~mining_mask
    m_idx = np.flatnonzero(mining_mask)
    m_dts = dts.iloc[m_idx]
    member_horizons = [parse_label_days(e.label_col) for e, _ in materialized]

    oriented: list[np.ndarray] = []
    member_icir: list[float] = []
    for i, (entry, _) in enumerate(materialized):
        arr = arrays[i]
        # 成员质量/方向统一按各自 horizon 的标签评估（与入库口径一致）
        h = member_horizons[i] or args.label_days
        label_i = forward_return_label(panel, h)
        mask = mining_mask & np.isfinite(arr) & np.isfinite(label_i)
        if mask.sum() < 100:
            member_icir.append(0.0)
            oriented.append(arr)
            continue
        ic = daily_spearman_ic(arr[mask], label_i[mask], dts[mask])
        icir = float(ic.mean() / ic.std()) if len(ic) and ic.std() > 1e-12 else 0.0
        member_icir.append(icir)
        oriented.append(arr if ic.mean() >= 0 else -arr)  # 负 IC 成员整体翻转
        print(f"  成员 {entry.name}: horizon={h}d mining ICIR={icir:.3f} "
              f"{'(翻转)' if ic.mean() < 0 else ''}")

    # ④ 组合标签期：成员期数中位数 → 预生成所需标签
    horizon_labels: dict[int, np.ndarray] = {}
    combo_horizons = [combo_horizon([member_horizons[i] for i in sub], args.label_days)
                      for sub in subsets]
    horizon_by_sub = dict(zip(subsets, combo_horizons))
    for h in sorted(set(combo_horizons)):
        horizon_labels[h] = forward_return_label(panel, h)
        print(f"组合标签期 {h}d 已生成")

    # mining 窗口行（按日期已排序）→ 逐日切片
    m_dates = pd.DatetimeIndex(pd.Series(m_dts).unique()).sort_values()
    date_codes = pd.Series(m_dts).map({d: i for i, d in enumerate(m_dates)}).to_numpy()
    day_bounds: list[tuple[int, int]] = []
    pos = 0
    counts = pd.Series(date_codes).value_counts().sort_index()
    for di in range(len(m_dates)):
        n = int(counts.get(di, 0))
        day_bounds.append((pos, pos + n))
        pos += n

    # ⑤ 逐组合 mining 评估（分块向量化）
    A = np.vstack([a[m_idx] for a in oriented]).T.astype(np.float32)  # [n_m, K]
    combo_rows: list[dict] = []
    mining_labels = {h: lab[m_idx] for h, lab in horizon_labels.items()}

    print(f"开始评估 {len(subsets)} 个组合（分块 {args.chunk}）…")
    for start in range(0, len(subsets), args.chunk):
        chunk = subsets[start:start + args.chunk]
        S = np.full((A.shape[0], len(chunk)), np.nan, dtype=np.float64)
        for ci, sub in enumerate(chunk):
            members = A[:, list(sub)]
            finite_cnt = np.isfinite(members).sum(axis=1)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mean_ = np.nanmean(members, axis=1)
            mean_[finite_cnt == 0] = np.nan
            S[:, ci] = mean_
        ics_by_h: dict[int, np.ndarray] = {}
        for h in sorted(set(horizon_by_sub[s] for s in chunk)):
            ics_by_h[h] = chunk_daily_spearman_ic(S, mining_labels[h], day_bounds)
        for ci, sub in enumerate(chunk):
            h = horizon_by_sub[sub]
            m = summarize_ics(ics_by_h[h][:, ci])
            coverage = float(np.isfinite(S[:, ci]).mean())
            combo_rows.append({
                "combo_id": start + ci,
                "size": len(sub),
                "members": [names[i] for i in sub],
                "label_days": h,
                "mining_ic": m["ic_mean"],
                "mining_icir": m["ic_ir"],
                "mining_n_days": m["n_days"],
                "mining_coverage": round(coverage, 4),
            })
        if (start // args.chunk) % 8 == 0:
            print(f"  … {start + len(chunk)}/{len(subsets)}")

    # 覆盖率门槛过滤
    n_before = len(combo_rows)
    combo_rows = [r for r in combo_rows if r["mining_coverage"] >= args.min_coverage
                  and r["mining_icir"] is not None]
    print(f"覆盖率门槛（>={args.min_coverage}）后剩余 {len(combo_rows)}/{n_before} 个组合")

    # ⑥ mining ICIR 排序；同 size 内的分布分位即穷举 null
    combo_rows.sort(key=lambda r: r["mining_icir"], reverse=True)
    by_size: dict[int, list[float]] = {}
    for r in combo_rows:
        by_size.setdefault(r["size"], []).append(r["mining_icir"])
    for r in combo_rows:
        peers = by_size[r["size"]]
        r["size_icir_p50"] = float(np.percentile(peers, 50))
        r["size_icir_p90"] = float(np.percentile(peers, 90))
        r["size_icir_percentile"] = round(
            100.0 * sum(1 for v in peers if v <= r["mining_icir"]) / len(peers), 2)

    print("\n=== mining 窗口 Top 20（按 ICIR）===")
    print(f"{'rank':>4} {'size':>4} {'ICIR':>7} {'IC':>8} {'days':>5} {'sizePct':>7}  成员")
    for i, r in enumerate(combo_rows[:20], 1):
        print(f"{i:>4} {r['size']:>4} {r['mining_icir']:>7.3f} {r['mining_ic']:>8.4f} "
              f"{r['mining_n_days']:>5} {r['size_icir_percentile']:>6.1f}%  {' + '.join(r['members'])}")

    # ⑦ OOS 一次确认：只读前 top_oos 名（多重检验克制）
    clean_idx = np.flatnonzero(oos_mask)
    first_clean = dts.iloc[clean_idx[0]] if len(clean_idx) else None
    for r in combo_rows[: max(0, args.top_oos)]:
        if first_clean is None:
            break
        sub_idx = [names.index(mn) for mn in r["members"]]
        members = np.vstack([oriented[i] for i in sub_idx])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            score = np.nanmean(members, axis=0)
        score[~np.isfinite(score)] = np.nan
        lab = horizon_labels[r["label_days"]]
        mask = oos_mask & np.isfinite(score) & np.isfinite(lab)
        if mask.sum() < 100:
            r["oos"] = None
            continue
        ic = daily_spearman_ic(score[mask], lab[mask], dts[mask])
        r["oos"] = {
            "ic_mean": float(ic.mean()),
            "ic_ir": float(ic.mean() / ic.std()) if len(ic) and ic.std() > 1e-12 else None,
            "n_days": int(len(ic)),
        }
        print(f"OOS {r['members']}: IC={r['oos']['ic_mean']:.4f} ICIR={r['oos']['ic_ir']}")

    # ⑧ 总赢家 = mining ICIR 最高（预注册），跑 engine_gate
    gate_result = None
    if combo_rows and not args.no_gate and first_clean is not None:
        try:
            from alphaagent.factor.mining.delivery.engine_gate import run_engine_gate
            from alphaagent.factor.mining.research_spec import default_research_spec

            spec_policy = default_research_spec(args.modes[0])["delivery_policy"]["production"]["engine_gate"]
            winner = combo_rows[0]
            sub_idx = [names.index(mn) for mn in winner["members"]]
            members = np.vstack([oriented[i] for i in sub_idx])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                score = np.nanmean(members, axis=0)
            score[~np.isfinite(score)] = np.nan
            print(f"\nengine_gate（{first_clean.date()} ~ {end.date()}）赢家={winner['members']} …")
            gate_result = run_engine_gate(
                panel, score, val_start=str(first_clean.date()), val_end=str(end.date()),
                policy=spec_policy,
            )
            print(f"gate passed={gate_result.get('passed')} fail_reasons={gate_result.get('fail_reasons')}")
        except Exception as exc:  # 门禁异常不吞扫描结果
            gate_result = {"error": str(exc)}
            print(f"[warn] engine_gate 异常：{exc}")

    # ⑨ 落盘：全量 combos.csv + report.json
    import csv

    fields = ["combo_id", "size", "members", "label_days", "mining_ic", "mining_icir",
              "mining_n_days", "mining_coverage", "size_icir_p50", "size_icir_p90",
              "size_icir_percentile", "oos"]
    with open(out_dir / "combos.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in combo_rows:
            w.writerow({**r, "members": " + ".join(r["members"])})
    report = {
        "run_id": run_id,
        "mining_end": str(mining_end.date()),
        "panel_start": str(panel_start.date()),
        "panel_end": str(end.date()),
        "n_factors": k,
        "factor_names": names,
        "member_icir": dict(zip(names, [round(v, 4) for v in member_icir])),
        "n_combos_evaluated": n_before,
        "n_combos_passed_coverage": len(combo_rows),
        "top": combo_rows[: args.top_oos],
        "size_distribution": {
            str(sz): {"n": len(v), "icir_p50": float(np.percentile(v, 50)),
                      "icir_p90": float(np.percentile(v, 90)),
                      "icir_max": float(max(v))}
            for sz, v in sorted(by_size.items())
        },
        "gate": gate_result,
        "time_isolation": "方向归一/穷举/排序全在 mining 窗口；OOS 仅前 top_oos 名各读一次；仅总赢家跑 engine_gate",
        "args": {k_: v for k_, v in vars(args).items()},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                                         encoding="utf-8")
    print(f"\n报告已写入 {out_dir}")


if __name__ == "__main__":
    main()
