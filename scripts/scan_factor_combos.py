"""面感知因子组合扫描：组内去冗余 → 组分数 → 组间权重（O(N)）→ OOS 一次确认。

与 train_ml_composite.py 同一套时间隔离契约：
- ``mining_end``（默认 = registry eval_end 最大值）之前的窗口用于组内去冗余、
  组间权重学习与 null 校准；
- mining_end 之后的干净留出段对每个配置只读一次（各 blend 方法各一次）；
- 赢家按预注册规则产生（mining 窗口 ICIR 最高的 blend 方法），只有赢家跑
  engine_gate。

协议分层（N 组泛化，无组名硬编码）：
  因子 → facets 主组派生 → 组内贪心去冗余（各组用自己的 horizon 算质量分）
      → 组分数（nanmean + 覆盖门）→ 组间权重（equal/icir/ridge_nn）
      → blended 分数 → OOS 一次确认 → null 校准 → engine_gate。

用法：
  python scripts/scan_factor_combos.py --blend-methods equal,icir,ridge_nn
  python scripts/scan_factor_combos.py --group-label-days 基本面组=15 --no-gate
"""
from __future__ import annotations

import argparse
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
    FactorEntry,
    _sampled_corr,
    collect_factor_entries,
    daily_spearman_ic,
    forward_return_label,
    materialize_entries,
    transform_factor_values,
)
from alphaagent.factor.stacking.groups import (  # noqa: E402
    FacetGroupPolicy,
    apply_weights,
    assign_groups,
    blend_weights,
    derive_blend_horizon,
    group_horizon,
    group_scores,
    parse_label_days,
)
from alphaagent.factor.mining.memory.expressions import expr_facets  # noqa: E402
from alphaagent.factor.window_config import resolve_test_end  # noqa: E402
from alphaagent.factor.stacking.dataset import _to_utc_naive  # noqa: E402


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modes", nargs="+", default=["technical", "fundamental"],
                    choices=["technical", "fundamental"], help="纳入哪些因子库模式（统一大库后仅派生路径/档位）")
    ap.add_argument("--no-candidate", action="store_true", help="只用正式库因子")
    ap.add_argument("--mining-end", default="auto", help="时间隔离边界（YYYY-MM-DD 或 auto）")
    ap.add_argument("--end", default=None, help="数据截止日（默认数据源最新交易日）")
    ap.add_argument("--decay-months", type=int, default=12, help="质量分窗口长度（月）")
    ap.add_argument("--warmup-days", type=int, default=250, help="panel 起点提前量（因子窗口预热）")
    ap.add_argument("--max-corr", type=float, default=0.6, help="组内去冗余相关阈值")
    ap.add_argument("--group-keep-max", type=int, default=None,
                    help="每组去冗余后按 |ICIR| 截断保留的最大成员数（等权稀释防护；默认不截断）")
    ap.add_argument("--min-coverage", type=float, default=0.30, help="组分数最小行覆盖率")
    ap.add_argument("--size-neutral/--no-size-neutral", dest="size_neutral", default=True)
    ap.add_argument("--label-days", type=int, default=5, help="组 horizon 无法派生时的回退标签期")
    ap.add_argument("--group-label-days", nargs="*", default=[], metavar="组=天",
                    help="按组覆盖标签期，如 --group-label-days 基本面组=15 事件资金组=3")
    ap.add_argument("--blend-label-days", type=int, default=None, help="组间合成标签期覆盖（默认各组中位数）")
    ap.add_argument("--blend-methods", default="equal,icir,ridge_nn",
                    help="逗号分隔：equal,icir,ridge_nn")
    ap.add_argument("--null-trials", type=int, default=200, help="随机成员子集 null 校准次数")
    ap.add_argument("--null-seed", type=int, default=42)
    ap.add_argument("--no-gate", action="store_true", help="跳过 engine_gate 回测裁决")
    ap.add_argument("--write-pred", action="store_true", help="把赢家组合分数写入 pred 通道（默认关闭）")
    ap.add_argument("--pred-out", default=None, help="pred 分数输出路径")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认 artifacts/alphaagent/stacking_scan/<时间戳>）")
    return ap.parse_args()


def _resolve_mining_end(entries: list[FactorEntry], args: argparse.Namespace) -> pd.Timestamp:
    """与 train_ml_composite 同口径：auto = max(eval_end)，回退 max(created_at)。"""
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


def _intra_group_dedup(
    members: list[tuple[FactorEntry, np.ndarray, np.ndarray]],  # (entry, raw, transformed)
    label_g: np.ndarray,
    panel: pd.DataFrame,
    mining_end: pd.Timestamp,
    decay_months: int,
    max_corr: float,
    keep_max: int | None = None,
) -> tuple[list[tuple[FactorEntry, np.ndarray]], list[dict]]:
    """组内贪心去冗余：质量分 = |ICIR|（mining 窗口逐日 IC 的均值/标准差）降序保留。

    ICIR 度量信号一致性，比 mean|IC|（正负日子互相抵消也高分）更贴合"组合
    成员该不该留"的判断。方向归一：IC 均值为负的成员整体翻转（等权合成
    前提；ML 路径模型可学符号，不受影响）。只用 mining 窗口，无 OOS 泄漏。
    ``keep_max``：去冗余后再按质量分截断到前 N 个（组内等权稀释防护）。
    """
    dts = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
    mining_start = mining_end - pd.DateOffset(months=decay_months)
    m_mask = (dts <= mining_end) & (dts >= mining_start)
    quality: dict[str, float] = {}
    orientation: dict[str, float] = {}
    for entry, raw, _ in members:
        mask = m_mask & np.isfinite(raw) & np.isfinite(label_g)
        if mask.sum() < 20:
            quality[entry.name] = 0.0
            orientation[entry.name] = 1.0
            continue
        ic = daily_spearman_ic(raw[mask], label_g[mask], dts[mask])
        if len(ic) and ic.std() > 1e-12:
            quality[entry.name] = abs(float(ic.mean() / ic.std()))
        else:
            quality[entry.name] = 0.0
        # 方向归一：mining 窗口 IC 均值为负的因子整体翻转（等权/ICIR 合成
        # 的前提；ML 路径模型可学符号，不受影响）。只用 mining 窗口，无 OOS 泄漏。
        orientation[entry.name] = -1.0 if (len(ic) and ic.mean() < 0) else 1.0

    order = sorted(range(len(members)), key=lambda i: quality[members[i][0].name], reverse=True)
    kept: list[tuple[FactorEntry, np.ndarray]] = []
    dropped: list[dict] = []
    for i in order:
        entry, _, transformed = members[i]
        oriented = transformed * orientation[entry.name]
        if keep_max is not None and len(kept) >= keep_max:
            dropped.append({"name": entry.name, "reason": f"keep_max<{keep_max}",
                            "quality": round(quality[entry.name], 5)})
            continue
        redundant_with = None
        for kept_entry, kept_arr in kept:
            corr = _sampled_corr(oriented, kept_arr, panel)
            if corr is not None and abs(corr) > max_corr:
                redundant_with = kept_entry.name
                break
        if redundant_with is not None:
            dropped.append({"name": entry.name, "reason": f"redundant_with={redundant_with}",
                            "quality": round(quality[entry.name], 5)})
            continue
        kept.append((entry, oriented))
    kept.sort(key=lambda t: [m[0].name for m in members].index(t[0].name))
    return kept, dropped, orientation


def _metrics_on_mask(blended: np.ndarray, label: np.ndarray, dts: pd.Series, mask: np.ndarray) -> dict | None:
    idx = np.flatnonzero(mask & np.isfinite(blended) & np.isfinite(label))
    if len(idx) < 100:
        return None
    ic = daily_spearman_ic(blended[idx], label[idx], dts.iloc[idx])
    if not len(ic):
        return None
    return {"ic_mean": float(ic.mean()),
            "ic_ir": float(ic.mean() / ic.std()) if ic.std() > 1e-12 else None,
            "n_days": int(len(ic))}


def _null_calibration(
    group_members: dict[str, list[tuple[FactorEntry, np.ndarray]]],
    all_members_by_group: dict[str, list[tuple[FactorEntry, np.ndarray]]],
    panel: pd.DataFrame,
    label: np.ndarray,
    dts: pd.Series,
    mining_end: pd.Timestamp,
    trials: int,
    seed: int,
) -> dict:
    """随机同规模成员子集 + 等权合成的 mining 窗口 ICIR 分布（因子汤对照，不碰 OOS）。

    抽样规模固定 = 组内保留成员数（与真组合同形状）；候选池为该组全部物化
    成员（含被去冗余剔除者）。规模随机的 null 会让小子集高方差抬高分布，
    失去对照意义。
    """
    rng = np.random.default_rng(seed)
    m_mask = (pd.Series(dts) <= pd.Timestamp(mining_end)).to_numpy()
    icirs: list[float] = []
    for _ in range(max(0, trials)):
        scores: dict[str, np.ndarray] = {}
        for g, kept in group_members.items():
            pool = all_members_by_group.get(g) or kept
            k = min(len(kept), len(pool))
            pick = rng.choice(len(pool), size=k, replace=False)
            arrs = [pool[i][1] for i in pick]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)  # 全 NaN 行
                s = np.nanmean(np.vstack(arrs), axis=0).astype(np.float32)
            s[~np.isfinite(s)] = np.nan
            if np.isfinite(s).mean() < 0.30:
                continue
            scores[g] = s
        if not scores:
            continue
        blended = apply_weights({g: 1.0 / len(scores) for g in scores}, scores, dts)
        m = _metrics_on_mask(blended, label, dts, m_mask)
        if m and m["ic_ir"] is not None:
            icirs.append(m["ic_ir"])
    if not icirs:
        return {"trials": 0}
    arr = np.asarray(icirs)
    return {
        "trials": len(icirs),
        "ic_ir_p50": float(np.percentile(arr, 50)),
        "ic_ir_p90": float(np.percentile(arr, 90)),
        "ic_ir_p99": float(np.percentile(arr, 99)),
        "_icirs_sorted": sorted(icirs),
    }


def main() -> None:
    args = _parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "artifacts" / "alphaagent" / "stacking_scan" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ① 因子枚举（含 facets / label_col）
    entries = collect_factor_entries(
        modes=tuple(args.modes), include_candidate=not args.no_candidate, include_production=True
    )
    print(f"因子枚举：{len(entries)} 个（去重后）")
    if len(entries) < 2:
        print("因子数不足（<2），无法组合。")
        sys.exit(1)

    mining_end = _resolve_mining_end(entries, args)
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

    # ② panel + 物化 + transform
    print("加载 CNE panel …")
    panel = load_panel_from_cne(start=panel_start, end=end, include_fundamentals=True)
    print(f"panel: {panel.shape[0]} 行 × {panel.shape[1]} 列")
    dts = pd.Series(panel.index.get_level_values("datetime"))

    cache = FactorValueCache()
    materialized, dropped = materialize_entries(panel, entries, cache=cache,
                                                progress=lambda m: print(" ", m, flush=True))
    for d in dropped:
        print(f"  - 物化剔除 {d['name']} ({d['library']}): {d['reason']}")
    if dropped:
        print(f"物化剔除共 {len(dropped)} 个（eval_error/覆盖率不足——如数据湖缺对应列族请先补数据）")
    members: list[tuple[FactorEntry, np.ndarray, np.ndarray]] = []
    for entry, raw in materialized:
        transformed = transform_factor_values(raw, panel, size_neutral=args.size_neutral)
        members.append((entry, raw, transformed))
    if len(members) < 2:
        print("有效因子不足（<2），无法组合。")
        sys.exit(1)

    # ③ 主组派生 + 组 horizon
    policy = FacetGroupPolicy(
        label_days={k.split("=", 1)[0]: int(v) for k, v in
                    (s.split("=", 1) for s in args.group_label_days)},
        blend_label_days=args.blend_label_days,
        min_coverage=args.min_coverage,
    )
    grouped = assign_groups([e for e, _, _ in members])
    members_by_name = {e.name: (e, raw, tr) for e, raw, tr in members}
    group_report: dict[str, dict] = {}
    group_members: dict[str, list[tuple[FactorEntry, np.ndarray]]] = {}
    group_horizons: dict[str, int | None] = {}
    label_g_by_group: dict[str, np.ndarray] = {}
    for g in sorted(grouped):
        es = grouped[g]
        member_label_days = []
        for e in es:
            lc = e.label_col
            if lc is None:
                lc = (members_by_name[e.name][0].label_col)
            member_label_days.append(parse_label_days(lc))
        horizon = group_horizon(member_label_days, g, policy)
        horizon_src = "policy_override" if g in policy.label_days else (
            "member_mode" if horizon is not None else f"fallback({args.label_days})"
        )
        if horizon is None:
            horizon = args.label_days
        label_g = forward_return_label(panel, horizon)
        kept, dropped_g, orientation = _intra_group_dedup(
            [members_by_name[e.name] for e in es], label_g, panel,
            mining_end, args.decay_months, args.max_corr, keep_max=args.group_keep_max,
        )
        group_horizons[g] = horizon if horizon_src != f"fallback({args.label_days})" else None
        group_members[g] = kept
        label_g_by_group[g] = label_g
        group_report[g] = {
            "n_members": len(es),
            "horizon_days": horizon,
            "horizon_source": horizon_src,
            "member_label_days": {e.name: ld for e, ld in zip(es, member_label_days)},
            "orientation": orientation,
            "kept": [e.name for e, _ in kept],
            "dropped": dropped_g,
        }
        print(f"组 {g}: {len(es)} 成员 → 保留 {len(kept)}，horizon={horizon}d（{horizon_src}）")
        for d in dropped_g:
            print(f"  - drop {d['name']}: {d['reason']}")

    # ④ 组分数
    assigned_arrays = {g: [arr for _, arr in kept] for g, kept in group_members.items()}
    scores, score_meta = group_scores(assigned_arrays, min_coverage=policy.min_coverage)
    for g, info in score_meta.items():
        if "absent" in info:
            print(f"组 {g} 当期缺席：{info['absent']}")
    if not scores:
        print("所有组均缺席（覆盖不足），无法合成。")
        sys.exit(1)

    # ⑤ 组间合成标签 + 各方法权重/指标
    blend_horizon = derive_blend_horizon(group_horizons, policy)
    blend_src = "policy_override" if policy.blend_label_days else (
        "group_median" if blend_horizon is not None else f"fallback({args.label_days})"
    )
    if blend_horizon is None:
        blend_horizon = args.label_days
    blend_label = forward_return_label(panel, blend_horizon)
    print(f"组间合成标签期：{blend_horizon}d（{blend_src}）")

    clean_mask = (dts > mining_end).to_numpy()
    first_clean = dts[clean_mask].min() if clean_mask.any() else None
    methods = [m.strip() for m in args.blend_methods.split(",") if m.strip()]
    method_results: dict[str, dict] = {}
    for method in methods:
        weights, diag = blend_weights(scores, blend_label, dts, mining_end=mining_end,
                                      method=method, policy=policy)
        blended = apply_weights(weights, scores, dts)
        mining_m = _metrics_on_mask(blended, blend_label, dts, ~clean_mask)
        oos_m = _metrics_on_mask(blended, blend_label, dts, clean_mask) if first_clean is not None else None
        method_results[method] = {
            "weights": {g: round(w, 4) for g, w in weights.items()},
            "diag": diag,
            "mining": mining_m,
            "oos": oos_m,
        }
        mi = mining_m or {}
        oi = oos_m or {}
        print(f"[{method}] weights={method_results[method]['weights']} "
              f"mining IC={mi.get('ic_mean')} ICIR={mi.get('ic_ir')} | "
              f"OOS IC={oi.get('ic_mean')} ICIR={oi.get('ic_ir')}")

    # ⑥ 预注册赢家：mining 窗口 ICIR 最高的方法（平局取方法字典序首个）
    def _icir(r: dict) -> float:
        m = r.get("mining") or {}
        v = m.get("ic_ir")
        return float(v) if v is not None else -np.inf

    winner = max(sorted(method_results), key=lambda m: _icir(method_results[m]))
    print(f"赢家（预注册：mining ICIR 最高）：{winner}")

    # ⑦ null 校准（随机同规模成员子集 + 等权，mining 窗口内）
    # 池成员使用与组内相同的方向归一数组（orientation 按 mining IC 符号）
    orientation_by_group = {g: group_report[g]["orientation"] for g in grouped}
    all_members_by_group = {
        g: [(e, members_by_name[e.name][2] * orientation_by_group[g].get(e.name, 1.0))
            for e in es]
        for g, es in grouped.items()
    }
    null_result = _null_calibration(
        group_members, all_members_by_group, panel, blend_label, dts,
        mining_end, args.null_trials, args.null_seed,
    ) if args.null_trials > 0 else {"trials": 0}
    win_icir = _icir(method_results[winner])
    if null_result.get("trials"):
        import bisect

        pool = null_result.pop("_icirs_sorted", [])
        pct = (bisect.bisect_left(pool, win_icir) / len(pool) * 100.0) if pool else None
        null_result["winner_ic_ir_percentile"] = round(pct, 2) if pct is not None else None
        print(f"null：{null_result['trials']} 次随机子集，ICIR p90={null_result.get('ic_ir_p90'):.3f} "
              f"p99={null_result.get('ic_ir_p99'):.3f}，赢家分位={null_result.get('winner_ic_ir_percentile')}")

    # ⑧ engine_gate（只跑赢家）
    gate_result = None
    if not args.no_gate and first_clean is not None:
        try:
            from alphaagent.factor.mining.engine_gate import run_engine_gate
            from alphaagent.factor.mining.research_spec import default_research_spec

            spec_policy = default_research_spec(args.modes[0])["delivery_policy"]["production"]["engine_gate"]
            blended_winner = apply_weights(method_results[winner]["weights"], scores, dts)
            print(f"engine_gate（{first_clean.date()} ~ {end.date()}）…")
            gate_result = run_engine_gate(
                panel, blended_winner, val_start=str(first_clean.date()), val_end=str(end.date()),
                policy=spec_policy,
            )
            print(f"gate passed={gate_result.get('passed')} fail_reasons={gate_result.get('fail_reasons')}")
        except Exception as exc:  # 门禁异常不吞扫描结果
            gate_result = {"error": str(exc)}
            print(f"[warn] engine_gate 异常：{exc}")

    # ⑨ 落盘
    report = {
        "run_id": run_id,
        "mining_end": str(mining_end.date()),
        "panel_start": str(panel_start.date()),
        "panel_end": str(end.date()),
        "blend_label_days": blend_horizon,
        "blend_label_source": blend_src,
        "groups": group_report,
        "score_meta": score_meta,
        "methods": method_results,
        "winner": winner,
        "null": null_result,
        "gate": gate_result,
        "time_isolation": "holdout（权重学习/去重/null 全在 mining 窗口；OOS 每方法各读一次）",
        "args": {k: v for k, v in vars(args).items()},
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str),
                                         encoding="utf-8")
    print(f"报告已写入 {out_dir / 'report.json'}")

    if args.write_pred and first_clean is not None:
        from scripts.train_ml_composite import write_pred_parquet

        blended_winner = apply_weights(method_results[winner]["weights"], scores, dts)
        pred_path = Path(args.pred_out) if args.pred_out else (ROOT / "data" / "stock" / "pred_demo.parquet")
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        write_pred_parquet(blended_winner, panel, pred_path)
        print(f"赢家组合分数已写入 pred 通道：{pred_path}")


if __name__ == "__main__":
    main()
