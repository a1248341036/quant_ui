"""ML 组合执行参数选定（盲测隔离 v2，2026-09-12）。

严格时间隔离纪律：
- **盲测段（mining_end 之后，2025+）始终不可见、不参与任何参数选择**；
- 执行参数（选股宽度）仅在 2024 前 train/val 段选定并锁定；
- 选定结果写 ``out_dir/exec_params.json``，供 train_ml_composite 的盲测终局裁决读取；
- 本脚本不触碰盲测段数据，候选宽度扫描全部发生在 mining_end 之前的折。

用法：
  python scripts/select_ml_exec_params.py --widths 0.002,0.003,0.004,0.006
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from alphaagent.data.adapters.cnequity import load_panel_from_cne  # noqa: E402
from alphaagent.factor.cache import FactorValueCache  # noqa: E402
from alphaagent.factor.stacking import (  # noqa: E402
    build_stacking_dataset,
    collect_factor_entries,
    fit_predict_walkforward,
    walk_forward_splits,
)
from alphaagent.factor.stacking.dataset import _to_utc_naive  # noqa: E402
from alphaagent.factor.mining.research_spec import default_research_spec  # noqa: E402
from alphaagent.factor.mining.engine_gate import run_engine_gate  # noqa: E402
from core import trading_config  # noqa: E402


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modes", nargs="+", default=["technical", "fundamental"],
                    choices=["technical", "fundamental"])
    ap.add_argument("--no-candidate", action="store_true")
    ap.add_argument("--label-days", type=int, default=5)
    ap.add_argument("--max-corr", type=float, default=0.6)
    ap.add_argument("--mining-end", default="auto")
    ap.add_argument("--widths", default="0.002,0.003,0.004,0.006",
                    help="候选选股宽度（top_pct），逗号分隔")
    ap.add_argument("--freqs", default="weekly,monthly",
                    help="候选调仓频率，逗号分隔（engine_gate overlap/回测均受影响）")
    ap.add_argument("--score-smooth", type=int, default=0)
    ap.add_argument("--out-dir", default=None,
                    help="输出目录（默认 artifacts/alphaagent/stacking/exec_select_<ts>）")
    return ap.parse_args()


def _resolve_mining_end(entries) -> pd.Timestamp:
    eval_ends = [
        c for c in (_to_utc_naive(e.eval_end) for e in entries) if c is not None
    ]
    if eval_ends:
        return max(eval_ends)
    created = [c for c in (_to_utc_naive(e.created_at) for e in entries) if c is not None]
    if not created:
        raise SystemExit("无 eval_end/created_at 可推断 mining_end")
    return max(created)


def main() -> None:
    args = _parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "artifacts" / "alphaagent" / "stacking" / f"exec_select_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    entries = collect_factor_entries(
        modes=tuple(args.modes), include_candidate=not args.no_candidate, include_production=True
    )
    if len(entries) < 2:
        sys.exit("因子数不足（<2），无法执行参数选定。")
    mining_end = _resolve_mining_end(entries)
    print(f"mining_end={mining_end.date()}（== 盲测段界线；本脚本绝不使用其之后数据）")

    print("加载 CNE panel …")
    panel = load_panel_from_cne()
    dts = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
    date_series = pd.Series(dts)
    panel_start = pd.Timestamp(dts.min())
    print(f"panel: {len(panel)} 行 × {len(panel.columns)} 列 [{panel_start.date()} ~ {pd.Timestamp(dts.max()).date()}]")

    print("构建 stacking 数据集（物化 → 冗余过滤 → 组装）…")
    cache = FactorValueCache()
    dataset = build_stacking_dataset(
        panel, entries, label_days=args.label_days, mining_end=mining_end,
        size_neutral=True, max_corr=args.max_corr, cache=cache,
        progress=lambda m: print(" ", m, flush=True),
    )
    if len(dataset.entries) < 2:
        sys.exit("有效因子 <2，无法组合。")

    # train/val 段折：OOS 必须【完全】落在 mining_end 之前（2024 前），盲测数据零接触。
    # 用较短的 train_months（默认 7）让折在 2023-04~2024-12 段铺开；strict 过滤确保
    # 每折 oos_dates.max() 都 < mining_end——OOS 起点在 blind 前但跨过界线的折一律丢弃。
    train_months = 7
    step_months = 5
    folds = walk_forward_splits(
        dts, train_start=panel_start, train_months=train_months,
        step_months=step_months, purge_days=max(5, args.label_days),
    )
    use_folds = [f for f in folds if pd.Timestamp(f.oos_dates.max()) < mining_end]
    use_folds = use_folds[:3]
    if not use_folds:
        sys.exit("train/val 段（mining_end 前，OOS 完整在界内）不足一折，无法选参"
                 "——panel 起点太晚或 mining_end 太早。")
    print(f"train/val 段折数={len(use_folds)}（每折 OOS 末端均 < {mining_end.date()}，盲测段未触碰）")
    for f in use_folds:
        print(f"  OOS {pd.Timestamp(f.oos_dates.min()).date()} ~ {pd.Timestamp(f.oos_dates.max()).date()}")

    print("训练 ridge（train/val 段，用于选参）…")
    pred, _report, _fw, _fc = fit_predict_walkforward(
        dataset.feature_matrix, dataset.label, date_series, use_folds, kind="ridge",
        feature_names=dataset.feature_names, label_horizon=args.label_days,
    )
    # 平滑（与 train_ml_composite 一致）——直接复用简单尾随 WMA
    stacked = _wma_smooth_scores(pred, panel, window=_resolve_smooth(args, panel))
    tv_mask = (pd.Series(dts) < mining_end) & np.isfinite(stacked)
    print(f"train/val 段有效组合分数行：{int(tv_mask.sum())}")

    # 候选宽度扫描：全部在 train/val 段，盲测段数据物理不参与。
    # 窗口 = 组合分数实际覆盖区间（首折 OOS 起点 ~ 末折 OOS 终点），而非整个 panel 段——
    # 分数空白区会让重叠/投入比指标失真（无分日子现金空置），导致假 execution_infeasible。
    policy_base = default_research_spec(args.modes[0])["delivery_policy"]["production"]["engine_gate"]
    tv_start = str(pd.Timestamp(use_folds[0].oos_dates.min()).date())
    tv_end = str(pd.Timestamp(use_folds[-1].oos_dates.max()).date())
    cap = trading_config.GATE_CAPITAL
    results: list[dict] = []
    widths = sorted({float(x) for x in args.widths.split(",") if x.strip()})
    freqs = [str(x).strip() for x in args.freqs.split(",") if x.strip()]
    for pct in widths:
        for freq in freqs:
            policy = {**policy_base, "selection_mode": "top_pct", "selection_pct": pct,
                      "freq": freq, "capital": cap}
            res = run_engine_gate(
                panel, stacked, val_start=tv_start, val_end=tv_end, policy=policy,
            )
            gm = res.get("metrics") or {}
            gd = res.get("diagnostics") or {}
            rec = {
                "selection_pct": pct,
                "freq": freq,
                "capital": cap,
                "passed": res.get("passed"),
                "excess_annual": gm.get("excess_annual"),
                "excess_sharpe": gm.get("excess_sharpe"),
                "max_drawdown": gm.get("max_drawdown"),
                "daily_overlap": gm.get("daily_overlap"),
                "turnover": gd.get("avg_daily_turnover"),
                "hold": gd.get("avg_num_hold"),
                "invested": gd.get("avg_invested_ratio"),
                "rejections": gd.get("n_rejections"),
                "fail_reasons": res.get("fail_reasons") or [],
            }
            results.append(rec)
            print(f"  pct={pct:.4f} freq={freq} → passed={rec['passed']} "
                  f"excess_annual={_fmt(rec['excess_annual'])} excess_sharpe={_fmt(rec['excess_sharpe'])} "
                  f"mdd={_fmt(rec['max_drawdown'])} overlap={_fmt(rec['daily_overlap'])} hold={rec['hold']} rej={rec['rejections']}")

    # 选定：优先取全门槛通过者；否则取 (freq, pct) 中 excess_sharpe 最高（诚实记录未过）
    passed = [r for r in results if r["passed"]]
    chosen = max(passed or results, key=lambda r: float(r["excess_sharpe"] or 0.0))
    selection = {
        "selection_mode": "top_pct",
        "selection_pct": chosen["selection_pct"],
        "freq": chosen["freq"],
        "capital": cap,
        "freq": policy_base.get("freq", "weekly"),
        "selected_window": {"start": tv_start, "end": tv_end},
        "selected_reason": "train/val段引擎门禁全门槛通过"
                           if chosen["passed"] else "train/val段全门槛未通过，取 excess_sharpe 最高（盲测终局将如实裁决）",
        "chosen_from": results,
        "mining_end": str(mining_end.date()),
    }
    (out_dir / "exec_params.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\n选定执行参数 → {out_dir / 'exec_params.json'}")
    print(f"  selection_pct={selection['selection_pct']:.4f}  capital={cap:,.0f}  freq={selection['freq']}")
    print("盲测段（2025+）完全未触碰；终局裁决请带该文件跑 train_ml_composite。")


def _resolve_smooth(args, panel: pd.DataFrame) -> int:
    if args.score_smooth == 1:
        return 1
    if args.score_smooth > 1:
        return args.score_smooth
    return max(2, args.label_days)


def _wma_smooth_scores(values: np.ndarray, panel: pd.DataFrame, window: int) -> np.ndarray:
    """尾随线性 WMA（train_ml_composite 同源简化版）。"""
    ser = pd.Series(np.asarray(values, dtype=np.float64), index=panel.index)
    wide = ser.unstack("instrument")
    wd = np.arange(window, 0, -1, dtype=np.float64)
    wd /= wd.sum()
    out: dict[object, np.ndarray] = {}
    for col in wide.columns:
        v = wide[col].to_numpy(dtype=np.float64)
        m = np.isfinite(v)
        if not m.any():
            out[col] = v
            continue
        v0 = np.where(m, v, 0.0)
        num = np.convolve(v0, wd, mode="full")[: len(v)]
        den = np.convolve(m.astype(np.float64), wd, mode="full")[: len(v)]
        sm = np.where(den > 1e-12, num / np.maximum(den, 1e-12), np.nan)
        sm[~m] = np.nan
        out[col] = sm
    sm_wide = pd.DataFrame(out, index=wide.index, columns=wide.columns)
    long = (
        sm_wide.rename_axis("datetime")
        .reset_index()
        .melt(id_vars="datetime", var_name="instrument", value_name="score")
        .set_index(["datetime", "instrument"])["score"]
    )
    return long.reindex(panel.index).to_numpy(dtype=np.float64)


def _fmt(v) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return str(v)


if __name__ == "__main__":
    main()