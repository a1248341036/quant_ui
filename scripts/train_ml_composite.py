"""ML 组合因子训练：AlphaAgent 因子池 → walk-forward Ridge/LightGBM → 组合分数。

时间隔离契约（防幸存者偏差传导）：
- ``mining_end``（默认 = 所选因子库里最晚的 created_at）之前的窗口只用于
  衰减对照表，绝不进入训练；
- walk-forward 每折 train 与 OOS 之间留 purge gap；
- engine_gate 在 OOS 段（首折 OOS 起）裁决，非训练段。

产出：
- artifacts/alphaagent/stacking/<run_id>/model.joblib（特征清单+表达式+参数）
- artifacts/alphaagent/stacking/<run_id>/report.json（逐折指标/衰减表/gate 结论）
- data/stock/pred_demo.parquet（date,code,score 长表 → 回测/前端 pred 因子通道）

用法：
  python scripts/train_ml_composite.py --model both --train-months 18 --step-months 6
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

from alphaagent.data.adapters.cnequity import load_panel_from_cne  # noqa: E402
from alphaagent.factor.cache import FactorValueCache  # noqa: E402
from alphaagent.factor.stacking import (  # noqa: E402
    build_stacking_dataset,
    collect_factor_entries,
    daily_spearman_ic,
    decay_table,
    fit_predict_walkforward,
    walk_forward_splits,
)
from alphaagent.factor.mining.research_spec import default_research_spec  # noqa: E402
from alphaagent.factor.stacking.dataset import _to_utc_naive  # noqa: E402


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modes", nargs="+", default=["technical", "fundamental"],
                    choices=["technical", "fundamental"], help="纳入哪些因子库模式")
    ap.add_argument("--no-candidate", action="store_true", help="只用正式库因子")
    ap.add_argument("--label-days", type=int, default=5, help="前向收益持有天数（对齐调仓频率）")
    ap.add_argument("--model", default="both", choices=["ridge", "lgbm", "both"])
    ap.add_argument("--mining-end", default="auto", help="时间隔离边界（YYYY-MM-DD 或 auto=因子库最晚 created_at）")
    ap.add_argument("--end", default=None, help="数据截止日（默认今天）")
    ap.add_argument("--decay-months", type=int, default=12, help="衰减对照表的 mining 窗口长度")
    ap.add_argument("--train-months", type=int, default=18)
    ap.add_argument("--step-months", type=int, default=6, help="OOS 折长")
    ap.add_argument("--purge-days", type=int, default=5, help="train/OOS 之间 purge gap 交易日数")
    ap.add_argument("--warmup-days", type=int, default=250, help="panel 起点提前量（因子窗口预热）")
    ap.add_argument("--max-corr", type=float, default=0.6, help="跨库因子冗余剔除阈值（默认 0.6，与候选池去重口径一致；相关 0.8 的 20 个因子 ≈ 2-3 个独立信号，放进来只会稀释权重、放大过拟合面）")
    ap.add_argument("--size-neutral/--no-size-neutral", dest="size_neutral", default=True)
    ap.add_argument("--no-gate", action="store_true", help="跳过 engine_gate 回测裁决")
    ap.add_argument("--no-write-pred", action="store_true", help="不写 pred 通道文件")
    ap.add_argument("--pred-out", default=None, help="pred 分数输出路径（默认 data/stock/pred_demo.parquet）")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "artifacts" / "alphaagent" / "stacking" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ① 因子枚举
    entries = collect_factor_entries(
        modes=tuple(args.modes), include_candidate=not args.no_candidate, include_production=True
    )
    print(f"因子枚举：{len(entries)} 个（去重后）")
    if len(entries) < 2:
        print("因子数不足（<2），无法组合。请先挖掘入库更多因子。")
        sys.exit(1)

    # ② 时间隔离边界：挖掘循环实际评估的右端（registry 的 eval_end），
    #    回退到入库时间。注意 eval_end << ingested_at 是常态：因子 2026-08
    #    入库，但其评估/反馈窗口只到 val_end（如 2025-12-31），其后数据未被
    #    挖掘循环消费，仍是干净的组合训练数据。
    if args.mining_end == "auto":
        eval_ends = [_to_utc_naive(e.eval_end) for e in entries]
        eval_ends = [c for c in eval_ends if c is not None]
        if eval_ends:
            mining_end = max(eval_ends)
            print("mining_end 依据：registry eval_end（挖掘循环真实评估边界）")
        else:
            created = [_to_utc_naive(e.created_at) for e in entries]
            created = [c for c in created if c is not None]
            if not created:
                print("因子库无入库时间/评估边界，无法 auto 推断 mining-end，请显式传 --mining-end")
                sys.exit(1)
            mining_end = max(created)
            print("警告：registry 无 eval_end，退回入库时间推断（偏保守）")
    else:
        mining_end = pd.Timestamp(args.mining_end)
    if mining_end.tzinfo is not None:
        mining_end = mining_end.tz_localize(None)
    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()
    panel_start = mining_end - pd.DateOffset(months=args.decay_months) - pd.DateOffset(days=args.warmup_days)
    print(f"时间隔离边界 mining_end={mining_end.date()}；panel 区间 [{panel_start.date()} ~ {end.date()}]")

    # ③ panel 加载（磁盘缓存命中则秒级）
    print("加载 CNE panel …")
    panel = load_panel_from_cne(start=panel_start, end=end, include_fundamentals=True)
    print(f"panel: {panel.shape[0]} 行 × {panel.shape[1]} 列")

    # ④ 数据集构建（物化 + 预处理 + 冗余过滤）
    cache = FactorValueCache()
    dataset = build_stacking_dataset(
        panel,
        entries,
        label_days=args.label_days,
        mining_end=mining_end,
        size_neutral=args.size_neutral,
        max_corr=args.max_corr,
        cache=cache,
        decay_months=args.decay_months,
        progress=lambda msg: print(" ", msg, flush=True),
    )
    print(f"有效特征 {len(dataset.feature_names)} 个；剔除 {len(dataset.dropped)} 个")
    for d in dataset.dropped:
        print(f"  - drop {d['name']} ({d['library']}): {d['reason']}")

    # ⑤ walk-forward 训练（train 严格晚于 mining_end）
    dts = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
    date_series = pd.Series(dts)
    folds = walk_forward_splits(
        dts,
        train_start=mining_end,
        train_months=args.train_months,
        step_months=args.step_months,
        purge_days=max(args.purge_days, args.label_days),
    )
    print(f"walk-forward 折数：{len(folds)}")
    if not folds:
        latest = max((_to_utc_naive(e.created_at) for e in entries if e.created_at), default=None)
        print(
            f"mining_end({mining_end.date()}) 之后样本不足以构成一折"
            f"（需 ≥ {args.train_months} 个月）。"
        )
        if latest is not None and latest >= end - pd.DateOffset(months=args.train_months + args.step_months):
            print(
                "原因：全部因子都在近期入库（最晚 {:%Y-%m-%d}），挖掘窗口之后没有"
                "干净的未来数据可用于组合训练。两条出路：\n"
                "  1. 等待数据积累后重跑（推荐，时间隔离才有意义）；\n"
                "  2. 显式传更早的 --mining-end（如 {} -前推训练期），接受"
                "组合训练与挖掘期重叠的 in-sample 风险——报告会标注该风险，"
                "结论仅作框架演示，不能作为入库依据。".format(latest, end.date())
            )
        sys.exit(1)

    kinds = ["ridge", "lgbm"] if args.model == "both" else [args.model]
    explicit_mining_end = args.mining_end != "auto"
    # 隔离有效性的真实判据：mining_end 是否 ≥ 挖掘循环实际评估右端（eval_end）
    latest_eval_end = max(
        (c for c in (_to_utc_naive(e.eval_end) for e in entries) if c is not None),
        default=None,
    )
    time_isolation = (
        "ok" if (not explicit_mining_end or (latest_eval_end is not None and mining_end >= latest_eval_end))
        else "violated_explicit_override（组合训练期与挖掘期重叠，OOS 结论不可作为入库依据）"
    )
    model_outputs: dict[str, np.ndarray] = {}
    fold_reports: dict[str, list] = {}
    for kind in kinds:
        print(f"训练 {kind} …")
        pred, report = fit_predict_walkforward(
            dataset.feature_matrix, dataset.label, date_series, folds, kind=kind
        )
        model_outputs[kind] = pred
        fold_reports[kind] = report
        for r in report:
            ic = r.get("ic_mean")
            print(f"  OOS {r['oos_start']}~{r['oos_end']}: n_train={r['n_train']} "
                  f"IC={ic if ic is None else round(ic, 4)} "
                  f"{'SKIP' if r.get('skipped') else ''}")

    # 组合分数：多模型 OOS 预测取平均（折内无 in-sample 污染）
    usable = [model_outputs[k] for k in kinds if np.isfinite(model_outputs[k]).any()]
    if not usable:
        print("所有模型折均被跳过（样本不足），无组合分数可产出。")
        sys.exit(1)
    stacked = np.nanmean(np.column_stack(usable), axis=1).astype(np.float32)

    # ⑥ 衰减对照表（幸存者偏差量化）
    print("衰减对照表（mining 窗口 IC vs OOS IC）…")
    materialized_for_decay = []
    from alphaagent.factor.ingest import materialize_factor

    for entry in dataset.entries:
        try:
            values = np.asarray(materialize_factor(entry.expr, panel, cache=cache).values, dtype=np.float32)
            materialized_for_decay.append((entry, values))
        except Exception:
            continue
    decay = decay_table(
        materialized_for_decay, panel, dataset.label,
        mining_end=mining_end, decay_months=args.decay_months,
    )
    for row in decay:
        print(f"  {row['name']:32s} mining={_fmt(row['ic_mining'])} oos={_fmt(row['ic_oos'])} "
              f"保留比={_fmt(row['decay_ratio'])}")

    # ⑦ engine_gate 裁决（OOS 段，周调仓口径）
    gate_result = None
    first_oos = folds[0].oos_dates.min()
    if not args.no_gate:
        from alphaagent.factor.mining.engine_gate import run_engine_gate

        policy = default_research_spec(args.modes[0])["delivery_policy"]["production"]["engine_gate"]
        print(f"engine_gate（{first_oos.date()} ~ {end.date()}，freq={policy.get('freq')}）…")
        gate_result = run_engine_gate(
            panel, stacked, val_start=str(first_oos.date()), val_end=str(end.date()), policy=policy
        )
        gm = gate_result.get("metrics") or {}
        print(f"gate passed={gate_result.get('passed')} fail_reasons={gate_result.get('fail_reasons')}")
        print(
            f"  excess_annual={gm.get('excess_annual')} excess_sharpe={gm.get('excess_sharpe')} "
            f"daily_overlap={gm.get('daily_overlap')} turnover={gate_result.get('diagnostics', {}).get('avg_daily_turnover')}"
        )

    # ⑧ 落盘：report + model + pred 通道
    import joblib

    report = {
        "run_id": run_id,
        "mining_end": str(mining_end.date()),
        "time_isolation": time_isolation,
        "panel_start": str(panel_start.date()),
        "panel_end": str(end.date()),
        "label_days": args.label_days,
        "folds": len(folds),
        "feature_names": dataset.feature_names,
        "dropped": dataset.dropped,
        "fold_metrics": fold_reports,
        "decay_table": decay,
        "gate": gate_result,
        "oos_ic_blended": _blended_oos_ic(stacked, dataset.label, dts, first_oos),
    }
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=1, default=str))
    joblib.dump(
        {
            "kind": kinds,
            "entries": [e.__dict__ for e in dataset.entries],
            "feature_names": dataset.feature_names,
            "label_days": args.label_days,
            "size_neutral": args.size_neutral,
            "mining_end": str(mining_end.date()),
            "folds": [(str(f.train_dates.min().date()), str(f.train_dates.max().date()),
                       str(f.oos_dates.min().date()), str(f.oos_dates.max().date())) for f in folds],
        },
        out_dir / "model.joblib",
    )
    print(f"报告与模型已写入 {out_dir}")

    if not args.no_write_pred:
        pred_path = Path(args.pred_out) if args.pred_out else (ROOT / "data" / "stock" / "pred_demo.parquet")
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        write_pred_parquet(stacked, panel, pred_path)
        print(f"组合分数已写入 pred 通道：{pred_path}")


def write_pred_parquet(values: np.ndarray, panel: pd.DataFrame, path: Path) -> None:
    """组合分数 → date,code,score 长表（core.data.load_pred_scores 的读取格式）。"""
    valid = np.isfinite(values)
    ser = pd.Series(values[valid].astype(np.float64), index=panel.index[valid])
    wide = ser.unstack("instrument")
    wide.columns = [str(c).zfill(6) for c in wide.columns]
    long = (
        wide.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="code", value_name="score")
        .dropna(subset=["score"])
    )
    long.to_parquet(path, index=False)


def _blended_oos_ic(stacked: np.ndarray, label: np.ndarray, dts, first_oos) -> dict | None:
    mask = (pd.Series(dts) >= pd.Timestamp(first_oos)) & np.isfinite(stacked) & np.isfinite(label)
    if mask.sum() < 100:
        return None
    ic = daily_spearman_ic(stacked[mask.to_numpy()], label[mask.to_numpy()], dts[mask.to_numpy()])
    if not len(ic):
        return None
    return {"ic_mean": float(ic.mean()), "ic_ir": float(ic.mean() / ic.std()) if ic.std() > 1e-12 else None,
            "n_days": int(len(ic))}


def _fmt(v) -> str:
    return "None" if v is None else f"{v:.4f}"


if __name__ == "__main__":
    main()
