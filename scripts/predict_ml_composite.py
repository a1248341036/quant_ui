"""ML 组合推理通道：加载 ``served_model.joblib`` → 新数据打分 → pred 通道。

这是「训练/推理分离」的推理侧：**不拟合、不评估**，只做
物化 → 截面变换 → 模型预测 → 频段融合 → 分数平滑。每日收盘后跑一次即可
（建议 17:00 计划任务），产出最新的 ``data/stock/pred_demo.parquet``，
供回测页 / 模拟盘 / ML 运营页消费。

用法：

  # 日常：用最新 served model 增量补分（只算最新交易日，追加到 pred）
  python scripts/predict_ml_composite.py --model latest --append

  # 补一段历史 / 指定模型
  python scripts/predict_ml_composite.py \\
      --model artifacts/alphaagent/stacking/<run>/served_model.joblib \\
      --start 2026-09-29 --end 2026-10-29 --append

  # 只算不写（核对分数）
  python scripts/predict_ml_composite.py --model latest --start 2026-10-20 --dry-run

纪律：本脚本不碰训练/盲测——模型权重与融合权重全部来自 served model，
同模型同数据 → 同分数（幂等）。
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:  # Windows 控制台/重定向下保证中文不乱码
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):
    pass

from alphaagent.factor.stacking.serve import (  # noqa: E402
    load_served_model,
    predict_served,
    resolve_latest_served_model,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ML 组合推理通道（加载 served model 出分）")
    ap.add_argument("--model", default="latest",
                    help="served_model.joblib 路径，或 latest=自动取最新（按 mtime）")
    ap.add_argument("--stacking-root", default=None,
                    help="stacking 产物目录（默认 artifacts/alphaagent/stacking）")
    ap.add_argument("--start", default=None,
                    help="推理起始日（默认=end，即只算最新一天；补历史时显式给）")
    ap.add_argument("--end", default=None,
                    help="推理截止日（默认取数据源最新交易日，受 TEST_END_CAP 约束）")
    ap.add_argument("--warmup-days", type=int, default=400,
                    help="面板起点提前量（因子窗口+平滑窗口预热，默认 400 与训练一致）")
    ap.add_argument("--out", default=None,
                    help="pred 输出路径（默认 data/stock/pred_demo.parquet）")
    ap.add_argument("--append", action="store_true",
                    help="与现有 pred 合并（同 date+code 以本次为准）；不加则整体覆盖")
    ap.add_argument("--low-price-floor", type=float, default=10.0,
                    help="低价池掩码：当日 close 高于该值的置 NaN（0=关闭；默认 10 与 11-08 pred 口径一致）")
    ap.add_argument("--dry-run", action="store_true", help="只算不写")
    ap.add_argument("--json", action="store_true", help="末尾打印一行 JSON 摘要（供计划任务/监控消费）")
    return ap.parse_args(argv)


def _resolve_model_path(args: argparse.Namespace) -> Path:
    if str(args.model).strip().lower() != "latest":
        return Path(args.model)
    root = Path(args.stacking_root) if args.stacking_root else (
        ROOT / "artifacts" / "alphaagent" / "stacking")
    return resolve_latest_served_model(root)


def _resolve_end(end_arg: str | None) -> pd.Timestamp:
    from alphaagent.factor.window_config import resolve_test_end

    if end_arg:
        return pd.Timestamp(end_arg)
    try:
        return pd.Timestamp(resolve_test_end())
    except Exception:  # noqa: BLE001 — 数据源不可用时退回今天（panel 加载会再收敛）
        return pd.Timestamp(datetime.now().date())


def _apply_low_price_floor(frame: pd.DataFrame, panel: pd.DataFrame,
                           floor: float) -> tuple[pd.DataFrame, int]:
    """低价池掩码：当日 close > floor 的分数置 NaN（与 11-08 pred 口径一致）。"""
    if floor is None or float(floor) <= 0 or frame.empty:
        return frame, 0
    close = panel.reset_index()[["datetime", "instrument", "close"]].copy()
    close.columns = ["date", "instrument", "close"]
    close["date"] = pd.to_datetime(close["date"])
    import re

    def _norm(v: object) -> str:
        m = re.search(r"(\d{6})", str(v))
        return (m.group(1) if m else str(v)).zfill(6)

    close["code"] = [_norm(v) for v in close["instrument"]]
    merged = frame.merge(close[["date", "code", "close"]], on=["date", "code"], how="left")
    blocked = merged["close"] > float(floor)
    merged.loc[blocked, "score"] = np.nan
    out = merged.drop(columns=["close"]).dropna(subset=["score"])
    return out.reset_index(drop=True), int(blocked.sum())


def _write_pred(frame: pd.DataFrame, out_path: Path, append: bool) -> tuple[int, int]:
    """写 pred；append 时按 date+code 去重（本次为准），返回 (总行数, 新增行数)。"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if append and out_path.is_file():
        old = pd.read_parquet(out_path)
        old["date"] = pd.to_datetime(old["date"])
        frame = frame.copy()
        frame["date"] = pd.to_datetime(frame["date"])
        before = len(old)
        merged = pd.concat([old, frame], ignore_index=True)
        merged = (merged.drop_duplicates(subset=["date", "code"], keep="last")
                  .sort_values(["date", "code"]).reset_index(drop=True))
        merged.to_parquet(out_path, index=False)
        return len(merged), len(merged) - before
    frame.to_parquet(out_path, index=False)
    return len(frame), len(frame)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    started = datetime.now()

    model_path = _resolve_model_path(args)
    served = load_served_model(model_path)
    card = served.model_card()
    print(f"served model：{model_path}")
    print(f"  训练至 {served.trained_until} · {len(served.groups)} 组 · "
          f"{sum(len(g['feature_names']) for g in card['groups'].values())} 特征 · "
          f"平滑 WMA-{served.smooth_window}")
    print("  融合权重：" + "、".join(
        f"{k}={served.blend_weights.get(k, 0.0):.3f}" for k in served.group_keys()))

    end = _resolve_end(args.end)
    start = pd.Timestamp(args.start) if args.start else end
    if start > end:
        print(f"[error] start={start.date()} 晚于 end={end.date()}")
        return 2
    panel_start = start - pd.Timedelta(days=max(int(args.warmup_days), 120))
    print(f"推理区间 {start.date()} ~ {end.date()}；面板起点 {panel_start.date()}（warmup {args.warmup_days} 天）")

    from alphaagent.data.adapters.cnequity import load_panel_from_cne

    print("加载 CNE panel …")
    panel = load_panel_from_cne(start=panel_start, end=end, include_fundamentals=True)
    print(f"panel: {len(panel)} 行 × {panel.shape[1]} 列")

    frame = predict_served(
        served, panel, start=start, end=end,
        progress=lambda msg: print(" ", msg, flush=True),
    )
    if frame.empty:
        print("[error] 推理结果为空（检查 panel 区间/因子物化）")
        return 3

    scored_days = frame["date"].nunique()
    n_before = len(frame)
    frame, n_blocked = _apply_low_price_floor(frame, panel, args.low_price_floor)
    print(f"分数：{len(frame)} 行 / {scored_days} 个交易日 / "
          f"{frame['code'].nunique()} 只；低价池掩码剔除 {n_blocked} 行（close>{args.low_price_floor}）")

    written = 0
    new_rows = 0
    out_path = Path(args.out) if args.out else (ROOT / "data" / "stock" / "pred_demo.parquet")
    if args.dry_run:
        print("[dry-run] 不写 pred")
    else:
        written, new_rows = _write_pred(frame, out_path, append=args.append)
        print(f"已写入 {out_path}：{written} 行（本次新增 {new_rows}），"
              f"覆盖 {start.date()} ~ {frame['date'].max().date()}")

    summary = {
        "ts": started.strftime("%Y-%m-%dT%H:%M:%S"),
        "model": str(model_path),
        "trained_until": served.trained_until,
        "score_start": str(frame["date"].min().date()),
        "score_end": str(frame["date"].max().date()),
        "n_days": int(scored_days),
        "rows": int(len(frame)),
        "rows_before_floor": int(n_before),
        "written_rows": int(written),
        "new_rows": int(new_rows),
        "dry_run": bool(args.dry_run),
        "pred_path": str(out_path),
        "elapsed_sec": round((datetime.now() - started).total_seconds(), 1),
    }
    if args.json:
        print("[PREDICT_JSON] " + json.dumps(summary, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
