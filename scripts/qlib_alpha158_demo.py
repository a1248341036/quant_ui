#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Qlib 研究层最小闭环：Alpha158 + LightGBM 训练 -> 预测分数导出。

用法:
    python scripts/qlib_alpha158_demo.py --qlib-dir qlib_data/cn_stock
    python scripts/qlib_alpha158_demo.py --qlib-dir <tmp>/qlib_data --out data/pred_demo.parquet

流程:
    1. init_instance 挂载 qlib 数据源
    2. Alpha158 因子 -> Dataset
    3. LightGBM 训练/预测
    4. 预测分数导出长表 parquet（date/code/score），供现有引擎回灌
    5. 打印与未来收益的秩 IC（粗评估，随机数据下接近 0 属正常）

依赖: pyqlib + scripts/export_qlib.py 生成的 qlib_data
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")  # qlib 0.9.7 + mlflow3 文件store兼容

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:  # noqa: BLE001
    pass

sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_QLIB_DIR = PROJECT_ROOT / "qlib_data" / "cn_stock"
DEFAULT_OUT = PROJECT_ROOT / "data" / "pred_demo.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description="Qlib Alpha158 + LightGBM 最小闭环")
    parser.add_argument("--qlib-dir", type=Path, default=DEFAULT_QLIB_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--start", default="2024-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--fit-start", default="2024-01-01")
    parser.add_argument("--fit-end", default="2024-09-30")
    parser.add_argument("--label-days", type=int, default=5,
                        help="IC 评估用的未来收益窗口（交易日）")
    args = parser.parse_args()

    if not (args.qlib_dir / "features").exists():
        print(f"[qlib_demo] qlib 数据不存在: {args.qlib_dir}\n"
              "  先用 scripts/export_qlib.py 生成", file=sys.stderr)
        return 1

    import qlib
    qlib.init(provider_uri=str(args.qlib_dir), region="cn")

    from qlib.contrib.data.handler import Alpha158
    from qlib.contrib.model.gbdt import LGBModel
    from qlib.data.dataset import Dataset, DatasetH

    handler = Alpha158(
        instruments="all",
        start_time=args.start, end_time=args.end,
        fit_start_time=args.fit_start, fit_end_time=args.fit_end,
    )
    segments = {
        "train": (args.fit_start, "2024-06-30"),
        "valid": ("2024-07-01", args.fit_end),
        "test": ("2024-10-01", args.end),
    }
    dataset = DatasetH(handler, segments=segments)
    model = LGBModel()
    print("[qlib_demo] 训练 LightGBM ...", flush=True)
    model.fit(dataset)
    print("[qlib_demo] 预测 ...", flush=True)
    pred = model.predict(dataset)
    print("[qlib_demo] pred shape:", pred.shape)

    out = pred.reset_index()
    out.columns = ["date", "symbol", "score"]
    out["code"] = out["symbol"].astype(str).str[:6]
    out = out[["date", "code", "score"]].sort_values(["date", "code"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"[qlib_demo] 预测分数已导出: {args.out} rows={len(out)}")

    # 粗评估：与未来 label_days 交易日收益的秩 IC
    try:
        from qlib.data import D
        close = D.features(["all"], ["$close"], start_time=args.start,
                           end_time=args.end, freq="day")
        close = close["$close"].unstack("instrument")
        ret_f = close.shift(-args.label_days) / close - 1.0
        if isinstance(pred, pd.DataFrame):
            score = pred["pred"] if "pred" in pred.columns else pred.iloc[:, 0]
        else:
            score = pred  # qlib 0.9.7 predict 返回 Series
        pred_pivot = score.unstack("instrument")
        ic = pred_pivot.corrwith(ret_f.reindex_like(pred_pivot), axis=0)
        ic = ic.dropna()
        print(f"[qlib_demo] 秩IC mean={ic.mean():.4f} std={ic.std():.4f} n={len(ic)}")
    except Exception as exc:  # noqa: BLE001
        print(f"[qlib_demo] IC 评估跳过: {exc}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())