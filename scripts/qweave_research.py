#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""qweave 研究层：Alpha158/101/191 因子计算 + IC/分组/换手评估 + LightGBM 预测分数。

替代已删除的旧 Qlib 工具链（export_qlib.py / dump_bin.py / qlib_alpha158_demo.py）：
- 直接读 data/pg_parquet/stock_daily.parquet（与回测面板同口径，qweave 无需 qlib 二进制数据）
- 用 Polars/Rust 原生计算因子，不再需要 pyqlib/qlib_data

用法示例:
    # 全市场（universe.csv 股票池）2020 至今，Alpha158 评估
    python scripts/qweave_research.py

    # 限定区间/股票池
    python scripts/qweave_research.py --start 2023-01-01 --end 2024-12-31 --codes 000001,600519,000333

    # 训练 LightGBM 并导出预测分数（date/code/score），供现有回测引擎回灌
    python scripts/qweave_research.py --start 2022-01-01 --train-model --out data/pred_demo.parquet

    # 小样本冒烟（50 只 / 前 30 个因子，验证链路）
    python scripts/qweave_research.py --max-codes 50 --alpha-limit 30

输出:
    data/qweave/<alpha-set>_<start>_<end>/
        factors.parquet        (可选 --save-factors，全部因子矩阵)
        summary.csv            (因子 x horizon 汇总：IC / IR / 分组 / 换手)
        ic.parquet / ic_monthly.parquet / quantile_returns.parquet / turnover.parquet
        rank_autocorr.parquet / portfolio.parquet / coverage.parquet / meta.json
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except Exception:  # noqa: BLE001
    pass
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_RUN_ROOT = PROJECT_ROOT / "data" / "qweave"
DEFAULT_OUT = PROJECT_ROOT / "data" / "pred_demo.parquet"

ALPHA_SETS = {
    "alpha158": ("qlib_alpha158", ["close", "high", "low", "open", "volume", "vwap"]),
    "alpha101": ("worldquant_alpha101", ["close", "high", "low", "open", "volume", "vwap"]),
    "alpha191": ("gtja_alpha191", ["close", "high", "low", "open", "volume", "vwap"]),
}


def load_codes(codes_arg: str | None, max_codes: int | None) -> list[str] | None:
    """--codes 逗号列表；缺省用 universe.csv；文件也没有就 None（全市场）。"""
    if codes_arg:
        return [str(c).strip().zfill(6) for c in codes_arg.split(",") if c.strip()]
    uni = PROJECT_ROOT / "data" / "universe.csv"
    if uni.exists():
        import pandas as pd
        codes = pd.read_csv(uni, dtype={"code": str})["code"].astype(str).str.zfill(6).tolist()
        codes = sorted(codes)
        if max_codes:
            codes = codes[:max_codes]
        print(f"[qweave] 股票池: universe.csv {len(codes)} 只")
        return codes
    return None


def load_panel(start: str, end: str, codes: list[str] | None):
    """读 pg_parquet/stock_daily.parquet，构建与回测面板同口径的前复权面板。"""
    from core.data import _load_panel_pg_parquet
    t0 = time.time()
    panel = _load_panel_pg_parquet(start=start, end=end, codes=codes)
    print(f"[qweave] 面板加载 {time.time()-t0:.1f}s rows={len(panel)} "
          f"codes={panel['code'].nunique()} range={panel['date'].min().date()}~{panel['date'].max().date()}")
    return panel


def to_qweave_df(panel) -> "pl.DataFrame":
    import numpy as np
    import polars as pl
    panel = panel.copy()
    panel["volume"] = panel["volume"].astype(float)
    panel["amount"] = panel["amount"].astype(float)
    # 口径：vwap = amount(元) / (volume(手) * 100)
    panel["vwap"] = np.where(panel["volume"] > 0,
                             panel["amount"] / (panel["volume"] * 100.0), np.nan)
    return pl.from_pandas(panel).select(
        ["date", "code", "open", "high", "low", "close",
         "volume", "amount", "turnover", "vwap"])


def build_alphas(alpha_set: str, alpha_limit: int | None):
    import qweave
    fn_name, _fields = ALPHA_SETS[alpha_set]
    alphas = getattr(qweave, fn_name)({})
    if alpha_limit:
        alphas = alphas[:alpha_limit]
    names = [a.output_name() for a in alphas]
    print(f"[qweave] {alpha_set}: {len(alphas)} 个因子")
    return alphas, names


def run_evaluate(lab, names: list[str], horizons: list[int], quantiles: int,
                 min_cs_count: int, cost_bps: float, run_dir: Path):
    import qweave
    label_cols = [f"ret_{h}" for h in horizons]
    t0 = time.time()
    res = qweave.evaluate(
        lab, "code", "date",
        factor_cols=names, label_cols=label_cols,
        quantiles=quantiles, min_cs_count=min_cs_count, cost_bps=cost_bps,
    )
    print(f"[qweave] evaluate {time.time()-t0:.1f}s")

    run_dir.mkdir(parents=True, exist_ok=True)
    res.save(str(run_dir))
    print(f"[qweave] 评估结果已保存: {run_dir}")

    summary = res.summary
    top = summary.sort("ic_mean", descending=True).select(
        ["factor", "horizon", "n_days", "ic_mean", "rank_ic_mean", "ic_ir",
         "rank_ic_ir", "ls_net_ann", "ls_ir", "top_turnover", "bottom_turnover"])
    print("[qweave] Top 12 因子（按 ic_mean）：")
    print(top.head(12))
    (run_dir / "top_factors.csv").write_text(top.to_pandas().to_csv(index=False), encoding="utf-8")
    return res


def rank_ic_report(pred_pl, score_col: str, label_col: str):
    """逐日横截面秩 IC（polars rank + group corr）。"""
    import polars as pl
    ranked = pred_pl.with_columns(
        pl.col(score_col).rank("ordinal").over("date").alias("_score_r"),
        pl.col(label_col).rank("ordinal").over("date").alias("_label_r"),
    ).drop_nulls(["_score_r", "_label_r"])
    ic = ranked.group_by("date").agg(pl.corr("_score_r", "_label_r").alias("rank_ic"))
    ic = ic.drop_nulls("rank_ic")
    icm = ic.select(
        pl.len().alias("n_days"),
        pl.col("rank_ic").mean().alias("mean"),
        pl.col("rank_ic").std().alias("std"),
        (pl.col("rank_ic").mean() / (pl.col("rank_ic").std() + 1e-12)).alias("ir"),
    )
    print("[qweave] 预测分数秩 IC（测试期）:")
    print(icm)
    return ic


def run_model(lab, names: list[str], model_horizon: int,
              fit_start: str, fit_end: str, test_start: str, test_end: str,
              out_path: Path):
    import lightgbm as lgb
    import polars as pl

    label = f"ret_{model_horizon}"
    if label not in lab.columns:
        raise SystemExit(f"[qweave] 缺少标签列 {label}，请确认 horizons 包含 {model_horizon}")

    cols = ["date", "code", label] + names
    df = lab.select(cols).to_pandas()
    df = df.dropna(subset=[label])
    df["date"] = pd.to_datetime(df["date"])
    print(f"[qweave] 模型样本 rows={len(df)} ({df['date'].min().date()}~{df['date'].max().date()})")

    train_mask = (df["date"] >= pd.Timestamp(fit_start)) & (df["date"] <= pd.Timestamp(fit_end))
    test_mask = (df["date"] >= pd.Timestamp(test_start)) & (df["date"] <= pd.Timestamp(test_end))
    X_train, y_train = df.loc[train_mask, names], df.loc[train_mask, label]
    X_test, y_test = df.loc[test_mask, names], df.loc[test_mask, label]
    print(f"[qweave] train={len(X_train)} test={len(X_test)}")

    med = X_train.median(numeric_only=True)
    X_train = X_train.fillna(med)
    X_test = X_test.fillna(med)

    t0 = time.time()
    model = lgb.LGBMRegressor(
        n_estimators=200, learning_rate=0.05, num_leaves=31,
        min_child_samples=50, subsample=0.9, colsample_bytree=0.8,
        n_jobs=-1, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    print(f"[qweave] LightGBM 训练 {time.time()-t0:.1f}s")

    # 对全部样本预测，输出与回测引擎一致的 date/code/score 长表
    all_x = df[names].fillna(med)
    pred = pd.DataFrame({
        "date": df["date"],
        "code": df["code"].astype(str).str.zfill(6),
        "score": model.predict(all_x),
    }).sort_values(["date", "code"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pred.to_parquet(out_path, index=False)
    print(f"[qweave] 预测分数已导出: {out_path} rows={len(pred)}")

    # 测试期秩 IC
    if len(X_test):
        test_pl = pl.from_pandas(pd.DataFrame({
            "date": df.loc[test_mask, "date"],
            "score": model.predict(X_test),
            "ret": df.loc[test_mask, label],
        }))
        rank_ic_report(test_pl, "score", "ret")
    else:
        print("[qweave] 测试期为空，跳过 IC 评估")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description="qweave 因子研究 / 模型预测")
    parser.add_argument("--start", default="2020-01-01", help="研究起始日")
    parser.add_argument("--end", default=None, help="研究截止日（默认数据最新日）")
    parser.add_argument("--codes", default=None, help="逗号分隔 6 位代码")
    parser.add_argument("--max-codes", type=int, default=None, help="股票池只取前 N 只（冒烟）")
    parser.add_argument("--alpha-set", default="alpha158",
                        choices=sorted(ALPHA_SETS), help="因子集")
    parser.add_argument("--alpha-limit", type=int, default=None, help="只算前 N 个因子（冒烟）")
    parser.add_argument("--horizons", default="1,5,10,20", help="标签 horizon 交易日")
    parser.add_argument("--quantiles", type=int, default=10)
    parser.add_argument("--min-cs-count", type=int, default=30)
    parser.add_argument("--cost-bps", type=float, default=8.0)
    parser.add_argument("--run-dir", type=Path, default=None, help="评估输出目录")
    parser.add_argument("--save-factors", action="store_true", help="额外保存因子矩阵 parquet")
    parser.add_argument("--skip-eval", action="store_true", help="跳过 evaluate 只算因子")
    parser.add_argument("--train-model", action="store_true", help="训练 LightGBM 并导出分数")
    parser.add_argument("--model-horizon", type=int, default=5)
    parser.add_argument("--fit-start", default=None, help="模型训练起始（默认区间前 70% 分位）")
    parser.add_argument("--fit-end", default=None, help="模型训练截止（默认区间前 70% 分位）")
    parser.add_argument("--test-start", default=None, help="模型测试起始（默认区间后 30% 分位）")
    parser.add_argument("--test-end", default=None, help="模型测试截止（默认区间后 30% 分位）")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="预测分数输出 parquet")
    args = parser.parse_args()

    import qweave  # noqa: F401  提前触发 import 错误

    from core.data import _pg_parquet_end
    end = args.end or _pg_parquet_end()
    if not end:
        raise SystemExit("[qweave] 无法确定数据最新日，请用 --end 指定")
    print(f"[qweave] 研究区间: {args.start} ~ {end}")

    codes = load_codes(args.codes, args.max_codes)
    panel = load_panel(args.start, end, codes)
    df = to_qweave_df(panel)

    alphas, names = build_alphas(args.alpha_set, args.alpha_limit)
    horizons = [int(h) for h in args.horizons.split(",") if h.strip()]
    if not horizons:
        raise SystemExit("--horizons 至少一个")

    t0 = time.time()
    out = qweave.with_alphas(df, "code", "date", alphas)
    print(f"[qweave] with_alphas {time.time()-t0:.1f}s -> {out.height}x{out.width}")
    lab = qweave.with_labels(out, "code", "date", horizons)
    print(f"[qweave] with_labels done -> {lab.height}x{lab.width}")

    if args.save_factors:
        factor_cols = ["date", "code"] + names
        factor_dir = Path(args.run_dir) if args.run_dir else DEFAULT_RUN_ROOT / "factors"
        factor_dir.mkdir(parents=True, exist_ok=True)
        factor_path = factor_dir / f"{args.alpha_set}_{args.start}_{end}.parquet"
        lab.select(factor_cols).write_parquet(factor_path)
        print(f"[qweave] 因子矩阵已保存: {factor_path}")

    if not args.skip_eval:
        run_dir = args.run_dir or DEFAULT_RUN_ROOT / f"{args.alpha_set}_{args.start}_{end}"
        run_evaluate(lab, names, horizons, args.quantiles, args.min_cs_count,
                     args.cost_bps, run_dir)

    if args.train_model:
        fit_start = args.fit_start or args.start
        fit_end = args.fit_end
        test_start = args.test_start
        test_end = args.test_end or end
        if fit_end is None or test_start is None:
            dates = sorted(lab["date"].unique().to_list())
            split = int(len(dates) * 0.7)
            fit_end = fit_end or pd.Timestamp(dates[split - 1]).date().isoformat()
            test_start = test_start or pd.Timestamp(dates[split]).date().isoformat()
        print(f"[qweave] 模型切分: fit={fit_start}~{fit_end} test={test_start}~{test_end}")
        run_model(lab, names, args.model_horizon,
                  fit_start, fit_end, test_start, test_end, args.out)

    print("[qweave] 完成")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())




