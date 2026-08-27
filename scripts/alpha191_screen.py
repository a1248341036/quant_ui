# -*- coding: utf-8 -*-
"""Alpha191 全量批量筛选：qweave 计算因子矩阵 → 统一指标口径 → 四门准入统计。

长任务设计：
- qweave/polars 一次向量化计算全部 191 因子（分钟级）
- 逐条换算到挖掘链路同一指标口径（label_1d_open_to_open 同公式）
- 准入门槛与 submit stage_one 一致（train-only 四条 + val 保留比）
- 只评估不写库：Alpha191 属公开教科书因子，Reviewer 必拒正式库，
  本任务目的是量化"统计上能进几个"
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

TRAIN_END = "2022-12-31"
VAL_START = "2023-01-01"
OUT_DIR = ROOT / "logs" / "factor_mining" / "alpha191"


def suffix_code(c: str) -> str:
    c = str(c).zfill(6)
    if c.startswith(("60", "68", "9")):
        return f"{c}.SH"
    return f"{c}.SZ"


def main() -> int:
    import qweave
    from core.data import _pg_parquet_end, _load_panel_pg_parquet

    end = _pg_parquet_end()
    start = "2018-01-01"
    print(f"[1] 加载本地前复权面板 {start}~{end} ...", flush=True)
    t0 = time.perf_counter()
    panel = _load_panel_pg_parquet(start=start, end=end)
    print(f"[1] rows={len(panel)} codes={panel['code'].nunique()} "
          f"load={time.perf_counter()-t0:.0f}s", flush=True)

    # vwap 口径与 qweave_research 一致
    panel["volume"] = panel["volume"].astype(float)
    panel["amount"] = panel["amount"].astype(float)
    panel["vwap"] = np.where(panel["volume"] > 0,
                             panel["amount"] / (panel["volume"] * 100.0), np.nan)
    df = panel[["date", "code", "open", "high", "low", "close",
                "volume", "amount", "turnover", "vwap"]].copy()
    df["code"] = df["code"].astype(str).str.zfill(6)

    # GTJA191 需要的市场级输入：等权指数代理 + 中性 SMB/HML（仅少数公式引用，
    # 置零使其优雅退化；正式研究需替换为真实 FF 因子序列）
    idx_close = df.groupby("date")["close"].mean()
    idx_open = df.groupby("date")["open"].mean()
    mkt_ret = idx_close.pct_change()
    df["index_close"] = df["date"].map(idx_close)
    df["index_open"] = df["date"].map(idx_open)
    df["mkt"] = df["date"].map(mkt_ret)
    df["hml"] = 0.0
    df["smb"] = 0.0

    print("[2] 构建 master 索引与 label（open_{t+2}/open_{t+1}-1）...", flush=True)
    t0 = time.perf_counter()
    wide_open = df.pivot(index="date", columns="code", values="open").sort_index()
    label_wide = wide_open.shift(-2) / wide_open.shift(-1) - 1.0
    print(f"[2] label 就绪 {time.perf_counter()-t0:.1f}s", flush=True)

    print("[3] qweave 计算 gtja_alpha191 ...", flush=True)
    t0 = time.perf_counter()
    import polars as pl
    import qweave as qw
    alphas = getattr(qw, "gtja_alpha191")({})
    names = [a.output_name() for a in alphas]
    out = qw.with_alphas(pl.from_pandas(df), "code", "date", alphas)
    print(f"[3] {len(names)} 因子矩阵 {out.height}x{out.width} "
          f"{time.perf_counter()-t0:.0f}s", flush=True)

    from alphaagent.factor.metrics import (
        coverage,
        cross_sectional_ic,
        cross_sectional_lag1_pearson_autocorr,
        cross_sectional_rank_ic,
        cs_ic_summary,
    )

    # 预构建 label 长表一次（保留 NaN 以维持完整索引）
    # metrics 函数要求 level name = "datetime" / "instrument"
    lab_long = label_wide.stack(dropna=False)
    lab_long.index = lab_long.index.set_names(["datetime", "instrument"])
    lab_long.index = lab_long.index.set_levels(
        [lab_long.index.levels[0], lab_long.index.levels[1].map(suffix_code)]
    )
    # 转为 dict 以 O(1) 查找，避免 MultiIndex 对齐问题
    lab_dict = lab_long.to_dict()

    dates_all = pd.DatetimeIndex(label_wide.index)
    train_cutoff = pd.Timestamp(TRAIN_END)
    val_start = pd.Timestamp(VAL_START)

    def stats_for(ser: pd.Series, cutoff: pd.Timestamp, is_train: bool) -> dict:
        """ser: index=MultiIndex(datetime,instrument) 的因子值序列。"""
        dates = ser.index.get_level_values(0)
        if is_train:
            mask_arr = np.asarray(dates <= cutoff)
        else:
            mask_arr = np.asarray(dates >= cutoff)
        sub = ser.iloc[mask_arr]
        if len(sub) < 1000:
            return {"ok": False}
        # 用 dict 查找 label，避免 MultiIndex 对齐
        idx_tuples = list(sub.index)
        lab_vals = [lab_dict.get(t, np.nan) for t in idx_tuples]
        lab = pd.Series(lab_vals, index=sub.index, name="l")
        # 去掉 label 为 NaN 的行
        valid = np.asarray(lab.notna())
        sub = sub.iloc[valid]
        lab = lab.iloc[valid]
        if len(sub) < 1000:
            return {"ok": False}
        m = pd.concat([sub.rename("f"), lab], axis=1)
        dic = cross_sectional_ic(m["f"], m["l"], min_pairs=30)
        dric = cross_sectional_rank_ic(m["f"], m["l"], min_pairs=30)
        cs = cs_ic_summary(dic, dric)
        ac = cross_sectional_lag1_pearson_autocorr(m["f"], min_pairs=30)
        return {
            "ok": True,
            "ic": cs.get("ic"), "icir": cs.get("icir"), "rank_ic": cs.get("rank_ic"),
            "n_days": cs.get("n_days"),
            "coverage": float(coverage(sub.to_numpy(dtype=np.float32))),
            "autocorr": None if ac is None or not np.isfinite(float(ac)) else round(float(ac), 4),
        }

    results = []
    t0 = time.perf_counter()
    for i, nm in enumerate(names):
        try:
            col_pl = out.select(["date", "code", nm]).drop_nulls()
            pdf = col_pl.to_pandas()
            pdf["code"] = pdf["code"].map(suffix_code)
            ser = pd.Series(pdf[nm].to_numpy(dtype=np.float32),
                            index=pd.MultiIndex.from_arrays(
                                [pd.to_datetime(pdf["date"]),
                                 pdf["code"].astype(str)],
                                names=["datetime", "instrument"]))
            ser = ser[~ser.index.duplicated()]
            tr = stats_for(ser, train_cutoff, is_train=True)
            va = stats_for(ser, val_start, is_train=False) if tr.get("ok") else {"ok": False}
        except Exception as exc:  # noqa: BLE001
            tr, va = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:120]}, {}
        row = {"name": nm,
               "train_ic": tr.get("ic"), "train_icir": tr.get("icir"),
               "rank_ic": tr.get("rank_ic"), "coverage": tr.get("coverage"),
               "autocorr": tr.get("autocorr"), "val_ic": va.get("ic"),
               "err": tr.get("error")}
        # 四门判定（train）+ 保留比
        reasons = []
        if not tr.get("ok"):
            reasons.append(tr.get("error") or "eval_error")
        else:
            ic, icir = abs(float(tr["ic"])), abs(float(tr["icir"]))
            if ic < 0.015: reasons.append("ic")
            if not icir > 0.25: reasons.append("icir")
            if float(tr["coverage"]) <= 0.85: reasons.append("coverage")
            ac = tr.get("autocorr")
            if ac is None or float(ac) < 0.18: reasons.append("cs_autocorr")
            if not reasons and va.get("ic") is not None and tr.get("ic"):
                tic, vic = float(tr["ic"]), float(va["ic"])
                if tic * vic < 0:
                    reasons.append("val_sign_flip")
                elif abs(vic) / max(abs(tic), 1e-12) < 0.5:
                    reasons.append("val_retention")
        row["admit_reasons"] = ",".join(reasons)
        row["admit"] = not reasons
        results.append(row)
        # 边算边落盘：崩溃也不丢已评估结果
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        csv_partial = OUT_DIR / "alpha191_partial.csv"
        import csv as _csv
        _write = not csv_partial.exists()
        with csv_partial.open("a", newline="", encoding="utf-8-sig") as fh:
            w = _csv.DictWriter(fh, fieldnames=list(row.keys()))
            if _write:
                w.writeheader()
            w.writerow(row)
        if (i + 1) % 10 == 0 or i == len(names) - 1:
            el = time.perf_counter() - t0
            eta = el / (i + 1) * (len(names) - i - 1)
            npass = sum(1 for r in results if r["admit"])
            print(f"  [{i+1}/{len(names)}] {nm} pass={npass} "
                  f"elapsed={el:.0f}s eta={eta:.0f}s", flush=True)

    csv_p = OUT_DIR / f"alpha191_admission_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    partial = OUT_DIR / "alpha191_partial.csv"
    if partial.exists():
        import shutil
        shutil.copyfile(partial, csv_p)
    res_df = pd.read_csv(csv_p)
    res_df["train_icir_num"] = pd.to_numeric(res_df.get("train_icir"), errors="coerce")
    res_df = res_df.sort_values("train_icir_num", key=lambda s: s.abs(), ascending=False).drop(columns=["train_icir_num"])

    admitted = res_df[res_df.admit] if "admit" in res_df else res_df[pd.to_numeric(res_df.get("admit"), errors="coerce") == True]  # noqa: E712
    # 每个因子只取一条（重复写盘防御）
    admitted = admitted.drop_duplicates(subset=["name"])
    res_df = res_df.drop_duplicates(subset=["name"])
    print("\n========== Alpha191 准入统计 ==========")
    print(f"总数 {len(res_df)} | 评估成功 {res_df.train_ic.notna().sum()} | "
          f"通过四门+保留比（可进候选池口径）: {len(admitted)}")
    if len(admitted):
        print(admitted[["name", "train_ic", "train_icir", "rank_ic",
                        "autocorr", "val_ic"]].to_string(index=False))
    # 最接近达标的 top10（按 |ICIR|）+ 死因分布
    close = res_df[res_df.train_ic.notna()].copy()
    close["a_icir"] = pd.to_numeric(close.get("train_icir"), errors="coerce").abs()
    close = close.sort_values("a_icir", ascending=False).head(10)
    print("\n-- 最接近（按 |ICIR| 前 10，含死因）--")
    print(close[["name", "train_ic", "train_icir", "rank_ic", "autocorr"]].to_string(index=False))
    if "admit_reasons" in res_df:
        from collections import Counter
        c = Counter()
        for r in res_df["admit_reasons"].fillna(""):
            for k in r.split(","):
                if k:
                    c[k] += 1
        print("\n-- 死因分布 --")
        print(dict(c.most_common()))
    res_df.to_csv(csv_p, index=False, encoding="utf-8-sig")
    print(f"\n明细已保存: {csv_p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
