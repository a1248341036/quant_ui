"""dividend 空值修复：按 ann_date 重拉 Tushare 真值，替换 curated 里的 null div_proc 行。

背景：2026-08-23 的历史导入（import-history-001）丢了 div_proc 字段，curated dividend
里 58,815 行 div_proc 为空。div_proc 是主键成员（required 非空），任何触碰这些分区的
compact 合并都会校验失败，阻塞全湖 staging 提升（8-31 的六连失败即此）。

策略：
1. 读 curated dividend，取 null 行的全部 distinct ann_date（~2k 个）
2. 逐日调 Tushare dividend(ann_date=...) 全市场批量，重建真值行
3. curated 重建 = 非 null 旧行 ∪ 重拉行，按 PK (symbol, end_date, div_proc) 去重
   （重拉行 fetched_at=now 天然覆盖同 PK 旧行；原 null 行 PK 含 null、与新行不同键，
   必须显式丢弃，其信息已由重拉行的完整版本轨迹覆盖）
4. 对重拉后仍为 null div_proc 的行（API 对个别老行也可能空），按
   ex_date/pay_date -> 实施、否则 -> 预案 兜底，并打印统计
5. 按原分区布局（end_date 年分区 / part-merged.parquet / zstd）原子回写
6. 全量自校验 validate_dataframe 通过后才落盘

用后即删。
"""

from __future__ import annotations

import glob
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, r"D:\Quant\quant_ui\CNEquity\src")

import polars as pl

from cnequity.config import load_config
from cnequity.domain.schemas import DIVIDEND_SCHEMA, validate_dataframe, with_provenance
from cnequity.external.tushare_fetch import _fetch_with_retry, _get_pro

CURATED = Path(r"D:\Quant\quant_ui\CNEquity\data\quant_dataset\_cnequity\curated")
DIV = CURATED / "dividend"
CFG = load_config(r"D:\Quant\quant_ui\CNEquity\configs\cnequity.quant_dataset.toml")
PK = ["symbol", "end_date", "div_proc"]
SLEEP = 0.35

# ── 1. 读 curated，收集 null 行的 ann_date ──────────────────────────────
files = sorted(DIV.rglob("*.parquet"))
print(f"curated dividend files: {len(files)}")
df = pl.concat([pl.read_parquet(p, hive_partitioning=False) for p in files], how="diagonal_relaxed")
print(f"rows: {df.height}, null div_proc: {df.filter(pl.col('div_proc').is_null()).height}")
bad = df.filter(pl.col("div_proc").is_null())
ann_dates = sorted(bad["ann_date"].unique().to_list())
print(f"distinct ann_date to refetch: {len(ann_dates)}")

keep = df.filter(pl.col("div_proc").is_not_null())
print(f"kept non-null rows: {keep.height}")

# ── 2. 逐日重拉真值 ─────────────────────────────────────────────────────
pro = _get_pro(CFG)
frames: list[pl.DataFrame] = []
fail_days: list[str] = []
t0 = time.time()
for i, d in enumerate(ann_dates, 1):
    ds = d.strftime("%Y%m%d")
    try:
        raw = _fetch_with_retry(pro, "dividend", interval=SLEEP, ann_date=ds)
        if raw is not None and raw.height:
            frames.append(raw)
    except Exception as exc:  # noqa: BLE001
        fail_days.append(ds)
        print(f"  FAIL {ds}: {exc}")
    if i % 200 == 0:
        el = time.time() - t0
        print(f"  {i}/{len(ann_dates)} days, {el:.0f}s elapsed")

if fail_days:
    print(f"FAILED days: {len(fail_days)} (first few: {fail_days[:5]})")
    print("ABORT — no files written. Rerun the script to retry (idempotent).")
    sys.exit(1)

refetched = pl.concat(frames, how="diagonal_relaxed")
print(f"refetched rows: {refetched.height}")

# ── 3. 类型对齐到 schema ────────────────────────────────────────────────
rename = {"ts_code": "symbol"}
refetched = refetched.rename(rename)
for col, dtype in DIVIDEND_SCHEMA.items():
    if col not in refetched.columns:
        refetched = refetched.with_columns(pl.lit(None).alias(col))
    if col in ("end_date", "ann_date", "record_date", "ex_date", "pay_date", "div_listdate", "imp_ann_date"):
        refetched = refetched.with_columns(
            pl.col(col).cast(pl.Utf8, strict=False).str.strptime(pl.Date, format="%Y%m%d", strict=False).alias(col)
        )
    elif col not in ("symbol", "div_proc", "source", "data_version", "fetched_at"):
        refetched = refetched.with_columns(pl.col(col).cast(dtype, strict=False).alias(col))

still_null = refetched.filter(pl.col("div_proc").is_null()).height
print(f"refetched rows still null div_proc (fallback fill): {still_null}")
if still_null:
    refetched = refetched.with_columns(
        pl.when(pl.col("ex_date").is_not_null() | pl.col("pay_date").is_not_null())
        .then(pl.lit("实施"))
        .otherwise(pl.lit("预案"))
        .alias("div_proc")
    )

# provenance：source 标记修复来源
now = datetime.now(timezone.utc)
refetched = refetched.with_columns(
    pl.lit("tushare").alias("source"),
    pl.lit("v1").alias("data_version"),
    pl.lit(now).alias("fetched_at"),
)
refetched = refetched.select(list(DIVIDEND_SCHEMA.keys()))
keep = keep.select(list(DIVIDEND_SCHEMA.keys()))

# ── 4. 合并去重：非 null 旧行 ∪ 重拉行，PK keep-last（重拉行 fetched_at 新）──
merged = pl.concat([keep, refetched], how="diagonal_relaxed")
merged = merged.sort("fetched_at").unique(subset=PK, keep="last")
merged = merged.sort(["end_date", "symbol"])
null_after = merged.filter(pl.col("div_proc").is_null()).height
print(f"merged rows: {merged.height}, null div_proc after merge: {null_after}")
assert null_after == 0, "null div_proc remains — abort"

# ── 5. 全量校验 ────────────────────────────────────────────────────────
validated = validate_dataframe(merged, "dividend")
print(f"validated rows: {validated.height}")

# ── 6. 分区回写（end_date 年分区） ─────────────────────────────────────
validated = validated.with_columns(pl.col("end_date").dt.year().cast(pl.Utf8).alias("__part__"))
tmp_root = DIV.parent / "_dividend_repair_tmp"
if tmp_root.exists():
    shutil.rmtree(tmp_root)
for key, group in validated.partition_by("__part__", as_dict=True).items():
    part = str(key[0])
    out_dir = tmp_root / f"end_date={part}"
    out_dir.mkdir(parents=True, exist_ok=True)
    group.drop("__part__").write_parquet(out_dir / "part-merged.parquet", compression="zstd")

# 原子替换：先改名旧目录，换入新目录，再删旧
old = DIV.parent / "_dividend_old"
if old.exists():
    shutil.rmtree(old)
DIV.rename(old)
tmp_root.rename(DIV)
shutil.rmtree(old)
print("curated dividend replaced.")

# ── 7. 终验：重读全部文件过校验 ────────────────────────────────────────
files2 = sorted((CURATED / "dividend").rglob("*.parquet"))
final = pl.concat([validate_dataframe(pl.read_parquet(p, hive_partitioning=False), "dividend") for p in files2], how="diagonal_relaxed")
print(f"FINAL: {len(files2)} files, {final.height} rows, null div_proc: {final.filter(pl.col('div_proc').is_null()).height}")
