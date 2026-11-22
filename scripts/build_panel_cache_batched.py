"""分批重建 panel 缓存（低内存版）。

为什么：整份重建峰值 7~10GB，用户机器会卡到无法使用。分批把峰值压到每批的
规模（约 2~4GB），代价是总时间变长（每批都要带一段回溯窗口重复构建）。

做法：
  1. 把 [start, end] 按 BATCH_DAYS 切成若干段；
  2. 每段构建 [seg_start - WARMUP_DAYS, seg_end]，裁回 [seg_start, seg_end]，
     补 is_trade/not_st、做板块过滤，写成一个临时 parquet，随后释放内存；
  3. 全部段完成后，用 pyarrow 流式读取各临时文件、按 row group 追加写入最终
     缓存文件（原子替换），峰值内存只与单个 batch 相当；
  4. 写 .meta.json（口径与 cnequity._save_cached_panel 一致，保证能命中）。

用法：
  python scripts/build_panel_cache_batched.py [--start 2023-04-01] [--end 2026-11-30]
         [--batch-days 90] [--warmup-days 400]
"""
from __future__ import annotations

import argparse
import gc
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import psutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from alphaagent.data.adapters import cnequity as cne  # noqa: E402


def _board_filter(panel: pd.DataFrame) -> pd.DataFrame:
    if panel.empty:
        return panel
    inst = panel.index.get_level_values("instrument").astype(str)
    excluded: list[str] = []
    if cne._exclude_bse_enabled():
        excluded += list(cne.BSE_PREFIXES)
    if cne._exclude_kechuang_enabled():
        excluded += ["688", "689"]
    if cne._exclude_chinext_enabled():
        excluded += ["300", "301"]
    if excluded:
        mask = inst.str.startswith(tuple(excluded))
        if mask.any():
            panel = panel.loc[~mask]
    return panel


def _segments(start: pd.Timestamp, end: pd.Timestamp, days: int):
    cur = start
    while cur <= end:
        seg_end = min(cur + pd.Timedelta(days=days - 1), end)
        yield cur, seg_end
        cur = seg_end + pd.Timedelta(days=1)


class _MemMonitor:
    """后台采样：进程峰值 RSS + 系统最低可用内存。

    只看批次结束时的瞬时 RSS 会严重低估（构建期峰值是结束时的 2~4 倍），
    所以用采样线程记录真实峰值，同时记录系统 available（含可回收缓存，
    比 "FreePhysicalMemory" 更能反映真实压力）。
    """

    def __init__(self, proc: psutil.Process, interval: float = 1.0):
        self.proc = proc
        self.interval = interval
        self.peak_rss = 0.0
        self.min_avail = 1e9
        self._stop = False
        self._t = None

    def _run(self):
        import time as _t

        while not self._stop:
            try:
                self.peak_rss = max(self.peak_rss, self.proc.memory_info().rss / 1e9)
                self.min_avail = min(self.min_avail, psutil.virtual_memory().available / 1e9)
            except Exception:
                pass
            _t.sleep(self.interval)

    def start(self):
        import threading

        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self):
        self._stop = True
        if self._t:
            self._t.join(timeout=2)

    def take(self):
        """取一次峰值并归零（用于逐批统计）。"""
        rss, self.peak_rss = self.peak_rss, 0.0
        return rss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-04-01")
    ap.add_argument("--end", default="2026-11-30")
    ap.add_argument("--batch-days", type=int, default=90)
    ap.add_argument("--warmup-days", type=int, default=400)
    ap.add_argument("--parts-dir", default=None,
                    help="临时分片目录；给出且分片已存在时跳过该批（断点续跑）")
    ap.add_argument("--merge-only", action="store_true",
                    help="只做合并（分片需已由 --parts-dir 提供）")
    ap.add_argument("--low-priority", action="store_true",
                    help="进程优先级降为 BelowNormal，避免抢占前台应用（内存不变，但系统更跟手）")
    args = ap.parse_args()

    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    segs = list(_segments(start, end, args.batch_days))
    proc = psutil.Process()
    if args.low_priority:
        try:
            proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            print("进程优先级已降为 BelowNormal", flush=True)
        except Exception as exc:  # noqa: BLE001
            print("设置优先级失败（忽略）: %s" % exc, flush=True)
    print("区间 %s ~ %s，共 %d 批，每批 %d 天，回溯 %d 天"
          % (start.date(), end.date(), len(segs), args.batch_days, args.warmup_days), flush=True)

    registry = cne.get_registry()
    sig = cne._facet_signature(None)
    path = cne._cache_path(args.start, True, sig)
    mon = _MemMonitor(proc)
    mon.start()
    tmpdir = Path(args.parts_dir) if args.parts_dir else Path(
        tempfile.mkdtemp(prefix="panelbatch_", dir=str(cne._CACHE_ROOT)))
    tmpdir.mkdir(parents=True, exist_ok=True)
    parts: list[Path] = []
    columns: list[str] | None = None
    # 续跑时列基准必须取自**已有分片**：否则新批以自身列为准，与旧分片列集合
    # 不一致，合并时会报 "schema lengths differ"。
    existing = sorted(tmpdir.glob("part_*.parquet"))
    if existing:
        import polars as _pl

        columns = list(_pl.scan_parquet(str(existing[0])).collect_schema().names())
        print("沿用已有分片列基准: %d 列（来自 %s）" % (len(columns), existing[0].name), flush=True)
    t_all = time.time()

    try:
        for i, (seg_s, seg_e) in enumerate(segs, 1):
            out = tmpdir / ("part_%03d.parquet" % i)
            if args.merge_only or out.exists():
                parts.append(out)
                print("[%d/%d] %s~%s 复用已有分片" % (i, len(segs), seg_s.date(), seg_e.date()), flush=True)
                continue
            build_s = max(start, seg_s - pd.Timedelta(days=args.warmup_days))
            t0 = time.time()
            df = registry.build_panel(
                start=build_s.date().isoformat(), end=seg_e.date().isoformat(),
                universe_mask=False, include_fundamentals=True, asset_type="stock",
                include_columns=None,
            )
            cne._enrich_trade_flags(df)
            df = _board_filter(df)
            dts = df.index.get_level_values("datetime")
            df = df[(dts >= seg_s) & (dts <= seg_e)]
            if columns is None:
                columns = list(df.columns)
            else:
                missing = [c for c in columns if c not in df.columns]
                for c in missing:
                    df[c] = float("nan")
                df = df[columns]
            cne._write_panel_parquet_chunked(df, str(out), 200_000)
            parts.append(out)
            peak = mon.take()
            print("[%d/%d] %s~%s 构建自 %s: %d 行, %.1fs, 峰值RSS %.2fGB"
                  % (i, len(segs), seg_s.date(), seg_e.date(), build_s.date(),
                     len(df), time.time() - t0, peak), flush=True)
            del df
            gc.collect()
            # Arrow/jemalloc 会把释放的内存留在自己的池里（RSS 高水位不回落），
            # 逐批累积后表现为"峰值远超预期"。每批结束强制归还给系统。
            try:
                import pyarrow as _pa

                _pa.default_memory_pool().release_unused()
            except Exception:
                pass

        # 流式合并：用 Polars 的 lazy scan + sink_parquet，自动做类型协调
        # （vertical_relaxed）且不把整份数据读进内存。
        # 之前用 pyarrow ParquetWriter 手工对齐 schema 很脆：各批 dtype 会在
        # 「全 NaN 列 → null / string」这类边界上不一致，强转直接报
        # Unsupported cast from string to null。
        import polars as pl

        fd, tmp_final = tempfile.mkstemp(dir=str(cne._CACHE_ROOT), prefix=".panel.", suffix=".parquet.tmp")
        os.close(fd)
        lf = pl.concat([pl.scan_parquet(str(p)) for p in parts], how="diagonal_relaxed")
        lf.sink_parquet(tmp_final, compression="snappy")
        rows = int(pl.scan_parquet(tmp_final).select(pl.len()).collect().item())
        columns = list(pl.scan_parquet(tmp_final).collect_schema().names())
        os.replace(tmp_final, path)

        meta = {
            "schema": cne._CACHE_SCHEMA_VERSION, "signature": sig, "focus_facets": None,
            "include_fundamentals": True, "board_filter": cne._board_filter_flag(),
            "start": args.start, "end": args.end,
            "columns": len(columns or []), "rows": int(rows),
        }
        cne._meta_path(path).write_text(
            __import__("json").dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        print("完成: %s  行=%d 列=%d 用时=%.1fs  进程峰值RSS %.2fGB  系统最低可用 %.2fGB"
              % (path.name, rows, len(columns or []), time.time() - t_all,
                 mon.peak_rss, mon.min_avail), flush=True)
        return 0
    finally:
        mon.stop()
        if args.parts_dir:
            print("分片保留在: %s（可用于 --merge-only 重试）" % tmpdir, flush=True)
        else:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
