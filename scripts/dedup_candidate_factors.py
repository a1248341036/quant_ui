"""候选因子库一次性去重脚本。

对 candidate_technical/mining_candidate_registry.json 中所有因子两两计算
截面 Pearson 相关，识别高重复因子对，删除冗余因子（保留指标更好的那个）。

用法:
    .venv/Scripts/python.exe scripts/dedup_candidate_factors.py [--threshold 0.6] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CAND_REGISTRY = ROOT / "artifacts" / "alphaagent" / "factorzoo" / "candidate_technical" / "mining_candidate_registry.json"


def _load_panel() -> pd.DataFrame:
    """加载精简 panel：仅加载因子表达式所需的最小列集。

    直接从 CNE parquet 读取 stock_daily_wide，跳过 fund_flow / fundamental 等辅助插件
    和 label/ret 等衍生列，避免内存爆炸。
    """
    import polars as pl
    from cnequity.config import load_config
    from cnequity.query.reader import load as cne_load
    import os

    cne_root = ROOT / "CNEquity"
    cfg_path = cne_root / "configs" / "cnequity.quant_dataset.toml"

    print("正在加载 CNE stock_daily_wide（2023-01-01 ~ 2025-12-31）…", flush=True)
    old = Path.cwd()
    try:
        os.chdir(cne_root)
        cfg = load_config(cfg_path)
        df = cne_load("stock_daily_wide", start="2023-01-01", end="2025-12-31", config=cfg)
    finally:
        os.chdir(old)

    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()
    print(f"  原始数据: {df.shape[0]:,} 行 × {df.shape[1]} 列", flush=True)

    # 列映射（与 stock_daily_wide 插件一致）
    col_map = {
        "open": "open", "high": "high", "low": "low", "close": "close",
        "vol": "volume", "amount": "amount", "adj_factor": "adjfactor",
        "circ_mv": "float_cap", "total_mv": "tot_cap",
        "turnover_rate": "turnover_rate",
    }
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    df = df.rename(columns=rename)

    # 构建 MultiIndex
    df["datetime"] = pd.to_datetime(df["trade_date"])
    df["instrument"] = df["ts_code"].astype(str)
    df = df.set_index(["datetime", "instrument"])

    # 衍生列（因子表达式需要）
    df["adj_close"] = df["close"] * df["adjfactor"]
    df["adj_open"] = df["open"] * df["adjfactor"]
    df["adj_low"] = df["low"] * df["adjfactor"]
    df["adj_high"] = df["high"] * df["adjfactor"]
    df["adj_vwap"] = (df["amount"] / df["volume"].replace(0, np.nan)) * df["adjfactor"]

    # downcast float64 → float32 节省内存
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and col not in ("is_st",):
            df[col] = df[col].astype("float32")

    df = df.sort_index()
    print(f"panel 加载完成: {df.shape[0]:,} 行 × {df.shape[1]} 列", flush=True)
    return df


def _sample_panel(panel: pd.DataFrame, n_dates: int = 40, block_days: int = 20) -> pd.DataFrame:
    """从 panel 中按日分层抽样，减少计算量。

    随机选 n_dates 个锚点日，每个锚点向前取 block_days 个连续交易日，
    保留约 n_dates * block_days ≈ 800 天的数据（≈ 800 * 5000 = 4M 行 → 1/3.6）。
    """
    dates = panel.index.get_level_values("datetime").unique().sort_values()
    if len(dates) <= n_dates * block_days:
        return panel
    rng = np.random.default_rng(42)
    anchors = rng.choice(
        np.arange(block_days - 1, len(dates)),
        size=min(n_dates, len(dates) - block_days + 1),
        replace=False,
    )
    selected: set[pd.Timestamp] = set()
    for anchor in anchors:
        selected.update(dates[max(0, int(anchor) - block_days + 1):int(anchor) + 1])
    sampled = panel[panel.index.get_level_values("datetime").isin(selected)]
    print(f"采样 panel: {sampled.shape[0]:,} 行 ({len(selected)} 天, 原 {len(dates)} 天)", flush=True)
    return sampled


def _eval_factor_on_panel(expr: str, panel: pd.DataFrame) -> np.ndarray | None:
    """在 panel 上求值因子表达式，返回对齐的 float64 数组。"""
    from alphaagent.dsl import eval_factor
    from alphaagent.factor.align import align_series_to_panel
    try:
        raw = eval_factor(expr, panel)
        if not isinstance(raw, pd.Series):
            return None
        aligned = align_series_to_panel(raw, panel)
        return np.asarray(aligned, dtype=np.float64)
    except Exception as exc:
        print(f"  求值失败: {type(exc).__name__}: {exc}", flush=True)
        return None


def _cross_sectional_pearson_mean(a: np.ndarray, b: np.ndarray, index: pd.MultiIndex, min_pairs: int = 30) -> float:
    """逐日横截面 Pearson 相关均值（与 zoo.similarity 口径一致）。

    向量化实现：用 groupby 一次性计算每日均值/协方差，避免逐日 Python 循环。
    """
    dt = index.get_level_values("datetime")
    fa = pd.Series(a, index=index)
    fb = pd.Series(b, index=index)

    # 逐日均值（忽略 NaN）
    def _daily_mean(s: pd.Series) -> pd.Series:
        mask = s.notna()
        return s.where(mask).groupby(level=0).sum() / mask.groupby(level=0).sum()

    # 构建有效掩码
    valid = np.isfinite(a) & np.isfinite(b)
    if int(valid.sum()) < min_pairs:
        return float("nan")

    fa_v = pd.Series(np.where(valid, a, np.nan), index=index)
    fb_v = pd.Series(np.where(valid, b, np.nan), index=index)

    # 逐日 n, sum_x, sum_y, sum_xx, sum_yy, sum_xy
    n_per_day = pd.Series(valid, index=index).groupby(level=0).sum()
    valid_days = n_per_day[n_per_day >= min_pairs].index
    if len(valid_days) == 0:
        return float("nan")

    sx = fa_v.groupby(level=0).sum()
    sy = fb_v.groupby(level=0).sum()
    sxx = (fa_v * fa_v).groupby(level=0).sum()
    syy = (fb_v * fb_v).groupby(level=0).sum()
    sxy = (fa_v * fb_v).groupby(level=0).sum()
    n = n_per_day.astype(float)

    cov = sxy - sx * sy / n
    vx = sxx - sx * sx / n
    vy = syy - sy * sy / n
    denom = np.sqrt(vx * vy)

    corr = pd.Series(np.nan, index=n.index, dtype=float)
    mask = denom > 0
    corr[mask] = cov[mask] / denom[mask]
    corr = corr.loc[valid_days]
    finite = corr[np.isfinite(corr)]
    if len(finite) == 0:
        return float("nan")
    return float(finite.mean())


def main() -> None:
    parser = argparse.ArgumentParser(description="候选因子库去重")
    parser.add_argument("--threshold", type=float, default=0.6, help="截面 |Pearson| 阈值，≥ 此值判定为重复")
    parser.add_argument("--dry-run", action="store_true", help="只输出报告不删除")
    parser.add_argument("--registry", type=str, default=str(CAND_REGISTRY), help="registry JSON 路径")
    args = parser.parse_args()

    registry_path = Path(args.registry)
    if not registry_path.is_file():
        print(f"registry 不存在: {registry_path}")
        sys.exit(1)

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    factor_ids = sorted(registry.keys())
    n = len(factor_ids)
    print(f"候选因子库: {n} 个因子", flush=True)
    if n < 2:
        print("少于 2 个因子，无需去重。")
        return

    # ── 加载 panel 并求值所有因子 ──
    panel = _load_panel()

    values: dict[str, np.ndarray] = {}
    for fid in factor_ids:
        entry = registry[fid]
        expr = str(entry.get("expr") or "").strip()
        if not expr:
            print(f"  {fid}: 无表达式，跳过", flush=True)
            continue
        print(f"  求值 {fid} ({factor_ids.index(fid)+1}/{len(factor_ids)}) …", flush=True)
        import time as _time
        t0 = _time.perf_counter()
        val = _eval_factor_on_panel(expr, panel)
        elapsed = _time.perf_counter() - t0
        if val is not None:
            values[fid] = val
            print(f"    完成 {elapsed:.1f}s, finite={np.isfinite(val).sum():,}", flush=True)
        else:
            print(f"    求值失败 ({elapsed:.1f}s)", flush=True)

    evaluated = sorted(values.keys())
    print(f"\n成功求值: {len(evaluated)}/{n} 个因子", flush=True)

    # ── 两两计算截面 Pearson 相关 ──
    print(f"\n计算两两截面 Pearson 相关（阈值={args.threshold}）…", flush=True)
    corr_matrix: dict[tuple[str, str], float] = {}
    for i, fid_a in enumerate(evaluated):
        for fid_b in evaluated[i + 1:]:
            corr = _cross_sectional_pearson_mean(values[fid_a], values[fid_b], panel.index)
            corr_matrix[(fid_a, fid_b)] = corr
            if np.isfinite(corr) and abs(corr) >= args.threshold:
                ic_a = abs(float(registry[fid_a].get("metrics", {}).get("ic", 0)))
                ic_b = abs(float(registry[fid_b].get("metrics", {}).get("ic", 0)))
                print(f"  ⚠ {fid_a:35s} ↔ {fid_b:35s}  corr={corr:+.4f}  IC_a={ic_a:.4f} IC_b={ic_b:.4f}", flush=True)

    # ── 识别要删除的因子 ──
    # 策略：对每对高相关因子，删除 |IC| 较小的那个（指标更差的）
    # 如果 |IC| 相同，删除后加入的（ingested_at 更晚的）
    to_delete: set[str] = set()
    high_corr_pairs = [
        (fid_a, fid_b, corr)
        for (fid_a, fid_b), corr in corr_matrix.items()
        if np.isfinite(corr) and abs(corr) >= args.threshold
    ]
    # 按相关性从高到低排序
    high_corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print(f"\n高重复因子对: {len(high_corr_pairs)} 对", flush=True)
    for fid_a, fid_b, corr in high_corr_pairs:
        ic_a = abs(float(registry[fid_a].get("metrics", {}).get("ic", 0)))
        ic_b = abs(float(registry[fid_b].get("metrics", {}).get("ic", 0)))
        # 保留指标更好的
        if ic_a >= ic_b:
            keep, drop = fid_a, fid_b
        else:
            keep, drop = fid_b, fid_a
        # 如果 keep 已经被删除了，找另一个可保留的
        if keep in to_delete and drop not in to_delete:
            keep, drop = drop, keep
        if drop not in to_delete and keep not in to_delete:
            to_delete.add(drop)
            print(f"  删除 {drop:35s} (保留 {keep}, corr={corr:+.4f})", flush=True)
        elif drop not in to_delete and keep in to_delete:
            # keep 已被标记删除，但 drop 也不能保留（因为和已删除的 keep 高相关）
            # 检查 drop 是否和其他保留因子也高相关
            to_delete.add(drop)
            print(f"  删除 {drop:35s} (连带删除，与 {keep} corr={corr:+.4f})", flush=True)

    if not to_delete:
        print("\n未发现重复因子，无需去重。")
        return

    print(f"\n共删除 {len(to_delete)} 个冗余因子: {sorted(to_delete)}", flush=True)
    print(f"保留 {n - len(to_delete)} 个因子", flush=True)

    if args.dry_run:
        print("\n--dry-run 模式，不执行删除。")
        return

    # ── 执行删除 ──
    # 从 registry 删除
    for fid in to_delete:
        entry = registry.pop(fid, None)
        # 删除 DSL 文件
        rel = str((entry or {}).get("expression_file") or "")
        dsl_path = ROOT / rel if rel else ROOT / "artifacts" / "alphaagent" / "factorzoo" / "candidate_technical" / "expressions" / f"{fid}.dsl"
        if dsl_path.exists():
            dsl_path.unlink()
            print(f"  删除文件: {dsl_path}", flush=True)

    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nregistry 已更新: {registry_path} ({len(registry)} 个因子)", flush=True)

    # ── 清理研究记忆中的残留 ──
    try:
        from alphaagent.factor.mining.research_memory import ResearchMemoryStore
        mem_path = ROOT / "artifacts" / "alphaagent" / "research_memory.db"
        if mem_path.exists():
            purged = ResearchMemoryStore(mem_path).purge_factor(
                factor_names=[registry.get(fid, {}).get("name", fid) for fid in to_delete],
                expressions=[],
            )
            print(f"研究记忆清理: {purged}", flush=True)
    except Exception as exc:
        print(f"研究记忆清理跳过: {exc}", flush=True)

    print("\n去重完成。", flush=True)


if __name__ == "__main__":
    main()
