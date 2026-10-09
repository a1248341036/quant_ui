"""ML 组合 score 的风险模型暴露画像 + 中性化 IC 对比（只读分析，不入库）。

目的：回答"ML 组合因子的收益里有多少是已知风险（风格/行业）的再包装"。
方法（对应 docs 量化中级进阶调研报告第三章 + core/risk_model.py 轻量 Barra）：

1. 暴露画像：把 ML score 每期截面 z-score 后对 Barra 暴露矩阵 [1, X] 逐期回归。
   X 的风格列已 z-score，回归系数 ≈ score 与该风格的截面相关；
   R² = 暴露能解释的 score 截面方差比例。分别报告 style-only / industry-only / full。
2. 中性化 IC 对比：score 用 core.risk_model.neutralize 对暴露取残差后重算
   N 日 Rank IC（label 口径与 stacking 训练一致：T+1 收盘进、T+1+label_days 收盘出）。
   中性化后 IC 掉多少，即风格暴露贡献了多少"alpha"。

边界（诚实声明）：
- 风格暴露是代理定义（am20 代理流动性/规模，mom20/vol20/turn20），非 CNE5 标准因子；
- 协方差/回归用逐期截面，不做时变协方差；本分析是期均画像，不能当逐日风控用；
- 只读：不写因子库、不改任何链路状态。

用法：
  .venv/Scripts/python.exe scripts/analyze_ml_score_exposure.py --run 20261008_203807
"""
from __future__ import annotations

import argparse
import sys
import warnings
from datetime import timedelta
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
from core.risk_model import build_exposures, neutralize  # noqa: E402


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", default="20261008_203807",
                    help="stacking run id（scores.parquet 所在目录名）")
    ap.add_argument("--scores", default=None,
                    help="直接指定 scores.parquet 路径（覆盖 --run）")
    ap.add_argument("--label-days", type=int, default=5,
                    help="前向收益持有天数（对齐 ML 训练 label 口径，默认 5）")
    ap.add_argument("--window", type=int, default=20,
                    help="风格暴露滚动窗口（交易日，默认 20，与 risk_model 风格口径一致）")
    ap.add_argument("--out", default=None, help="md 报告输出路径（默认 docs/review/ 下自动命名）")
    return ap.parse_args()


def _load_score_wide(path: Path) -> pd.DataFrame:
    """长表 date/code/score → 宽表 index=date, columns=code。"""
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df["code"] = df["code"].astype(str).str.split(".").str[0].str.zfill(6)
    wide = df.pivot_table(index="date", columns="code", values="score", aggfunc="last")
    return wide.sort_index()


def _pivot_panel(panel: pd.DataFrame, col: str) -> pd.DataFrame:
    s = panel[col].dropna()
    s.index = s.index.set_levels([
        pd.to_datetime(s.index.levels[0]).normalize(),
        s.index.levels[1].astype(str).str.split(".").str[0].str.zfill(6),
    ])
    return s.unstack(level=1).sort_index()


def build_style_matrices(panel: pd.DataFrame, window: int) -> dict[str, pd.DataFrame]:
    """从 panel 构建风格代理矩阵（与 core/risk_model.build_exposures 输入口径对齐）。

    各列 dropna 后 unstack 的股票集合可能不一致（如停牌股 amount 全缺），
    统一 reindex 到 close 的 index/columns 保证矩阵同形。
    """
    close = _pivot_panel(panel, "close")

    def _align(m: pd.DataFrame) -> pd.DataFrame:
        return m.reindex(index=close.index, columns=close.columns)

    amount = _align(_pivot_panel(panel, "amount"))
    turn = _align(_pivot_panel(panel, "turnover_rate"))
    ret = close.pct_change(fill_method=None)
    return {
        "close": close,
        "am20": amount.rolling(window, min_periods=max(5, window // 2)).mean(),
        "turn20": turn.rolling(window, min_periods=max(5, window // 2)).mean(),
        "mom20": close / close.shift(window) - 1.0,
        "vol20": ret.rolling(window, min_periods=max(5, window // 2)).std(),
    }


def _industry_map_from_panel(panel: pd.DataFrame) -> dict[str, str]:
    """每股取最后一个非空 industry_sw_l1（PIT 快照的期末截面近似）。"""
    if "industry_sw_l1" not in panel.columns:
        return {}
    ind = _pivot_panel(panel, "industry_sw_l1")
    last = ind.ffill().iloc[-1].dropna()
    return {str(c): str(int(v)) for c, v in last.items() if pd.notna(v)}


def _zscore_rows(mat: np.ndarray) -> np.ndarray:
    """逐行截面 z-score；全 NaN 行保留 NaN。"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # 全 NaN 行（预热段）触发 empty-slice
        mu = np.nanmean(mat, axis=1, keepdims=True)
        sd = np.nanstd(mat, axis=1, keepdims=True)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (mat - mu) / sd


def _ols_cross_section(y: np.ndarray, Xt: np.ndarray) -> tuple[np.ndarray | None, float]:
    """单期截面回归 y ~ [1, Xt]，返回 (coef, r2)。有效样本不足返回 (None, nan)。"""
    valid = np.isfinite(y) & np.isfinite(Xt).all(axis=1)
    n = int(valid.sum())
    p = Xt.shape[1]
    if n < p + 5:
        return None, float("nan")
    Xv = np.column_stack([np.ones(n), Xt[valid]])
    yv = y[valid]
    coef, *_ = np.linalg.lstsq(Xv, yv, rcond=None)
    resid = yv - Xv @ coef
    ss_res = float(resid @ resid)
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-18 else float("nan")
    return coef, r2


def exposure_profile(S: np.ndarray, X: np.ndarray, dates: pd.DatetimeIndex,
                     names: list[str], n_style: int, score_mask: np.ndarray) -> dict:
    """逐期回归画像。返回每因子相关序列、三组 R² 序列（score 有效日期段）。"""
    S_z = _zscore_rows(S)
    T, K, P = X.shape
    coef = np.full((T, P), np.nan)
    r2_full = np.full(T, np.nan)
    r2_style = np.full(T, np.nan)
    r2_ind = np.full(T, np.nan)
    for t in range(T):
        if not score_mask[t]:
            continue
        Xfull = X[t]
        c, r = _ols_cross_section(S_z[t], Xfull)
        if c is not None:
            coef[t] = c[1:]
            r2_full[t] = r
        _, r2_style[t] = _ols_cross_section(S_z[t], Xfull[:, :n_style])
        _, r2_ind[t] = _ols_cross_section(S_z[t], Xfull[:, n_style:])
    return {
        "coef": pd.DataFrame(coef, index=dates, columns=names),
        "r2_full": pd.Series(r2_full, index=dates).dropna(),
        "r2_style": pd.Series(r2_style, index=dates).dropna(),
        "r2_ind": pd.Series(r2_ind, index=dates).dropna(),
    }


def daily_spearman_ic(score_w: pd.DataFrame, label_w: pd.DataFrame) -> pd.Series:
    """逐日截面 Spearman IC（rank 后逐行 Pearson）。"""
    rs = score_w.rank(axis=1)
    rl = label_w.rank(axis=1)
    return rs.corrwith(rl, axis=1).dropna()


def label_forward(close: pd.DataFrame, hold: int) -> pd.DataFrame:
    """T+1 收盘进、T+1+hold 收盘出的 N 日收益（与 stacking dataset 口径一致）。"""
    return close.shift(-(hold + 1)) / close.shift(-1) - 1.0


def _monthly(s: pd.Series) -> pd.Series:
    return s.groupby(s.index.to_period("M")).mean()


def _fmt(x: float, nd: int = 4) -> str:
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def main() -> None:
    args = _parse_args()
    scores_path = Path(args.scores) if args.scores else \
        ROOT / "artifacts" / "alphaagent" / "stacking" / args.run / "scores.parquet"
    if not scores_path.is_file():
        raise FileNotFoundError(f"scores 不存在: {scores_path}")

    score_w = _load_score_wide(scores_path)
    d0, d1 = score_w.index.min(), score_w.index.max()
    print(f"[1/5] score: {score_w.shape[0]} 日 × {score_w.shape[1]} 股 "
          f"({d0.date()} ~ {d1.date()})，来源 {scores_path}")

    # panel：score 段 + 前置预热（风格窗口）+ 后置 label 期
    pad_front = timedelta(days=max(45, args.window * 2))
    pad_back = timedelta(days=(args.label_days + 1) * 2 + 5)
    panel = load_panel_from_cne(
        start=str((d0 - pad_front).date()), end=str((d1 + pad_back).date()),
        include_fundamentals=False,
    )
    print(f"[2/5] panel: {panel.shape[0]} 行 × {panel.shape[1]} 列")

    mats = build_style_matrices(panel, args.window)
    close = mats["close"]
    ind_map = _industry_map_from_panel(panel)
    codes = list(close.columns)

    X, names = build_exposures(
        close.values, mats["am20"].values, mats["turn20"].values,
        mom20=mats["mom20"].values, vol20=mats["vol20"].values,
        industry_map=ind_map or None, codes=codes,
    )
    n_style = sum(1 for n in names if not n.startswith("ind_"))
    print(f"[3/5] 暴露矩阵: T={X.shape[0]} K={X.shape[1]} P={X.shape[2]} "
          f"（风格 {n_style} + 行业 {X.shape[2] - n_style}；行业映射覆盖 {len(ind_map)} 股）")

    dates = close.index
    S = score_w.reindex(index=dates, columns=codes).values.astype(float)
    score_mask = np.isfinite(S).sum(axis=1) >= 50

    profile = exposure_profile(S, X, dates, names, n_style, score_mask)
    style_names = names[:n_style]
    coef_df = profile["coef"]
    c_mean, c_std = coef_df.mean(), coef_df.std()
    t_stat = c_mean / (c_std / np.sqrt(coef_df.notna().sum()))

    label_w = label_forward(close, args.label_days)
    S_neut = neutralize(S, X)
    S_neut_w = pd.DataFrame(S_neut, index=dates, columns=codes)
    seg = slice(score_w.index.min(), score_w.index.max())
    ic_raw = daily_spearman_ic(score_w.loc[seg, score_w.columns.intersection(codes)],
                               label_w.loc[seg])
    ic_neut = daily_spearman_ic(S_neut_w.loc[seg], label_w.loc[seg])

    def _ic_stats(ic: pd.Series) -> dict:
        return {"mean": float(ic.mean()), "std": float(ic.std()),
                "icir": float(ic.mean() / ic.std()) if ic.std() > 0 else float("nan"),
                "n": int(len(ic))}

    st_raw, st_neut = _ic_stats(ic_raw), _ic_stats(ic_neut)

    # ---- 控制台摘要 ----
    print("\n[4/5] === 暴露画像（score 与风格的平均截面相关） ===")
    print(f"{'因子':<12}{'mean':>9}{'std':>9}{'t-stat':>9}")
    for n in style_names:
        print(f"{n:<12}{_fmt(c_mean[n]):>9}{_fmt(c_std[n]):>9}{_fmt(t_stat[n], 2):>9}")
    print(f"\nR²：full={_fmt(profile['r2_full'].mean())} "
          f"style-only={_fmt(profile['r2_style'].mean())} "
          f"industry-only={_fmt(profile['r2_ind'].mean())}（期均）")
    print(f"\n[5/5] === IC 对比（{args.label_days} 日，Spearman，n={st_raw['n']} 日） ===")
    print(f"原始:    IC={_fmt(st_raw['mean'])}  ICIR={_fmt(st_raw['icir'])}")
    print(f"中性化:  IC={_fmt(st_neut['mean'])}  ICIR={_fmt(st_neut['icir'])}")
    drop = st_raw["mean"] - st_neut["mean"]
    pct = drop / abs(st_raw["mean"]) if st_raw["mean"] else float("nan")
    print(f"IC 掉幅: {_fmt(drop)}（{_fmt(pct * 100, 1)}% 的 IC 来自风格暴露）")

    # ---- md 报告 ----
    out = Path(args.out) if args.out else \
        ROOT / "docs" / "review" / f"2026-10-10_ml_score_exposure_{args.run}.md"
    lines = [
        f"# ML score 风险暴露画像：{args.run}",
        "",
        f"- 生成：2026-10-10；脚本 `scripts/analyze_ml_score_exposure.py`（分支 "
        f"`feat/ml-score-risk-exposure`，只读分析）",
        f"- score：`{scores_path.name}`，{score_w.shape[0]} 日 × {score_w.shape[1]} 股"
        f"（{d0.date()} ~ {d1.date()}，盲测段 OOS）",
        f"- 暴露：`core/risk_model.build_exposures`，风格代理 "
        f"（liquidity=am20/momentum=mom20/volatility=vol20/turnover=turn20，"
        f"窗口 {args.window} 日）+ 申万一级行业哑变量（panel `industry_sw_l1`）",
        f"- label：{args.label_days} 日（T+1 收盘进、T+1+{args.label_days} 收盘出，"
        f"与 stacking 训练口径一致）",
        "",
        "## 1. 暴露画像（score_z 对暴露逐期回归的系数 ≈ 平均截面相关）",
        "",
        "| 风格 | mean | std | t-stat |",
        "|---|---|---|---|",
    ]
    for n in style_names:
        lines.append(f"| {n} | {_fmt(c_mean[n])} | {_fmt(c_std[n])} | {_fmt(t_stat[n], 2)} |")
    lines += [
        "",
        f"- 行业映射覆盖 {len(ind_map)} 股（其余入\"其他\"组）",
        f"- 期均 R²：**full {_fmt(profile['r2_full'].mean())}** / "
        f"style-only {_fmt(profile['r2_style'].mean())} / "
        f"industry-only {_fmt(profile['r2_ind'].mean())}",
        f"- 解读：full R² 为暴露能解释的 score 截面方差比例，"
        f"1 − full 即特质（无法被已知风险解释）部分",
        "",
        "## 2. 中性化前后 IC 对比",
        "",
        "| 口径 | mean IC | ICIR | n 日 |",
        "|---|---|---|---|",
        f"| 原始 score | {_fmt(st_raw['mean'])} | {_fmt(st_raw['icir'])} | {st_raw['n']} |",
        f"| 风格+行业中性化 | {_fmt(st_neut['mean'])} | {_fmt(st_neut['icir'])} | {st_neut['n']} |",
        "",
        f"**IC 掉幅 {_fmt(drop)}（{_fmt(pct * 100, 1)}%）**——"
        f"即当前 IC 中约该比例由风格/行业暴露贡献。",
        "",
        "## 3. 逐月 IC（原始 → 中性化）",
        "",
        "| 月份 | IC 原始 | IC 中性化 |",
        "|---|---|---|",
    ]
    m_raw, m_neut = _monthly(ic_raw), _monthly(ic_neut)
    for m in m_raw.index:
        lines.append(f"| {m} | {_fmt(m_raw[m])} | {_fmt(m_neut.get(m, np.nan))} |")
    lines += [
        "",
        "## 边界",
        "",
        "- 风格为代理定义（非 CNE5 标准），暴露 R²/相关是**下限估计**——"
        "更细的风格定义可能解释更多",
        "- 期均画像，非逐日风险预测；协方差收缩路径未参与本分析"
        "（画像只做截面回归，不估 Σ）",
        "- score 为 OOS 盲测段产物，本分析不构成对训练段的评估",
        "",
        "---",
        "",
        "> 结论模板（跑完后填写）：中性化后 IC 掉幅小 → alpha 干净，"
        "组合层风险分解（路 B）可省；掉幅大 → score 显著暴露于风格/行业，"
        "建议做路 B 定量分解并考虑把暴露约束纳入 ML 特征或组合层。",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n报告已写入: {out}")


if __name__ == "__main__":
    main()
