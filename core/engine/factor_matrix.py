"""阶段 2 因子矩阵：财务帧、因子构建、pred 注入、composite 与中性化。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .factors import (
    build_factor_frames,
    _inject_pred_factor,
    _ensure_ma_cross_factor,
    build_composite_factor,
)
from ..screener import screen_factors, ScreenerConfig


def _build_factor_matrix(cfg, prep: dict) -> dict:
    """阶段 2 因子矩阵:财务帧、因子构建、pred 注入、composite 与中性化。"""
    close = prep["close"]
    am20 = prep["am20"]
    turn20 = prep["turn20"]
    volume_w = prep["volume_w"]
    cal = prep["cal"]
    codes_used = prep["codes_used"]
    profile = prep["profile"]
    factor = cfg.factor
    factor_weights = cfg.factor_weights
    factor_directions = cfg.factor_directions
    external_scores = cfg.external_scores
    analyze = cfg.analyze
    industry_map = cfg.industry_map
    risk_neutral = cfg.risk_neutral
    industry_neutral = cfg.industry_neutral
    factor_builder = cfg.factor_builder
    use_financial = cfg.use_financial

    from ..financial import FINANCIAL_FACTORS, financial_factor_frames
    need_financial = use_financial or factor in FINANCIAL_FACTORS
    if factor_weights:
        need_financial = need_financial or any(
            n in FINANCIAL_FACTORS for n in factor_weights)
    financial_frames = None
    if need_financial:
        try:
            financial_frames = financial_factor_frames(codes_used, cal, close)
        except Exception:
            financial_frames = None

    _asset_type = profile.asset_type

    def _default_builder(c: pd.DataFrame, a: pd.DataFrame, t: pd.DataFrame):
        return build_factor_frames(c, a, t, financial=financial_frames,
                                   asset_type=_asset_type, volume=volume_w)

    builder = factor_builder or _default_builder
    factors = builder(close, am20, turn20)
    _inject_pred_factor(factors, close, factor, factor_weights, external_scores)
    _ensure_ma_cross_factor(factors, close, factor)
    if factor_weights:
        # 多因子自由组合：权重合成后的得分矩阵
        combo = build_composite_factor(
            close, am20, turn20, factor_weights, factor_directions,
            factor_builder=builder,
            extra_factors={"pred": factors["pred"]} if "pred" in factors else None,
        )
        fmat = combo.values
        quality = None
        if analyze:
            from ..performance import factor_quality
            quality = factor_quality(combo, close, horizon=20, groups=5, min_n=10)
    else:
        fmat = factors[factor].values
        quality = None
        if analyze:
            from ..performance import factor_quality
            quality = factor_quality(factors[factor], close, horizon=20, groups=5, min_n=10)

    use_screener = prep.get("use_screener", False)

    # Screener 模式：保留全部因子帧供信号日动态评分
    if use_screener:
        screener_cfg = ScreenerConfig(
            lookback=prep.get("screener_lookback", 10),
            min_ic=prep.get("screener_min_ic", 0.02),
            max_corr=prep.get("screener_max_corr", 0.7),
        )
        # 筛选要参与 Screener 的因子（用户可指定子集，否则取 factor_weights 的键）
        screener_factor_names = prep.get("screener_factors") or list(factors.keys())
        screener_frames = {n: factors[n] for n in screener_factor_names
                           if n in factors and n != "composite"}
        signal_indices = prep.get("signal_indices", [])
        close_df = prep["close"]
        high_df = prep.get("high")
        low_df = prep.get("low")
        # 用等权均值近似指数
        index_close_s = close_df.mean(axis=1)
        index_high_s = high_df.mean(axis=1) if high_df is not None else None
        index_low_s = low_df.mean(axis=1) if low_df is not None else None

        # 逐信号日算 Screener → 动态权重 → 合成得分
        combo_arr = np.full_like(close_df.values, np.nan)
        screener_log: list[dict] = []
        for sig_i in signal_indices:
            result = screen_factors(
                screener_frames, close_df, sig_i, screener_cfg,
                index_close=index_close_s,
                index_high=index_high_s,
                index_low=index_low_s,
                all_dates=close_df.index,
            )
            if not result.weights:
                # 没有因子通过，该信号日得分全 NaN（选不出股 → 空仓）
                screener_log.append({
                    "date": str(result.signal_date.date()),
                    "regime": result.regime_label,
                    "selected": [],
                    "rejected": dict(list(result.rejected.items())[:5]),
                    "weights": {},
                })
                continue
            # 用动态权重合成该信号日的截面得分
            row = np.zeros(len(codes_used))
            for fname, w in result.weights.items():
                fmat_row = factors[fname].iloc[sig_i].values
                ascending = result.directions.get(fname, False)
                rank = pd.Series(fmat_row).rank(pct=True).values
                if ascending:
                    rank = 1.0 - rank
                row += w * rank
            combo_arr[sig_i] = row
            screener_log.append({
                "date": str(result.signal_date.date()),
                "regime": result.regime_label,
                "selected": result.selected,
                "factor_ic": {k: round(v, 4) for k, v in result.factor_ic.items()},
                "weights": {k: round(v, 4) for k, v in result.weights.items()},
                "directions": {k: "买低" if v else "买高"
                               for k, v in result.directions.items()},
                "rejected": dict(list(result.rejected.items())[:5]),
            })

        # 填充非信号日的值（用前一个信号日的得分延持到次日执行）
        fmat_frame = pd.DataFrame(combo_arr, index=close_df.index,
                                  columns=codes_used)
        fmat = fmat_frame.values
        quality = None
        if analyze:
            from ..performance import factor_quality
            quality = factor_quality(fmat_frame, close, horizon=20, groups=5, min_n=10)
        return {
            "fmat": fmat,
            "quality": quality,
            "_X_risk": None,
            "_risk_names": [],
            "screener_log": screener_log,
        }

    if factor_weights:
        factor = "composite"

    _X_risk: np.ndarray | None = None
    _risk_names: list[str] = []
    if risk_neutral and industry_map:
        from ..risk_model import (build_exposures, neutralize)
        _X_risk, _risk_names = build_exposures(
            close.values, am20.values, turn20.values,
            mom20=factors.get("mom20").values if "mom20" in factors else None,
            vol20=factors.get("vol20").values if "vol20" in factors else None,
            pb=factors.get("pb").values if "pb" in factors else None,
            roe=factors.get("roe").values if "roe" in factors else None,
            growth=factors.get("rev_yoy").values if "rev_yoy" in factors else None,
            industry_map=industry_map, codes=codes_used,
        )
        fmat = neutralize(np.array(fmat, dtype=float, copy=True), _X_risk)
        fmat_frame = pd.DataFrame(fmat, index=close.index, columns=codes_used)
        if analyze:
            from ..performance import factor_quality
            quality = factor_quality(fmat_frame, close, horizon=20, groups=5, min_n=10)
    elif industry_neutral and industry_map:
        ind_arr = np.array([industry_map.get(str(c), "?") for c in codes_used])
        raw_fmat = np.array(fmat, dtype=float, copy=True)
        for ind in np.unique(ind_arr):
            mask = ind_arr == ind
            if mask.sum() == 0:
                continue
            sub = raw_fmat[:, mask]
            valid_cnt = np.sum(~np.isnan(sub), axis=1, keepdims=True)
            valid_sum = np.nansum(np.where(np.isnan(sub), 0.0, sub), axis=1, keepdims=True)
            row_means = np.divide(valid_sum, valid_cnt,
                                  out=np.full_like(valid_sum, np.nan),
                                  where=valid_cnt > 0)
            raw_fmat[:, mask] = sub - row_means
        fmat = raw_fmat

    return {
        "fmat": fmat,
        "quality": quality,
        "_X_risk": _X_risk,
        "_risk_names": _risk_names,
        "screener_log": [],
    }
