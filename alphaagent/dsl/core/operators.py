"""股票 DSL 算子：时序算子 per instrument、截面算子 per datetime；Window 为整数固定窗或单列 DataFrame 动态窗/滞后。"""
from __future__ import annotations

from typing import Any, Callable, Union

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from .dyn_window import (
    arg_extreme_dynamic,
    arg_extreme_fixed,
    rolling_dynamic,
    shift_dynamic,
)
from . import accel as _accel
from . import chip_daily as _chip_daily
from .ops_kit import (
    Window,
    as_int_window as _as_int_window,
    dynamic_window_int_series as _dynamic_window_int_series,
    first_series as _first_series,
    gb_instrument as _gb_instrument,
    is_dynamic_window as _is_dynamic_window,
    lag_int_series as _lag_int_series,
    out_frame as _out_frame,
    per_datetime_transform as _per_datetime_transform,
    per_instrument_bivariate as _ts_bivariate_event_accel,
    per_instrument_unary as _ts_unary_accel,
    series_from_group as _series_from_group,
)

# TS_CROSS_* 第二 operand 可为与之对齐的面板单列，或与 ADD 二元算子一致的 Python / NumPy 标量（按 x 索引广播）
TsCrossOperand = Union[pd.DataFrame, int, float, np.integer, np.floating]


def _as_ts_cross_y_panel(x: pd.DataFrame, y: TsCrossOperand) -> pd.DataFrame:
    if isinstance(y, pd.DataFrame):
        return y
    col = x.columns[:1]
    try:
        v = float(y)
    except (TypeError, ValueError) as e:
        raise TypeError(
            "TS_CROSS_ABOVE/BELOW 的 y 须为单列 DataFrame 或与 rank 对齐可 float() 的标量"
        ) from e
    return pd.DataFrame(v, index=x.index, columns=col)


def _chip_wass_win_series(w: Window, index: pd.Index) -> np.ndarray:
    """筹码窗长：整数或单列动态窗 → 每 bar 整数长度 ≥1。"""
    if _is_dynamic_window(w):
        return _dynamic_window_int_series(w.reindex(index), index)
    return np.full(len(index), max(1, int(w)), dtype=np.int64)


def _chip_wass_rho_series(rho_w: Window, index: pd.Index) -> np.ndarray:
    """参照窗右端偏移 ρ（≥0）：整数滞后或动态单列面板。"""
    if _is_dynamic_window(rho_w):
        return _lag_int_series(rho_w.reindex(index), index)
    return np.full(len(index), max(0, int(rho_w)), dtype=np.int64)


def _ts_agg_fixed_accel(df: pd.DataFrame, window: int, kind: str, ddof: int = 1) -> pd.DataFrame:
    """固定窗滚动聚合，使用 C++ 或 Numba 加速后端。
    
    Args:
        kind: "mean", "std", "sum", "min", "max", "rank_pct", "median", "var", "skew", "kurt", "prod"
    """
    kind_map = {
        "mean": "mean",
        "std": "std",
        "sum": "sum",
        "min": "min",
        "max": "max",
        "rank_pct": "rank_pct",
        "median": "median",
        "var": "var",
        "skew": "skew",
        "kurt": "kurt",
        "prod": "prod",
    }
    accel_kind = kind_map.get(kind)
    if accel_kind is None:
        raise ValueError(f"Unknown kind: {kind}")

    def _roll_accelerated(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float, copy=False)
        out = _accel.roll_fixed(vals, window, accel_kind, ddof=ddof)
        return pd.Series(out, index=s.index)

    ser = _gb_instrument(df).transform(lambda x: _roll_accelerated(_series_from_group(x)))
    return _out_frame(ser, df)


def _ts_agg(
    df: pd.DataFrame,
    window: Window,
    agg_fixed: Callable[[int], pd.DataFrame],
    *,
    dyn_kind: str,
    ddof: int = 1,
) -> pd.DataFrame:
    if _is_dynamic_window(window):
        return rolling_dynamic(df, window, _dynamic_window_int_series, dyn_kind, ddof=ddof)
    # Use accelerated backend for fixed windows
    return _ts_agg_fixed_accel(df, _as_int_window(window), dyn_kind, ddof=ddof)


def _shift_fixed(df: pd.DataFrame, periods: int) -> pd.DataFrame:
    p = int(periods)
    if p < 0:
        raise ValueError("DELAY 的周期数不能为负")

    def _shift_accelerated(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float, copy=False)
        out = _accel.shift_fixed(vals, p)
        return pd.Series(out, index=s.index)

    return _gb_instrument(df).transform(lambda x: _shift_accelerated(_series_from_group(x)))


def _shift_dynamic(df: pd.DataFrame, lag_df: pd.DataFrame) -> pd.DataFrame:
    """每行延迟 lag[i] 根（同 instrument 内整数位移；按时间序而非展平下标）。"""
    return shift_dynamic(df, lag_df, _lag_int_series)


def _binary_op_panel_df(df1: pd.DataFrame, df2: pd.DataFrame, ufunc) -> pd.DataFrame:
    if (
        df1.shape == df2.shape
        and df1.index.equals(df2.index)
        and df1.shape[1] == 1
        and df2.shape[1] == 1
        and list(df1.columns) != list(df2.columns)
    ):
        out = ufunc(
            df1.iloc[:, 0].to_numpy(dtype=float, copy=False),
            df2.iloc[:, 0].to_numpy(dtype=float, copy=False),
        )
        return pd.DataFrame(out, index=df1.index, columns=df1.columns[:1])
    return ufunc(df1, df2)


def _is_ufunc_broadcast_scalar(x: object) -> bool:
    """二元逐元素算子：与面板混算时允许的标量（含 numpy 标量，不含 str / 非标量数组）。"""
    if x is None:
        return False
    if isinstance(x, (str, bytes, pd.DataFrame, pd.Series)):
        return False
    if isinstance(x, (bool, int, float, np.integer, np.floating, np.bool_)):
        return True
    if isinstance(x, np.ndarray) and x.shape == ():
        return True
    return bool(np.isscalar(x))


def _broadcast_ufunc_panel_scalar(
    df: pd.DataFrame,
    scalar: object,
    ufunc,
    *,
    scalar_left: bool,
) -> pd.DataFrame:
    vals = df.to_numpy(dtype=float, copy=False)
    s = float(np.asarray(scalar, dtype=np.float32))
    res = ufunc(s, vals) if scalar_left else ufunc(vals, s)
    return pd.DataFrame(res, index=df.index, columns=df.columns)


def _binary_op_panel_mixed(a: object, b: object, ufunc) -> pd.DataFrame:
    """与 ``ADD`` 类似：双面板走 ``_binary_op_panel_df``；任一侧为标量则向面板广播。"""
    if isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame):
        return _binary_op_panel_df(a, b, ufunc)
    if isinstance(a, pd.DataFrame) and _is_ufunc_broadcast_scalar(b):
        return _broadcast_ufunc_panel_scalar(a, b, ufunc, scalar_left=False)
    if isinstance(b, pd.DataFrame) and _is_ufunc_broadcast_scalar(a):
        return _broadcast_ufunc_panel_scalar(b, a, ufunc, scalar_left=True)
    out = ufunc(a, b)
    if not isinstance(out, pd.DataFrame):
        raise TypeError(
            "二元算子需要至少一侧为 DataFrame 或双方为可 ufunc 的标量数组，得到 %s" % type(out).__name__
        )
    return out


# -----------------------------------------------------------------------------
# 基础时序
# -----------------------------------------------------------------------------


def DELTA(df: pd.DataFrame, p: int = 1) -> pd.DataFrame:
    """同品种差分 diff(p)；df 单列面板，p 为正整数步长。
    优先使用 C++ 加速，否则回退到 pandas。"""
    p = int(p)
    
    def _delta_accelerated(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float, copy=False)
        out = _accel.delta(vals, p)
        return pd.Series(out, index=s.index)
    
    return _gb_instrument(df).transform(lambda x: _delta_accelerated(_series_from_group(x)))


def DELAY(df: pd.DataFrame, p: Window) -> pd.DataFrame:
    """滞后 shift；p 为 int 根数或非负，或为与 df 对齐的单列 DataFrame 表示逐行动态滞后。"""
    if _is_dynamic_window(p):
        return _shift_dynamic(df, p)
    return _shift_fixed(df, _as_int_window(p))


def TS_PCTCHANGE(df: pd.DataFrame, p: int = 1) -> pd.DataFrame:
    """相对 p 根前的涨跌幅；±inf 置 NaN。
    优先使用 C++ 加速，否则回退到 pandas。"""
    p = int(p)
    
    def _pctchange_accelerated(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float, copy=False)
        out = _accel.pctchange(vals, p)
        return pd.Series(out, index=s.index)
    
    return _gb_instrument(df).transform(lambda x: _pctchange_accelerated(_series_from_group(x)))


def TS_CUMPROD(
    df: pd.DataFrame,
    base: Union[int, float] = 1.0,
) -> pd.DataFrame:
    """逐品种从开始到当前的**连乘** ``cumprod``（非滚动窗）。

    常用于把收益率序列还原为指数**点位**：
    例如 ``TS_CUMPROD(ADD(1, $ret), 100)`` 将日收益还原为指数点位；
    即 ``base × ∏_{s≤t}(1+R_s)``；``NaN`` 因子按 **1** 跳过（该步不改变累计积）。

    第二参 ``base`` 为初始尺度（如 ``100`` 表示基点 100 起算）。
    """
    scale = float(base)

    def _cumprod_skip_nan(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=np.float32, copy=False)
        acc = scale
        out = np.empty(len(vals), dtype=np.float32)
        for i, v in enumerate(vals):
            if np.isfinite(v):
                acc *= v
            out[i] = acc
        return pd.Series(out, index=s.index, dtype=np.float32)

    return _gb_instrument(df).transform(
        lambda x: _cumprod_skip_nan(_series_from_group(x))
    )


def EMA(df: pd.DataFrame, p: Window) -> pd.DataFrame:
    """EWM，span=p（须可转 int，勿传动态 DataFrame）。
    优先使用 C++ 加速，否则回退到 pandas。"""
    span = _as_int_window(p)
    
    # Use C++ or Numba accelerated backend
    def _ema_accelerated(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float, copy=False)
        out = _accel.ema(vals, span)
        return pd.Series(out, index=s.index)
    
    return _gb_instrument(df).transform(lambda x: _ema_accelerated(_series_from_group(x)))


def WMA(df: pd.DataFrame, p: int = 20) -> pd.DataFrame:
    """线性加权均线，近端权重大，窗口 p 根。
    优先使用 C++ 加速，否则回退到 pandas。"""
    p = max(1, int(p))
    
    # Use C++ or Numba accelerated backend
    def _wma_accelerated(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float, copy=False)
        out = _accel.wma(vals, p)
        return pd.Series(out, index=s.index)
    
    ser = _gb_instrument(df).transform(lambda x: _wma_accelerated(_series_from_group(x)))
    return _out_frame(ser, df)


def SMA(df: pd.DataFrame, m: Optional[float] = None, n: Optional[float] = None) -> pd.DataFrame:
    """SMA(df,m) 滚动均线；SMA(df,m,n) 为 alpha=n/m 的 ewm。"""
    if isinstance(m, (int, float)) and m is not None and n is None:
        w = int(m)

        def _sma_mean(s: pd.Series) -> pd.Series:
            vals = s.to_numpy(dtype=float, copy=False)
            out = _accel.roll_fixed(vals, w, "mean")
            return pd.Series(out, index=s.index)

        return _gb_instrument(df).transform(lambda x: _sma_mean(_series_from_group(x)))
    if m is None or n is None:
        raise ValueError("SMA 请使用 SMA(df, m) 指定整数均线周期，或提供 (m,n) 自定义递推")
    alpha = float(n) / float(m)
    return _gb_instrument(df).transform(lambda x: x.ewm(alpha=alpha, adjust=False).mean())


def ABS(df: pd.DataFrame) -> pd.DataFrame:
    """逐元素绝对值。"""
    return df.abs()


def SIGN(df: pd.DataFrame) -> pd.DataFrame:
    """逐元素符号。"""
    return np.sign(df)


def NEG(df: pd.DataFrame) -> pd.DataFrame:
    """逐元素取负；等价 ``MULTIPLY(df, -1)``。"""
    return -df


def _cond_truthy_mask(cond: pd.DataFrame) -> np.ndarray:
    """与 ``TS_SINCE`` 一致：有限且非零为真。"""
    c = cond.iloc[:, 0].to_numpy(dtype=float, copy=False)
    return np.isfinite(c) & (c != 0.0)


def _if_then_else_operand_array(template: pd.DataFrame, operand: object, arg_name: str) -> np.ndarray:
    if isinstance(operand, pd.DataFrame):
        op = operand
        if op.shape[1] != 1:
            op = op.iloc[:, :1]
        if not op.index.equals(template.index):
            op = op.reindex(template.index)
        return op.iloc[:, 0].to_numpy(dtype=float, copy=False)
    if _is_ufunc_broadcast_scalar(operand):
        return np.full(len(template), float(np.asarray(operand, dtype=np.float32)), dtype=np.float32)
    raise TypeError(f"IF_THEN_ELSE {arg_name} 须为面板 DataFrame 或数值标量，收到: {type(operand).__name__}")


def IF_THEN_ELSE(cond: pd.DataFrame, then_val: object, else_val: object = 0.0) -> pd.DataFrame:
    """条件选择：``cond`` 为真取 ``then_val``，否则取 ``else_val``（默认 0）。

    ``cond`` 须为单列面板；**真** = 有限且 ≠0（与 ``TS_SINCE`` 一致），比较结果请先 ``CAST(..., 'float64')``。
    ``then_val`` / ``else_val`` 可为同索引单列面板或数值标量。
    """
    if not isinstance(cond, pd.DataFrame):
        raise TypeError("IF_THEN_ELSE 第一参数 cond 须为面板 DataFrame")
    if cond.shape[1] < 1:
        raise ValueError("IF_THEN_ELSE cond 须至少一列")
    mask = _cond_truthy_mask(cond if cond.shape[1] == 1 else cond.iloc[:, :1])
    then_arr = _if_then_else_operand_array(cond, then_val, "then_val")
    else_arr = _if_then_else_operand_array(cond, else_val, "else_val")
    out = np.where(mask, then_arr, else_arr)
    return pd.DataFrame(out, index=cond.index, columns=cond.columns[:1], dtype=np.float32)


def FILLNA(df: pd.DataFrame, value: float = 0.0) -> pd.DataFrame:
    """非有限值（NaN/±inf）替换为 ``value``（默认 0）；有限值不变。

    常用于滚动暖启动导致的缺失：在**最终因子**上 ``FILLNA(expr, 0)`` 表示缺失 bar
    不参与排序但覆盖率按 0 计入。
    """
    v = float(value)
    out = df.astype(float).copy()
    return out.where(np.isfinite(out), other=v)


def CAST(df: pd.DataFrame, dtype: str) -> pd.DataFrame:
    """逐元素 ``astype``；常用于将比较/逻辑得到的 bool 面板转为 float 再参与算术。"""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("CAST 第一参数须为面板 DataFrame")
    d = str(dtype).strip()
    if len(d) >= 2 and d[0] == d[-1] and d[0] in "'\"":
        d = d[1:-1]
    return df.astype(d)


def LOG(df: pd.DataFrame) -> pd.DataFrame:
    """自然对数；0 先变 NaN。"""
    return np.log(df.replace(0, np.nan))


def SQRT(df: pd.DataFrame) -> pd.DataFrame:
    """逐元素平方根。"""
    return np.sqrt(df)


def EXP(df: pd.DataFrame) -> pd.DataFrame:
    """逐元素 exp。"""
    return np.exp(df)


def INV(df: pd.DataFrame) -> pd.DataFrame:
    """逐元素倒数；0 变 NaN。"""
    return 1.0 / df.replace(0, np.nan)


def POW(df: pd.DataFrame, n: float) -> pd.DataFrame:
    """逐元素 x**n。"""
    return np.power(df, float(n))


# -----------------------------------------------------------------------------
# TS_* ：固定或动态窗口
# -----------------------------------------------------------------------------


def TS_MIN(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """滚动最小值；window 为 int 或单列 DataFrame 动态窗宽。"""
    return _ts_agg(
        df,
        window,
        lambda w: _gb_instrument(df).transform(
            lambda x: x.rolling(w, min_periods=1).min()
        ),
        dyn_kind="min",
    )


def TS_MAX(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """滚动最大值；window 同 TS_MIN。"""
    return _ts_agg(
        df,
        window,
        lambda w: _gb_instrument(df).transform(
            lambda x: x.rolling(w, min_periods=1).max()
        ),
        dyn_kind="max",
    )


def TS_MEAN(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """滚动均值；window 同 TS_MIN。"""
    return _ts_agg(
        df,
        window,
        lambda w: _gb_instrument(df).transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        ),
        dyn_kind="mean",
    )


def TS_SUM(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """滚动求和；window 同 TS_MIN。"""
    return _ts_agg(
        df,
        window,
        lambda w: _gb_instrument(df).transform(
            lambda x: x.rolling(w, min_periods=1).sum()
        ),
        dyn_kind="sum",
    )


def TS_PROD(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """滚动乘积；window 同 TS_MIN。窗口内 ``NaN`` 按乘法单位元 **1** 参与（不改变乘积）。
    固定窗优先 C++/Numba；与 pandas ``rolling(...).prod()`` 等对 NaN 的约定不同。"""
    return _ts_agg(
        df,
        window,
        lambda w: _gb_instrument(df).transform(
            lambda x: x.rolling(w, min_periods=1).sum()
        ),
        dyn_kind="prod",
    )


def TS_STD(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """滚动样本标准差（ddof=1）；window 同 TS_MIN。"""
    return _ts_agg(
        df,
        window,
        lambda w: _gb_instrument(df).transform(
            lambda x: x.rolling(w, min_periods=1).std()
        ),
        dyn_kind="std",
        ddof=1,
    )


def TS_VAR(df: pd.DataFrame, window: Window, ddof: int = 1) -> pd.DataFrame:
    """滚动方差；window 同 TS_MIN，ddof 默认 1。"""
    return _ts_agg(
        df,
        window,
        lambda w: _gb_instrument(df).transform(
            lambda x: x.rolling(w, min_periods=1).var(ddof=ddof)
        ),
        dyn_kind="var",
        ddof=ddof,
    )


def TS_MEDIAN(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """滚动中位数；window 同 TS_MIN。"""
    return _ts_agg(
        df,
        window,
        lambda w: _gb_instrument(df).transform(
            lambda x: x.rolling(w, min_periods=1).median()
        ),
        dyn_kind="median",
    )


def TS_RANK(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """滚动分位秩 pct=True；window 同 TS_MIN。"""
    return _ts_agg(
        df,
        window,
        lambda w: _gb_instrument(df).transform(
            lambda x: x.rolling(w, min_periods=1).rank(pct=True)
        ),
        dyn_kind="rank_pct",
    )


def TS_SKEW(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """滚动偏度；window 同 TS_MIN。"""
    return _ts_agg(
        df,
        window,
        lambda w: _gb_instrument(df).transform(
            lambda x: x.rolling(w, min_periods=1).skew()
        ),
        dyn_kind="skew",
    )


def TS_KURT(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """滚动峰度（pandas adjusted Fisher–Pearson 超额峰度 G2）；窗宽同 TS_MIN，
    有效样本数 < 4 输出 NaN，窗口内方差为 0 输出 0。"""
    return _ts_agg(
        df,
        window,
        lambda w: _gb_instrument(df).transform(
            lambda x: x.rolling(w, min_periods=1).kurt()
        ),
        dyn_kind="kurt",
    )


def TS_QUANTILE(df: pd.DataFrame, window: int, q: float) -> pd.DataFrame:
    """滚动 q 分位数（线性插值），等价 pandas ``rolling(w, min_periods=1).quantile(q)``。
    q ∈ [0,1]；窗口内全 NaN 输出 NaN。"""
    w = max(1, int(window))
    qf = float(q)

    def _quantile_accelerated(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float, copy=False)
        out = _accel.roll_quantile_fixed(vals, w, qf)
        return pd.Series(out, index=s.index)

    ser = _gb_instrument(df).transform(
        lambda x: _quantile_accelerated(_series_from_group(x))
    )
    return _out_frame(ser, df)


def TS_ZSCORE(df: pd.DataFrame, window: Window, ddof: int = 1) -> pd.DataFrame:
    """滚动 z-score：``(x - TS_MEAN) / TS_STD``；std=0 时输出 NaN，ddof 默认 1。"""
    w = _as_int_window(window)

    def _zscore_accelerated(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float, copy=False)
        m = _accel.roll_fixed(vals, w, "mean")
        sd = _accel.roll_fixed(vals, w, "std", ddof=ddof)
        denom = np.where(sd == 0.0, np.nan, sd)
        out = (vals - m) / denom
        return pd.Series(out, index=s.index)

    ser = _gb_instrument(df).transform(
        lambda x: _zscore_accelerated(_series_from_group(x))
    )
    return _out_frame(ser, df)


# -----------------------------------------------------------------------------
# 事件驱动：TS_SINCE / TS_SINCE_N / TS_STREAK / TS_COUNT / TS_RATE / TS_ANY / TS_ALL / TS_RUNLENGTH_* / TS_CROSS_* /
#          TS_MONTH_POS / PRICE_GAP_* / TS_LAST_ARGGAP
# -----------------------------------------------------------------------------


def TS_SINCE(cond: pd.DataFrame) -> pd.DataFrame:
    """距上一次 cond 为真的 bar 数；``cond`` 为面板 DataFrame（单列），
    非 NaN 且不等于 0 视为 True；首次事件前输出 NaN，发生当根输出 0。"""
    return _ts_unary_accel(cond, _accel.ts_since)


def TS_SINCE_N(cond: pd.DataFrame, n: int) -> pd.DataFrame:
    """距倒数第 ``n`` 次 cond 为真的 bar 数（``n=1`` 同 ``TS_SINCE``）。

    从当前 bar 向历史回溯计数事件：``n=1`` 为最近一次，``n=2`` 为倒数第二次，依此类推。
    历史上不足 ``n`` 次事件时输出 NaN；落在该次事件当根时输出 0。
    **真** = 有限且 ≠0（与 ``TS_SINCE`` 一致）；比较结果请先 ``CAST(..., 'float64')``。

    取该次事件当时的字段值：``DELAY(x, TS_SINCE_N(cond, n))``（``DELAY`` 第二参可为动态滞后列）。"""
    if not isinstance(cond, pd.DataFrame):
        raise TypeError("TS_SINCE_N 第一参数 cond 须为面板 DataFrame")
    if cond.shape[1] < 1:
        raise ValueError("cond 须至少一列")
    n_ev = max(1, int(n))
    return _ts_unary_accel(cond, lambda v: _accel.ts_since_nth(v, n_ev))


def TS_RUNLENGTH_UP(df: pd.DataFrame) -> pd.DataFrame:
    """当前连续严格上涨根数（``x[i] > x[i-1]``）；中断或 NaN 重置为 0。"""
    return _ts_unary_accel(df, lambda v: _accel.ts_runlength(v, 1))


def TS_RUNLENGTH_DOWN(df: pd.DataFrame) -> pd.DataFrame:
    """当前连续严格下跌根数（``x[i] < x[i-1]``）；中断或 NaN 重置为 0。"""
    return _ts_unary_accel(df, lambda v: _accel.ts_runlength(v, -1))


def _ts_cond_roll(cond: pd.DataFrame, window: Window, op: int) -> pd.DataFrame:
    """事件滚动聚合；``op``：0=count，1=rate，2=any，3=all。真值规则同 ``TS_SINCE``。"""
    if not isinstance(cond, pd.DataFrame):
        raise TypeError("事件算子第一参数 cond 须为面板 DataFrame")
    if cond.shape[1] < 1:
        raise ValueError("cond 须至少一列")
    if op not in (0, 1, 2, 3):
        raise ValueError("内部 op 须为 0=count, 1=rate, 2=any, 3=all")

    if _is_dynamic_window(window):
        result = np.full(len(cond), np.nan, dtype=np.float32)
        for _, sub in _gb_instrument(cond):
            idx = sub.index
            wsub = window.reindex(idx)
            vals = sub.iloc[:, 0].to_numpy(dtype=float, copy=False)
            wvals = _dynamic_window_int_series(wsub, idx)
            pos = cond.index.get_indexer(idx)
            result[pos] = _accel.ts_event_roll_dyn(vals, wvals, op)
        return pd.DataFrame(result, index=cond.index, columns=cond.columns[:1])

    w = _as_int_window(window)

    def _roll(vals: np.ndarray) -> np.ndarray:
        return _accel.ts_event_roll(vals, w, op)

    return _ts_unary_accel(cond, _roll)


def TS_COUNT(cond: pd.DataFrame, window: Window) -> pd.DataFrame:
    """滚动窗内 ``cond`` 为真的 bar 数。

    **真** = 有限且 ≠0（与 ``TS_SINCE`` 一致）；比较结果请先 ``CAST(..., 'float64')``。
    窗内无有限 ``cond`` 输出 NaN；有有限值但无真值时输出 0。
    ``window`` 为 int 或单列 DataFrame 动态窗宽。"""
    return _ts_cond_roll(cond, window, 0)


def TS_RATE(cond: pd.DataFrame, window: Window) -> pd.DataFrame:
    """滚动窗内 ``cond`` 真值占比 ∈ [0, 1]。

    ``真 bar 数 / 有限 bar 数``；真值规则同 ``TS_SINCE``。窗内无有限 ``cond`` 为 NaN。"""
    return _ts_cond_roll(cond, window, 1)


def TS_ANY(cond: pd.DataFrame, window: Window) -> pd.DataFrame:
    """滚动窗内是否存在真 bar：有 → 1，无 → 0。

    真值规则同 ``TS_SINCE``；窗内无有限 ``cond`` 为 NaN。"""
    return _ts_cond_roll(cond, window, 2)


def TS_ALL(cond: pd.DataFrame, window: Window) -> pd.DataFrame:
    """滚动窗内是否全部为真：是 → 1，否 → 0。

    真值规则同 ``TS_SINCE``；窗内无有限 ``cond`` 为 NaN。"""
    return _ts_cond_roll(cond, window, 3)


def TS_STREAK(cond: pd.DataFrame) -> pd.DataFrame:
    """当前连续为真的根数（含当根）；假或 0 中断为 0，``cond`` 为 NaN 时输出 NaN 并重置。

    真值规则同 ``TS_SINCE``；比较结果请先 ``CAST(..., 'float64')``。"""
    return _ts_unary_accel(cond, _accel.ts_streak)


def _month_progress_values(dt: pd.DatetimeIndex) -> np.ndarray:
    """自然月进度 ∈ [0,1]：``(day-1)/(days_in_month-1)``；1 号=0，月末=1。"""
    ts = pd.DatetimeIndex(dt)
    day = ts.day.to_numpy(dtype=np.float32)
    dim = ts.days_in_month.to_numpy(dtype=np.float32)
    denom = np.maximum(dim - 1.0, 1.0)
    return (day - 1.0) / denom


def TS_MONTH_POS(df: pd.DataFrame) -> pd.DataFrame:
    """自然月进度 ∈ [0,1]：``(day-1)/(days_in_month-1)``；每月 1 日=0，该月最后一日=1。

    闰年 2 月自动按 29 天计。``df`` 仅用于索引对齐（取 ``datetime`` 层）。
    """
    if df.empty:
        return pd.DataFrame(index=df.index, columns=df.columns[:1], dtype=np.float32)
    dt = df.index.get_level_values("datetime")
    vals = _month_progress_values(dt)
    return pd.DataFrame(vals, index=df.index, columns=df.columns[:1], dtype=np.float32)


PRICE_GAP_DEFAULT_MIN_PCT: float = 0.0


def _instrument_price_gap_state(
    high: np.ndarray,
    low: np.ndarray,
    *,
    min_pct: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """单品种 OHLC：相邻 bar 经典 K 线跳空检测，并向前传播缺口状态。

    向上缺口：``low[i] > high[i-1]``（区间 ``[high[i-1], low[i]]``，下沿/上沿）。
    向下缺口：``high[i] < low[i-1]``（区间 ``[high[i], low[i-1]]``）。
    """
    n = len(high)
    size = np.zeros(n, dtype=np.float32)
    fill = np.full(n, np.nan, dtype=np.float32)
    floor = np.full(n, np.nan, dtype=np.float32)
    ceiling = np.full(n, np.nan, dtype=np.float32)
    event = np.zeros(n, dtype=np.float32)
    bars = np.full(n, np.nan, dtype=np.float32)

    if n == 0:
        return size, fill, floor, ceiling, event, bars

    active = False
    gap_dir = 0
    gap_floor = np.nan
    gap_ceiling = np.nan
    gap_height = 0.0
    signed_size = 0.0
    min_low = np.nan
    max_high = np.nan
    bars_count = np.nan

    min_pct_f = max(float(min_pct), 0.0)

    for i in range(n):
        event[i] = 0.0
        formed_gap = False

        if i > 0:
            prev_hi = high[i - 1]
            prev_lo = low[i - 1]
            hi_i = high[i]
            lo_i = low[i]

            is_up = False
            is_down = False
            if (
                np.isfinite(prev_hi)
                and np.isfinite(prev_lo)
                and np.isfinite(hi_i)
                and np.isfinite(lo_i)
                and prev_hi != 0.0
                and prev_lo != 0.0
            ):
                if lo_i > prev_hi:
                    up_size = (lo_i - prev_hi) / abs(prev_hi)
                    is_up = up_size >= min_pct_f
                if hi_i < prev_lo:
                    down_size = (prev_lo - hi_i) / abs(prev_lo)
                    is_down = down_size >= min_pct_f

            if is_up and not is_down:
                active = True
                gap_dir = 1
                gap_floor = prev_hi
                gap_ceiling = lo_i
                gap_height = lo_i - prev_hi
                signed_size = gap_height / abs(prev_hi)
                min_low = lo_i
                max_high = np.nan
                bars_count = 0.0
                event[i] = 1.0
                formed_gap = True
            elif is_down and not is_up:
                active = True
                gap_dir = -1
                gap_floor = hi_i
                gap_ceiling = prev_lo
                gap_height = prev_lo - hi_i
                signed_size = -gap_height / abs(prev_lo)
                max_high = hi_i
                min_low = np.nan
                bars_count = 0.0
                event[i] = -1.0
                formed_gap = True
            elif active:
                bars_count += 1.0
                if gap_dir > 0:
                    if np.isfinite(lo_i):
                        min_low = lo_i if not np.isfinite(min_low) else min(min_low, lo_i)
                    if gap_height > 1e-12:
                        if min_low <= gap_floor:
                            fill[i] = 1.0
                        else:
                            fill[i] = np.clip(
                                (gap_ceiling - min_low) / gap_height, 0.0, 1.0
                            )
                    else:
                        fill[i] = 1.0
                elif gap_dir < 0:
                    if np.isfinite(hi_i):
                        max_high = hi_i if not np.isfinite(max_high) else max(max_high, hi_i)
                    if gap_height > 1e-12:
                        if max_high >= gap_ceiling:
                            fill[i] = 1.0
                        else:
                            fill[i] = np.clip(
                                (max_high - gap_floor) / gap_height, 0.0, 1.0
                            )
                    else:
                        fill[i] = 1.0

        size[i] = signed_size if active else 0.0
        if active:
            floor[i] = gap_floor
            ceiling[i] = gap_ceiling
            bars[i] = bars_count
            if not np.isfinite(fill[i]):
                fill[i] = 0.0 if gap_height > 1e-12 else 1.0
        elif not formed_gap:
            signed_size = 0.0
            bars_count = np.nan

    return size, fill, floor, ceiling, event, bars


def _price_gap_output(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    field: str,
    min_pct: float = PRICE_GAP_DEFAULT_MIN_PCT,
) -> pd.DataFrame:
    """OHLC 四列对齐后，按 ``field`` 输出缺口派生序列。"""
    template = close_df
    if template.empty:
        return pd.DataFrame(index=template.index, columns=template.columns[:1], dtype=np.float32)

    for name, other in (
        ("open_df", open_df),
        ("high_df", high_df),
        ("low_df", low_df),
    ):
        if other.shape != template.shape or not other.index.equals(template.index):
            raise ValueError(f"缺口算子要求 OHLC 四列同形同索引，{name} 不一致")

    valid = {"size", "fill", "floor", "ceiling", "event", "bars"}
    if field not in valid:
        raise ValueError(f"未知缺口字段: {field!r}")

    result = np.full(len(template), np.nan, dtype=np.float32)
    for _, sub_c in _gb_instrument(close_df):
        idx = sub_c.index
        sub_h = high_df.reindex(idx)
        sub_l = low_df.reindex(idx)
        sz, fl, flr, clg, ev, br = _instrument_price_gap_state(
            sub_h.iloc[:, 0].to_numpy(dtype=float, copy=False),
            sub_l.iloc[:, 0].to_numpy(dtype=float, copy=False),
            min_pct=float(min_pct),
        )
        pick = {
            "size": sz,
            "fill": fl,
            "floor": flr,
            "ceiling": clg,
            "event": ev,
            "bars": br,
        }[field]
        pos = template.index.get_indexer(idx)
        result[pos] = pick

    return pd.DataFrame(result, index=template.index, columns=template.columns[:1])


def PRICE_GAP_SIZE(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    min_pct: float = PRICE_GAP_DEFAULT_MIN_PCT,
) -> pd.DataFrame:
    """活跃缺口**带符号相对幅度**（向上为正、向下为负）；无活跃缺口为 0。

    缺口判定：``low[t]>high[t-1]`` 向上；``high[t]<low[t-1]`` 向下。
    ``min_pct`` 过滤过小缺口（相对下沿/上沿边界价的幅度）。
    """
    return _price_gap_output(open_df, high_df, low_df, close_df, "size", min_pct)


def PRICE_GAP_FLOOR(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    min_pct: float = PRICE_GAP_DEFAULT_MIN_PCT,
) -> pd.DataFrame:
    """当前活跃缺口的**下沿价格**；无活跃缺口为 NaN。

    向上缺口 = ``high[t-1]``；向下缺口 = 形成当根 ``high[t]``。
    """
    return _price_gap_output(open_df, high_df, low_df, close_df, "floor", min_pct)


def PRICE_GAP_CEILING(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    min_pct: float = PRICE_GAP_DEFAULT_MIN_PCT,
) -> pd.DataFrame:
    """当前活跃缺口的**上沿价格**；无活跃缺口为 NaN。

    向上缺口 = 形成当根 ``low[t]``；向下缺口 = ``low[t-1]``。
    """
    return _price_gap_output(open_df, high_df, low_df, close_df, "ceiling", min_pct)


def PRICE_GAP_FILL(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    min_pct: float = PRICE_GAP_DEFAULT_MIN_PCT,
) -> pd.DataFrame:
    """当前活跃缺口的**回补比例** ∈ [0,1]（0=未回补，1=完全回补）；无活跃缺口为 NaN。"""
    return _price_gap_output(open_df, high_df, low_df, close_df, "fill", min_pct)


def TS_LAST_ARGGAP(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    min_pct: float = PRICE_GAP_DEFAULT_MIN_PCT,
) -> pd.DataFrame:
    """最近一次缺口形成距今 bar 数（形成当根=0）；无活跃缺口为 NaN。

    缺口判定同 ``PRICE_GAP_EVENT``；语义同 ``TS_LAST_ARGPEAK`` 类「距今根数」算子。
    新缺口形成时重置为 0。
    """
    return _price_gap_output(open_df, high_df, low_df, close_df, "bars", min_pct)


def TS_ARGGAP(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    min_pct: float = PRICE_GAP_DEFAULT_MIN_PCT,
) -> pd.DataFrame:
    """[兼容] 同 ``TS_LAST_ARGGAP``。"""
    return TS_LAST_ARGGAP(open_df, high_df, low_df, close_df, min_pct)


def PRICE_GAP_EVENT(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    min_pct: float = PRICE_GAP_DEFAULT_MIN_PCT,
) -> pd.DataFrame:
    """缺口**形成当根**事件：+1 向上、-1 向下、0 无缺口；其余 bar 为 0。"""
    return _price_gap_output(open_df, high_df, low_df, close_df, "event", min_pct)


def TS_CROSS_ABOVE(x: pd.DataFrame, y: TsCrossOperand) -> pd.DataFrame:
    """上穿事件：``x[t-1] <= y[t-1] and x[t] > y[t]`` 为 1，否则 0；缺失为 NaN。

    第一个参数 ``x`` 必须是面板（pd.DataFrame，时间序列），不能是常数；
    第二个参数 ``y`` 可为与 ``x`` 同索引的单列面板，也可为 Python/NumPy 数值标量
    （按 ``x`` 索引广播，等价于 ``ADD(df, k)`` 的常数语义），常用于"上穿固定阈值"，
    例如 ``TS_CROSS_ABOVE(compression_rank, 0.8)``。
    """
    return _ts_bivariate_event_accel(
        x,
        _as_ts_cross_y_panel(x, y),
        lambda a, b: _accel.ts_cross(a, b, 1),
    )


def TS_CROSS_BELOW(x: pd.DataFrame, y: TsCrossOperand) -> pd.DataFrame:
    """下穿事件：``x[t-1] >= y[t-1] and x[t] < y[t]`` 为 1，否则 0；缺失为 NaN。

    第一个参数 ``x`` 必须是面板（pd.DataFrame，时间序列），不能是常数；
    第二个参数 ``y`` 可为与 ``x`` 同索引的单列面板，也可为 Python/NumPy 数值标量
    （按 ``x`` 索引广播），常用于"下穿固定阈值"，
    例如 ``TS_CROSS_BELOW(compression_rank, 0.2)``。
    """
    return _ts_bivariate_event_accel(
        x,
        _as_ts_cross_y_panel(x, y),
        lambda a, b: _accel.ts_cross(a, b, -1),
    )


def _ts_bivariate_fixed(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    window: int,
    kernel,
) -> pd.DataFrame:
    """双序列固定窗滚动（corr / cov），按品种分组后调用加速内核。"""
    w = max(1, int(window))
    result = np.full(len(df1), np.nan, dtype=np.float32)

    for _, sub1 in _gb_instrument(df1):
        idx = sub1.index
        sub2 = df2.reindex(idx)
        x = sub1.iloc[:, 0].to_numpy(dtype=float, copy=False)
        y = sub2.iloc[:, 0].to_numpy(dtype=float, copy=False)
        pos = df1.index.get_indexer(idx)
        result[pos] = kernel(x, y, w)

    return pd.DataFrame(result, index=df1.index, columns=df1.columns[:1])


def TS_CORR(df1: pd.DataFrame, df2: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动 Pearson 相关系数；按品种分组，对 df1、df2 对应列在窗口内计算相关性。
    NaN 对会被跳过；有效对数 < 2 或任一序列方差为零时输出 NaN。"""
    return _ts_bivariate_fixed(df1, df2, window, _accel.roll_corr_fixed)


def TS_COV(df1: pd.DataFrame, df2: pd.DataFrame, window: int, ddof: int = 1) -> pd.DataFrame:
    """滚动协方差；按品种分组，对 df1、df2 对应列在窗口内计算样本协方差（ddof 默认 1）。
    NaN 对会被跳过；有效对数 ≤ ddof 时输出 NaN。"""
    return _ts_bivariate_fixed(
        df1, df2, window,
        lambda x, y, w: _accel.roll_cov_fixed(x, y, w, ddof=ddof),
    )


def TS_RANKCORR(df1: pd.DataFrame, df2: pd.DataFrame, window: int) -> pd.DataFrame:
    """滚动 Spearman（秩）相关系数：每个窗口内先对 df1、df2 各自取平均秩（等同
    ``rank(method='average')``），再对两组秩计算 Pearson 相关；对异常值鲁棒，
    捕捉单调关系。NaN 对跳过；有效对数 < 2 或任一维秩方差为零输出 NaN。"""
    return _ts_bivariate_fixed(df1, df2, window, _accel.roll_rankcorr_fixed)


def MUTUAL_INFO_LAG(
    df_close: pd.DataFrame,
    df_volume: pd.DataFrame,
    window: int,
    lag: int,
    *,
    n_bins: int = 8,
    min_pairs: int | None = None,
) -> pd.DataFrame:
    """滚动直方图（秩分箱）估计 Shannon 互信息 I(close(t); volume(t-k))，单位为 nats。

    **配对规则**：窗口 ``[t-window+1, t]`` 内对每个满足 j≥k 的 j 取
    (close[j], volume[j-k])，k 即 ``lag``（k=0 表示同 bar 的量价）。窗内先在有效样本上
    对价格、成交量各自做等频秩映射到 ``n_bins`` 档，再对联合频数表计算 MI；可检出
    Pearson≈0 但存在的非线性耦合（如阈值效应）。**有效对数**不足 ``min_pairs``
    （默认 ``max(n_bins+2, 8)``）时为 NaN。"""
    w = max(2, int(window))
    lag_i = max(0, int(lag))
    B = int(n_bins)
    if B < 2:
        raise ValueError("n_bins must be >= 2")
    mp = int(min_pairs) if min_pairs is not None else max(B + 2, 8)

    result = np.full(len(df_close), np.nan, dtype=np.float32)
    for _, sub_c in _gb_instrument(df_close):
        idx = sub_c.index
        sub_v = df_volume.reindex(idx)
        c_arr = sub_c.iloc[:, 0].to_numpy(dtype=float, copy=False)
        v_arr = sub_v.iloc[:, 0].to_numpy(dtype=float, copy=False)
        pos = df_close.index.get_indexer(idx)
        result[pos] = _accel.roll_mutual_info_lag_fixed(
            c_arr, v_arr, w, lag_i, n_bins=B, min_pairs=mp
        )

    return pd.DataFrame(
        result, index=df_close.index, columns=df_close.columns[:1]
    )


def TS_TREND_RANK(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """Mann-Kendall 风格秩趋势：每个窗口内对 x 值与"时间位置"做 Spearman 相关，
    得到 ∈ [-1, +1] 的**非参数单调趋势强度**——+1 表示窗口内单调上升、-1 单调下降、
    0 无单调趋势。比 ``SLOPE`` / ``REGBETA`` 对尖刺、跳空、离群点鲁棒得多，与
    ``TS_EFFICIENCY_RATIO`` 互补（ER 测路径效率、本算子测单调性）。NaN 跳过；
    有效点数 < 2 输出 NaN。"""
    w = max(2, int(window))
    result = np.full(len(df), np.nan, dtype=np.float32)
    for _, sub in _gb_instrument(df):
        idx = sub.index
        vals = sub.iloc[:, 0].to_numpy(dtype=float, copy=False)
        # 每品种独立的严格单调计数器；在任意窗口内它的平均秩恒为 1..c（c=窗内有效数），
        # 因此 Spearman(x, counter) 恰好是 Mann-Kendall 风格的单调趋势度量。
        counter = np.arange(len(vals), dtype=np.float32)
        pos = df.index.get_indexer(idx)
        result[pos] = _accel.roll_rankcorr_fixed(vals, counter, w)
    return pd.DataFrame(result, index=df.index, columns=df.columns[:1])


def TS_EFFICIENCY_RATIO(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """Kaufman 效率比 ER∈[0,1]；window 可为整数固定窗或单列 DataFrame 动态窗（如峰谷间距）。
    定义：|窗末-窗首|/Σ|逐 bar 变化|；接近 1=单边趋势，接近 0=震荡；有效值不足或路径为 0 时为 NaN。"""
    result = np.full(len(df), np.nan, dtype=np.float32)

    if _is_dynamic_window(window):
        for _, sub in _gb_instrument(df):
            idx = sub.index
            vals = sub.iloc[:, 0].to_numpy(dtype=float, copy=False)
            wsub = window.reindex(idx)
            w_arr = _dynamic_window_int_series(wsub, idx)
            pos = df.index.get_indexer(idx)
            result[pos] = _accel.roll_efficiency_ratio_dynamic(vals, w_arr)
        return pd.DataFrame(result, index=df.index, columns=df.columns[:1])

    w = max(2, _as_int_window(window))

    def _er_accelerated(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float, copy=False)
        out = _accel.roll_efficiency_ratio_fixed(vals, w)
        return pd.Series(out, index=s.index)

    ser = _gb_instrument(df).transform(
        lambda x: _er_accelerated(_series_from_group(x))
    )
    return _out_frame(ser, df)


def VOLUME_CLOCK_VPIN(
    price: pd.DataFrame,
    volume: pd.DataFrame,
    window: int,
    bucket_size: float,
    classification: str = "tick",
    min_buckets: int = 5,
    sigma_window: int = 20,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """成交量同步 VPIN（Volume-Synchronized Probability of Informed Trading）∈ [0,1]。

    **成交量时钟**（非时间窗）：按固定 ``bucket_size`` 切成交量桶；单 bar 可跨多桶或多 bar
    才满一桶，**不**等价于固定 K 线根数。逐 bar **从前向后**因果累积，无前视。

    **买卖分类**（第 5 参 ``classification``，禁止 ``name=value``）：

    - ``'tick'``（默认）：涨→全买、跌→全卖；平盘沿用上一方向（首根 50/50）
    - ``'lee_ready'``：涨→全买、跌→全卖；平盘恒 50/50
    - ``'bulk'``：量钟 BVC——``buy = vol·Φ(ΔP/σ_ΔP)``，``sell = vol - buy``；σ_ΔP 为过去
      bar 价格变化的滚动样本标准差（第 7 参 ``sigma_window``，默认 20；``tick``/``lee_ready`` 忽略）

    **每桶**结算 imbalance = ``|Buy-Sell|/(Buy+Sell)``；**t 时刻输出** = 最近 ``n`` 个
    **已满桶** imbalance 的均值（``n = min(已满桶数, window)``，窗口末位为最新完成桶）。
    **进行中的半桶不计入**；已满桶数 < ``min_buckets`` 时为 NaN（默认 ``min_buckets=5``，
    且自动截断为 ``min(min_buckets, window)``）。``window`` 为桶个数上限（非 bar 数）。
    高≈单边主动/知情成交主导，低≈买卖均衡。price 建议 ``$adj_vwap``，volume 用 ``$volume``。"""
    w = max(1, int(window))
    bsize = float(bucket_size)
    if bsize <= 0.0:
        raise ValueError("bucket_size must be > 0")
    _accel.vpin_classification_id(classification)
    mb = int(min_buckets)
    if mb < 1:
        raise ValueError("min_buckets must be >= 1")
    sw = int(sigma_window)
    if sw < 1:
        raise ValueError("sigma_window must be >= 1")
    eps_f = float(eps)

    result = np.full(len(price), np.nan, dtype=np.float32)
    for _, sub_p in _gb_instrument(price):
        idx = sub_p.index
        sub_v = volume.reindex(idx)
        p_arr = sub_p.iloc[:, 0].to_numpy(dtype=float, copy=False)
        v_arr = sub_v.iloc[:, 0].to_numpy(dtype=float, copy=False)
        pos = price.index.get_indexer(idx)
        result[pos] = _accel.volume_clock_vpin_fixed(
            p_arr,
            v_arr,
            w,
            bsize,
            classification,
            min_buckets=mb,
            eps=eps_f,
        )

    return pd.DataFrame(result, index=price.index, columns=price.columns[:1])


def WICK_EFFICIENCY(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    lag: int,
    *,
    eps: float = 1e-12,
) -> pd.DataFrame:
    """影线能量效率：当前柱上影线与 k 根前下影线的交叉耦合，按实体尺度归一化。

    ::
        UP_WICK(t)  = high - max(open, close)
        DN_WICK(t)  = min(open, close) - low
        BODY(t)     = |close - open|
        OUT(t)      = UP_WICK(t) * DN_WICK(t-k) / (BODY(t) * BODY(t-k) + eps)

    若前期下影线与当期上影线在区间边界上呈现对称吸收语义，比值往往偏大。**时间交叉乘法**
    结构无法由单序列 ``DELTA`` 单独生成。t < k 或任一侧 OHLC 非有限为 NaN。``lag`` ≥ 1。
    """
    k = int(lag)
    if k < 1:
        raise ValueError("lag must be >= 1")

    eps_f = float(eps)
    idx0 = open_df.index
    if not (
        high_df.index.equals(idx0)
        and low_df.index.equals(idx0)
        and close_df.index.equals(idx0)
    ):
        raise ValueError("WICK_EFFICIENCY: open/high/low/close panels must share the same index")

    result = np.full(len(open_df), np.nan, dtype=np.float32)

    for _, sub_o in _gb_instrument(open_df):
        idx = sub_o.index
        sub_h = high_df.reindex(idx)
        sub_l = low_df.reindex(idx)
        sub_c = close_df.reindex(idx)
        o_arr = sub_o.iloc[:, 0].to_numpy(dtype=float, copy=False)
        h_arr = sub_h.iloc[:, 0].to_numpy(dtype=float, copy=False)
        l_arr = sub_l.iloc[:, 0].to_numpy(dtype=float, copy=False)
        c_arr = sub_c.iloc[:, 0].to_numpy(dtype=float, copy=False)
        pos = open_df.index.get_indexer(idx)
        result[pos] = _accel.wick_efficiency_fixed(o_arr, h_arr, l_arr, c_arr, k, eps=eps_f)

    return pd.DataFrame(result, index=open_df.index, columns=open_df.columns[:1])


def KLINE_GEOMETRY(
    open_df: pd.DataFrame,
    high_df: pd.DataFrame,
    low_df: pd.DataFrame,
    close_df: pd.DataFrame,
    window: Window,
    *,
    eps: float = 1e-15,
) -> pd.DataFrame:
    """窗口内将每根 K 线视为 ``(o,h,l,c)∈R^4``，堆成 ``X∈R^{k×4}`` 做 SVD，输出 ``σ₂/σ₁``。

    刻画形态从近似一维（比值 → 0）到多方向分散/震荡（比值 → 1）。``window`` 须 ≥ 2；
    可为固定整数或单列动态窗。窗内任一路 OHLC 非有限则该点为 NaN。"""
    idx0 = open_df.index
    if not (
        high_df.index.equals(idx0)
        and low_df.index.equals(idx0)
        and close_df.index.equals(idx0)
    ):
        raise ValueError("KLINE_GEOMETRY: open/high/low/close panels must share the same index")

    eps_f = float(eps)
    result = np.full(len(open_df), np.nan, dtype=np.float32)

    if _is_dynamic_window(window):
        for _, sub_o in _gb_instrument(open_df):
            idx = sub_o.index
            sub_h = high_df.reindex(idx)
            sub_l = low_df.reindex(idx)
            sub_c = close_df.reindex(idx)
            o_arr = sub_o.iloc[:, 0].to_numpy(dtype=float, copy=False)
            h_arr = sub_h.iloc[:, 0].to_numpy(dtype=float, copy=False)
            l_arr = sub_l.iloc[:, 0].to_numpy(dtype=float, copy=False)
            c_arr = sub_c.iloc[:, 0].to_numpy(dtype=float, copy=False)
            wsub = window.reindex(idx)
            w_arr = _dynamic_window_int_series(wsub, idx)
            pos = open_df.index.get_indexer(idx)
            result[pos] = _accel.roll_kline_geometry(
                o_arr, h_arr, l_arr, c_arr, w_arr, eps=eps_f
            )
    else:
        w = _as_int_window(window)
        if w < 2:
            raise ValueError("KLINE_GEOMETRY: window must be >= 2")
        for _, sub_o in _gb_instrument(open_df):
            idx = sub_o.index
            sub_h = high_df.reindex(idx)
            sub_l = low_df.reindex(idx)
            sub_c = close_df.reindex(idx)
            o_arr = sub_o.iloc[:, 0].to_numpy(dtype=float, copy=False)
            h_arr = sub_h.iloc[:, 0].to_numpy(dtype=float, copy=False)
            l_arr = sub_l.iloc[:, 0].to_numpy(dtype=float, copy=False)
            c_arr = sub_c.iloc[:, 0].to_numpy(dtype=float, copy=False)
            pos = open_df.index.get_indexer(idx)
            result[pos] = _accel.roll_kline_geometry_fixed(
                o_arr, h_arr, l_arr, c_arr, w, eps=eps_f
            )

    return pd.DataFrame(result, index=open_df.index, columns=open_df.columns[:1])


def TS_PERMUTATION_ENTROPY(
    df: pd.DataFrame, window: int, order: int = 3
) -> pd.DataFrame:
    """Bandt-Pompe 排列熵：窗口内所有长度 ``order`` 的子序列按"序数模式"计数并做
    Shannon 熵，再除以 ``log(order!)`` 归一化到 [0, 1]。接近 1 表示序列无规律（白噪
    声），接近 0 表示高度可预测（单调或周期）。``order`` 典型 3–5；``window`` 建议
    ≥ ``order!`` 以覆盖各种模式。窗口内无可用子序列输出 NaN。"""
    w = max(2, int(window))
    m = int(order)
    if m < 2 or m > 7:
        raise ValueError("order must be in [2, 7]")

    def _pe_accelerated(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float, copy=False)
        out = _accel.roll_permutation_entropy_fixed(vals, w, m)
        return pd.Series(out, index=s.index)

    ser = _gb_instrument(df).transform(
        lambda x: _pe_accelerated(_series_from_group(x))
    )
    return _out_frame(ser, df)


def _chip_metric_daily(
    close_df: pd.DataFrame,
    low_df: pd.DataFrame,
    high_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    window: int,
    float_cap_df: pd.DataFrame,
    nbins: int,
    method: str,
    op: str,
) -> pd.DataFrame:
    """日频筹码指标（默认 cyq + float_cap）。"""
    w = max(1, int(window))
    nb = int(nbins)
    if nb < 2:
        raise ValueError("nbins must be >= 2")
    _chip_daily.chip_method_id(method)

    result = np.full(len(close_df), np.nan, dtype=np.float32)
    for _, sub_c in _gb_instrument(close_df):
        idx = sub_c.index
        sub_l = low_df.reindex(idx)
        sub_h = high_df.reindex(idx)
        sub_v = volume_df.reindex(idx)
        sub_a = float_cap_df.reindex(idx)
        close = sub_c.iloc[:, 0].to_numpy(dtype=float, copy=False)
        low = sub_l.iloc[:, 0].to_numpy(dtype=float, copy=False)
        high = sub_h.iloc[:, 0].to_numpy(dtype=float, copy=False)
        vol = sub_v.iloc[:, 0].to_numpy(dtype=float, copy=False)
        aux = sub_a.iloc[:, 0].to_numpy(dtype=float, copy=False)
        pos = close_df.index.get_indexer(idx)
        result[pos] = _accel.roll_chip_metric_fixed(
            close, vol, low, high, aux, w, nb, op, method
        )

    return pd.DataFrame(
        result, index=close_df.index, columns=close_df.columns[:1]
    )


def _chip_roll_daily(
    close_df: pd.DataFrame,
    low_df: pd.DataFrame,
    high_df: pd.DataFrame,
    volume_df: pd.DataFrame,
    window: int,
    float_cap_df: pd.DataFrame,
    nbins: int,
    method: str,
    kernel,
) -> pd.DataFrame:
    """按品种调用日频筹码 kernel(close, low, high, vol, aux, window, nbins)。"""
    w = max(1, int(window))
    nb = int(nbins)
    if nb < 2:
        raise ValueError("nbins must be >= 2")
    _chip_daily.chip_method_id(method)

    result = np.full(len(close_df), np.nan, dtype=np.float32)
    for _, sub_c in _gb_instrument(close_df):
        idx = sub_c.index
        sub_l = low_df.reindex(idx)
        sub_h = high_df.reindex(idx)
        sub_v = volume_df.reindex(idx)
        sub_a = float_cap_df.reindex(idx)
        close = sub_c.iloc[:, 0].to_numpy(dtype=float, copy=False)
        low = sub_l.iloc[:, 0].to_numpy(dtype=float, copy=False)
        high = sub_h.iloc[:, 0].to_numpy(dtype=float, copy=False)
        vol = sub_v.iloc[:, 0].to_numpy(dtype=float, copy=False)
        aux = sub_a.iloc[:, 0].to_numpy(dtype=float, copy=False)
        pos = close_df.index.get_indexer(idx)
        result[pos] = kernel(close, low, high, vol, aux, w, nb)

    return pd.DataFrame(
        result, index=close_df.index, columns=close_df.columns[:1]
    )


def CHIP_PEAK_LOC(
    close: pd.DataFrame,
    low: pd.DataFrame,
    high: pd.DataFrame,
    volume: pd.DataFrame,
    window: int,
    float_cap: pd.DataFrame,
    nbins: int = 64,
    method: str = "cyq",
) -> pd.DataFrame:
    """筹码主峰价相对现价的偏离：``(p* - P) / P``（日频 CYQ，默认）。
    **DSL 位置参数**：``close, low, high, volume, window, float_cap, [nbins], [method]``。
    默认 ``method='cyq'``、``nbins=64``；``tri`` 时第 6 参改传 ``$vwap``。"""
    return _chip_metric_daily(
        close, low, high, volume, window, float_cap, nbins, method, "peak_loc"
    )


def CHIP_ENTROPY(
    close: pd.DataFrame,
    low: pd.DataFrame,
    high: pd.DataFrame,
    volume: pd.DataFrame,
    window: int,
    float_cap: pd.DataFrame,
    nbins: int = 64,
    method: str = "cyq",
) -> pd.DataFrame:
    """筹码密度归一化 Shannon 熵 ``H / log(nbins)`` ∈ [0, 1]（日频，默认 cyq）。
    **DSL 位置参数**：``close, low, high, volume, window, float_cap, [nbins], [method]``。
    推荐窗口 20~120 交易日。"""
    return _chip_metric_daily(
        close, low, high, volume, window, float_cap, nbins, method, "entropy"
    )


def CHIP_COM_W_GAP(
    close: pd.DataFrame,
    low: pd.DataFrame,
    high: pd.DataFrame,
    volume: pd.DataFrame,
    window: int,
    float_cap: pd.DataFrame,
    nbins: int = 64,
    method: str = "cyq",
) -> pd.DataFrame:
    """筹码重心相对现价的偏离：``(bar_p - P) / P``（日频，默认 cyq）。
    **DSL 位置参数**：``close, low, high, volume, window, float_cap, [nbins], [method]``。"""
    return _chip_metric_daily(
        close, low, high, volume, window, float_cap, nbins, method, "com_w_gap"
    )


def CHIP_MASS_ASYM(
    close: pd.DataFrame,
    low: pd.DataFrame,
    high: pd.DataFrame,
    volume: pd.DataFrame,
    window: int,
    float_cap: pd.DataFrame,
    nbins: int = 64,
    method: str = "cyq",
) -> pd.DataFrame:
    """以现价为界的上下筹码质量不对称度 ``M_below - M_above`` ∈ [-1, 1]（日频，默认 cyq）。
    **DSL 位置参数**：``close, low, high, volume, window, float_cap, [nbins], [method]``。"""
    return _chip_metric_daily(
        close, low, high, volume, window, float_cap, nbins, method, "mass_asym"
    )


def CHIP_PEAK_SHARPNESS(
    close: pd.DataFrame,
    low: pd.DataFrame,
    high: pd.DataFrame,
    volume: pd.DataFrame,
    window: int,
    float_cap: pd.DataFrame,
    nbins: int = 64,
    implementation: str = "curvature",
    method: str = "cyq",
) -> pd.DataFrame:
    """主峰尖锐度（日频，默认 cyq）：``curvature`` / ``fwhm`` / ``combined``。
    **DSL 位置参数**：``close, low, high, volume, window, float_cap, [nbins], [implementation], [method]``。"""
    _accel.chip_peak_sharpness_impl_id(implementation)
    return _chip_roll_daily(
        close,
        low,
        high,
        volume,
        window,
        float_cap,
        nbins,
        method,
        lambda c, l, h, v, a, w, nb: _accel.roll_chip_peak_sharpness_fixed(
            c, v, l, h, a, w, nb, implementation, method
        ),
    )


def CHIP_BIMODAL_SCORE(
    close: pd.DataFrame,
    low: pd.DataFrame,
    high: pd.DataFrame,
    volume: pd.DataFrame,
    window: int,
    float_cap: pd.DataFrame,
    nbins: int = 64,
    implementation: str = "simple",
    lambda_scale: float = 1.0,
    method: str = "cyq",
) -> pd.DataFrame:
    """双峰结构得分（日频，默认 cyq）：``simple`` 或 ``dip``。
    **DSL 位置参数**：``close, low, high, volume, window, float_cap, [nbins], [implementation], [lambda_scale], [method]``。"""
    _accel.chip_bimodal_impl_id(implementation)
    return _chip_roll_daily(
        close,
        low,
        high,
        volume,
        window,
        float_cap,
        nbins,
        method,
        lambda c, l, h, v, a, w, nb: _accel.roll_chip_bimodal_fixed(
            c, v, l, h, a, w, nb, implementation, method, lambda_scale=lambda_scale
        ),
    )


def CHIP_WASS_DIST(
    close: pd.DataFrame,
    low: pd.DataFrame,
    high: pd.DataFrame,
    volume: pd.DataFrame,
    window: Window,
    float_cap: pd.DataFrame,
    nbins: int = 64,
    lag: Window = 0,
    implementation: str = "moment",
    method: str = "cyq",
) -> pd.DataFrame:
    """当前窗与参照窗筹码直方图漂移（日频 CYQ，ρ=``lag``）。
    **DSL 位置参数**：``close, low, high, volume, window, float_cap, [nbins], [lag], [implementation], [method]``。"""
    nb = int(nbins)
    if nb < 2:
        raise ValueError("nbins must be >= 2")
    _accel.chip_wass_implementation_id(implementation)
    _chip_daily.chip_method_id(method)

    result = np.full(len(close), np.nan, dtype=np.float32)
    for _, sub_c in _gb_instrument(close):
        idx = sub_c.index
        sub_l = low.reindex(idx)
        sub_h = high.reindex(idx)
        sub_v = volume.reindex(idx)
        sub_a = float_cap.reindex(idx)
        close_arr = sub_c.iloc[:, 0].to_numpy(dtype=float, copy=False)
        low_arr = sub_l.iloc[:, 0].to_numpy(dtype=float, copy=False)
        high_arr = sub_h.iloc[:, 0].to_numpy(dtype=float, copy=False)
        vol_arr = sub_v.iloc[:, 0].to_numpy(dtype=float, copy=False)
        aux_arr = sub_a.iloc[:, 0].to_numpy(dtype=float, copy=False)
        w_arr = _chip_wass_win_series(window, idx)
        rho_arr = _chip_wass_rho_series(lag, idx)
        pos = close.index.get_indexer(idx)
        result[pos] = _accel.roll_chip_wass_dist(
            close_arr,
            vol_arr,
            low_arr,
            high_arr,
            aux_arr,
            w_arr,
            w_arr,
            rho_arr,
            nb,
            implementation,
            method,
        )

    return pd.DataFrame(
        result, index=close.index, columns=close.columns[:1]
    )


def _ts_arg_extreme(df: pd.DataFrame, window: int, want_max: bool) -> pd.DataFrame:
    """窗口内极值距今的 bar 数（0 表示当前 bar 即为极值）。固定窗走 Numba，避免 ``rolling.apply`` 全表 Python 回调。"""
    return arg_extreme_fixed(df, window, want_max)


def _ts_arg_local_extreme_last(df: pd.DataFrame, half_window: int, want_max: bool) -> pd.DataFrame:
    """最近一次已确认中心局部峰/谷距今 bar 数。

    某位置 ``j`` 只有在其前后各 ``half_window`` 根都齐备时，才会被判定为局部峰/谷；
    因此输出天然带 ``half_window`` 根确认延迟，可避免把未来 ``close`` 偷看进当前时点。
    """
    hw = max(1, int(half_window))

    def _arg_local_extreme_accelerated(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float, copy=False)
        out = _accel.arg_local_extreme(vals, hw, want_max=want_max)
        return pd.Series(out, index=s.index)

    ser = _gb_instrument(df).transform(
        lambda x: _arg_local_extreme_accelerated(_series_from_group(x))
    )
    return _out_frame(ser, df)


def _ts_local_extreme_value_last(
    df: pd.DataFrame, half_window: int, want_max: bool
) -> pd.DataFrame:
    """最近一次已确认中心局部峰/谷的价格值。"""
    hw = max(1, int(half_window))

    def _local_extreme_value_accelerated(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float, copy=False)
        out = _accel.local_extreme_value(vals, hw, want_max=want_max)
        return pd.Series(out, index=s.index)

    ser = _gb_instrument(df).transform(
        lambda x: _local_extreme_value_accelerated(_series_from_group(x))
    )
    return _out_frame(ser, df)


def _ts_maxamp_arg_local(
    df: pd.DataFrame, half_window: int, want_max: bool
) -> pd.DataFrame:
    """在已确认峰/谷中，选左右「峰到谷/谷到峰」价宽之和最大者，输出距今 bar 数。"""
    hw = max(1, int(half_window))
    col = df.columns[0]

    def _f(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float, copy=False)
        out = _accel.maxamp_arg_local_extreme(vals, hw, want_max=want_max)
        return pd.Series(out, index=s.index)

    ser = df[col].groupby(level="instrument", sort=False).transform(_f)
    return _out_frame(ser, df)


def _ts_maxamp_value_local(
    df: pd.DataFrame, half_window: int, want_max: bool
) -> pd.DataFrame:
    """在已确认峰/谷中，选左右价宽和最大者，输出该极值价格。"""
    hw = max(1, int(half_window))
    col = df.columns[0]

    def _f(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float, copy=False)
        out = _accel.maxamp_local_extreme_value(vals, hw, want_max=want_max)
        return pd.Series(out, index=s.index)

    ser = df[col].groupby(level="instrument", sort=False).transform(_f)
    return _out_frame(ser, df)


def _ts_arg_extreme_dynamic(
    df: pd.DataFrame, win_df: pd.DataFrame, want_max: bool
) -> pd.DataFrame:
    """与 _ts_arg_extreme 相同语义，窗口宽度为每行可变的正整数（来自另一列 DataFrame）。"""
    return arg_extreme_dynamic(df, win_df, _dynamic_window_int_series, want_max)


def TS_ARGMAX(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """极大值距今 bar 数；window 同 TS_MIN。"""
    if _is_dynamic_window(window):
        return _ts_arg_extreme_dynamic(df, window, True)
    return _ts_arg_extreme(df, _as_int_window(window), True)


def TS_ARGMIN(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """极小值距今 bar 数；window 同 TS_MIN。"""
    if _is_dynamic_window(window):
        return _ts_arg_extreme_dynamic(df, window, False)
    return _ts_arg_extreme(df, _as_int_window(window), False)


def TS_ARGMEDIAN(df: pd.DataFrame, window: Window) -> pd.DataFrame:
    """窗口内最接近中位数的 bar 距今数；window 同 TS_MIN。
    若有多个值与中位数距离相同，取距当前 bar 最近（索引最大）的。"""
    W = _as_int_window(window)

    def _f(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float, copy=False)
        out = _accel.arg_median_fixed(vals, W)
        return pd.Series(out, index=s.index)

    ser = _gb_instrument(df).transform(lambda x: _f(_series_from_group(x)))
    return _out_frame(ser, df)


def TS_ARGNTH(df: pd.DataFrame, window: Window, n: int, ascending: bool = False, unique: bool = False) -> pd.DataFrame:
    """窗口内第 n 大 (ascending=False) 或第 n 小 (ascending=True) 的 bar 距今数；window 同 TS_MIN。
    n >= 1；有效值不足 n 个时输出 NaN。
    - ascending: 排序方向，False=降序（大的在前找第n大），True=升序（小的在前找第n小）
    - unique=False (默认): 有重复值时取距当前 bar 最近的
    - unique=True: 跳过重复值，找严格第 n 个不同的值"""
    W = _as_int_window(window)
    n = max(1, int(n))

    def _f(s: pd.Series) -> pd.Series:
        vals = s.to_numpy(dtype=float, copy=False)
        out = _accel.arg_nth_fixed(vals, W, n, ascending=bool(ascending), unique=bool(unique))
        return pd.Series(out, index=s.index)

    ser = _gb_instrument(df).transform(lambda x: _f(_series_from_group(x)))
    return _out_frame(ser, df)


def TS_LAST_ARGPEAK(df: pd.DataFrame, confirm_window: int = 10) -> pd.DataFrame:
    """最近一次已确认局部波峰距今 bar 数。

    波峰判定规则：某位置为中心、左右各 ``confirm_window`` 根组成的窗口内，
    该中心必须是最高价；只有当右侧 ``confirm_window`` 根都到齐后，该峰才被确认，
    因此输出天然带 ``confirm_window`` 根延迟。
    """
    return _ts_arg_local_extreme_last(df, int(confirm_window), True)


def TS_LAST_ARGTROUGH(df: pd.DataFrame, confirm_window: int = 10) -> pd.DataFrame:
    """最近一次已确认局部波谷距今 bar 数。

    波谷判定规则：某位置为中心、左右各 ``confirm_window`` 根组成的窗口内，
    该中心必须是最低价；只有当右侧 ``confirm_window`` 根都到齐后，该谷才被确认，
    因此输出天然带 ``confirm_window`` 根延迟。
    """
    return _ts_arg_local_extreme_last(df, int(confirm_window), False)


def TS_LAST_PEAK(df: pd.DataFrame, confirm_window: int = 10) -> pd.DataFrame:
    """最近一次已确认局部波峰的价格。

    波峰判定规则同 ``TS_LAST_ARGPEAK``；输出为该峰价格而非距离。
    """
    return _ts_local_extreme_value_last(df, int(confirm_window), True)


def TS_LAST_TROUGH(df: pd.DataFrame, confirm_window: int = 10) -> pd.DataFrame:
    """最近一次已确认局部波谷的价格。

    波谷判定规则同 ``TS_LAST_ARGTROUGH``；输出为该谷价格而非距离。
    """
    return _ts_local_extreme_value_last(df, int(confirm_window), False)


def TS_AMPARGPEAK(df: pd.DataFrame, confirm_window: int = 10) -> pd.DataFrame:
    """已确认波峰中，左右价宽之和最大者距今 bar 数；确认规则同 ``TS_LAST_ARGPEAK``。"""
    return _ts_maxamp_arg_local(df, int(confirm_window), True)


def TS_AMPARGTROUGH(df: pd.DataFrame, confirm_window: int = 10) -> pd.DataFrame:
    """已确认波谷中，左右价宽之和最大者距今 bar 数；确认规则同 ``TS_LAST_ARGTROUGH``。"""
    return _ts_maxamp_arg_local(df, int(confirm_window), False)


def TS_AMPPEAK(df: pd.DataFrame, confirm_window: int = 10) -> pd.DataFrame:
    """已确认波峰中，选左右价宽和最大者，输出该峰价格。"""
    return _ts_maxamp_value_local(df, int(confirm_window), True)


def TS_AMPTROUGH(df: pd.DataFrame, confirm_window: int = 10) -> pd.DataFrame:
    """已确认波谷中，选左右价宽和最大者，输出该谷价格。"""
    return _ts_maxamp_value_local(df, int(confirm_window), False)


def _ts_chan_fractal_3bar_bivariate(
    df_high: pd.DataFrame,
    df_low: pd.DataFrame,
    *,
    want_top: bool,
    want_arg: bool,
) -> pd.DataFrame:
    """三 K 线顶/底分型：``df_high`` / ``df_low`` 为同索引单列（例如 ``$adj_high@60m`` 与 ``$adj_low@60m``）。"""
    if (
        df_high.shape != df_low.shape
        or not df_high.index.equals(df_low.index)
        or df_high.shape[1] != 1
        or df_low.shape[1] != 1
    ):
        raise ValueError(
            "分型算子要求 df_high 与 df_low 为同索引、同形的单列面板"
        )
    result = np.full(len(df_high), np.nan, dtype=np.float32)
    for _, sub_h in _gb_instrument(df_high):
        idx = sub_h.index
        sub_l = df_low.reindex(idx)
        h = sub_h.iloc[:, 0].to_numpy(dtype=float, copy=False)
        lo = sub_l.iloc[:, 0].to_numpy(dtype=float, copy=False)
        out = _accel.fractal_chan_3bar_last(
            h, lo, want_top_fractal=want_top, want_arg=want_arg
        )
        pos = df_high.index.get_indexer(idx)
        result[pos] = out
    return pd.DataFrame(result, index=df_high.index, columns=df_high.columns[:1])


def TS_LAST_ARGBOTTOMFRACTAL(df_high: pd.DataFrame, df_low: pd.DataFrame) -> pd.DataFrame:
    """三 K 线底分型（严格不等）：连续三根编号 1→3 由旧到新须满足 ``h1>h2<h3`` 且 ``l1>l2<l3``。

    在第三根 K 收盘后确认，分型中心为第 2 根；避免前视。输出为距最近一次已确认分型中心的 bar 数。
    """
    return _ts_chan_fractal_3bar_bivariate(
        df_high, df_low, want_top=False, want_arg=True
    )


def TS_LAST_BOTTOMFRACTAL(df_high: pd.DataFrame, df_low: pd.DataFrame) -> pd.DataFrame:
    """底分型确认规则同 ``TS_LAST_ARGBOTTOMFRACTAL``；输出为分型中心 K 的最低价。"""
    return _ts_chan_fractal_3bar_bivariate(
        df_high, df_low, want_top=False, want_arg=False
    )


def TS_LAST_ARGTOPFRACTAL(df_high: pd.DataFrame, df_low: pd.DataFrame) -> pd.DataFrame:
    """三 K 线顶分型（严格不等）：须满足 ``h1<h2>h3`` 且 ``l1<l2>l3``；第三根收盘后确认，中心为第 2 根。"""
    return _ts_chan_fractal_3bar_bivariate(df_high, df_low, want_top=True, want_arg=True)


def TS_LAST_TOPFRACTAL(df_high: pd.DataFrame, df_low: pd.DataFrame) -> pd.DataFrame:
    """顶分型确认规则同 ``TS_LAST_ARGTOPFRACTAL``；输出为分型中心 K 的最高价。"""
    return _ts_chan_fractal_3bar_bivariate(df_high, df_low, want_top=True, want_arg=False)


# 兼容旧名
def TS_ARGPEAK(df: pd.DataFrame, confirm_window: int = 10) -> pd.DataFrame:
    """[兼容] 同 ``TS_LAST_ARGPEAK``。"""
    return TS_LAST_ARGPEAK(df, confirm_window)


def TS_ARGTROUGH(df: pd.DataFrame, confirm_window: int = 10) -> pd.DataFrame:
    """[兼容] 同 ``TS_LAST_ARGTROUGH``。"""
    return TS_LAST_ARGTROUGH(df, confirm_window)


def TS_PEAK(df: pd.DataFrame, confirm_window: int = 10) -> pd.DataFrame:
    """[兼容] 同 ``TS_LAST_PEAK``。"""
    return TS_LAST_PEAK(df, confirm_window)


def TS_TROUGH(df: pd.DataFrame, confirm_window: int = 10) -> pd.DataFrame:
    """[兼容] 同 ``TS_LAST_TROUGH``。"""
    return TS_LAST_TROUGH(df, confirm_window)


# -----------------------------------------------------------------------------
# 截面算子（per datetime）
# -----------------------------------------------------------------------------

_CS_MIN_PAIRS: int = 2


def _validate_cs_panel(df: pd.DataFrame, *, name: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} 第一参数须为面板 DataFrame")
    if df.shape[1] < 1:
        raise ValueError(f"{name} 须至少一列")
    if not isinstance(df.index, pd.MultiIndex):
        raise ValueError(f"{name} 需要 MultiIndex 面板 (datetime, instrument)")
    if "datetime" not in df.index.names:
        raise ValueError(f"{name} 索引须含 datetime 层")


def RANK(df: pd.DataFrame) -> pd.DataFrame:
    """每个 **datetime 截面**内的百分位秩 ∈ [0, 1]（``rank(pct=True, method='average')``）。

    与 ``TS_RANK``（单 instrument 窗口内时序秩）不同。NaN 不参与排序；截面无有效值时为 NaN。"""
    _validate_cs_panel(df, name="RANK")

    def _rank_cs(s: pd.Series) -> pd.Series:
        finite = s.notna()
        if not finite.any():
            return pd.Series(np.nan, index=s.index, dtype=np.float32)
        out = pd.Series(np.nan, index=s.index, dtype=np.float32)
        out.loc[finite] = s.loc[finite].rank(pct=True, method="average").astype(np.float32)
        return out

    return _per_datetime_transform(df, _rank_cs)


def CS_ZSCORE(df: pd.DataFrame, ddof: int = 1) -> pd.DataFrame:
    """截面标准化：``(x - mean) / std``（按 datetime 分组）。

    有效样本 < 2 或 std=0 时该截面输出 NaN；输入 NaN 保持 NaN。"""
    _validate_cs_panel(df, name="CS_ZSCORE")
    d = int(ddof)

    def _zscore_cs(s: pd.Series) -> pd.Series:
        finite = s.notna()
        n = int(finite.sum())
        if n < _CS_MIN_PAIRS:
            return pd.Series(np.nan, index=s.index, dtype=np.float32)
        vals = s.loc[finite].to_numpy(dtype=float, copy=False)
        mu = float(np.mean(vals))
        std = float(np.std(vals, ddof=d))
        if not np.isfinite(std) or std == 0.0:
            return pd.Series(np.nan, index=s.index, dtype=np.float32)
        out = pd.Series(np.nan, index=s.index, dtype=np.float32)
        out.loc[finite] = ((s.loc[finite] - mu) / std).astype(np.float32)
        return out

    return _per_datetime_transform(df, _zscore_cs)


def CS_DEMEAN(df: pd.DataFrame) -> pd.DataFrame:
    """截面去均值：``x - mean``（按 datetime 分组）。有效样本 < 1 时该截面为 NaN。"""
    _validate_cs_panel(df, name="CS_DEMEAN")

    def _demean_cs(s: pd.Series) -> pd.Series:
        finite = s.notna()
        if not finite.any():
            return pd.Series(np.nan, index=s.index, dtype=np.float32)
        mu = float(s.loc[finite].mean())
        out = pd.Series(np.nan, index=s.index, dtype=np.float32)
        out.loc[finite] = (s.loc[finite] - mu).astype(np.float32)
        return out

    return _per_datetime_transform(df, _demean_cs)


def CS_WINSORIZE(
    df: pd.DataFrame,
    lower_pct: float,
    upper_pct: float,
) -> pd.DataFrame:
    """截面分位裁剪：将每个 datetime 截面内的值限制在 ``[lower_pct, upper_pct]`` 分位之间。

    参数为 [0, 1] 分位（如 ``0.01, 0.99``）。截面无有效值时为 NaN。"""
    _validate_cs_panel(df, name="CS_WINSORIZE")
    lo = float(lower_pct)
    hi = float(upper_pct)
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError("CS_WINSORIZE 要求 0 <= lower_pct < upper_pct <= 1")

    def _winsor_cs(s: pd.Series) -> pd.Series:
        finite = s.notna()
        if not finite.any():
            return pd.Series(np.nan, index=s.index, dtype=np.float32)
        valid = s.loc[finite]
        q_lo = float(valid.quantile(lo))
        q_hi = float(valid.quantile(hi))
        out = pd.Series(np.nan, index=s.index, dtype=np.float32)
        clipped = valid.clip(lower=q_lo, upper=q_hi).astype(np.float32)
        out.loc[finite] = clipped
        return out

    return _per_datetime_transform(df, _winsor_cs)


def CS_BUCKET(df: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """截面等频分 N 组：每个 datetime 内对变量做 ``qcut``，输出组号 ``0…K-1``（``K≤N``）。

    可与 ``CS_NEUTRALIZE`` 配合做组内去均值，例如
    ``CS_NEUTRALIZE(raw, CS_BUCKET(LOG($float_cap), 10))``。
    有效样本数 < ``n_bins`` 时该截面为 NaN；输入 NaN 保持 NaN。"""
    _validate_cs_panel(df, name="CS_BUCKET")
    n = int(n_bins)
    if n < 2:
        raise ValueError("CS_BUCKET 要求 n_bins >= 2")

    def _bucket_cs(s: pd.Series) -> pd.Series:
        finite = s.notna()
        if not finite.any():
            return pd.Series(np.nan, index=s.index, dtype=np.float32)
        if int(finite.sum()) < n:
            return pd.Series(np.nan, index=s.index, dtype=np.float32)
        valid = s.loc[finite]
        try:
            codes = pd.qcut(valid, n, labels=False, duplicates="drop")
        except ValueError:
            codes = pd.qcut(valid.rank(method="first"), n, labels=False, duplicates="drop")
        out = pd.Series(np.nan, index=s.index, dtype=np.float32)
        out.loc[finite] = codes.astype(np.float32)
        return out

    return _per_datetime_transform(df, _bucket_cs)


def CS_NEUTRALIZE(x: pd.DataFrame, group: pd.DataFrame) -> pd.DataFrame:
    """截面组内去均值：每个 datetime 内按 ``group`` 分组，输出 ``x - group_mean(x)``。

    组内仅 1 个有效值时输出 **0**（已中性化）；输入 NaN 保持 NaN。
    ``group`` 须为离散组号，可用 ``CS_BUCKET(var, N)`` 构造，例如
    ``CS_NEUTRALIZE(raw, CS_BUCKET(LOG($float_cap), 10))``。"""
    _validate_cs_panel(x, name="CS_NEUTRALIZE")
    _validate_cs_panel(group, name="CS_NEUTRALIZE")
    if not x.index.equals(group.index):
        raise ValueError("CS_NEUTRALIZE: x 与 group 须同索引")

    xs = _first_series(x).to_numpy(dtype=float, copy=False)
    gs = _first_series(group).to_numpy(dtype=float, copy=False)
    result = np.full(len(x), np.nan, dtype=np.float32)

    for _, sub in x.groupby(level="datetime", sort=False):
        pos = x.index.get_indexer(sub.index)
        x_day = xs[pos]
        g_day = gs[pos]
        uniq = np.unique(g_day[np.isfinite(g_day)])
        for g_val in uniq:
            x_mask = np.isfinite(g_day) & (g_day == g_val) & np.isfinite(x_day)
            n = int(x_mask.sum())
            if n == 0:
                continue
            if n == 1:
                result[pos[x_mask]] = 0.0
                continue
            mu = float(np.mean(x_day[x_mask]))
            result[pos[x_mask]] = (x_day[x_mask] - mu).astype(np.float32)

    return pd.DataFrame(result, index=x.index, columns=x.columns[:1])


# -----------------------------------------------------------------------------
# 二元与逻辑
# -----------------------------------------------------------------------------


def ADD(df1, df2):
    """逐元素加；双面板单列首列对齐。"""
    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        return _binary_op_panel_df(df1, df2, np.add)
    return np.add(df1, df2)


def SUBTRACT(df1, df2):
    """逐元素减；规则同 ADD。"""
    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        return _binary_op_panel_df(df1, df2, np.subtract)
    return np.subtract(df1, df2)


def MULTIPLY(df1, df2):
    """逐元素乘；规则同 ADD。"""
    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        return _binary_op_panel_df(df1, df2, np.multiply)
    return np.multiply(df1, df2)


def DIVIDE(df1, df2):
    """逐元素除；规则同 ADD。"""
    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        return _binary_op_panel_df(df1, df2, np.divide)
    return np.divide(df1, df2)


def MAXIMUM(x: object, y: object, z: Optional[Any] = None) -> pd.DataFrame:
    """两/三列逐元素 max（非 TS_MAX）；任一参可为面板同索引广播用的数值标量。"""
    if z is None:
        return _binary_op_panel_mixed(x, y, np.maximum)
    return MAXIMUM(MAXIMUM(x, y), z)


def MINIMUM(x: object, y: object, z: Optional[Any] = None) -> pd.DataFrame:
    """两/三列逐元素 min；同 MAXIMUM，支持对标量边界（如 clip）。"""
    if z is None:
        return _binary_op_panel_mixed(x, y, np.minimum)
    return MINIMUM(MINIMUM(x, y), z)


def LT(df1, df2):
    """双面板逐元素 ``<``；列名可不同，规则同 ``ADD``。"""
    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        return _binary_op_panel_df(df1, df2, np.less)
    return np.less(df1, df2)


def GT(df1, df2):
    """双面板逐元素 ``>``。"""
    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        return _binary_op_panel_df(df1, df2, np.greater)
    return np.greater(df1, df2)


def LE(df1, df2):
    """双面板逐元素 ``<=``。"""
    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        return _binary_op_panel_df(df1, df2, np.less_equal)
    return np.less_equal(df1, df2)


def GE(df1, df2):
    """双面板逐元素 ``>=``。"""
    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        return _binary_op_panel_df(df1, df2, np.greater_equal)
    return np.greater_equal(df1, df2)


def EQ(df1, df2):
    """双面板逐元素 ``==``。"""
    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        return _binary_op_panel_df(df1, df2, np.equal)
    return np.equal(df1, df2)


def NE(df1, df2):
    """双面板逐元素 ``!=``。"""
    if isinstance(df1, pd.DataFrame) and isinstance(df2, pd.DataFrame):
        return _binary_op_panel_df(df1, df2, np.not_equal)
    return np.not_equal(df1, df2)


def AND(df1, df2):
    """按位与（先转 bool）。"""
    return np.bitwise_and(df1.astype(np.bool_), df2.astype(np.bool_))


def OR(df1, df2):
    """按位或（先转 bool）。"""
    return np.bitwise_or(df1.astype(np.bool_), df2.astype(np.bool_))


# -----------------------------------------------------------------------------
# 回归 / 斜率（时序滚动，无截面）
# -----------------------------------------------------------------------------


def SEQUENCE(n: int) -> np.ndarray:
    """长度 n 的 1…n 浮点向量，用于回归自变量形状。"""
    n = int(n)
    if n < 1:
        raise ValueError("SEQUENCE(n) 需要 n>=1")
    return np.linspace(1.0, float(n), n, dtype=np.float32)


def calculate_beta(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if y.shape[0] != x.shape[0]:
        raise ValueError(
            f"calculate_beta: y/x 长度须一致 (len(y)={y.shape[0]}, len(x)={x.shape[0]})。"
            "REGBETA/REGRESI 请将 df2 配成 SEQUENCE(p)，或使用 REGBETA(df,p) 的标量窗口写法。"
        )
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2:
        return float("nan")
    yv, xv = y[mask], x[mask]
    X = np.column_stack([xv, np.ones(len(xv))])
    beta, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
    return float(beta[0])


def rolling_beta(df1_group: pd.DataFrame, df2_vec: np.ndarray, p: int) -> pd.Series:
    y = df1_group.iloc[:, 0]
    x = df2_vec[:p]
    n = len(y)
    out = np.full(n, np.nan)
    for i in range(p - 1, n):
        yy = y.iloc[i - p + 1 : i + 1].to_numpy()
        out[i] = calculate_beta(yy, x)
    return pd.Series(out, index=df1_group.index)


def REGBETA(
    df1: pd.DataFrame, df2: Union[pd.DataFrame, np.ndarray], p: int = 5, n_jobs: int = 1
) -> pd.DataFrame:
    """滚动 OLS 斜率 β；df2 为 ndarray 取前 p 元为自变量；DataFrame 时对齐索引且自变量为 SEQUENCE(p)；零维数值则视为窗口长度。
    （``REGBETA(df, 20)`` 等价 ``REGBETA(df, SEQUENCE(20), 20)``。）"""
    p = int(p)
    if isinstance(df2, np.ndarray):
        xvec = np.asarray(df2, dtype=float).ravel()[:p]
    elif isinstance(df2, pd.DataFrame):
        assert df1.index.equals(df2.index), "df1 与 df2 索引须对齐"
        xvec = SEQUENCE(p)
    elif np.ndim(df2) == 0:
        p = int(np.asarray(df2).item())
        if p < 1:
            raise ValueError("REGBETA: 窗口长度须 >= 1")
        xvec = SEQUENCE(p)
    else:
        raise TypeError(
            "REGBETA: df2 须为 DataFrame、ndarray，或窗口长度（如 int）；"
            "若第二参为 ndarray 且较短，勿短于滚动窗 p。"
        )
    parts = Parallel(n_jobs=n_jobs)(
        delayed(rolling_beta)(grp, xvec, p) for _, grp in _gb_instrument(df1)
    )
    ser = pd.concat(parts).sort_index()
    return _out_frame(ser, df1)


def SLOPE(df: pd.DataFrame, p: int = 5, n_jobs: int = 1) -> pd.DataFrame:
    """对时间 1…p 的滚动斜率，等价 REGBETA(df, SEQUENCE(p), p)。"""
    return REGBETA(df, SEQUENCE(int(p)), p, n_jobs=n_jobs)


def calculate_residuals(y: np.ndarray, x: np.ndarray) -> float:
    y = np.asarray(y, dtype=float).ravel()
    x = np.asarray(x, dtype=float).ravel()
    if y.shape[0] != x.shape[0]:
        raise ValueError(
            f"calculate_residuals: y/x 长度须一致 (len(y)={y.shape[0]}, len(x)={x.shape[0]})。"
            "若只要固定窗残差请用 RESI(df, p) 或 REGRESI(df, SEQUENCE(p), p)。"
        )
    mask = np.isfinite(y) & np.isfinite(x)
    if mask.sum() < 2:
        return float("nan")
    yv, xv = y[mask], x[mask]
    X = np.column_stack([xv, np.ones(len(xv))])
    b, _, _, _ = np.linalg.lstsq(X, yv, rcond=None)
    pred = b[0] * xv + b[1]
    return float(yv[-1] - pred[-1])


def rolling_residuals(df1_group: pd.DataFrame, x: np.ndarray, p: int) -> pd.Series:
    y = df1_group.iloc[:, 0]
    n = len(y)
    out = np.full(n, np.nan)
    for i in range(p - 1, n):
        yy = y.iloc[i - p + 1 : i + 1].to_numpy()
        out[i] = calculate_residuals(yy, x)
    return pd.Series(out, index=df1_group.index)


def REGRESI(
    df1: pd.DataFrame, df2: Union[pd.DataFrame, np.ndarray], p: int = 5, n_jobs: int = 1
) -> pd.DataFrame:
    """滚动 OLS 末点残差； ndarray 取自变量 ravel(df2)[:p]；零维数值则视为窗口长度并按 SEQUENCE(p)（``REGRESI(df, 20)`` 等同 ``RESI(df,20)``）。"""
    p_roll = int(p)
    if isinstance(df2, pd.DataFrame):
        assert df1.index.equals(df2.index), "df1 与 df2 索引须对齐"
        xvec = SEQUENCE(p_roll)
    elif isinstance(df2, np.ndarray):
        xvec = np.asarray(df2, dtype=float).ravel()[:p_roll]
    elif np.ndim(df2) == 0:
        p_roll = int(np.asarray(df2).item())
        if p_roll < 1:
            raise ValueError("REGRESI: 窗口长度须 >= 1")
        xvec = SEQUENCE(p_roll)
    else:
        xvec = np.asarray(df2, dtype=float).ravel()[:p_roll]
    parts = Parallel(n_jobs=n_jobs)(
        delayed(rolling_residuals)(grp, xvec, p_roll) for _, grp in _gb_instrument(df1)
    )
    ser = pd.concat(parts).sort_index()
    return _out_frame(ser, df1)


def RESI(df1: pd.DataFrame, p: int = 5, n_jobs: int = 1) -> pd.DataFrame:
    """对 SEQUENCE(p) 的滚动残差，等价 REGRESI(df1, SEQUENCE(p), p)。"""
    return REGRESI(df1, SEQUENCE(int(p)), p, n_jobs=n_jobs)


# -----------------------------------------------------------------------------
# 广义拥挤度（CROWD_*）：dimension 分桶 → 成交/属性是否扎堆
#
# 与 CHIP_*（价轴筹码直方图）、VOLUME_CLOCK_VPIN（成交量时钟）勿混用。
# -----------------------------------------------------------------------------


def _parse_crowd_bucket_params(
    side_or_nbuckets: str | int | float = "high",
    split_or_bucket_idx: str | int | float = 0.5,
) -> dict[str, Any]:
    """解析分桶参数：字符串 ``side`` + ``split``，或整数 ``n_buckets`` + ``bucket_idx``。"""
    if isinstance(side_or_nbuckets, str):
        side = side_or_nbuckets.strip().lower()
        if side not in ("high", "low"):
            raise ValueError(
                f"CROWD_* side 须为 'high' 或 'low'，收到: {side_or_nbuckets!r}"
            )
        split = float(split_or_bucket_idx)
        if not (0.0 < split < 1.0):
            raise ValueError(f"CROWD_* split 须在 (0, 1)，收到: {split}")
        return {
            "bucket_mode": "quantile",
            "side": side,
            "split": split,
            "n_buckets": 2,
            "bucket_idx": 1,
        }
    try:
        nb = int(side_or_nbuckets)
        bidx = int(split_or_bucket_idx)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "CROWD_* 分桶参数：第 4 参为 'high'/'low' + split，"
            "或整数 n_buckets + bucket_idx"
        ) from e
    if nb < 2 or bidx < 1 or bidx > nb:
        raise ValueError(
            f"CROWD_* 等频分桶须 2 <= n_buckets 且 1 <= bucket_idx <= n_buckets，"
            f"收到 n_buckets={nb}, bucket_idx={bidx}"
        )
    return {
        "bucket_mode": "equal_freq",
        "side": "high",
        "split": 0.5,
        "n_buckets": nb,
        "bucket_idx": bidx,
    }


def _crowd_roll_panel(
    dimension: pd.DataFrame,
    attribute: pd.DataFrame,
    weight: pd.DataFrame,
    window: int,
    op: str,
    *,
    bucket_mode: str = "quantile",
    side: str = "high",
    split: float = 0.5,
    n_buckets: int = 2,
    bucket_idx: int = 1,
    min_valid: int = 0,
    use_attr: bool = True,
    use_weight: bool = True,
) -> pd.DataFrame:
    w = max(1, int(window))
    result = np.full(len(dimension), np.nan, dtype=np.float32)
    for _, sub_d in _gb_instrument(dimension):
        idx = sub_d.index
        sub_a = attribute.reindex(idx)
        sub_w = weight.reindex(idx)
        d_arr = sub_d.iloc[:, 0].to_numpy(dtype=float, copy=False)
        a_arr = sub_a.iloc[:, 0].to_numpy(dtype=float, copy=False)
        w_arr = sub_w.iloc[:, 0].to_numpy(dtype=float, copy=False)
        pos = dimension.index.get_indexer(idx)
        result[pos] = _accel.roll_crowd_fixed(
            d_arr,
            a_arr,
            w_arr,
            w,
            op,
            bucket_mode=bucket_mode,
            side=side,
            split=float(split),
            n_buckets=int(n_buckets),
            bucket_idx=int(bucket_idx),
            min_valid=int(min_valid),
            use_attr=use_attr,
            use_weight=use_weight,
        )
    return pd.DataFrame(
        result, index=dimension.index, columns=dimension.columns[:1]
    )


def CROWD_SHARE(
    dimension: pd.DataFrame,
    weight: pd.DataFrame,
    window: int,
    side_or_nbuckets: str | int | float = "high",
    split_or_bucket_idx: str | int | float = 0.5,
) -> pd.DataFrame:
    """**量拥挤占比** ∈ [0,1]：窗口内 ``weight`` 有多少落在指定 **dimension 环境桶**。

**问什么**：成交（``weight``，通常 ``$volume``）是否**扎堆**在贵/便宜、开盘/尾盘、高波动、放量等环境（由 ``dimension`` 定义）。

**分桶**（第 4、5 位置参数，禁止关键字）：
``'high'/'low'`` + ``split``（``high``→``dim>=``窗口分位；``low``→``dim<``分位；如 ``'high',0.9``≈最高十分位）；
或 ``n_buckets, bucket_idx``（等频；``1``=最低档，``K``=最高档）。

``dimension`` 例：``$adj_vwap``、``TS_MONTH_POS($adj_close)``、``TS_STD($ret,5)``、``$float_cap``。
目标桶无有效 weight → NaN。无前视。"""
    cfg = _parse_crowd_bucket_params(side_or_nbuckets, split_or_bucket_idx)
    dummy = pd.DataFrame(0.0, index=dimension.index, columns=dimension.columns[:1])
    return _crowd_roll_panel(
        dimension,
        dummy,
        weight,
        window,
        "share",
        bucket_mode=cfg["bucket_mode"],
        side=cfg["side"],
        split=cfg["split"],
        n_buckets=cfg["n_buckets"],
        bucket_idx=cfg["bucket_idx"],
        use_attr=False,
        use_weight=True,
    )


def CROWD_MEAN_RATIO(
    dimension: pd.DataFrame,
    attribute: pd.DataFrame,
    window: int,
    side_or_nbuckets: str | int | float = "high",
    split_or_bucket_idx: str | int | float = 0.5,
) -> pd.DataFrame:
    """**属性抬升倍数**：目标环境桶内 ``attribute`` 均值 / 全窗 ``attribute`` 均值。

**问什么**：在 dimension 划出的环境桶里（如高成交量时段），``attribute``（如 ``$adj_vwap``）比全天平均水平高/低多少倍。

分桶同 ``CROWD_SHARE``（第 4、5 参）。全窗均值为 0 或目标桶为空 → NaN。
例：``CROWD_MEAN_RATIO($volume, $adj_vwap, 48, 'high', 0.9)``。"""
    cfg = _parse_crowd_bucket_params(side_or_nbuckets, split_or_bucket_idx)
    dummy_w = pd.DataFrame(1.0, index=dimension.index, columns=dimension.columns[:1])
    return _crowd_roll_panel(
        dimension,
        attribute,
        dummy_w,
        window,
        "mean_ratio",
        bucket_mode=cfg["bucket_mode"],
        side=cfg["side"],
        split=cfg["split"],
        n_buckets=cfg["n_buckets"],
        bucket_idx=cfg["bucket_idx"],
        use_attr=True,
        use_weight=False,
    )


def CROWD_CONTRAST(
    dimension: pd.DataFrame,
    attribute: pd.DataFrame,
    window: int,
    split: float = 0.5,
) -> pd.DataFrame:
    """**高低环境差**：高维区 ``attribute`` 均值 − 低维区 ``attribute`` 均值。

**问什么**：dimension 高端 vs 低端环境里，``attribute`` 水平差多少（默认 ``split=0.5`` 中位数二分）。
``dimension`` 与 ``attribute`` 应为不同语义列（如 ``$adj_vwap`` 分桶、``$ret`` 作属性）；

第 4 参 ``split``（``(0,1)``）。任一侧无样本 → NaN。无前视。"""
    sq = float(split)
    if not (0.0 < sq < 1.0):
        raise ValueError(f"CROWD_CONTRAST split 须在 (0, 1)，收到: {split}")
    dummy_w = pd.DataFrame(1.0, index=dimension.index, columns=dimension.columns[:1])
    return _crowd_roll_panel(
        dimension,
        attribute,
        dummy_w,
        window,
        "contrast",
        bucket_mode="quantile",
        side="high",
        split=sq,
        use_attr=True,
        use_weight=False,
    )


def CROWD_RANK_WEIGHTED(
    dimension: pd.DataFrame,
    attribute: pd.DataFrame,
    window: int,
    weight: pd.DataFrame,
) -> pd.DataFrame:
    """**软倾斜加权**：``Σ rank_norm(dim)·attr·weight / Σ weight``（不硬切桶）。

**问什么**：dimension 越高的 bar，其 ``attribute`` 在 ``weight`` 加权平均里话语权越大（``rank_norm``∈[0,1] 为窗口内平均秩）。

``weight`` 通常 ``$volume``。有效样本 <2 或 ``Σweight<=0`` → NaN。
价量**线性共动**用 ``TS_CORR``，勿与本算子混用。无前视。"""
    return _crowd_roll_panel(
        dimension,
        attribute,
        weight,
        window,
        "rank_weighted",
        bucket_mode="quantile",
        side="high",
        split=0.5,
        use_attr=True,
        use_weight=True,
    )


# -----------------------------------------------------------------------------
# 兼容别名：旧表达式若写 MAX(A,B) 逐元素，映射到 MAXIMUM
# -----------------------------------------------------------------------------

MAX = MAXIMUM
MIN = MINIMUM


# -----------------------------------------------------------------------------
# __main__
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    idx = pd.MultiIndex.from_product(
        [
            pd.date_range("2020-01-01", periods=5, freq="min"),
            ["S1"],
        ],
        names=["datetime", "instrument"],
    )
    demo = pd.DataFrame({"high": [1.0, 3.0, 2.0, 4.0, 1.0]}, index=idx)
    th = TS_ARGMAX(demo, 3)
    dyn = pd.DataFrame({"w": [1.0, 2.0, 2.0, 3.0, 2.0]}, index=idx)
    m = TS_MIN(demo, dyn)
    assert callable(DELTA) and callable(TS_MIN)
    print("function_registry OK (futures time-series only):", th.iloc[-1, 0], m.iloc[-1, 0])
