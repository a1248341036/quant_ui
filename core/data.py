from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

import pandas as pd

from .store import (DATA_DIR, ETF_FILE, ETF_PANEL_FILE, FUND_FILE, FUND_NAV_FILE,
                    FUND_PANEL_FILE, INDEX_FILE, LEGACY_DATA_DIR, PANEL_FILE,
                    PG_PARQUET_DIR, SENTIMENT_DIR, TECH_FILE, UNIVERSE_FILE,
                    load_meta)


def duck_query(sql: str, params=None) -> pd.DataFrame:
    """DuckDB SQL 查询（基于 data/ 下 parquet/csv 的视图）。"""
    from .db import query
    return query(sql, params)


PANEL_PATH = LEGACY_DATA_DIR / "panel/turn20/turn20_fast_panel_cs800_2020-01-01_2026-08-13.parquet"
UNIVERSE_PATH = LEGACY_DATA_DIR / "panel/universe_cs800.csv"
TECH_PATH = LEGACY_DATA_DIR / "panel/tech_universe_sw.csv"
INDEX_PATH = LEGACY_DATA_DIR / "panel/csi300_index.csv"

SENT_ROOT = SENTIMENT_DIR

# PG 面板只拉该日期以来的行情，避免全表 1100 万行进入 pandas（与 parquet 面板口径一致）。
PANEL_START = os.getenv("QUANT_PANEL_START", "2020-01-01").strip()
# turn20/am20 滚动窗口在区间起点需要约 20 个交易日历史，查询起点统一前移自然日缓冲，
# 保证区间化加载与全量加载的因子口径一致。
FACTOR_BUFFER_DAYS = 40

# pg=优先 PostgreSQL（失败回退本地 parquet）；pg_parquet=优先本地 PG 导出 parquet
# （失败回退 PostgreSQL）；panel=只用本地文件。小内存机器默认走 pg_parquet，
# 运行时不再整表读 PG 行情。
DATA_SOURCE = os.getenv("QUANT_DATA_SOURCE", "pg_parquet").strip().lower()
# 与 backend/services 共用：空闲 TTL 后释放整份 PG 面板，
# 避免全市场面板（几百 MB）在“什么都没跑”时仍被永久持有。
PANEL_CACHE_TTL = float(os.getenv("QUANT_CACHE_TTL", "1800"))
_pg_panel_cache: dict = {}
_panel_codes_cache: set | None = None
_pg_panel_lock = threading.Lock()

# 信号因子最长只看 60 个交易日（ma_cross20_60），
# 保留 800 个自然日（约 550 个交易日）足够覆盖全部滚动窗口。
SIGNAL_LOOKBACK_DAYS = int(os.getenv("QUANT_SIGNAL_LOOKBACK_DAYS", "800"))


# 导出 PG 时同时写一份「每只股票最新复权因子」小文件，作为前复权锚点：
# qfq 价 = 不复权价 * adj_factor / last_adj。锚点取自导出快照而不是查询区间，
# 避免同一历史日在不同查询区间/不同刷新时间下显示不同的价格。
LAST_ADJ_FILE = PG_PARQUET_DIR / "stock_daily_last_adj.parquet"
_last_adj_cache: dict | None = None
_last_adj_lock = threading.Lock()


def _load_last_adj() -> pd.DataFrame | None:
    """读导出快照的最新复权因子（ts_code, last_adj），进程内缓存。"""
    global _last_adj_cache
    if _last_adj_cache is not None:
        return _last_adj_cache
    if not LAST_ADJ_FILE.exists():
        return None
    with _last_adj_lock:
        if _last_adj_cache is not None:
            return _last_adj_cache
        try:
            _last_adj_cache = pd.read_parquet(LAST_ADJ_FILE)
            if "last_adj" not in _last_adj_cache.columns:
                _last_adj_cache = _last_adj_cache.rename(
                    columns={"adj_factor": "last_adj"})
            return _last_adj_cache
        except Exception as exc:
            print(f"加载复权因子快照失败: {exc}", file=sys.stderr)
            return None


def reset_last_adj_cache() -> None:
    """数据导出后清掉进程内复权锚点缓存。"""
    global _last_adj_cache
    _last_adj_cache = None


def _finalize_stock_df(df: pd.DataFrame, last_adj: pd.DataFrame | None = None,
                       adj: str = "qfq") -> pd.DataFrame:
    """复权价 + turn20/am20 因子 + 类型压缩（PG stock_daily -> 面板口径）。

    adj: qfq=前复权（默认，锚点为导出快照的最新因子）；
         hfq=后复权（不复权价 * 当日因子，历史值永不随新分红漂移，但绝对价格
             不是真实成交价，仅适合研究/回测收益率口径）；
         raw=不复权原始价（真实成交价，除权日会有跳空）。
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df["code"] = df["ts_code"].str[:6]
    # PG 统一存 %（0.4289），面板历史口径是比例（0.0043），转回比例保持一致
    df["turnover"] = df["turnover"] / 100.0
    adj = (adj or "qfq").strip().lower()
    if adj == "qfq":
        if last_adj is None:
            snap = _load_last_adj()
            if snap is not None and len(snap):
                present = df["ts_code"].unique()
                last_adj = snap[snap["ts_code"].isin(present)]
            if last_adj is None or len(last_adj) == 0:
                # 无快照时回退：取查询区间内各股票最新因子（原逻辑）
                last_adj = (df.groupby("ts_code", observed=True)["adj_factor"]
                            .last().reset_index()
                            .rename(columns={"adj_factor": "last_adj"}))
            else:
                # 快照缺股票（如停牌未出现在最后交易日）时，用区间内最新因子补齐，
                # 避免 merge 后 last_adj 为 NaN 导致价格全部变成 NaN。
                range_last = (df.groupby("ts_code", observed=True)["adj_factor"]
                              .last().reset_index()
                              .rename(columns={"adj_factor": "last_adj"}))
                missing = range_last[~range_last["ts_code"].isin(last_adj["ts_code"])]
                if len(missing):
                    last_adj = pd.concat([last_adj[["ts_code", "last_adj"]], missing],
                                         ignore_index=True)
        df = df.merge(last_adj, on="ts_code", how="left")
        for c in ("open", "high", "low", "close"):
            df[c] = df[c] * df["adj_factor"] / df["last_adj"]
    elif adj == "hfq":
        for c in ("open", "high", "low", "close"):
            df[c] = df[c] * df["adj_factor"]
    # raw: 不复权原始价，不改动
    df = df.sort_values(["ts_code", "date"])
    g = df.groupby("ts_code", observed=True)
    df["turn20"] = g["turnover"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    df["am20"] = g["amount"].transform(lambda s: s.rolling(20, min_periods=15).mean())
    for c in ("turn20", "am20", "volume"):
        df[c] = df[c].astype("float32")
    df["code"] = df["code"].astype("category")
    return df[["date", "open", "high", "low", "close", "turnover", "amount", "code",
               "turn20", "am20", "volume"]]


def _finalize_panel_df(df: pd.DataFrame) -> pd.DataFrame:
    """把 DuckDB 查询结果整理成引擎面板口径（与 _finalize_stock_df 输出一致）。"""
    df["code"] = df["code"].astype(str).str.zfill(6)
    df["date"] = pd.to_datetime(df["date"])
    for c in ("turn20", "am20", "volume"):
        if c in df.columns:
            df[c] = df[c].astype("float32")
    df["code"] = df["code"].astype("category")
    cols = [c for c in ("date", "open", "high", "low", "close", "turnover",
                        "amount", "code", "turn20", "am20", "volume")
            if c in df.columns]
    return df[cols]


def _duck_query(sql: str, params: list) -> pd.DataFrame | None:
    """DuckDB 查询；不可用/失败返回 None，由调用方回退。"""
    try:
        from .db import query as duck_query
        return duck_query(sql, params)
    except Exception:
        return None


def _panel_sql_where(start: str | None, end: str | None,
                     codes: list[str] | None) -> tuple[str, list]:
    conds: list[str] = []
    params: list = []
    if start:
        conds.append("date >= CAST(? AS DATE)")
        params.append(start)
    if end:
        conds.append("date <= CAST(? AS DATE)")
        params.append(end)
    if codes:
        six = sorted({str(c).zfill(6) for c in codes})
        marks = ", ".join(["?"] * len(six))
        conds.append(f"code IN ({marks})")
        params.extend(six)
    return (" AND ".join(conds) if conds else "1=1"), params


def _duck_panel_slice(start: str | None = None, end: str | None = None,
                      codes: list[str] | None = None) -> pd.DataFrame:
    """用 DuckDB 在预计算 panel.parquet 上做区间/股票池过滤。

    因子（turn20/am20）已由每日刷新预计算好，这里只做 pyarrow/DuckDB
    下推读取，返回筛选后的小 DataFrame；不在请求路径上对 1170 万行
    stock_daily 现算窗口因子（实测窗口 SQL 在 3.6G 机器上峰值 3GB+）。
    """
    where, params = _panel_sql_where(start, end, codes)
    sql = f"""
    SELECT date, open, close, turnover, amount, code, turn20, am20, volume
    FROM panel
    WHERE {where}
    ORDER BY code, date
    """
    df = _duck_query(sql, params)
    if df is None or df.empty:
        raise RuntimeError("DuckDB 面板为空")
    return _finalize_panel_df(df)


def _pg_parquet_end() -> str:
    """pg_parquet/stock_daily.parquet 最新交易日（优先 DuckDB，回退列投影）。"""
    try:
        df = _duck_query("SELECT max(trade_date) AS end FROM stock_daily", [])
        if df is not None and len(df) and df.iloc[0]["end"] is not None:
            return str(pd.Timestamp(df.iloc[0]["end"]).date())
    except Exception:
        pass
    path = PG_PARQUET_DIR / "stock_daily.parquet"
    dates = pd.read_parquet(path, columns=["trade_date"])
    return str(pd.Timestamp(dates["trade_date"].max()).date())

def _code_to_ts_map() -> dict[str, str]:
    """6 位代码 -> 完整 ts_code（优先 pg_parquet/stock_basic，缺失回退 PG）。"""
    try:
        path = PG_PARQUET_DIR / "stock_basic.parquet"
        if path.exists():
            df = pd.read_parquet(path, columns=["ts_code", "symbol"])
            return {str(r["symbol"]).zfill(6): str(r["ts_code"]) for _, r in df.iterrows()}
    except Exception:
        pass
    try:
        from .pg import query_df
        df = query_df("SELECT ts_code, symbol FROM stock_basic")
        return {str(r["symbol"]).zfill(6): str(r["ts_code"]) for _, r in df.iterrows()}
    except Exception:
        return {}


def _load_panel_pg_parquet(start: str | None = None, end: str | None = None,
                           codes: list[str] | None = None) -> pd.DataFrame:
    """从每日导出的 pg_parquet/stock_daily.parquet 构建回测面板（与 PG 同口径）。"""
    path = PG_PARQUET_DIR / "stock_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"PG parquet 不存在: {path}")
    if end is None:
        end = _pg_parquet_end()
    calc_start = start or PANEL_START
    if start:
        calc_start = (pd.Timestamp(start)
                      - pd.Timedelta(days=FACTOR_BUFFER_DAYS)).date().isoformat()
    filters: list = []
    if codes:
        code_map = _code_to_ts_map()
        ts_codes = [code_map.get(str(c).zfill(6), f"{str(c).zfill(6)}.SZ")
                    for c in codes]
        filters.append(("ts_code", "in", ts_codes))
    filters.append(("trade_date", ">=", pd.Timestamp(calc_start).date()))
    filters.append(("trade_date", "<=", pd.Timestamp(end).date()))
    cols = ["ts_code", "trade_date", "open", "high", "low", "close",
            "vol", "amount", "turnover_rate", "adj_factor"]
    df = pd.read_parquet(path, columns=cols, filters=filters)
    if len(df) == 0:
        raise RuntimeError("PG parquet 面板为空")
    df = df.rename(columns={"trade_date": "date", "vol": "volume",
                            "turnover_rate": "turnover"})
    return _finalize_stock_df(df)


def _load_panel_precomputed(start: str | None = None, end: str | None = None,
                            codes: list[str] | None = None) -> pd.DataFrame:
    """读预计算 panel.parquet（已含 turn20/am20），pyarrow 下推区间/股票池过滤。"""
    if PANEL_FILE.exists():
        path = PANEL_FILE
    elif PANEL_PATH.exists():
        path = PANEL_PATH
    else:
        raise FileNotFoundError(f"面板数据不存在: {PANEL_PATH} 或 {PANEL_FILE}")
    filters: list = []
    if codes:
        filters.append(("code", "in", [str(c).zfill(6) for c in codes]))
    if start:
        filters.append(("date", ">=", pd.Timestamp(start)))
    if end:
        filters.append(("date", "<=", pd.Timestamp(end)))
    try:
        panel = pd.read_parquet(path, filters=filters or None)
    except Exception:
        panel = pd.read_parquet(path)
        if start:
            panel = panel[panel["date"] >= pd.Timestamp(start)]
        if end:
            panel = panel[panel["date"] <= pd.Timestamp(end)]
        if codes:
            panel = panel[panel["code"].isin([str(c).zfill(6) for c in codes])]
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    for c in ("turn20", "am20", "volume"):
        if c in panel.columns and panel[c].dtype == "float64":
            panel[c] = panel[c].astype("float32")
    panel["code"] = panel["code"].astype("category")
    return panel


def _load_panel_pg(start: str | None = None, end: str | None = None,
                   codes: list[str] | None = None) -> pd.DataFrame:
    """从 PostgreSQL stock_daily 构建回测面板（前复权 + turn20/am20 因子）。

    start/end 可选，只拉回测所需区间；实际查询起点前移 FACTOR_BUFFER_DAYS，
    保证因子滚动窗口与全量加载口径一致。codes 限定股票池（如科技TMT 90 只），
    避免 5600 只全市场行情进入 pandas。
    """
    global _pg_panel_cache
    from .pg import configured as pg_configured, query_df
    if not pg_configured():
        raise RuntimeError("PG_DSN 未配置")
    meta_end = query_df(
        "SELECT max(trade_date) AS end FROM stock_daily"
    ).iloc[0]["end"]
    if meta_end is None:
        raise RuntimeError("stock_daily 为空")
    calc_start = start or PANEL_START
    if start:
        calc_start = (pd.Timestamp(start)
                      - pd.Timedelta(days=FACTOR_BUFFER_DAYS)).date().isoformat()
    codes_key = tuple(sorted(codes)) if codes else None
    key = (calc_start, end or str(meta_end), codes_key)
    cached = _pg_panel_cache
    if (cached.get("key") == key
            and time.time() - cached.get("ts", 0) < PANEL_CACHE_TTL):
        return cached["df"]
    with _pg_panel_lock:
        # 双检锁：并发请求同时 miss 时只允许一个构建，
        # 避免 N 份全量面板同时进内存（3.6G 机器直接卡死）。
        cached = _pg_panel_cache
        if (cached.get("key") == key
                and time.time() - cached.get("ts", 0) < PANEL_CACHE_TTL):
            return cached["df"]
        where = ["adj_factor IS NOT NULL", "close IS NOT NULL"]
        params: list = []
        if calc_start:
            where.append("trade_date >= %s")
            params.append(calc_start)
        if end:
            where.append("trade_date <= %s")
            params.append(end)
        if codes:
            like_conds = " OR ".join(["ts_code LIKE %s"] * len(codes))
            where.append(f"({like_conds})")
            params.extend(f"{c}%" for c in codes)
        df = query_df(
            "SELECT ts_code, trade_date AS date, open, high, low, close, vol AS volume, amount, "
            "turnover_rate AS turnover, adj_factor FROM stock_daily "
            "WHERE " + " AND ".join(where) + " ORDER BY ts_code, trade_date",
            tuple(params),
        )
        if len(df) == 0:
            raise RuntimeError("所选区间/股票池在 stock_daily 中没有数据")
        if df["adj_factor"].isna().mean() > 0.05:
            raise RuntimeError("stock_daily 复权因子覆盖不足，历史补全后自动切换")
        # last_adj 直接取区间内各股票最新 adj_factor：回测终点通常是数据最新日，
        # 与全量口径一致；终点早于最新日时按区间内最新复权，避免全表 DISTINCT ON。
        out = _finalize_stock_df(df)
        _pg_panel_cache = {"key": key, "df": out, "ts": time.time()}
        return out


def load_panel(start: str | None = None, end: str | None = None,
               codes: list[str] | None = None) -> pd.DataFrame:
    unbounded = start is None and end is None and codes is None
    if DATA_SOURCE == "pg":
        try:
            return _load_panel_pg(start=start, end=end, codes=codes)
        except Exception as exc:
            print(f"PG 面板加载失败，回退 parquet: {exc}", file=sys.stderr)
    if DATA_SOURCE == "pg_parquet":
        if unbounded:
            # 全量面板直接用预计算文件（已有 turn20/am20），避免把
            # 1170 万行 stock_daily 重新构造成 pandas（3.6G 机器会 OOM）。
            try:
                return _load_panel_precomputed()
            except Exception as exc:
                print(f"预计算面板加载失败: {exc}", file=sys.stderr)
        try:
            return _load_panel_pg_parquet(start=start, end=end, codes=codes)
        except Exception as exc:
            print(f"PG parquet 面板加载失败，回退 PG: {exc}", file=sys.stderr)
            try:
                return _load_panel_pg(start=start, end=end, codes=codes)
            except Exception as exc2:
                print(f"PG 面板加载失败: {exc2}", file=sys.stderr)
    return _load_panel_precomputed(start=start, end=end, codes=codes)


def load_signal_panel(codes: list[str], end: str | None = None) -> pd.DataFrame:
    """信号面板轻量加载：只拉最近 ~2 年行情，不加载 2020 年至今全量。

    信号因子最长只看 60 个交易日，2 年窗口（约 550 个交易日）足够覆盖。
    科技TMT（约 90 只）只需 5~6 万行，内存有界，响应也快。
    """
    if not codes:
        raise RuntimeError("信号股票池为空")
    if DATA_SOURCE == "pg_parquet":
        calc_start = (pd.Timestamp(end or pd.Timestamp.today())
                      - pd.Timedelta(days=SIGNAL_LOOKBACK_DAYS + FACTOR_BUFFER_DAYS * 2)
                      ).date().isoformat()
        try:
            return _duck_panel_slice(start=calc_start, end=end, codes=codes)
        except Exception as exc:
            print(f"[duck] 信号面板加载失败，回退预计算面板: {exc}",
                  file=sys.stderr)
        return _load_panel_precomputed(start=calc_start, end=end, codes=codes)
    from .pg import configured as pg_configured, query_df
    if not pg_configured():
        raise RuntimeError("PG_DSN 未配置")
    if end is None:
        end = str(query_df(
            "SELECT max(trade_date) AS end FROM stock_daily"
        ).iloc[0]["end"])
    calc_start = (pd.Timestamp(end)
                  - pd.Timedelta(days=SIGNAL_LOOKBACK_DAYS + FACTOR_BUFFER_DAYS * 2)
                  ).date().isoformat()
    like_conds = " OR ".join(["ts_code LIKE %s"] * len(codes))
    sql = (
        "SELECT ts_code, trade_date AS date, open, high, low, close, "
        "vol AS volume, amount, turnover_rate AS turnover, adj_factor "
        "FROM stock_daily "
        "WHERE adj_factor IS NOT NULL AND close IS NOT NULL "
        "AND trade_date >= %s AND trade_date <= %s AND (" + like_conds + ") "
        "ORDER BY ts_code, trade_date"
    )
    params: list = [calc_start, end] + [f"{c}%" for c in codes]
    df = query_df(sql, tuple(params))
    if len(df) == 0:
        raise RuntimeError("信号面板为空")
    return _finalize_stock_df(df)


def load_panel_codes() -> set[str]:
    """轻量返回股票面板全部代码（不加载整张面板），用于构建股票池。"""
    global _panel_codes_cache
    if _panel_codes_cache is not None:
        return _panel_codes_cache
    codes: set[str] = set()
    if DATA_SOURCE in ("pg", "pg_parquet"):
        if DATA_SOURCE == "pg_parquet":
            try:
                path = PG_PARQUET_DIR / "stock_basic.parquet"
                if path.exists():
                    df = pd.read_parquet(path, columns=["ts_code"])
                    codes = {str(c)[:6].zfill(6) for c in df["ts_code"]}
            except Exception:
                codes = set()
    if DATA_SOURCE == "pg" and not codes:
        try:
            from .pg import configured as pg_configured, query_df
            if pg_configured():
                # stock_basic 是静态股票注册表（5000+ 行），避免对 1174 万行日线做 DISTINCT
                df = query_df("SELECT ts_code FROM stock_basic WHERE ts_code IS NOT NULL")
                codes = {str(c)[:6].zfill(6) for c in df["ts_code"]}
        except Exception:
            codes = set()
    if not codes:
        path = PANEL_FILE if PANEL_FILE.exists() else PANEL_PATH
        if path.exists():
            codes = set(pd.read_parquet(path, columns=["code"])
                        ["code"].astype(str).str.zfill(6).unique())
    _panel_codes_cache = codes
    return codes


def load_stock_detail(code: str, days: int = 250, adj: str = "qfq") -> pd.DataFrame:
    """单只股票复权行情 + turn20/am20，只查 PG 单股，避免加载整张面板。

    adj: qfq/hfq/raw，见 _finalize_stock_df。raw/hfq 的历史价不随最新分红漂移，
    适合需要稳定历史图的展示场景；回测/信号仍使用 qfq（默认）。
    """
    code = str(code).zfill(6)
    adj = (adj or "qfq").strip().lower()
    if adj not in ("qfq", "hfq", "raw"):
        adj = "qfq"
    if DATA_SOURCE in ("pg", "pg_parquet"):
        if DATA_SOURCE == "pg_parquet":
            try:
                path = PG_PARQUET_DIR / "stock_daily.parquet"
                code_map = _code_to_ts_map()
                ts_codes = [code_map.get(code, f"{code}.SZ"), f"{code}.SH", f"{code}.BJ"]
                cols = ["ts_code", "trade_date", "open", "high", "low", "close",
                        "vol", "amount", "turnover_rate", "adj_factor"]
                df = pd.read_parquet(path, columns=cols,
                                     filters=[("ts_code", "in", ts_codes)])
                if not df.empty:
                    df = df.rename(columns={"trade_date": "date", "vol": "volume",
                                            "turnover_rate": "turnover"})
                    return _finalize_stock_df(df, adj=adj).tail(days).reset_index(drop=True)
            except Exception:
                pass
        try:
            from .pg import configured as pg_configured, query_df
            if pg_configured():
                df = query_df(
                    "SELECT ts_code, trade_date AS date, open, high, low, close, "
                    "vol AS volume, amount, turnover_rate AS turnover, adj_factor "
                    "FROM stock_daily "
                    "WHERE ts_code LIKE %s AND adj_factor IS NOT NULL AND close IS NOT NULL "
                    "ORDER BY ts_code, trade_date",
                    (f"{code}%",),
                )
                if not df.empty:
                    return _finalize_stock_df(df, adj=adj).tail(days).reset_index(drop=True)
        except Exception:
            pass
    panel = load_panel()
    sub = panel[panel["code"] == code].sort_values("date")
    return sub.tail(days).reset_index(drop=True)


def reset_caches() -> None:
    """数据更新后清空面板/代码缓存，避免 stale。"""
    global _pg_panel_cache, _panel_codes_cache
    _pg_panel_cache = {}
    _panel_codes_cache = None
    reset_last_adj_cache()


def load_universe() -> pd.DataFrame:
    if UNIVERSE_FILE.exists():
        uni = pd.read_csv(UNIVERSE_FILE, dtype={"code": str})
    elif UNIVERSE_PATH.exists():
        uni = pd.read_csv(UNIVERSE_PATH, dtype={"code": str})
    else:
        raise FileNotFoundError(f"股票池数据不存在: {UNIVERSE_PATH}")
    uni["code"] = uni["code"].astype(str).str.zfill(6)
    return uni


def load_tech() -> pd.DataFrame:
    if TECH_FILE.exists():
        tech = pd.read_csv(TECH_FILE, dtype={"code": str})
    elif TECH_PATH.exists():
        tech = pd.read_csv(TECH_PATH, dtype={"code": str})
    else:
        raise FileNotFoundError(f"行业数据不存在: {TECH_PATH}")
    tech["code"] = tech["code"].astype(str).str.zfill(6)
    return tech


def load_etf() -> pd.DataFrame:
    if not ETF_FILE.exists():
        return pd.DataFrame(columns=["code", "name"])
    etf = pd.read_csv(ETF_FILE, dtype={"code": str})
    etf["code"] = etf["code"].astype(str).str.zfill(6)
    return etf


def load_etf_panel(start: str | None = None,
                   end: str | None = None) -> pd.DataFrame:
    """ETF 日线面板；start/end 可选，pyarrow 下推区间过滤避免整面板进内存。"""
    if not ETF_PANEL_FILE.exists():
        return pd.DataFrame(columns=["date", "open", "close", "turnover",
                                     "amount", "code", "turn20", "am20",
                                     "volume"])
    filters: list = []
    if start:
        filters.append(("date", ">=", pd.Timestamp(start)))
    if end:
        filters.append(("date", "<=", pd.Timestamp(end)))
    try:
        panel = pd.read_parquet(ETF_PANEL_FILE, filters=filters or None)
    except Exception:
        panel = pd.read_parquet(ETF_PANEL_FILE)
        if start:
            panel = panel[panel["date"] >= pd.Timestamp(start)]
        if end:
            panel = panel[panel["date"] <= pd.Timestamp(end)]
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    for c in ("turn20", "am20", "volume"):
        if c in panel.columns and panel[c].dtype == "float64":
            panel[c] = panel[c].astype("float32")
    panel["code"] = panel["code"].astype("category")
    return panel


def load_etf_panel_codes() -> set[str]:
    """轻量返回 ETF 面板全部代码（只投影 code 列）。"""
    if not ETF_PANEL_FILE.exists():
        return set()
    codes = pd.read_parquet(ETF_PANEL_FILE, columns=["code"])["code"]
    return {str(c).zfill(6) for c in codes}


def load_fund() -> pd.DataFrame:
    if not FUND_FILE.exists():
        return pd.DataFrame(columns=["code", "name", "type"])
    fund = pd.read_csv(FUND_FILE, dtype={"code": str})
    fund["code"] = fund["code"].astype(str).str.zfill(6)
    return fund


def load_fund_nav(start: str | None = None,
                  end: str | None = None) -> pd.DataFrame:
    """基金净值；start/end 可选，pyarrow 下推区间过滤。"""
    if not FUND_NAV_FILE.exists():
        return pd.DataFrame(columns=["date", "code", "nav"])
    filters: list = []
    if start:
        filters.append(("date", ">=", pd.Timestamp(start)))
    if end:
        filters.append(("date", "<=", pd.Timestamp(end)))
    try:
        panel = pd.read_parquet(FUND_NAV_FILE, filters=filters or None)
    except Exception:
        panel = pd.read_parquet(FUND_NAV_FILE)
        if start:
            panel = panel[panel["date"] >= pd.Timestamp(start)]
        if end:
            panel = panel[panel["date"] <= pd.Timestamp(end)]
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["code"] = panel["code"].astype("category")
    return panel


def load_fund_nav_codes() -> set[str]:
    """轻量返回基金净值面板全部代码（只投影 code 列）。"""
    if not FUND_NAV_FILE.exists():
        return set()
    codes = pd.read_parquet(FUND_NAV_FILE, columns=["code"])["code"]
    return {str(c).zfill(6) for c in codes}


def load_fund_panel(start: str | None = None,
                    end: str | None = None) -> pd.DataFrame:
    """基金净值标准面板（列结构与股票/ETF 面板一致，供统一回测）。"""
    if FUND_PANEL_FILE.exists():
        filters: list = []
        if start:
            filters.append(("date", ">=", pd.Timestamp(start)))
        if end:
            filters.append(("date", "<=", pd.Timestamp(end)))
        try:
            panel = pd.read_parquet(FUND_PANEL_FILE, filters=filters or None)
        except Exception:
            panel = pd.read_parquet(FUND_PANEL_FILE)
            if start:
                panel = panel[panel["date"] >= pd.Timestamp(start)]
            if end:
                panel = panel[panel["date"] <= pd.Timestamp(end)]
    else:
        try:
            from .fund_engine import build_fund_panel
            panel = build_fund_panel(load_fund_nav())
        except Exception:
            return pd.DataFrame(columns=["date", "open", "close", "turnover",
                                         "amount", "code", "turn20", "am20",
                                         "volume"])
    panel["code"] = panel["code"].astype(str).str.zfill(6)
    panel["date"] = pd.to_datetime(panel["date"])
    for c in ("turn20", "am20", "volume"):
        if c in panel.columns and panel[c].dtype == "float64":
            panel[c] = panel[c].astype("float32")
    panel["code"] = panel["code"].astype("category")
    return panel


def load_index() -> pd.DataFrame:
    if INDEX_FILE.exists():
        idx = pd.read_csv(INDEX_FILE)
    elif INDEX_PATH.exists():
        idx = pd.read_csv(INDEX_PATH)
    else:
        raise FileNotFoundError(f"基准指数数据不存在: {INDEX_PATH}")
    if "code" not in idx.columns:
        # 旧版单指数文件（沪深300）
        idx = idx.copy()
        idx["code"] = "sh000300"
        idx["name"] = "沪深300"
    idx["code"] = idx["code"].astype(str)
    idx["name"] = idx["name"].astype(str)
    idx["date"] = pd.to_datetime(idx["date"])
    cols = ["date", "code", "name"]
    for c in ("open", "close"):
        if c in idx.columns:
            cols.append(c)
    return idx[cols].sort_values(["code", "date"]).reset_index(drop=True)


def _file_entry(name: str, path: Path, desc: str, source: str, update: str) -> dict:
    return {
        "name": name,
        "path": str(path),
        "desc": desc,
        "source": source,
        "update": update,
        "exists": path.exists(),
        "size_mb": round(path.stat().st_size / 1e6, 1) if path.exists() else None,
    }


def data_status() -> dict:
    """返回所有数据源状态：本地行情文件 + 衍生缓存 + PostgreSQL + 舆情文件。"""
    files = [
        ("panel", _file_entry(
            "股票日线+因子面板", PANEL_FILE,
            "全股票池前复权日线，含 turn20/am20 滚动因子", "腾讯行情", "一键更新 / refresh_data.py")),
        ("universe", _file_entry(
            "股票池", UNIVERSE_FILE,
            "沪深300 + 中证500 + 中证1000 成分股", "中证指数官网", "一键更新")),
        ("tech", _file_entry(
            "行业分类", TECH_FILE,
            "科技TMT 行业归属", "东方财富 akshare", "一键更新")),
        ("index", _file_entry(
            "指数日线", INDEX_FILE,
            "沪深300/中证500/中证1000/创业板指/科创50/上证指数",
            "Tushare（腾讯回退）", "一键更新")),
        ("etf", _file_entry(
            "ETF 列表", ETF_FILE, "全市场 ETF 快照", "东方财富 akshare", "一键更新")),
        ("etf_panel", _file_entry(
            "ETF 日线面板", ETF_PANEL_FILE,
            "ETF 日线，结构与股票面板一致", "腾讯行情", "一键更新")),
        ("fund", _file_entry(
            "场外基金池", FUND_FILE,
            "科技相关场外基金（股票/混合/指数/QDII）", "天天基金（akshare）", "一键更新")),
        ("fund_nav", _file_entry(
            "场外基金净值", FUND_NAV_FILE,
            "逐只基金单位净值历史", "天天基金（akshare）", "一键更新")),
        ("fund_panel", _file_entry(
            "基金衍生面板", FUND_PANEL_FILE,
            "由基金净值派生，供统一回测引擎使用", "本地派生", "一键更新 / refresh_data.py")),
        ("duck_cache", _file_entry(
            "DuckDB 查询缓存", DATA_DIR / "duck.db",
            "本地查询缓存/视图", "本地派生", "自动生成")),
    ]
    out = {key: {"store": entry} for key, entry in files}

    sentiment_files = [
        ("sentiment_articles", "舆情库", SENT_ROOT / "data" / "articles.db",
         "去重后的舆情文章库"),
        ("sentiment_news_raw", "东财个股新闻", SENT_ROOT / "data" / "news_raw.jsonl",
         "东方财富个股新闻原始流"),
        ("sentiment_news_cls", "财联社电报", SENT_ROOT / "data" / "news_cls.jsonl",
         "财联社电报原始流"),
        ("sentiment_news_extra", "扩展新闻源", SENT_ROOT / "data" / "news_extra.jsonl",
         "其他来源新闻原始流"),
        ("sentiment_news_sentiment", "词典打分", SENT_ROOT / "data" / "news_sentiment.csv",
         "舆情词典/规则打分结果"),
        ("sentiment_news_daily", "日度全量情绪", SENT_ROOT / "data" / "news_sentiment_daily.csv",
         "按日聚合的情绪分"),
        ("sentiment_event_study", "事件研究", SENT_ROOT / "outputs" / "event_study_daily.csv",
         "事件驱动研究日度结果"),
        ("sentiment_universe", "舆情股票池", SENT_ROOT / "data" / "universe.csv",
         "舆情覆盖股票池"),
    ]
    for key, name, path, desc in sentiment_files:
        out[key] = {"store": _file_entry(
            name, path, desc, "sentiment-mvp", "独立流水线 run_pipeline.py daily")}

    pg_entry = _file_entry(
        "PG 日线表 stock_daily", Path("PostgreSQL: public.stock_daily"),
        "TimescaleDB 日线同步表（Tushare/本地面板）", "Tushare / 本地面板",
        "一键更新 / refresh_data.py")
    pg_entry["info"] = {}
    try:
        from . import pg as pg_mod
        if pg_mod.configured():
            pg_entry["exists"] = True
            try:
                df = pg_mod.query_df("SELECT max(trade_date) AS last_date FROM stock_daily")
                if len(df) and df.iloc[0]["last_date"] is not None:
                    pg_entry["info"]["last_date"] = str(df.iloc[0]["last_date"])
            except Exception:
                pass
            try:
                df = pg_mod.query_df(
                    "SELECT reltuples::bigint AS n_rows FROM pg_class "
                    "WHERE relname = 'stock_daily'")
                if len(df) and df.iloc[0]["n_rows"] is not None and int(df.iloc[0]["n_rows"]) > 0:
                    pg_entry["info"]["n_rows"] = int(df.iloc[0]["n_rows"])
            except Exception:
                pass
    except Exception:
        pass
    out["pg_stock_daily"] = {"store": pg_entry}

    out["meta"] = load_meta()
    return out
