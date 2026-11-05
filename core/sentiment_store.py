# -*- coding: utf-8 -*-
"""舆情数据源统一层：旧 sentiment-mvp ``articles.db`` + CNE curated ``sentiment_articles``。

两段数据实测互补、不重叠：

- 旧库 ``data/sentiment-mvp/articles.db``：2026-02-15 ~ 2026-08-17，768 只票，33061 条；
- CNE curated ``sentiment_articles``：2026-09-10 ~ 2026-11-05，300 只票，16010 条。

CNE 的 ``article_id`` 直接沿用旧库 ``articles.key``（sha1），因此两源可按 id 去重；
``symbol`` 是带交易所后缀的代码（``688519.SH``），统一归一化为 6 位数字代码。

设计取舍：
- **合并而非替换**：只切 CNE 会让页面丢掉 2~8 月历史；只读旧库则停在 8 月。
- **fail-soft**：任一源缺失/损坏只损失该段数据并记录 ``sources`` 状态，不抛给页面。
- **缓存**：按「旧库 mtime + CNE 分区目录 mtime」做进程内缓存，数据日更后自动失效。
"""

from __future__ import annotations

import hashlib
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any

import pandas as pd

from core.store import SENTIMENT_DIR

logger = logging.getLogger(__name__)

LEGACY_DB = SENTIMENT_DIR / "articles.db"

# 统一 schema（顺序即对外列序）
COLUMNS = (
    "article_id",
    "code",
    "publish_time",
    "publish_date",
    "media",
    "title",
    "url",
    "source",
    "label",
    "score",
)

_LOCK = threading.Lock()
_CACHE: tuple[tuple[Any, ...], pd.DataFrame] | None = None


def _norm_code(raw: object) -> str:
    """代码归一化为 6 位数字（``688519.SH`` → ``688519``）。"""
    text = str(raw or "").strip()
    if not text:
        return ""
    if "." in text:
        text = text.split(".", 1)[0]
    if text.isdigit():
        return text.zfill(6)
    return text


def _legacy_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取旧 ``articles.db``；返回 (dataframe, 源状态)。"""
    status: dict[str, Any] = {"name": "legacy_articles_db", "path": str(LEGACY_DB), "rows": 0}
    if not LEGACY_DB.is_file():
        status["error"] = "数据库不存在"
        return pd.DataFrame(columns=list(COLUMNS)), status
    try:
        conn = sqlite3.connect(str(LEGACY_DB))
        try:
            df = pd.read_sql_query(
                "SELECT key, code, publish_time, media, title, url, source, label, score "
                "FROM articles",
                conn,
            )
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001 — 单源失败只降级该段
        status["error"] = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(columns=list(COLUMNS)), status
    df = df.rename(columns={"key": "article_id"})
    df["code"] = df["code"].map(_norm_code)
    df["article_id"] = df["article_id"].fillna("").astype(str)
    status["rows"] = int(len(df))
    return df, status


def _cne_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    """读取 CNE curated ``sentiment_articles``；返回 (dataframe, 源状态)。"""
    status: dict[str, Any] = {"name": "cne_curated", "dataset": "sentiment_articles", "rows": 0}
    try:
        # 延迟导入：backend 启动不必为舆情页立即拉起 polars/CNE 配置解析
        from alphaagent.data.adapters.plugins._pitlib import _curated_root, read_curated

        root = _curated_root() / "sentiment_articles"
        status["path"] = str(root)
        if not root.is_dir():
            status["error"] = "CNE curated 数据集目录不存在"
            return pd.DataFrame(columns=list(COLUMNS)), status
        raw = read_curated("sentiment_articles")
    except Exception as exc:  # noqa: BLE001 — 单源失败只降级该段
        status["error"] = f"{type(exc).__name__}: {exc}"
        return pd.DataFrame(columns=list(COLUMNS)), status

    if raw.is_empty():
        status["error"] = "数据集为空"
        return pd.DataFrame(columns=list(COLUMNS)), status

    df = raw.select(
        [
            "article_id",
            "symbol",
            "publish_time",
            "publish_date",
            "media",
            "title",
            "url",
            "source",
            "label",
            "score",
        ]
    ).to_pandas()
    df["code"] = df["symbol"].map(_norm_code)
    df["article_id"] = df["article_id"].fillna("").astype(str)
    status["rows"] = int(len(df))
    return df, status


def _signature() -> tuple[Any, ...]:
    """缓存签名：旧库 mtime + CNE 舆情分区目录 mtime（任一更新即失效）。"""
    legacy = LEGACY_DB.stat().st_mtime if LEGACY_DB.is_file() else 0.0
    try:
        from alphaagent.data.adapters.plugins._pitlib import _curated_root

        cne_dir = _curated_root() / "sentiment_articles"
        cne = cne_dir.stat().st_mtime if cne_dir.is_dir() else 0.0
    except Exception:  # noqa: BLE001
        cne = 0.0
    return (legacy, cne)


def _finalize(frame: pd.DataFrame) -> pd.DataFrame:
    """统一类型、去重、排序。"""
    if frame.empty:
        return pd.DataFrame(columns=list(COLUMNS))
    df = frame.copy()
    df["publish_time"] = df["publish_time"].fillna("").astype(str)
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    # 无 publish_date 的行用 publish_time 兜底（旧库热榜/快照行时间为空，保持 NaT）
    need = df["publish_date"].isna() & (df["publish_time"].str.len() >= 10)
    if bool(need.any()):
        df.loc[need, "publish_date"] = pd.to_datetime(
            df.loc[need, "publish_time"].str.slice(0, 10), errors="coerce"
        )
    df["label"] = df["label"].fillna("").astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    for col in ("media", "title", "url", "source"):
        df[col] = df[col].fillna("").astype(str)
    # article_id 优先去重；缺失 id 的行用 code+时间+标题 合成键，避免整行被并
    empty_id = df["article_id"].str.len() == 0
    if bool(empty_id.any()):
        synth = (
            df.loc[empty_id, "code"]
            + "|"
            + df.loc[empty_id, "publish_time"]
            + "|"
            + df.loc[empty_id, "title"]
        )
        df.loc[empty_id, "article_id"] = synth.map(
            # 用 sha1 而非内置 hash：后者带进程级随机盐，跨进程不稳定
            lambda s: "synth:" + hashlib.sha1(s.encode("utf-8")).hexdigest()[:20]
        )
    df = df.drop_duplicates(subset=["article_id"], keep="last")
    df = df.sort_values("publish_date", ascending=False, na_position="last")
    return df.loc[:, list(COLUMNS)].reset_index(drop=True)


def load_articles(*, use_cache: bool = True) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """合并两源舆情文章；返回 (dataframe, 各源状态列表)。"""
    global _CACHE
    sig = _signature()
    with _LOCK:
        if use_cache and _CACHE is not None and _CACHE[0] == sig:
            return _CACHE[1].copy(), _CACHE[1].attrs.get("sources", [])

    legacy, legacy_status = _legacy_frame()
    cne, cne_status = _cne_frame()
    frame = _finalize(pd.concat([legacy, cne], ignore_index=True))
    sources = [legacy_status, cne_status]
    for st in sources:
        try:
            if st["name"] == "legacy_articles_db":
                st["date_range"] = _date_range(legacy)
            else:
                st["date_range"] = _date_range(cne)
        except Exception:  # noqa: BLE001
            st["date_range"] = None

    frame.attrs["sources"] = sources
    with _LOCK:
        _CACHE = (sig, frame)
    return frame.copy(), sources


def _date_range(df: pd.DataFrame) -> dict[str, str] | None:
    """源数据日期范围；旧库只有 publish_time，需从其前 10 位推导。"""
    if df.empty:
        return None
    if "publish_date" in df:
        dts = pd.to_datetime(df["publish_date"], errors="coerce").dropna()
    elif "publish_time" in df:
        dts = pd.to_datetime(
            df["publish_time"].astype(str).str.slice(0, 10), errors="coerce"
        ).dropna()
    else:
        return None
    if dts.empty:
        return None
    return {"start": str(dts.min().date()), "end": str(dts.max().date())}


def stats() -> dict[str, Any]:
    """页面统计：总数/覆盖票数/最近日期/标签分布/情绪分均值/近 30 日条数。"""
    df, sources = load_articles()
    if df.empty:
        return {"ok": False, "error": "暂无舆情数据（旧库与 CNE curated 均不可用）", "sources": sources}

    dated = df.dropna(subset=["publish_date"])
    labels = df.groupby("label").size().sort_values(ascending=False)
    daily = (
        dated.groupby(dated["publish_date"].dt.strftime("%Y-%m-%d"))
        .size()
        .sort_index(ascending=False)
        .head(30)
    )
    return {
        "ok": True,
        "n_articles": int(len(df)),
        "n_codes": int(df["code"].nunique()),
        "last_date": str(dated["publish_date"].max().date()) if not dated.empty else "",
        "first_date": str(dated["publish_date"].min().date()) if not dated.empty else "",
        "label_dist": {str(k): int(v) for k, v in labels.items()},
        "mean_score": float(df["score"].mean()) if df["score"].notna().any() else None,
        "daily": [{"date": str(d), "n": int(n)} for d, n in daily.items()],
        "sources": sources,
    }


def news(top: int = 10, sort: str = "high", days: int = 7) -> list[dict[str, Any]]:
    """情绪最强/最弱新闻；窗口按数据内最近日期回推 ``days`` 天。"""
    from core.instruments import normalize_code

    df, _ = load_articles()
    if df.empty:
        return []
    dated = df.dropna(subset=["publish_date", "score"])
    if dated.empty:
        return []
    last = dated["publish_date"].max()
    cutoff = last - pd.Timedelta(days=max(1, int(days)))
    sub = dated[dated["publish_date"] >= cutoff]
    if sub.empty:
        sub = dated
    sub = sub.sort_values("score", ascending=(sort != "high"))
    limit = max(1, min(int(top), 100))
    out: list[dict[str, Any]] = []
    for _, row in sub.head(limit).iterrows():
        out.append(
            {
                "code": normalize_code(row["code"]),
                "publish_time": str(row["publish_time"])[:19],
                "media": str(row["media"]),
                "title": str(row["title"]),
                "url": str(row["url"]),
                "label": str(row["label"]),
                "score": float(row["score"]),
                "source": str(row["source"]),
            }
        )
    return out


def invalidate_cache() -> None:
    """测试/运维用：清空进程内缓存。"""
    global _CACHE
    with _LOCK:
        _CACHE = None
