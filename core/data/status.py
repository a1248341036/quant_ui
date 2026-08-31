"""数据源状态报告：data_status + _file_entry。"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ..store import (
    DATA_DIR, ETF_FILE, ETF_PANEL_FILE, FUND_FILE, FUND_FEE_FILE,
    FUND_NAV_FILE, FUND_PANEL_FILE, INDEX_FILE, PANEL_FILE,
    SENTIMENT_DIR, load_meta,
)
from .panel import PANEL_PATH


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
    """返回所有数据源状态：本地行情文件 + 衍生缓存 + Tushare parquet + 舆情文件。"""
    files = [
        ("panel", _file_entry(
            "股票日线+因子面板", PANEL_FILE,
            "全股票池前复权日线，含 turn20/am20 滚动因子", "腾讯行情", "一键更新 / refresh_data.py")),
        ("universe", _file_entry(
            "股票池", DATA_DIR / "universe.csv",
            "沪深300 + 中证500 + 中证1000 成分股", "中证指数官网", "一键更新")),
        ("tech", _file_entry(
            "行业分类", DATA_DIR / "tech_universe.csv",
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
            "全市场权益类场外基金（股票/混合/指数/QDII）", "天天基金（akshare）", "一键更新")),
        ("fund_fee", _file_entry(
            "基金费率", FUND_FEE_FILE,
            "申购、管理、托管、销售服务及赎回费率", "天天基金（akshare）", "CNE 流水线 / step_fund_fees（每周）")),
        ("fund_nav", _file_entry(
            "场外基金净值", FUND_NAV_FILE,
            "逐只基金单位净值历史", "天天基金快照（akshare）", "CNE 流水线 / step_fund_nav")),
        ("fund_panel", _file_entry(
            "基金衍生面板", FUND_PANEL_FILE,
            "由基金净值派生，供统一回测引擎使用", "本地派生", "一键更新 / refresh_data.py")),
        ("duck_cache", _file_entry(
            "DuckDB 查询缓存", DATA_DIR / "db" / "duck.db",
            "本地查询缓存/视图", "本地派生", "自动生成")),
    ]
    out = {key: {"store": entry} for key, entry in files}
    try:
        from ..cne_reader import source_status
        out["cne_daily_source"] = {"store": {
            "name": "CNE 原生日线来源",
            "path": source_status().get("path"),
            "desc": "Tushare 主源，AkShare 兜底；失败时保留 CNE 原有 fallback",
            "source": "CNE daily_bars",
            "update": "scripts/cne/run_cne_daily.ps1",
            **source_status(),
        }}
    except Exception as exc:  # noqa: BLE001
        out["cne_daily_source"] = {"store": {"status": "unknown", "error": str(exc)}}

    sentiment_files = [
        ("sentiment_articles", "舆情库", SENTIMENT_DIR / "articles.db",
         "去重后的舆情文章库"),
        ("sentiment_news_raw", "东财个股新闻", SENTIMENT_DIR / "news_raw.jsonl",
         "东方财富个股新闻原始流"),
        ("sentiment_news_cls", "财联社电报", SENTIMENT_DIR / "news_cls.jsonl",
         "财联社电报原始流"),
        ("sentiment_news_extra", "扩展新闻源", SENTIMENT_DIR / "news_extra.jsonl",
         "其他来源新闻原始流"),
        ("sentiment_news_sentiment", "词典打分", SENTIMENT_DIR / "news_sentiment.csv",
         "舆情词典/规则打分结果"),
        ("sentiment_news_daily", "日度全量情绪", SENTIMENT_DIR / "news_sentiment_daily.csv",
         "按日聚合的情绪分"),
        ("sentiment_event_study", "事件研究", SENTIMENT_DIR / "outputs" / "event_study_daily.csv",
         "事件驱动研究日度结果"),
        ("sentiment_universe", "舆情股票池", SENTIMENT_DIR / "universe.csv",
         "舆情覆盖股票池"),
    ]
    for key, name, path, desc in sentiment_files:
        out[key] = {"store": _file_entry(
            name, path, desc, "sentiment-mvp", "独立流水线 run_pipeline.py daily")}

    out["meta"] = load_meta()
    return out
