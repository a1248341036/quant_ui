# -*- coding: utf-8 -*-
"""东方财富搜索接口抓取个股新闻。

akshare 的 stock_news_em 在 pandas 3.0 下因正则转义报错，这里直接请求
search-api-web.eastmoney.com 原始接口，并做 HTML 清理。
"""

import json
import re
import time
from datetime import datetime
from pathlib import Path

import httpx

HTML_RE = re.compile(r"<[^>]+>")
CB = "jQuery35101792940631092459_1764599530165"


def clean_text(s):
    s = (s or "").replace("<em>", "").replace("</em>", "")
    s = HTML_RE.sub(" ", s)
    for a, b in (
        ("&nbsp;", " "),
        ("&amp;", "&"),
        ("&quot;", '"'),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&#39;", "'"),
    ):
        s = s.replace(a, b)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _parse_jsonp(text):
    start = text.find("(")
    if start < 0:
        raise ValueError("bad jsonp payload")
    return json.loads(text[start + 1 : -1])


def fetch_stock_news(session, code, max_pages=10, delay=0.25, timeout=15):
    """返回 [{code, publish_time, media, title, content, url}]。

    session: httpx.Client 实例（兼容 MVP run_pipeline 的并发模型）。
    """
    url = "https://search-api-web.eastmoney.com/search/jsonp"
    headers = {
        "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/142.0.0.0 Safari/537.36",
        "referer": "https://so.eastmoney.com/news/s",
    }
    arts = []
    for page in range(1, max_pages + 1):
        inner = {
            "uid": "",
            "keyword": code,
            "type": ["cmsArticleWebOld"],
            "client": "web",
            "clientType": "web",
            "clientVersion": "curr",
            "param": {
                "cmsArticleWebOld": {
                    "searchScope": "default",
                    "sort": "default",
                    "pageIndex": page,
                    "pageSize": 10,
                    "preTag": "<em>",
                    "postTag": "</em>",
                }
            },
        }
        params = {
            "cb": CB,
            "param": json.dumps(inner, ensure_ascii=False),
            "_": str(int(time.time() * 1000)),
        }
        try:
            r = session.get(url, params=params, headers=headers, timeout=timeout)
            if r.status_code != 200:
                break
            items = _parse_jsonp(r.text).get("result", {}).get("cmsArticleWebOld", [])
        except Exception:
            break
        if not items:
            break
        for it in items:
            article_code = it.get("code") or ""
            link = it.get("url") or (f"http://finance.eastmoney.com/a/{article_code}.html" if article_code else "")
            arts.append(
                {
                    "code": code,
                    "publish_time": it.get("date"),
                    "media": it.get("mediaName"),
                    "title": clean_text(it.get("title")),
                    "content": clean_text(it.get("content")),
                    "url": link,
                }
            )
        if len(items) < 10:
            break
        time.sleep(delay)
    return arts


def dedup(arts):
    seen = set()
    out = []
    for a in arts:
        key = (a["code"], a["publish_time"], a["title"])
        if key in seen:
            continue
        seen.add(key)
        out.append(a)
    return out


def save_raw(arts, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for a in arts:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")


def load_raw(path):
    arts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                arts.append(json.loads(line))
    return arts


def parse_time(s):
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(s).strip(), fmt)
        except ValueError:
            continue
    return None
