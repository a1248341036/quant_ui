# -*- coding: utf-8 -*-
"""财联社电报采集：分页拉取当日电报，按股票代码/简称匹配标的池。

注意：CLS 接口只提供当日滚动窗口，无法回溯历史，适合做「今日舆情快照」
和后续日度入库，不适合作为历史事件研究数据源。
"""

import hashlib
import json
import re
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import requests

CN_TZ = timezone(timedelta(hours=8))


def _sign(params):
    return hashlib.md5(
        hashlib.sha1(urlencode(params).encode("utf-8")).hexdigest().encode("utf-8")
    ).hexdigest()


def _fetch_page(session, last_time, rn=50, timeout=15):
    url = "https://www.cls.cn/v1/roll/get_roll_list"
    params = {
        "app": "CailianpressWeb",
        "category": "",
        "last_time": last_time,
        "os": "web",
        "refresh_type": "1",
        "rn": str(rn),
        "sv": "8.4.6",
    }
    params["sign"] = _sign(params)
    r = session.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    return r.json()["data"]["roll_data"]


def fetch_cls(session, max_pages=10, timeout=15, sleep=0.2):
    """返回今日电报列表，按时间正序。"""
    items = []
    last_time = int(time.time())
    seen = set()
    for _ in range(max_pages):
        try:
            page = _fetch_page(session, last_time, timeout=timeout)
        except Exception:
            break
        if not page:
            break
        new = 0
        for it in page:
            ctime = int(it.get("ctime", 0))
            if ctime in seen:
                continue
            seen.add(ctime)
            items.append(it)
            new += 1
        if new == 0:
            break
        last_time = page[-1]["ctime"]
        time.sleep(sleep)
    items.sort(key=lambda x: int(x.get("ctime", 0)))
    return items


def map_mentions(items, name_map, code_list):
    """按代码/简称匹配，返回 [{code, publish_time, media, title, content, url, source}]。

    name_map: code -> 名称（含简称关键词列表）；code_list: 需要跟踪的代码集合。
    """
    code_names = {c: n for c, n in name_map.items() if c in code_list}
    patterns = []
    for code in code_list:
        patterns.append((code, code))
    for code, name in code_names.items():
        # 匹配完整名称，避免「京东方A」这类简称截断
        patterns.append((name, code))
    out = []
    for it in items:
        text = (it.get("title") or "") + " " + (it.get("content") or "")
        found = set()
        for kw, code in patterns:
            if kw and kw in text:
                found.add(code)
        if not found:
            continue
        ctime = int(it.get("ctime", 0))
        publish_time = datetime.fromtimestamp(ctime, tz=CN_TZ).strftime("%Y-%m-%d %H:%M:%S")
        for code in sorted(found):
            out.append(
                {
                    "code": code,
                    "publish_time": publish_time,
                    "media": "财联社",
                    "title": (it.get("title") or "").strip(),
                    "content": (it.get("content") or "").strip(),
                    "url": it.get("shareurl") or "https://www.cls.cn/telegraph",
                    "source": "cls",
                }
            )
    return out


def save_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
