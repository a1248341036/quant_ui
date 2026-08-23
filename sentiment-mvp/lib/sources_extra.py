# -*- coding: utf-8 -*-
"""扩展舆情源：东财全球资讯 / 金十快讯 / 东财股吧 / 巨潮公告 / 互动易 / 热榜。

数据层底座复用项目内 tools/a-stock-data/astock_data.py，
本模块只做适配：统一成 {code, publish_time, media, title, content, url, source}。
"""

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

try:
    from astock_data import (cninfo_announcements, cninfo_irm, em_hot_rank,
                             eastmoney_global_news, ths_hot_list)
except ImportError:
    # 直接运行时兜底：把仓库路径加入 sys.path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools" / "a-stock-data"))
    from astock_data import (cninfo_announcements, cninfo_irm, em_hot_rank,
                             eastmoney_global_news, ths_hot_list)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/142.0.0.0 Safari/537.36"


def fetch_em_global(max_items: int = 100) -> list[dict]:
    """东财全球资讯 7x24。返回 [{title, content, time, url}]。"""
    rows = []
    page = 1
    while len(rows) < max_items and page <= 5:
        batch = eastmoney_global_news(page_size=min(50, max_items - len(rows)))
        if not batch:
            break
        rows.extend(batch)
        # 滚动接口按 showTime 倒序，取最后一条做下一页游标，简单起见直接截断
        if len(batch) < 50:
            break
        page += 1
        time.sleep(0.3)
    return rows[:max_items]


def fetch_jin10(max_items: int = 100) -> list[dict]:
    """金十快讯 flash_newest.js（免费、免登录，滚动最近若干条）。"""
    r = requests.get("https://www.jin10.com/flash_newest.js",
                     headers={"User-Agent": UA}, timeout=10)
    r.raise_for_status()
    m = re.search(r"var newest\s*=\s*(\[.*?\])\s*;", r.text, re.DOTALL)
    if not m:
        return []
    try:
        items = json.loads(m.group(1))
    except Exception:
        return []
    rows = []
    for it in items[:max_items]:
        d = it.get("data") or {}
        content = (d.get("content") or "").strip()
        if not content:
            continue
        rows.append({
            "title": content[:60],
            "content": content,
            "time": it.get("time", ""),
            "url": f"https://www.jin10.com/",
            "media": "金十数据",
        })
    return rows


def _fetch_guba_one(code: str, posts_per_stock: int) -> list[dict]:
    """抓单只股票的股吧热帖。"""
    rows = []
    try:
        page = 1
        got = 0
        with requests.Session() as session:
            while got < posts_per_stock and page <= 3:
                url = f"https://guba.eastmoney.com/list,{code},f_{page}.html"
                r = session.get(url, headers={"User-Agent": UA,
                                              "Referer": "https://guba.eastmoney.com/"}, timeout=12)
                if r.status_code != 200:
                    break
                m = re.search(r"var article_list=(\{.*?\});", r.text, re.DOTALL)
                if not m:
                    break
                try:
                    d = json.loads(m.group(1))
                except Exception:
                    break
                lst = d.get("re") or []
                if not lst:
                    break
                for it in lst:
                    title = (it.get("post_title") or "").strip()
                    if not title:
                        continue
                    rows.append({
                        "code": code,
                        "publish_time": it.get("post_publish_time", ""),
                        "media": "东财股吧",
                        "title": title,
                        "content": title,
                        "url": f"https://guba.eastmoney.com/news,{code},{it.get('post_id', '')}.html",
                        "source": "guba",
                    })
                    got += 1
                    if got >= posts_per_stock:
                        break
                page += 1
                time.sleep(0.3)
    except Exception as e:
        print(f"[sources_extra] guba {code} failed: {e}", file=__import__("sys").stderr)
    return rows


def fetch_guba(codes: list[str], posts_per_stock: int = 20, workers: int = 8) -> list[dict]:
    """东财股吧热帖列表（页面内嵌 article_list），并发抓取。只取标题，情感信息足够。"""
    rows = []
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as ex:
        futs = [ex.submit(_fetch_guba_one, c, posts_per_stock) for c in codes]
        for fut in as_completed(futs):
            try:
                rows.extend(fut.result())
            except Exception as e:
                print(f"[sources_extra] guba worker failed: {e}", file=__import__("sys").stderr)
            done += 1
            if done % 300 == 0:
                print(f"[sources_extra] guba {done}/{len(codes)}", flush=True)
    return rows


def fetch_cninfo(codes: list[str], per: int = 10) -> list[dict]:
    """巨潮公告。公告本身带 code，无需再匹配。"""
    rows = []
    for code in codes:
        try:
            for a in cninfo_announcements(code, page_size=per)[:per]:
                rows.append({
                    "code": code,
                    "publish_time": f"{a['date']} 00:00:00" if a.get("date") else "",
                    "media": "巨潮资讯",
                    "title": a.get("title", ""),
                    "content": f"{a.get('type', '')} {a.get('title', '')}".strip(),
                    "url": a.get("url", ""),
                    "source": "cninfo",
                })
        except Exception as e:
            print(f"[sources_extra] cninfo {code} failed: {e}", file=__import__("sys").stderr)
        time.sleep(0.15)
    return rows


def fetch_irm(codes: list[str], per: int = 10) -> list[dict]:
    """互动易问答。"""
    rows = []
    for code in codes:
        try:
            for q in cninfo_irm(code, page_size=per)[:per]:
                q_text = (q.get("question") or "").strip()
                a_text = (q.get("answer") or "").strip()
                if not q_text:
                    continue
                title = f"{q.get('company', code)} 互动易：{q_text[:40]}"
                content = f"Q: {q_text} A: {a_text}" if a_text else f"Q: {q_text}"
                rows.append({
                    "code": code,
                    "publish_time": q.get("ask_time", ""),
                    "media": "互动易",
                    "title": title,
                    "content": content,
                    "url": f"https://irm.cninfo.com.cn/",
                    "source": "irm",
                })
        except Exception as e:
            print(f"[sources_extra] irm {code} failed: {e}", file=__import__("sys").stderr)
        time.sleep(0.15)
    return rows


def fetch_hot(top: int = 30) -> list[dict]:
    """同花顺热榜 + 东财人气榜。返回带 code 的榜单条目。"""
    rows = []
    try:
        for it in ths_hot_list("hour")[:top]:
            if not it.get("code"):
                continue
            concepts = "、".join(it.get("concepts") or []) or it.get("tag", "")
            rows.append({
                "code": it["code"],
                "publish_time": "",
                "media": "同花顺热榜",
                "title": f"热榜#{it.get('rank')} {it.get('name')} 热度{it.get('heat')} {concepts}".strip(),
                "content": f"同花顺热榜 {it.get('name')} 人气{it.get('heat')} 涨幅{it.get('pct')}% 概念:{concepts}",
                "url": "",
                "source": "hot",
            })
    except Exception as e:
        print(f"[sources_extra] ths_hot failed: {e}", file=__import__("sys").stderr)
    try:
        for it in em_hot_rank(top)[:top]:
            if not it.get("code"):
                continue
            rows.append({
                "code": it["code"],
                "publish_time": "",
                "media": "东财人气榜",
                "title": f"人气榜#{it.get('rank')} {it.get('name')} 涨幅{it.get('pct')}%",
                "content": f"东财人气榜 {it.get('name')} 排名{it.get('rank')} 涨幅{it.get('pct')}%",
                "url": "",
                "source": "hot",
            })
    except Exception as e:
        print(f"[sources_extra] em_hot failed: {e}", file=__import__("sys").stderr)
    return rows


def match_mentions(items, name_map, code_list, media, source):
    """把无 code 的资讯按 代码/简称 匹配到标的池。"""
    code_names = {c: n for c, n in name_map.items() if c in code_list}
    patterns = [(code, code) for code in code_list]
    patterns += [(name, code) for code, name in code_names.items()]
    out = []
    for it in items:
        text = f"{it.get('title') or ''} {it.get('content') or ''}"
        found = set()
        for kw, code in patterns:
            if kw and kw in text:
                found.add(code)
        if not found:
            continue
        for code in sorted(found):
            out.append({
                "code": code,
                "publish_time": it.get("time", "") or it.get("publish_time", ""),
                "media": media,
                "title": (it.get("title") or "").strip(),
                "content": (it.get("content") or "").strip(),
                "url": it.get("url", ""),
                "source": source,
            })
    return out


def fetch_all(cfg, codes) -> list[dict]:
    """按 config.sources 拉取所有启用源，返回已匹配标的的统一行。"""
    uni = None
    name_map = {}
    code_list = set(codes)
    src = cfg.get("sources", {})
    rows = []

    # 先拿名称映射（从 config universe csv/tech csv）
    import pandas as pd
    for key in ("universe_csv", "tech_csv"):
        p = cfg["universe"].get(key)
        if p and Path(p).exists():
            df = pd.read_csv(p, dtype={"code": str})
            df["code"] = df["code"].astype(str).str.zfill(6)
            name_map.update(dict(zip(df["code"], df["name"])))

    if src.get("em_global", {}).get("enabled", True):
        items = fetch_em_global(int(src["em_global"].get("max_items", 100)))
        rows += match_mentions(items, name_map, code_list, "东财全球资讯", "em_global")
        print(f"[sources_extra] em_global raw={len(items)} matched={sum(1 for x in rows if x['source']=='em_global')}")

    if src.get("jin10", {}).get("enabled", True):
        items = fetch_jin10(int(src["jin10"].get("max_items", 100)))
        rows += match_mentions(items, name_map, code_list, "金十数据", "jin10")
        print(f"[sources_extra] jin10 raw={len(items)} matched={sum(1 for x in rows if x['source']=='jin10')}")

    if src.get("guba", {}).get("enabled", True):
        per = int(src["guba"].get("posts_per_stock", 20))
        workers = int(src["guba"].get("workers", 8))
        guba = fetch_guba(sorted(code_list), per, workers=workers)
        rows += guba
        print(f"[sources_extra] guba rows={len(guba)}")

    if src.get("cninfo", {}).get("enabled", False):
        per = int(src["cninfo"].get("announcements_per_stock", 10))
        rows += fetch_cninfo(sorted(code_list), per)

    if src.get("irm", {}).get("enabled", False):
        per = int(src["irm"].get("page_size", 10))
        rows += fetch_irm(sorted(code_list), per)

    if src.get("hot", {}).get("enabled", True):
        top = int(src["hot"].get("top", 30))
        hot = fetch_hot(top)
        hot = [h for h in hot if h["code"] in code_list]
        rows += hot
        print(f"[sources_extra] hot rows={len(hot)}")

    return rows


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
