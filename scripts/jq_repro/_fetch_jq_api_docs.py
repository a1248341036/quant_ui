# -*- coding: utf-8 -*-
"""抓取聚宽 API 文档树 + 各章节正文 -> docs/jq_api_snapshot/。

用途: 对照整理"聚宽 API -> 本平台兼容层"迁移表。
"""
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs" / "jq_api_snapshot"
OUT.mkdir(parents=True, exist_ok=True)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = "https://www.joinquant.com"
HEADERS = {
    "X-Requested-With": "XMLHttpRequest",
    "Referer": f"{BASE}/help/api/help",
    "User-Agent": "Mozilla/5.0",
}


def get(url: str) -> str:
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", errors="replace")


def fetch_tree() -> list:
    raw = get(f"{BASE}/help/api/getHelpDocTree?name=api")
    data = json.loads(raw)
    return data["data"]["tree"]


def walk(nodes, out, depth=0):
    for n in nodes:
        title = re.sub(r"<[^>]+>", "", n.get("title", ""))
        out.append({"id": n.get("helpDocId"), "level": n.get("level"),
                    "title": title, "depth": depth})
        if n.get("products"):
            walk(n["products"], out, depth + 1)


def main():
    tree = fetch_tree()
    flat = []
    walk(tree, flat)
    print(f"tree nodes: {len(flat)}")
    results = []
    for node in flat:
        nid, title = node["id"], node["title"]
        if not nid:
            results.append({**node, "content": ""})
            print(f"  [skip] {title}")
            continue
        try:
            raw = get(f"{BASE}/help/api/getContent?name={nid}")
            data = json.loads(raw)
            content = data.get("data") or ""
        except Exception as exc:  # noqa: BLE001
            content = f"<<fetch error: {exc}>>"
        results.append({**node, "content": content})
        print(f"  [{nid}] {len(content):>7} chars  {title}")
    (OUT / "tree.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    total = sum(len(r["content"]) for r in results)
    print(f"saved {OUT/'tree.json'}, total {total} chars")


if __name__ == "__main__":
    main()
