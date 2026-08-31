# -*- coding: utf-8 -*-
"""从 api_full.html 提取聚宽 API 函数级索引 -> docs/jq_api_snapshot/api_index.txt."""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
html = (ROOT / "docs" / "jq_api_snapshot" / "api_full.html").read_text(
    encoding="utf-8")

# 每个函数级 API 都有 <span id="..."/> 或标题锚点; 标题文本在附近
ids = re.findall(r'id="([^"]+)"', html)
seen, out = set(), []
for i in ids:
    if i not in seen and not i.startswith(("first", "API")):
        seen.add(i)
        out.append(i)
print(f"anchors: {len(out)}")
(ROOT / "docs" / "jq_api_snapshot" / "api_index.txt").write_text(
    "\n".join(out), encoding="utf-8")

# 提取 <h2>/<h3> 及其 id -> 层级结构
heads = re.findall(
    r'<h([1-6])[^>]*>\s*(?:<span[^>]*>)?\s*<span id="([^"]+)"', html)
print(f"headed anchors: {len(heads)}")
for lvl, aid in heads:
    print(f"{'  ' * (int(lvl) - 1)}h{lvl} {aid}")
