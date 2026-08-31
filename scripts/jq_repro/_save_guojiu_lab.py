# -*- coding: utf-8 -*-
"""把 _test_guojiu.py 内嵌的国九策略代码存入 labs/ (代码面板-已保存 可直接载入)."""
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

src = (ROOT / "scripts" / "jq_repro" / "_test_guojiu.py").read_text(
    encoding="utf-8")
start = src.index("CODE = r'''") + len("CODE = r'''")
end = src.index("'''", start)
code = src[start:end]

labs = ROOT / "labs"
labs.mkdir(exist_ok=True)
meta = {
    "name": "国九小市值策略",
    "code": code,
    "registry": "",
    "factors": "",
    "saved_at": datetime.now().isoformat(timespec="seconds"),
    "engine": "jq",
}
(labs / "国九小市值策略.json").write_text(
    json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
(labs / "国九小市值策略.py").write_text(code, encoding="utf-8")
print(f"saved labs/国九小市值策略.json+py, code {len(code)} chars")
