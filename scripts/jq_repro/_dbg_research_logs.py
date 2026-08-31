# -*- coding: utf-8 -*-
"""调试: 看研究链路测试的日志尾部."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from core.event_engine.jq.entry import run_jq_backtest  # noqa: E402

src = (ROOT / "scripts" / "jq_repro" / "_test_research_api.py").read_text(
    encoding="utf-8")
start = src.index("CODE = r'''") + len("CODE = r'''")
end = src.index("'''", start)
res = run_jq_backtest(src[start:end], start="2025-01-06", end="2025-06-30",
                      capital=100_000.0)
for x in res["logs"][-14:]:
    print(repr(x[:120]))
