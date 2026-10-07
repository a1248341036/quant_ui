#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""AlphaAgent MCP server 入口（stdio JSON-RPC，手写协议）。

MCP 客户端配置（示例，DSH / Cursor / Cline 通用）：

    {
      "mcpServers": {
        "alphaagent": {
          "command": "D:\\\\Quant\\\\quant_ui\\\\.venv\\\\Scripts\\\\python.exe",
          "args": ["D:\\\\Quant\\\\quant_ui\\\\scripts\\\\alphaagent_mcp.py"]
        }
      }
    }

子命令：
    （无参数）        以 stdio MCP server 运行（stdout 是协议流，日志走 stderr）
    --list-tools     打印工具清单（JSON），用于人工核对
    --selftest       自检协议握手 + 轻量工具；加 --with-eval 额外跑一次真实评估（载 panel，约 30s）
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alphaagent.mcp_server import ServerContext, serve_stdio  # noqa: E402
from alphaagent.mcp_server.catalog import DESCRIPTORS  # noqa: E402
from alphaagent.mcp_server.protocol import handle_message  # noqa: E402

_CHEAP_CHECKS: list[tuple[str, dict]] = [
    ("get_thresholds", {"mode": "technical_monthly"}),
    ("list_fields", {"prefixes": "label"}),
    ("describe_operator", {"name": "TS_CORR"}),
    ("precheck_expression", {"multi_line_expr": "CS_ZSCORE(TS_MEAN($ret, 20))"}),
    ("list_runs", {"limit": 3}),
    ("memory_stats", {}),
]


def _selftest(with_eval: bool) -> int:
    ctx = ServerContext(ROOT)

    def call(message: dict) -> dict:
        return handle_message(ctx, message) or {}

    init = call({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}})
    info = init["result"]["serverInfo"]
    print(f"[selftest] initialize → {info['name']} {info['version']} (protocol {init['result']['protocolVersion']})")

    listed = call({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}})
    names = [t["name"] for t in listed["result"]["tools"]]
    print(f"[selftest] tools/list → {len(names)} 个: {', '.join(names)}")
    assert len(names) == len(DESCRIPTORS)

    for name, args in _CHEAP_CHECKS:
        res = call({"jsonrpc": "2.0", "id": name, "method": "tools/call",
                    "params": {"name": name, "arguments": args}})
        payload = json.loads(res["result"]["content"][0]["text"])
        flag = "ERROR" if res["result"].get("isError") else "ok"
        keys = list(payload)[:6] if isinstance(payload, dict) else type(payload).__name__
        print(f"[selftest] {name:22s} {flag:5s} keys={keys}")

    # 写工具必须被 confirm 拦住
    for name, args in (("submit_factor", {"multi_line_expr": "CS_ZSCORE($ret)", "factor_name": "x", "comment": "y"}),
                       ("memory_record", {"multi_line_expr": "CS_ZSCORE($ret)"})):
        res = call({"jsonrpc": "2.0", "id": name, "method": "tools/call",
                    "params": {"name": name, "arguments": args}})
        assert res["result"].get("isError"), f"{name} 未 confirm 必须报错"
    print("[selftest] 写工具 confirm 门 → ok（未 confirm 均被拒）")

    if with_eval:
        res = call({"jsonrpc": "2.0", "id": "eval", "method": "tools/call", "params": {
            "name": "eval_batch", "arguments": {
                "exprs": [{"name": "selftest_atc", "why": "自检",
                           "expr": "CS_ZSCORE(TS_MEAN(DIVIDE($amount, $float_cap), 20))"}],
                "mode": "technical_monthly"}}})
        payload = json.loads(res["result"]["content"][0]["text"])
        row = payload["rows"][0]
        print(f"[selftest] eval_batch → ic={row['ic']} icir={row['icir']} "
              f"turn={row['turnover_rebalance']} fails={row['fails']}")

    ctx.close()
    print("[selftest] OK")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="AlphaAgent MCP server (stdio)")
    ap.add_argument("--root", default=str(ROOT), help="仓库根（默认脚本上级目录）")
    ap.add_argument("--list-tools", action="store_true", help="打印工具清单 JSON")
    ap.add_argument("--selftest", action="store_true", help="协议 + 轻量工具自检")
    ap.add_argument("--with-eval", action="store_true", help="自检时额外跑一次真实评估（载 panel）")
    ap.add_argument("--log-level", default="WARNING")
    ap.add_argument("--no-session-lock", action="store_true",
                    help="关闭跨进程会话锁（多客户端同时挖矿会各自占 6–8GB panel，内存自负）")
    args = ap.parse_args()

    # stdout 是协议流：日志一律写 stderr
    logging.basicConfig(stream=sys.stderr, level=getattr(logging, args.log_level.upper(), logging.WARNING),
                        format="%(levelname)s %(name)s: %(message)s")

    if args.list_tools:
        print(json.dumps(DESCRIPTORS, ensure_ascii=False, indent=1))
        return 0

    ctx = ServerContext(args.root, session_lock=not args.no_session_lock)
    if args.selftest:
        return _selftest(args.with_eval)

    serve_stdio(ctx)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())