# -*- coding: utf-8 -*-
"""AlphaAgent MCP server（stdio）：把挖掘/评估/交付/记忆能力暴露给任意 agent。

用法（DSH / Cursor / Cline 等 MCP 客户端）：

    command: D:\\Quant\\quant_ui\\.venv\\Scripts\\python.exe
    args:    ["D:\\Quant\\quant_ui\\scripts\\alphaagent_mcp.py"]

自检：`python scripts/alphaagent_mcp.py --list-tools` / `--selftest`。
盘点与设计约束见 `docs/review/agent_tool_mcp_inventory_20261007.md`。
"""
from alphaagent.mcp_server.catalog import DESCRIPTORS, HANDLERS
from alphaagent.mcp_server.protocol import (
    PROTOCOL_VERSION,
    SERVER_NAME,
    dispatch,
    handle_message,
    serve_stdio,
)
from alphaagent.mcp_server.tools import ServerContext, ToolError

__all__ = [
    "DESCRIPTORS", "HANDLERS", "PROTOCOL_VERSION", "SERVER_NAME",
    "ServerContext", "ToolError", "dispatch", "handle_message", "serve_stdio",
]
