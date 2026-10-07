# AlphaAgent MCP server · 客户端接线指南 · 2026-10-07

同一份 **stdio MCP server**(`scripts/alphaagent_mcp.py`)可被任意 MCP 客户端复用；差别只在
"往哪个注册文件里写一行"。本文记录本机已完成的接线、其它客户端的写法、验证与回滚。

## 0. 服务端（一份，处处复用）

| 项 | 值 |
|---|---|
| 入口 | `scripts/alphaagent_mcp.py`（stdio JSON-RPC，手写协议） |
| 解释器 | `D:\Quant\quant_ui\.venv\Scripts\python.exe` |
| 工具数 | **15**：`get_thresholds` `release_session` `list_fields` `describe_operator` `precheck_expression` `list_runs` `run_summary` `memory_search` `memory_stats` `eval_batch` `eval_val` `library_similarity` `dry_run_delivery` `submit_factor` `memory_record` |
| 自检 | `python scripts/alphaagent_mcp.py --list-tools` / `--selftest`（加 `--with-eval` 跑真实评估） |
| 超时 | `eval_batch`/`dry_run_delivery` 会载 6–8GB panel，单次可达数分钟 ⇒ 客户端 **tool timeout 必须调大**（建议 600–900s） |
| 并发 | 服务器内置**跨进程会话锁**（`artifacts/alphaagent/.mcp_session.lock`，OS 文件锁）：同一时刻只允许一个进程持有重会话，后来者收到明确的"忙"错误而非 OOM；只用轻量工具（门槛/字段/算子/预检/台账/记忆）不受影响；用完可调 `release_session` 交还。确需并行加 `--no-session-lock`（内存自负） |

## 1. Codex —— ✅ 已接好（本机）

配置文件：`C:\Users\zhoubw\.codex\config.toml`（备份：`config.toml.bak-alphaagent-mcp-20261007-185839`）

```toml
[mcp_servers.alphaagent]
command = 'D:\Quant\quant_ui\.venv\Scripts\python.exe'
args = ['D:\Quant\quant_ui\scripts\alphaagent_mcp.py']
startup_timeout_sec = 120
tool_timeout_sec = 900
```

验证：`tomllib` 解析通过、`mcp_servers` 下已出现 `alphaagent`、乱码 0 行；以 Codex 的方式
（管道喂 `initialize`）拉起服务端有正常响应。
生效：新开 Codex 会话即在工具列表看到 `alphaagent` 的工具（若客户端不热加载配置，重启 Codex）。
回滚：删掉该 `[mcp_servers.alphaagent]` 段（或从备份恢复）。

## 2. DSH（DeepSeek Harness）—— ✅ 已配置，**需重启生效**

DSH 的工具面来自 profile 的插件栈，**默认不含 MCP 客户端**；官方桥插件是
`@deepseek-ai/dsh-mcp-client`（"MCP client bridge: connects to MCP servers and registers their
tools on `ctx.tools`"）。已完成两步：

1. **安装插件**（普通依赖，非 bundle；DSH 提示 "declares no `dsh.bundle`…"）：
   ```powershell
   dsh plugin --profile desktop add "@deepseek-ai/dsh-mcp-client"
   ```
   → `@deepseek-ai/dsh-mcp-client 0.0.1-rc.1`，落在 `~/.dsh/profiles/desktop/node_modules/`。
2. **注册实例**（追加到 `~/.dsh/profiles/desktop/cordis.patch.yml`）：
   ```yaml
   - insert:
       - id: mcp-alphaagent
         name: '@deepseek-ai/dsh-mcp-client'
         config:
           serverName: alphaagent
           transport: stdio
           command: 'D:\Quant\quant_ui\.venv\Scripts\python.exe'
           args: ['D:\Quant\quant_ui\scripts\alphaagent_mcp.py']
           cwd: 'D:\Quant\quant_ui'
           toolCallTimeoutMs: 900000
   ```
   工具名形如 **`mcp__alphaagent__eval_batch`**（serverName 命名空间）。

备份（同目录）：`package.json.bak-alphaagent-mcp-20261007-185954`、
`pnpm-lock.yaml.bak-alphaagent-mcp-20261007-185954`、`cordis.patch.yml.bak-alphaagent-mcp-20261007-185954`。

**为什么需要重启**：profile 已开 `patchReload: live`（patch 文件改动会热重载），但本次是**首次
加载一个新安装的插件包**，需要重新组合 profile 的模块图；宿主进程重启最稳。重启会中断当前会话，
所以由你决定时机。
**重启后验证**：让会话调用 `mcp__alphaagent__get_thresholds`（或 `list_runs`）；或让会话执行
`python scripts/alphaagent_mcp.py --selftest` 对比。
**回滚**：删除该 insert 块（停用）+ 可选 `dsh plugin --profile desktop remove @deepseek-ai/dsh-mcp-client`。

## 3. 其它客户端（本机未安装，写法通用）

**Cursor**（`~/.cursor/mcp.json` 或项目 `.cursor/mcp.json`）：
```json
{ "mcpServers": { "alphaagent": {
    "command": "D:\\Quant\\quant_ui\\.venv\\Scripts\\python.exe",
    "args": ["D:\\Quant\\quant_ui\\scripts\\alphaagent_mcp.py"] } } }
```

**Cline / Roo / Claude Desktop**（`claude_desktop_config.json` 同形）：
```json
{ "mcpServers": { "alphaagent": {
    "command": "D:\\Quant\\quant_ui\\.venv\\Scripts\\python.exe",
    "args": ["D:\\Quant\\quant_ui\\scripts\\alphaagent_mcp.py"],
    "env": {} } } }
```

**VS Code（Copilot MCP，`.vscode/mcp.json`）**：
```json
{ "servers": { "alphaagent": {
    "type": "stdio",
    "command": "D:\\Quant\\quant_ui\\.venv\\Scripts\\python.exe",
    "args": ["D:\\Quant\\quant_ui\\scripts\\alphaagent_mcp.py"] } } }
```

> 超时/批量建议：若客户端 tool timeout 只有 60s（多数默认），把 `eval_batch` 的每批控制在
> **≤8 个表达式**（每个约 6–25s）；大客户端（Codex/DSH 已设 900s）可一次 30–60 个。

## 4. 验证清单（任一客户端）

1. 服务端自检：`python scripts/alphaagent_mcp.py --selftest`（协议 + 轻量工具 + 写门）；
2. 单测：`.venv\Scripts\python.exe -m pytest tests/test_alphaagent_mcp_server.py -q`
   （21 项；重活需 `ALPHAAGENT_MCP_EVAL_TEST=1`）；
3. 端到端：把 `{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}` 管道喂给入口，应回
   `serverInfo.name = "alphaagent"`；
4. 客户端内：调用 `mcp__alphaagent__get_thresholds(mode="technical_monthly")`，应返回
   0.053 / 0.65 / 0.80 等生效门槛。

## 5. 已知取舍

- **`eval_batch` 走服务层直连**（不走 `_dispatch` 护栏），以换取批量与免护栏开销；配套用
  `memory_record` 写回记忆，保证死路学习不丢。要"完全受护栏约束"的评估路径需把 `_dispatch`
  的 dispatch 接进 MCP（依赖其 tools bundle，属后续工作）。
- **面板会话很重**：多客户端同时挖掘靠会话锁串行化，`release_session` 可主动交还。
- **盲测/晋升/重筛**刻意未暴露（耗预算或需用户拍板）。
