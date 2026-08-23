# cli 模块

路径：`src/cnequity/cli/main.py`

Click 命令组 `cne` 的实现入口（`pyproject.toml` `[project.scripts]` 与 `__main__.py`）。

**完整命令与参数**见 [CLI 参考](../reference/cli.md)；入门流程见 [快速开始](../getting-started/quickstart.md)。

---

## 源码地图

| 关注点 | 位置 |
|--------|------|
| 命令定义 | `cli/main.py` |
| demo 编排 | `cli/demo.py` |
| 配置路径解析 | `resolve_config_path`（缺省时引导 `cne config init`） |
| step 注册 | 启动时 `import cnequity.steps` |

### 退出码（供 cron / Task Scheduler）

| 场景 | 退出码 |
|------|--------|
| 成功 / `skipped_non_trading_day` | 0 |
| run / audit / init 失败 | 1 |
| `status --datasets` 有 STALE | 1 |
| `audit --full` UNHEALTHY | 1 |
