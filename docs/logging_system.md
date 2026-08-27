# 统一日志系统文档

## 概述

quant_ui 后端已实施**轻量级统一日志系统**（方案 A），提供：
- 统一的日志格式和文件轮转
- 多个专用日志文件（主日志、API 日志、错误日志、数据日志、回测日志）
- 自动注入请求 ID 和上下文信息
- REST API 查看日志
- **装饰器方式**减少代码侵入性

## 日志文件结构

```
logs/
├── quant_ui.log              # 主日志文件（应用启动、业务逻辑）
├── api.log                   # HTTP 请求日志（所有 API 调用）
├── error.log                 # 错误日志（仅 ERROR 级别）
├── data.log                  # 数据加载日志（数据源加载、更新进度）
├── backtest.log              # 回测日志（回测执行、性能指标）
├── backend_uvicorn.log       # Uvicorn 服务器日志
├── cne_serve.log             # CNE 数据服务日志
└── factor_mining/
    └── ui/
        └── <run_id>/         # AlphaAgent 挖掘任务日志
            ├── run_*.jsonl
            └── console.log
```

## 日志配置

### 默认参数

```python
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB（单个文件最大）
LOG_BACKUP_COUNT = 5              # 保留 5 个备份文件
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
```

### 日志级别

- `DEBUG`: 调试信息（开发用）
- `INFO`: 一般信息（默认级别）
- `WARNING`: 警告信息
- `ERROR`: 错误信息（自动写入 error.log）
- `CRITICAL`: 严重错误

## 使用方法

### 方式 1：直接使用 logger（适合简单场景）

```python
from backend.logging_config import main_logger, api_logger, error_logger, data_logger, backtest_logger

# 记录日志
main_logger.info("服务启动成功")
data_logger.info("数据加载完成")
backtest_logger.info("回测执行完毕")
```

### 方式 2：使用装饰器（推荐，减少侵入性）

#### 通用函数调用装饰器

```python
from backend.logging_decorators import log_function_call

@log_function_call()
def load_data(start, end, need_panel=True):
    # 函数自动记录调用、参数、耗时
    return data
```

**输出示例：**
```
2026-08-27 16:53:50 [INFO] quant_ui.data: Calling load_data(start='2023-01-01', end='2023-12-31', need_panel=True)
2026-08-27 16:53:50 [INFO] quant_ui.data: load_data completed in 0.10s: 1000 rows
```

#### 数据加载专用装饰器

```python
from backend.logging_decorators import log_data_loading

@log_data_loading
def load_data(start, end, need_panel=True, codes=None, need_heavy=False):
    # 自动记录：加载参数、数据量、耗时
    return result
```

**输出示例：**
```
2026-08-27 16:53:50 [INFO] quant_ui.data: Data load: start=2023-01-01, end=2023-12-31, need_panel=True, codes=2, heavy=False
2026-08-27 16:53:50 [INFO] quant_ui.data: Data load completed in 0.10s: universe=0 codes
```

#### 回测执行专用装饰器

```python
from backend.logging_decorators import log_backtest_execution

@log_backtest_execution
def run_backtest(strategy, universe, start, end, top_n=5, capital=100000):
    # 自动记录：回测参数、性能指标、耗时
    return result
```

**输出示例：**
```
2026-08-27 16:53:50 [INFO] quant_ui.backtest: Backtest: strategy=低换手冷门, universe=全部股票, range=[2023-01-01, 2023-12-31], top_n=10, capital=100000
2026-08-27 16:53:50 [INFO] quant_ui.backtest: Backtest completed in 0.10s: return=0.15, sharpe=1.2, max_dd=-0.08
```

### 方式 3：使用上下文管理器（记录代码块）

```python
from backend.logging_decorators import log_block

with log_block(message="加载数据", logger=data_logger):
    data = load_data(start="2023-01-01", end="2023-12-31")
```

**输出示例：**
```
2026-08-27 16:53:50 [INFO] quant_ui.data: Starting: 加载数据
2026-08-27 16:53:50 [INFO] quant_ui.data: Completed: 加载数据 in 0.10s
```

### 带上下文的日志

HTTP 请求中间件会自动注入 `request_id`：

```python
# 请求日志示例
2026-08-27 16:40:36 [INFO] quant_ui.api: Request abc12345 GET /api/alphaagent/factors completed with status 200 in 45ms
```

## REST API

### 查看日志

```bash
# 获取最近 100 条日志
GET /api/alphaagent/logs?limit=100

# 按 run_id 过滤
GET /api/alphaagent/logs?run_id=abc123&limit=50

# 按日志级别过滤
GET /api/alphaagent/logs?level=ERROR

# 组合过滤
GET /api/alphaagent/logs?run_id=abc123&level=WARNING&limit=200
```

**响应格式**：
```json
{
  "log_file": "d:/Quant/quant_ui/logs/quant_ui.log",
  "total_lines": 150,
  "filters": {
    "run_id": "abc123",
    "level": "ERROR",
    "limit": 100
  },
  "logs": [
    {
      "timestamp": "2026-08-27 16:40:36",
      "level": "INFO",
      "logger": "quant_ui",
      "message": "Request abc123 GET /api/alphaagent/factors completed"
    }
  ]
}
```

### 实时日志流

```bash
# 实时查看日志（类似 tail -f）
GET /api/alphaagent/logs/tail?run_id=abc123
```

返回 Server-Sent Events (SSE) 流：
```
data: 2026-08-27 16:40:36 [INFO] quant_ui: Starting run abc123

data: 2026-08-27 16:40:37 [INFO] quant_ui: Evaluating factor expr_001

```

## 日志最佳实践

### ✅ 推荐做法

#### 1. 优先使用装饰器（减少代码侵入）

```python
# ✅ 推荐：一行代码搞定日志
@log_data_loading
def load_data(start, end, need_panel=True):
    # 函数内部不需要任何日志代码
    return data

@log_backtest_execution
def run_backtest(strategy, universe, start, end):
    # 自动记录回测参数和结果
    return result
```

#### 2. 复杂逻辑使用上下文管理器

```python
# ✅ 推荐：精确控制日志范围
with log_block(message="数据预处理", logger=data_logger):
    data = clean_data(data)
    data = fill_missing(data)
```

#### 3. 使用合适的 logger

```python
# API 层使用 api_logger
api_logger.info(f"Request {request_id} {method} {path}")

# 数据加载使用 data_logger
data_logger.info("Data load completed")

# 回测使用 backtest_logger
backtest_logger.info("Backtest completed")

# 错误统一使用 error_logger
error_logger.error("Database connection failed", exc_info=True)
```

#### 4. 使用 exc_info 记录异常

```python
try:
    risky_operation()
except Exception as e:
    logger.error(f"Operation failed: {e}", exc_info=True)
```

### ❌ 避免做法

#### 1. 不要手动添加大量日志代码

```python
# ❌ 错误：侵入性强，难以维护
def load_data(start, end):
    logger.info("Starting load_data")
    logger.info(f"Loading from {start}")
    data = load_from_db(start, end)
    logger.info(f"Loaded {len(data)} rows")
    logger.info("Load completed")
    return data

# ✅ 正确：使用装饰器
@log_data_loading
def load_data(start, end):
    return load_from_db(start, end)
```

#### 2. 不要使用 print

```python
# ❌ 错误
print("Debug info")

# ✅ 正确
logger.debug("Debug info")
```

#### 3. 不要记录敏感信息

```python
# ❌ 错误
logger.info(f"User password: {password}")

# ✅ 正确
logger.info(f"User {user_id} logged in")
```

## 日志轮转

当日志文件达到 10MB 时自动轮转：
- `quant_ui.log` → `quant_ui.log.1`
- `quant_ui.log.1` → `quant_ui.log.2`
- ...
- `quant_ui.log.4` → `quant_ui.log.5`（删除）

保留最近 5 个备份文件，自动清理旧文件。

## 监控建议

### 1. 定期检查错误日志

```bash
# 查看最近 100 条错误
GET /api/alphaagent/logs?level=ERROR&limit=100

# 命令行查看
Get-Content logs\error.log -Tail 50
```

### 2. 监控 API 响应时间

API 日志包含耗时信息：
```
Request abc123 GET /api/alphaagent/eval-factor completed with status 200 in 1250ms
```

关注耗时 > 1000ms 的请求。

### 3. 清理旧日志

```powershell
# 删除 30 天前的日志文件
Get-ChildItem -Path logs -Filter "*.log.*" | 
    Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } | 
    Remove-Item
```

## 调试技巧

### 临时提高日志级别

```python
# 在代码中临时设置
import logging
from backend.logging_config import main_logger

main_logger.setLevel(logging.DEBUG)  # 临时开启调试日志
```

### 查看特定模块日志

```python
# 创建模块专用 logger
from backend.logging_config import get_logger
module_logger = get_logger("backend.alphaagent_service")

# 只查看该模块日志
Get-Content logs\quant_ui.log | Select-String "backend.alphaagent_service"
```

### 追踪请求链路

所有 HTTP 请求都有唯一的 `request_id`，可以通过它追踪完整请求链路：

```powershell
# 查找特定请求的所有日志
Get-Content logs\quant_ui.log | Select-String "abc12345"
```

## 故障排查

### 问题 1：日志文件不生成

**原因**：日志目录权限不足
**解决**：
```powershell
# 检查目录权限
Get-Acl logs | Format-List

# 创建目录
New-Item -Path logs -ItemType Directory -Force
```

### 问题 2：日志级别不生效

**原因**：logger 重复配置
**解决**：确保只调用一次 `setup_root_logger()`

### 问题 3：日志文件过大

**原因**：日志级别设置过低（DEBUG）
**解决**：
```python
# 修改日志级别
main_logger.setLevel(logging.INFO)
```

## 扩展

### 添加新的日志文件

```python
# backend/logging_config.py
SPECIAL_LOG_FILE = LOG_DIR / "special.log"

def setup_special_logger():
    return get_logger("quant_ui.special", log_file=SPECIAL_LOG_FILE)
```

### 自定义日志格式

```python
# 修改 LOG_FORMAT
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s [%(request_id)s]: %(message)s"
```

### 集成外部日志系统

未来可升级到：
- **方案 B**：structlog（结构化 JSON 日志）
- **方案 C**：ELK Stack / Grafana Loki（集中式日志）

---

**实施日期**：2026-08-27  
**版本**：1.0  
**状态**：✅ 已部署并测试通过
