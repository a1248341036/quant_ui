---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: '9b831806-958f-4caa-ae3b-77efe444ece1'
  PropagateID: '9b831806-958f-4caa-ae3b-77efe444ece1'
  ReservedCode1: 'cc20aada-9163-4703-80be-e7fcc807e3bd'
  ReservedCode2: 'cc20aada-9163-4703-80be-e7fcc807e3bd'
---

# 代码精简与收口总方案（2026-09-14 合并版）

> 合并自三轮扫描：code_reuse_audit / code_stratification_governance / code_deep_scan_round3。
> 全部结论基于实际代码取证，非文档宣称。

## 一、已完成项（不再重复）

| 事项 | 落地 commit | 效果 |
|------|-------------|------|
| report/JSON 读取 ×5 处 → `backend/report_io.py` | c88dfeb | 消除 1 处缺 GBK 回退的 bug |
| 股票代码归一化 ×13 处 → `core/instruments.py` | c88dfeb | 统一 zfill(6) 风格 |
| `_now()` ×6 处 → `alphaagent/core/timeutil.py` | c88dfeb | ~30 行去重 |
| 原子写 ×4 处 → `alphaagent/core/atomicio.py` | c88dfeb | ~40 行去重 |
| FactorValueCache 8 处 new → `get_default_cache()` | c88dfeb | 内存 LRU 跨调用方共享 |
| worker.py `test_end` 硬编码 → window_config 引用 | c88dfeb | 消除过期日期 |
| FACTORZOO_DIR `production_technical` → `production_main` | 9ae8fac | 与统一大库对齐 |
| 6 个 backfill/migrate 脚本 + realign 体系删除 | 9ae8fac | -1,519 行 |
| 50 个未跟踪临时脚本清理 | c88dfeb | -1,649 行 |

## 二、待执行项

### P1 — 低风险，各半天

#### 1. `_to_float` ×7 处统一

全仓 7 处 `def _to_float` 定义，签名和实现略有差异：

| 位置 | 签名 |
|------|------|
| `backend/services_old.py:215` | `_to_float(x) -> float \| None` |
| `backend/alphaagent_service.py:1832` | `_safe_float(v) -> float \| None` |
| `backend/lab_runner.py:26` | `_to_float(x) -> float \| None` |
| `backend/routers/backtest.py` | `_to_float(value, default=None)` |
| `alphaagent/factor/mining/memory/calibration.py:72` | `_to_float(value, default=None)` |
| `scripts/train_ml_composite.py` | `_norm_code` 内联 |
| `core/factors/__init__.py` | `_to_float(x: bytes \| str)` |

方案：新建 `core/numutil.py`，提供 `to_float(v, *, default=None) -> float | None`，7 处改 import。

#### 2. logging 两文件合并 + 删死代码

`logging_config.py`（287 行）+ `logging_decorators.py`（240 行）= 527 行。

合并为 `backend/logging.py`（~480 行）。同时删除 `get_logger_with_context`（全仓仅 1 处定义、0 处调用）。

使用频率确认（合并安全性）：
- `RequestContext`：132 处引用（高频）
- `log_function_call`：13 处
- `LogContext / log_block`：19 处
- `log_data_operation`：9 处
- `log_backtest_execution`：9 处
- `parse_log_file`：7 处

#### 3. qweave_runner 反向依赖修复

`backend/qweave_runner.py:17` 从 scripts 层导入生产代码：
```python
from scripts.qweave_research import ALPHA_SETS, build_alphas, load_panel, to_qweave_df
```

后端不应依赖 scripts（辅助脚本目录）。方案：将 qweave_research.py 中被依赖的 4 个对象移到 `core/qweave/` 或 `backend/qweave/`，qweave_runner 改从新位置导入，scripts 侧保留薄壳。

#### 4. lab_runner.py 瘦身

`lab_runner.py`（161 行）仅被 `alphaagent_service.py:1304` 一处引用（`load_module`）。
其中 `_to_float` / `points` 与 services_old.py 重复，随 P1-1 统一后删除。
瘦身后保留 `load_module` + `main`，约 80 行。

### P2 — 需验证后执行

#### 5. services_old.py 搬家

`services_old.py`（407 行）不是废弃文件——`services/__init__.py` 主动导入其 16 个函数，是 backend 核心数据服务层。命名为 `_old` 是"搬家未完成"的中间态。

拆分方案：
- `services/market_data.py`：`load_data` / `invalidate_data` / `run_update_background` / `run_configured_update_background` / `configured_update_tasks`（~200 行）
- `services/codes.py`：`build_codes` / `get_name_map` / `get_fund_name_map` / `get_industry_map` / `series_to_points` / `clean_records` / `_to_float`（~150 行，`_to_float` 随 P1-1 移走后更少）
- 更新 `services/__init__.py` 导入路径
- 删除 `services_old.py`
- 前提：`test_backend_services.py` 全通过

#### 6. alphaagent_service.py 2,288 行拆分

三段边界已确认：

| 段 | 行范围 | 函数数 | 拆为 |
|----|--------|--------|------|
| Run 生命周期 | 68-992 | 28 + AgentRun class | `services/run_lifecycle.py` ~900 行 |
| 因子库操作 | 1039-1897 | 18 | `services/factor_library.py` ~900 行 |
| 评估编排 | 1897-2288 | 4 | `services/factor_evaluation.py` ~400 行 |

`alphaagent_service.py` 保留为 thin facade（re-export + 公共状态）。
前提：涉及 alphaagent 的 ~30 个测试全部通过。

#### 7. MlPanel.vue 1,340 行拆分

- `<template>` 619 行 / `<script>` 581 行 / `<style>` 140 行
- 拆为 `MlPanel.vue`（~300 行骨架）+ `MlReportPanel.vue`（~250 行）+ `MlRunDetail.vue`（~200 行）
- 内联函数 `parseCommentSegments` / `fmtNum` 等移到 `utils/alphaagent.js`
- 前提：`npx vite build` 验证

#### 8. SQLite connect 收口

核心代码 ~15 处裸 `sqlite3.connect`（排除 .temp / scripts / .venv）：
- `core/sqldb.py` 已有规范封装（timeout=10, check_same_thread=False）
- `composite_factor_service.py`、`memory/schema.py`、`memory/analytics.py`、`factor/index.py` 等各自裸连，timeout 10~30 不等
- 方案：`core/sqldb.py` 增加 `sqlite_connect(path, *, ro=False)` 统一入口，15 处改引用
- 不抽 ORM/仓储层——3 套库（composite_factors.db / research_memory.db / factor_index.db）结构差异真实存在

#### 9. jq 层补 2-3 个自动化回归测试

`core/event_engine/jq` 3,936 行，当前 0 个测试文件。正确性靠 jq_repro 的 37 个手动对比脚本。
建议取有 CSV 对照数据的 2-3 个场景自动化为 pytest。非紧急。

## 三、红线：不动的

| 对象 | 行数 | 理由 |
|------|------|------|
| alphaagent/factor/mining 核心 | 17,545 | 刚跑通整夜监控，回归风险远大于代码美感 |
| core/event_engine/jq 聚宽兼容层 | 3,936 | 每行对应一个聚宽 API 模拟，删一行破一个兼容 |
| 两套 walk-forward | — | core 侧按日历均分 n 折回测；stacking 侧 expanding + purge 防泄漏。目标不同，合并引入泄漏风险 |
| store/alphaagent.js vs utils/alphaagent.js | — | 一个是 reactive 状态容器，一个是纯函数库，无重复 |
| 前端测试框架 | — | 项目级决策，不在收口范围 |
| core/metrics.py vs portfolio.py 双口径 | — | 几何年化 vs 算术年化，不同金融场景，加 docstring 标注即可 |

## 四、全仓行数全景

| 目录 | 行数 | 占比 | 可精简空间 |
|------|------|------|-----------|
| alphaagent | 41,734 | 37.6% | 不可精简（mining 17.5k + dsl 10k = 27.5k 是基础设施） |
| scripts | 19,036 | 17.2% | jq_repro 37 个可归档 3,551 行 + cne 43 个可归档 1,223 行 |
| core | 13,947 | 12.6% | jq 3,936 行不可精简；其余收口空间小 |
| tests | 14,188 | 12.8% | 不动（jq 补测试是增量非精简） |
| static/src | 11,240 | 10.1% | MlPanel 1,340 行可拆 |
| backend | 10,036 | 9.1% | services_old 407 行搬家 + alphaagent_service 2,288 行拆分 |
| docs | 4,402 | 4.0% | — |
| **合计** | **~110,878** | 100% | — |

功能代码约 77,287 行（排除 tests + docs），其中 21,481 行（mining + jq）是不可精简的基础设施（27.8%）。真正可优化空间在剩余 ~55,800 行的重复与冗余中。

## 五、执行原则

- 走 feature 分支，禁直改 main
- 后端改动重启由用户负责（30s 超时）
- 新功能带开关默认关闭
- 同参数对比 run 验证无行为变化
- 每步配单测（参照 tests/ 现有风格）

> AI生成