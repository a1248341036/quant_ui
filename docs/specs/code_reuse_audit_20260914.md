---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: 'dc689790-b3d7-4d29-9733-5ca3c491b427'
  PropagateID: 'dc689790-b3d7-4d29-9733-5ca3c491b427'
  ReservedCode1: '5fff8f5e-0068-484c-8df9-6acd44525e99'
  ReservedCode2: '5fff8f5e-0068-484c-8df9-6acd44525e99'
---

# 代码复用与重构扫描报告（2026-09-14）

> 审计范围：quant_ui 主仓（backend / alphaagent / core / scripts / static/src），
> 不含 CNEquity 数据湖子项目（独立仓库式管理）。
> 全部结论基于实际代码取证（rg 逐处定位 + 关键文件通读），非文档宣称。

## 0. 结论速览

| 优先级 | 事项 | 预计收益 | 风险 |
|--------|------|----------|------|
| P0 | 清理 50 个未跟踪临时脚本（1,649 行） | 立减 ~1.6k 行零功能损失 | 零（git 未跟踪，删前归档） |
| P0 | 修 worker.py `test_end="2025-12-31"` 硬编码 | 消除与 window_config 设计冲突的过期日期 | 低（改走收口中心默认值） |
| P1 | 抽 `report_io` 公共模块（JSON utf-8/GBK 读取 ×5 处） | 消除不一致实现（1 处缺 GBK 回退） | 低 |
| P1 | 股票代码归一化收口（zfill(6) ×13 处两种风格） | 消除归一化不一致隐患 | 低 |
| P1 | FactorValueCache 改用 `get_default_cache()`（8 处 new） | 跨调用方共享热缓存 | 低 |
| P1 | `_now()` 等 6 处时间工具 + 原子写 4 处收口 | ~50 行去重 | 零 |
| P2 | SQLite connect 样板收口（已有 core/sqldb.py） | ~40 行去重 + 超时参数统一 | 低 |
| P2 | 拆 alphaagent_service.py（2,036 行） | 可维护性 | 中（需单测护航） |
| 不建议 | 强行合并两套 walk-forward / 重构 alphaagent 挖掘核心 | — | 收益小于风险 |

## 1. 临时脚本清理清单（P0，零成本）

git 未跟踪 .py 共 50 个、1,649 行。全部为一次性诊断/验证脚本，删除无功能损失。

**scripts/ 根 15 个（692 行）**：`_tmp_time_audit.py`(115)、`_tmp_time_audit2.py`(109)、
`_check_original_arm5_candidates.py`(98)、`_tmp_model_check.py`(96)、`_tmp_llm_bench3.py`(96)、
`_tmp_llm_bench2.py`(79)、`_tmp_llm_bench.py`(65)、`_tmp_concurrency.py`(52)、
`_tmp_reasoning_probe.py`(48)、`_tmp_verify_run.py`(44)、`_tmp_seq.py`(36)、
`_monitor_run_adfbf57c45aa.py`(32)、`_monitor_run_ced4cb8a1fe3.py`(26)、
`_profile_single_eval.py`(27)、`_tmp_audit2.py`(13)

**scripts/cne/ 35 个（~950 行）**：`_tmp_*` 前缀 31 个（均 <40 行的一次性诊断），
另有 `clear_compact_gate_blockers.py`(40)、`audit_staging_orphans.py`(31)、
`compact_macro_runs.py`(26)、`run_wide_daily_once.py`(23)。

历史统计对照：`scripts/_*.py` 23 个 1,364 行 + `scripts/cne/_*.py` 31 个 593 行 + 
`scripts/jq_repro/_test_*.py` 约 1,300 行（已跟踪，另计）。

操作建议：`git status` 确认后整批移入 `scripts/_archive/`（不进 git）或直接删除；
**删前先跑一次完整挖掘 smoke 确认无 import 依赖**（这些脚本不被其他代码引用，
但要防 `from scripts._xxx import` 型隐式依赖，rg 验证过无此情况）。

## 2. 日期窗口硬编码：收口中心已建但未贯彻（P0/P1）

`alphaagent/factor/window_config.py` 是设计完备的收口中心（静态窗口 + TEST_END
动态解析链 + 进程级缓存），docstring 明确"任何组件不应再散落硬编码日期"，
但实际贯彻率不足：

| 位置 | 问题 | 严重度 |
|------|------|--------|
| `compute/worker.py:118` | `test_end = spec.get("test_end", "2025-12-31")` —— 硬编码过期日期，**与设计直接冲突**；window_config 明言固定日期会随数据演进过期（历史教训：硬编码 2026-08-29 超过实际数据截至 2026-08-28） | P0 |
| `compute/worker.py:113-117` | train/val/test 其余边界默认值与 window_config 常量重复但未引用 | P1 |
| `compute/panel_store.py:66-67` | `spec.get("train_start", "2020-01-01")` 等兜底默认重复 | P1 |
| `core/data/panel.py:21` | `PANEL_START` 环境变量兜底 "2020-01-01" | P2（有 env 覆盖，可接受） |
| scripts/ 下 ~20 处 | `resubmit_promising_factors.py`、`ingest_original_arm_factors.py`、`backfill_candidate_portfolio_segments.py` 等挖掘口径脚本整段复制窗口常量 | P2（脚本性质，随 P0 清理或逐个改 import） |

修复方案（半天）：worker.py 与 panel_store.py 的 spec 默认值改为
`from alphaagent.factor.window_config import ...` 引用常量；test_end 缺省时
调 `resolve_test_end(asset_type)`。scripts 侧改 import window_config 的
`mining_window()` / `test_window()`。

## 3. report/JSON 读取：×5 处重复且实现不一致（P1）

同一"读 report.json（utf-8 失败回退 GBK）"逻辑存在 5 份实现，其中 1 份不一致：

| 位置 | 实现差异 |
|------|----------|
| `backend/composite_factor_service.py:72-77` | utf-8 → GBK 回退 |
| `backend/stacking_service.py:139-147` | utf-8 → GBK 回退 |
| `backend/stacking_service.py:580-583` | 同文件内第二份（同服务内部重复！） |
| `backend/routers/alphaagent.py:507` | **只有 utf-8 无 GBK 回退** —— 遇 GBK 编码 report 会 UnicodeDecodeError |
| `scripts/cne/stacking_job.py:64` | 又一份拷贝 |

修复方案（1 小时）：新建 `backend/report_io.py`（或放 core）：
```python
def read_report_json(path: Path) -> dict:
    """report.json 统一读取：utf-8 优先，GBK 回退。"""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except UnicodeDecodeError:
        return json.loads(Path(path).read_text(encoding="gbk"))
```
5 处全部替换为 import 调用；`routers/alphaagent.py` 顺带修复 GBK 场景。

## 4. 股票代码归一化：×13 处两种风格（P1）

两种归一化实现并存，语义不同但未区分使用：

- **风格 A（zfill）**：`services_old.py`、`routers/sentiment.py:106`、
  `routers/stock.py:35,55`、`routers/ledger.py:26`、`routers/backtest.py:905-944`（5 处）
  —— `str(code).zfill(6)`，纯补零；
- **风格 B（正则提取）**：`train_ml_composite.py` 内 `_norm_code` —— 正则提取 6 位数字，
  可剥掉 `SH./SZ.` 前缀类脏数据。

修复方案（1 小时）：`core/instruments.py` 提供
`normalize_code(code, *, strip_prefix=False)` 单一实现，routers 层统一改引用。
注意风格 A/B 语义差异需逐处确认（是否需要剥前缀），不要盲改。

## 5. 因子值缓存：主链路已收口，但 8 处旁路直连（P1）

`alphaagent/factor/cache.py` 的 `FactorValueCache` 设计完善（内容寻址 key +
共享内存 LRU + 磁盘 2GB 上限淘汰），且提供 `get_default_cache()` 进程级单例。
但实际调用方 8 处全部 `FactorValueCache()` 直接 new 实例：

- `backend/stacking_service.py:477`、`scripts/train_ml_composite.py:409`、
  `scripts/scan_factor_combos.py:263`、`scripts/select_ml_exec_params.py:98`、
  `scripts/scan_factor_subsets.py:211`、`alphaagent/factor/mining/tools/_dispatch.py:950`
- 仅 `mining/eval/session.py:29` 用 dataclass default_factory（每 session 一实例，
  但内存 LRU 共享所以行为等价）

后果：磁盘缓存共享（内容寻址）但**内存 LRU 命中浪费**——同一表达式在两个
调用点间只能靠磁盘往返，30-48s/次的 DSL 求值本可内存直命中。

修复方案（30 分钟）：全部改 `get_default_cache()`；`session.py` 的
default_factory 改传 `get_default_cache`。

## 6. 时间/原子写小工具：_now ×6、原子写 ×4（P1）

- `def _now() -> str: return datetime.now(timezone.utc).isoformat()` 一字不差
  重复 6 处：`backend/alphaagent_service.py:63`、`alphaagent/factor/index.py:49`、
  `mining/agent/agentscope_run.py:114`、`mining/agent/loop.py:22`、
  `mining/memory/diagnostics.py:15`（+ `mining/memory/calibration.py` 同型 `_safe_float`）
- 原子写（tmp + os.replace）重复 4 份：`core/composites.py:27`、
  `core/ledger.py:22`、`core/store.py:81`、`alphaagent/data/adapters/cnequity.py:416`
  （+ `factor/cache.py` 内部 3 处，属类内聚可不动）
- `_safe_float` / `_to_float` 重复 5 处：`alphaagent_service.py:1831`、
  `services_old.py:215`、`lab_runner.py:26`、`memory/calibration.py:72`、
  `memory/diagnostics.py:19`

修复方案（半天）：`alphaagent/core/` 下新建 `timeutil.py`（utc_now_iso）与
`atomicio.py`（atomic_write_text / atomic_write_bytes），全仓替换。
收益 ~60 行去重 + 单一实现可单测。

## 7. SQLite connect 样板：core/sqldb.py 已有封装，仍散落自连（P2）

全仓 48 处 `sqlite3.connect`，其中核心代码自连且参数不一：

- `core/sqldb.py:36`：timeout=10、check_same_thread=False（规范封装）
- `composite_factor_service.py:50`、`memory/analytics.py:39`（mode=ro uri）、
  `memory/schema.py:206`、`factor/index.py:60`：各自裸 connect，timeout 10~30 不等
- 9+ 个 `scripts/cne/_tmp_*.py` 硬编码 manifest.db 绝对路径（随 P0 清理消失）

修复方案（1 天）：三套 db 样板（composite_factors.db / research_memory.db /
factor_index.db）改走 `core/sqldb.py` 增加的 `sqlite_connect(path, *, ro=False)`
统一入口；backend/composite_factor_service.py 与 alphaagent 侧 schema 文件
改为引用。**不建议**抽重 ORM/仓储层——现有 3 套库结构差异真实存在。

## 8. 前端重复模式（P2）

- `MlPanel.vue`（976 行）自带 `fmtNum/num2/pct/icClass` 内联实现，
  而 `utils/alphaagent.js` 已有 `formatMetricValue` + `icClass`、
  `utils/format.js` 已有 `pct/fmt`——同仓 7 个 alphaagent 组件用共享版，
  仅 MlPanel 自己造轮子。改 import 后可删 ~30 行。
- 轮询 `setInterval` 8 处（AlphaAgent 10s / JqRun 1s / MlPanel / Data /
  OvernightMonitorWidget / MetricsPanel 等），各写各的清理逻辑（beforeUnmount
  或缺失）。已有版本间重连的差异写法，可抽 `composables/usePolling.js`。
- 13 个 views 页面均各写一份 `methods:` 大对象，属 Vue2 选项式历史风格，
  收益低**不建议**现在动（等迁移 Vue3 时一并处理）。

## 9. walk-forward / 时间切分：两套实现，不建议合并（反模式警示）

- `core/walkforward.py`：回测页事件/因子策略滚动验证，`split_windows` 按日历
  均分 n 折，每折独立全量回测；
- `alphaagent/factor/stacking/model.py:53`：ML stacking 专用，expanding 窗口 +
  purge_days=5 防前向标签泄漏 + shift_months 多路径。

两者目标不同（回测报告 vs 防泄漏折切），stacking 侧带 purge 语义更先进。
**强行合并会引入泄漏风险**——core 侧调用方没有 purge 概念，统一后参数默认值
传播易出错。结论：保留两套，仅当 core/walkforward 未来做 ML 参数验证时
再考虑引用 stacking 侧实现。

## 10. 巨型文件拆分建议（P2，单测护航后做）

| 文件 | 行数 | 拆分方向 |
|------|------|----------|
| `backend/alphaagent_service.py` | 2,036 | run 生命周期 / 评估编排 / 入库三模块 + routers 薄壳 |
| `backend/routers/backtest.py` | 1,254 | mode 分支（event/factor/walk-forward）各拆 service |
| `mining/agent/agentscope_run.py` | 1,201 | 模型构建 / 对话循环 / 事件记录分离 |
| `mining/memory/retrieval.py` | 1,083 | 检索策略 / 打分 / 缓存分离 |
| `mining/delivery/submit.py` | 1,003 | 提交校验 / 相似度 / 入库分离 |

前提：`alphaagent/factor/mining/` 核心（17.5k 行）**整仓不建议动**——
刚跑通整夜监控，任何回归风险大于代码美感收益。拆分只做 backend 层
（有 routers 边界、可借 FastAPI 依赖注入隔离）。

## 11. 实施顺序与验证

1. **P0-1 临时脚本**：归档/删除 50 个文件 → `git status` 干净 →
   跑 `scripts/selftest.py` 冒烟（不涉挖掘核心）
2. **P0-2 worker.py test_end**：改 window_config 引用 → 后端重启由用户执行 →
   触发一次评估确认 test_end 动态解析生效（日志可见 `TEST_END 数据源 →`）
3. **P1 三件套**（report_io / normalize_code / get_default_cache）：各配 2-3 个
   单测（tests/ 下已有 test_ml_blind_test_isolation.py 风格可参照）→
   同参数对比 run 验证无行为变化
4. **P2 按需**：SQLite 收口与巨型文件拆分等 P1 落地稳定后再排期

所有改动走 feature 分支（如 `feature/reuse-p0-cleanup`），逐项合并；
涉及后端的改动重启由用户负责（30s 超时约定）。

## 12. 本次扫描未覆盖/明确的边界

- CNEquity 子项目内部重复不在本报告范围
- `services_old.py` 疑似整体废弃文件，值得单独确认调用链后整文件删除
  （rg 确认当前无 router 引用它，但保守起见列为独立待办）
- 前端 Vue2 → Vue3 迁移属另一议题
- `scripts/jq_repro/` 下已跟踪的 ~1,300 行 `_test_*.py` 属聚宽复现资产，
  建议移 `scripts/jq_repro/_archive/` 子目录而非删除（可能还要对照复测）

> AI生成