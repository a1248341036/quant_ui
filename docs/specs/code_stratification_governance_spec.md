# SPEC: 多 Agent 快速迭代留下的“代码地质分层”治理规范（修正版）

> 状态：正式执行基准  
> 版本：v1.1（经过实际代码审计校准）  
> 日期：2026-09-14  

---

## 1. 现象背景与问题定性

在经过多个并行 Agent（以及多分支协作）持续高频迭代后，仓库中产生了**“代码地质分层（Code Stratification）”**现象。
典型表现为：**“新 Agent 开发新需求时，担心改动老逻辑引发未知回归，于是复制或新建一套全新实现（v2/unified/mmap/main），而旧常量、历史一次性迁移脚本、以及废弃命名依然散落各处”**。

---

## 2. 五大沉积层的客观校准

### 层一：因子库物理路径残留
* **实际代码事实**：
  - `core/factor_categories.py` 的逻辑早已收口指向 `RESEARCH_MODES`，但顶部注释（L10-15）依然赫然写着 4 库旧名；
  - `alphaagent/core/paths.py:16` 存在真正的硬编码残留：`FACTORZOO_DIR = ... / "production_technical"`，导致 `agentscope_tools.py:317` 等调用方仍然需要挂载这个已经过期的旧目录作为兜底；
  - `alphaagent_service.py` 内部传递 `category` 是正常的研究模式枚举，非残留。
* **治理动作**：
  - 更新 `paths.py` 的 `FACTORZOO_DIR` 指向 `production_main`；
  - 修正 `factor_categories.py` 的过期 docstring。

### 层二：数据加载管道状态校准
* **实际代码事实**：
  - 数据管道已通过 `panel_store.py` 自然收口（Arrow IPC mmap 优先 $\to$ Parquet 降级）；
  - `session.py:52-62` 的 `cne://` 与 `parquet` 分支属于 Adapter 模式的正常设计，不存在所谓的 `_transform_panel` 契约冲突。
* **治理判定**：**保持现状，不盲目删除降级容错分支**。

### 层三：后端遗留文件 `backend/services_old.py`
* **实际代码事实**：
  - 真实物理行数为 **361 行**；
  - `services/__init__.py` 依然从 `services_old` re-export 16 个函数；
  - `build_codes`, `load_tech`, `get_name_map` 等基础数据查询仍寄生在此。
* **治理动作**：按领域平移收敛至 `backend/services/market_data.py` 等明确模块，消除 `services_old.py`。

### 层四：指标评估的双口径本质（保留并显式标注）
* **实际代码事实**：
  - `core/metrics.py`（几何年化 `(1+total)^(252/n)-1`，回撤为负数）服务于事件引擎日频净值；
  - `alphaagent/factor/metrics/portfolio.py`（算术年化 `mean * sqrt(ann)`，回撤为正数）服务于因子多日持有期非重叠采样；
  - **两者是基于不同金融场景的设计意图，强行合并会引入严重语义错误**。
* **治理判定**：**不做破坏性合并**，在各自 docstring 中显式标注数学口径与适用场景。

### 层五：历史一次性脚本归档
* **实际代码事实**：
  - 5 个一次性 backfill 脚本 + 1 个 migrate 脚本共 **568 行**（任务已全部执行完毕并固化）；
  - `alphaagent/factor/zoo/realign.py`（**677 行**）仅被 `scripts/realign_factorlib.py` 一处引用，属于历史库重对齐运维工具；
  - 合计可清理/归档代码 **~1,245 行**。
* **治理动作**：将 6 个历史一次性脚本与 `realign_factorlib.py` 移入 `scripts/_archive/`，`realign.py` 同步清理或移入归档。

---

## 3. “三不拆”红线（严格维持）

1. **不拆挖掘核心状态机**；
2. **不合并两套 walk-forward**；
3. **不破坏 DSL 算子逐位一致性**。

---

## 4. 实施清单与执行顺序

| 阶段 | 事项 | 影响行数 | 风险评估 |
| :--- | :--- | :---: | :--- |
| **P0** | 修 `paths.py:16` `FACTORZOO_DIR` 指向 `production_main` | 1 行 | 极低（与统一大库对齐） |
| **P0** | 更新 `factor_categories.py` docstring | 4 行 | 零 |
| **P0** | 归档 6 个历史 backfill/migrate 脚本与 realign 体系到 `scripts/_archive/` | ~1,245 行 | 零（无运行时生产依赖） |
| **P1** | 拆解平移 `services_old.py`（361 行）并物理删除 | 361 行 | 低（需同步单测） |
| **P2** | 在 `portfolio.py` 闭包与 `metrics.py` 加注数学口径声明 | ~10 行 | 零 |
