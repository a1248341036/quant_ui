# 评审报告：工作区未合并分支深度审查（2026-09-25）

**评审对象**：
1. `perf/adj-factors-workers`（Commit `d21b92e`）
2. `feat/tdx-auction-capital-changes`（Commit `7052118`）

**评审环境基准**：`main`（Commit `9b32274`）  
**评审性质**：只读静态代码审查（架构一致性、真实缺陷、运行时隐患、调度安全性）

---

## 评审总览与裁决矩阵

| 分支 | 改动规模 | 裁决 (Verdict) | 核心评价 |
| :--- | :--- | :--- | :--- |
| **`perf/adj-factors-workers`** | 1 文件，+1 / -1 | **`merge` (建议合入)** | 纯 CPU 并发上限调优（16→32），逻辑干净，无任何网络/限速副作用，可直接合并加速 daily 流水线。 |
| **`feat/tdx-auction-capital-changes`** | 20 文件，+1214 / -9 | **`merge_after_fixes` (整改后再合)** | 协议逆向与解析质量高（0x056A/0x000F），单元测试健全；但存在 **P0 级致命调度隐患**（全市场 5500 标的盲扫被塞入 daily `core` 门禁），若直接合入将拖死或阻断每日核心流水线。 |

---

## 一、 分支 1：`perf/adj-factors-workers` 详细审查

### 1.1 改动范围与代码
```python
# CNEquity/src/cnequity/derive/adj_factors.py:653
-    workers = max(1, min(config.workers, 16))
+    workers = max(1, min(config.workers, 32))
```

### 1.2 审查结论
- **逻辑正确性**：✅ 复权因子推导（`derive_adj_factors`）基于本地已入库的 `daily_bars` 和除权除息表计算复权因子，属于纯本地 CPU 与 Polars 内存密集型计算，**不经过任何外部网络请求，不受任何数据源限流（RateLimiter）约束**。
- **线程安全性**：✅ 由 `ThreadPoolExecutor` 并发调用 `_derive_symbol_adj_factors`，每个标的独立计算并返回 DataFrame，主线程在 `as_completed` 收集后统一合并，无共享可变状态竞争。
- **性能收益**：实测此前全市场推导耗时约 334 秒（5分半钟），在 16 核以上硬件上放宽至 32 线程可充分利用本地计算资源，预计节省 1~2 分钟耗时。
- **裁决**：**`merge`**。干净独立，建议立即合并。

---

## 二、 分支 2：`feat/tdx-auction-capital-changes` 详细审查

### 2.1 改动概述
新增基于通达信私有协议（TDX Wire）的两个重要数据集：
1. **`auction_series`**：集合竞价过程快照（命令 `0x056A`），逐秒虚拟撮合快照（开盘 09:15–09:25、收盘 14:57–15:00），手→股归一，挂在 `intraday` 组，默认关闭。
2. **`capital_changes`**：股本变迁 / 权息资料（命令 `0x000F`），全类别 1–15 原始事件日志（万股→股归一，补齐了 tdxpy 缺失的类别 15 重整调整）。

### 2.2 优势与亮点
- **协议解析与单位契约严密**：
  - `get_auction_series.py` 二进制解包结构完整，成交量与未匹配量正确处理手→股（×100）；
  - `get_capital_changes.py` 准确处理了股本数量类（万股 ×10000）与每 10 股分配类（原值保留）的类别级差异，符合数据湖统一单位契约；
- **`auction_series` 架构隔离好**：
  - 明确标为可选（`required = False`），配置默认 `[auction_series].enabled = false`，默认 `scope = "watchlist"`，挂在 `intraday` 组，不搭日常 daily waves，设计极其克制，不会对日常运行造成扰动；
- **离线测试质量优秀**：
  - `test_tdx_auction_capital_changes.py` 包含 12 个纯内存二进制包构建的单元测试，覆盖了解析器越界防御、字段解析、单位折算与异常回落。

---

### 2.3 发现的缺陷与致命风险清单

#### 🔴 P0 致命缺陷：`capital_changes` 排入 daily `core` 门禁，将拖死并经常性阻断日常流水线

- **问题定位**：
  - `CNEquity/configs/cnequity.example.toml:288`
    ```toml
    [job.daily.groups.core]
    at = "16:00"
    steps = [
      "instruments", "trading_calendar", "trading_status", "stock_st", "trading_status_st",
      "corporate_actions", "capital_changes", "daily_bars", "index_bars", ...
    ]
    ```
  - `CNEquity/src/cnequity/steps/events.py:295`
    ```python
    @register_step("capital_changes", group="core", depends_on=["instruments"])
    def step_capital_changes(config: Config, trade_date: date, run_id: str, context: dict) -> dict:
        ...
        symbols = list(context.get("_retry_symbols") or load_symbols(config))
    ```
  - `CNEquity/src/cnequity/adapters/tdx_protocol/client.py:202`
    ```python
    def fetch_capital_changes(symbols, ...):
        with TDX_SESSION_LOCK:
            client = _connect_with_retry(config)
            for index, sym in enumerate(symbols, start=1):
                df = _fetch_one(client, sym, ...)
    ```

- **事实与根因剖析**：
  1. **全市场 5500 标的无差别盲扫**：`symbols` 默认取全市场 A 股（~5500 只标的）。TDX 命令 `0x000F` 在 Wire 协议层**没有日期过滤参数**，每次请求单个标的都会返回该标的自上市以来的全量历史权息记录，由客户端本地做 `event_date == trade_date` 过滤。
  2. **严重的串行耗时**：`for index, sym in enumerate(symbols)` 为单 TCP 链接上的单线程串行遍历。即使在内网最佳延迟下（单请求 30ms），5500 次请求也需要 **165 秒（近 3 分钟）**；若受主站网络波动或 socket 重连影响，耗时将轻松突破 5~10 分钟。
  3. **致命的 Gate 级联雪崩**：`core` 是整个流水线最核心的第一道闸门（`GateWaves = @("core")`）。一旦 `capital_changes` 在遍历 5500 标的中有部分标的 socket 抖动超时，该步骤将返回 `status="failed"`，直接导致 **`core` gate 失败退出**！导致后续的 `capital`、`signals`、`fundamentals`、`research` 全部被饿死，每日数据全线停摆！
  4. **对比 `corporate_actions`**：底层的 0x000F 与 `corporate_actions` 是同一接口。之所以 `corporate_actions` 每天只需 5.3 秒，是因为它事先根据东财/交易所公告或者当天除权候选名单进行了**标的收窄**（每天仅几十只），而 `capital_changes` 没有任何筛选，每天盲打 5500 次。

---

#### 🟡 P1 架构隐患：大粒度锁独占 `TDX_SESSION_LOCK`

- **代码位置**：`CNEquity/src/cnequity/adapters/tdx_protocol/client.py:200`
  ```python
  with TDX_SESSION_LOCK:
      client = _connect_with_retry(config)
      for index, sym in enumerate(symbols, start=1):
          ...
  ```
- **影响**：`TDX_SESSION_LOCK` 是整个 TDX 模块的全局跨线程会话锁。用它直接包住一个 5500 次循环的大循环，意味着在执行 `capital_changes` 的数分钟内，其他任何并发线程若尝试与 TDX 交互（如探针心跳、行情报文），都会被无条件阻塞数分钟，容易引发级联超时。

---

## 三、 整改方案与建议

分支 `feat/tdx-auction-capital-changes` 的核心代码价值非常高，但**必须解除其对 daily `core` 门禁的绑定**后才能合入：

1. **移除 daily `core` 调度**：
   - 从 `[job.daily.groups.core]` 的 steps 列表中**移除 `"capital_changes"`**；
   - 将其定位与 `trade_ticks` / `minute_bars` 类似：作为独立的低频回填工具（支持 `cne backfill capital_changes`）或者放到专属维护 group 中；
   - 或者，如果日常确实需要每日增量，必须实现类似于 `corporate_actions` 的“仅扫描当天发生除权/股本变动的标的候选集”机制，将请求量从 5500 次缩减到几十次以内。
2. **细化持锁粒度**：
   - 将 `client.py` 中的 `with TDX_SESSION_LOCK:` 下沉到单次请求级别，避免长时间独占全局锁。

---

## 四、 最终合并操作建议

1. **`perf/adj-factors-workers`**：
   - **立即合并**（仅 1 行代码调整，无任何负面影响，可即刻加速复权因子推导）。
2. **`feat/tdx-auction-capital-changes`**：
   - 暂不合入 main；
   - 先在该分支上做最小整改：在配置中将 `capital_changes` 移出 `core` 默认 steps 列表，再行合并。
