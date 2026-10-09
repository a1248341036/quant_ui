# 因子评估超时（EvalTimeout）根因诊断报告

> 日期：2026-10-06　关联 run2：`logs/factor_mining/run_20261006_192628.jsonl`
> （run_meta: DeepSeek-V4-Flash-0731，报告模式，测试段 train 3151819×176，4985 股票）
> 事件触发：用户反馈"因子超时评估看下什么原因，以前很少有超时的"。
> 修订：2026-10-07 依实验3（完整 8 表达式复算）与 jsonl/console 时间线逐行核对，修正
> 三处事实错误 —— ① @1w 因子不属于 batch9（它们在 11:35/11:40 批次）；
> ② run2 启动到 batch9 仅 17 分钟（原"4 小时 17 分钟"系本地/UTC 时区混减）；
> ③ batch9 真算 5 < 信号量 6，信号量未构成瓶颈。

---

## 一、结论速览

本次 run2 到 11:43:25 的 batch9（同秒提交 8 个 eval_on_train_set）出现 2 个 EvalTimeout
（601.0s / 604.8s）+ 2 个因子（bottom_fractal_dist / max_amount_dist20）**连结果都没有**，
随后 run 冻结。全部 5 个"真算"因子在**干净进程 12 路并发复算均 ≤11.4s**（实验3），
即这 5 个假超时**与因子计算量、新算子首次 JIT、线程池并发、信号量排队全部无关**。

**假超时的完整因果链（三段）**：

1. **执行被瞬时放大**：batch9 提交时线程池已空（batch8 于 11:43:12 全部完成），
   5 个真算立即并行展开。其中 vol_spike_cooldown 的 dsl_ms=401.3s 是 engine 内部
   纯 eval_factor 执行计时（不含排队），而同样表达式干净进程 12 路并发只要 6.5s
   —— **执行放大约 62x（单线程 1.6s 计则 250x）**，且 pb（RANK/CS_ZSCORE 两个
   老算子）也被放大 ~500x（1.2s → 604.8s）。放大只能来自**那一刻的机器/进程
   资源压力**（内存/swap/GC，叠加 5 个 315 万行全量 evaluate 并发展开 + 当日
   18:00 启动的高负载进程），无法从日志回溯到单一触发点（见 2.2-5）。
2. **600s 预算墙到点被掐**：`asyncio.wait_for(..., timeout=600)`（agentscope_tools.py
   :545-572）从 run_in_executor 提交时刻起算，排队/信号量等待都计入。top_fractal
   601.0s、pb 604.8s 均 ≈ `dispatch + 600s`，预算到点被掐，返回 EvalTimeout。
3. **主循环冻结吞噬剩余结果**：超时只取消 future、不杀线程池线程（幽灵线程）；
   且 run2 在 11:53:32（pb 超时结果写盘）后 **jsonl 停止写事件**，console 报告
   行拖到 20:01-20:05（本地，=12:01-12:05 UTC）才出——主 asyncio 事件循环已
   无法调度。bottom/max 提交于 11:43:27/28，其 600s 超时点（≈11:53:27）本应
   返回 EvalTimeout，但**任何结果都没写盘** → 它们不是"没算完"，而是主循环冻结
   后 wait_for 永远无法返回，run 于 ~20:05 本地被杀。

**"以前很少超时"部分属实但不完全**：历史上有超时（9/24 10 次、9/29 26 次、
10/05 14 次，含简单 RANK/$pb/TS_SINCE_N/CHIP 家族）；run1（GLM）0 超时是因为因子
简单（max dsl_ms 仅 10s）、无 @1w、无 8 并发激进批。本次是 **@1w 真慢（拖长之前
批次）+ 8 路并发 + 17 分钟内 60+ 次评估的内存/GC 压力 + 环境进程** 叠加，首次把
"秒级因子执行被放大数百倍"顶到 600s 预算墙。

---

## 二、根因分解

### 2.1 真慢类：@1w 周频因子单算子就要 ~130s（不算超时，但拖长批次）

- **实证**（`scripts/diag_eval_slow_factors2.py`，同面板复算）：
  - 单线程：vw_res_weekly_dev 147.6s、weekly_mom4 135.5s；
  - run2 中两个 @1w 因子 131.7s / 133.2s —— **与单线程一致，没有异常退化**。
  - ⚠️ 这两个因子提交于 11:35:27 与 11:40:37 的**前批**，不属于 batch9；
    它们与批内其他秒级因子并行（12 线程池），拖长的是各自的批次总时长。
- **关键澄清**（`scripts/diag_aux_cache_share.py`，会话级 aux_cache 共享验证）：
  - aux panel 构建 **仅 9.2s**（3151819×176→周频），第二次构建命中缓存 **0ms**；
  - 但两个 @1w 表达式 eval_factor 仍各 118–125s → **~120s 几乎全部花在算子本体**
    （TS_PCTCHANGE/SUBTRACT 等在周频面板上的执行），与 aux_cache 命中与否无关。
- 结论：@1w 因子就是 pandas 周频重采样 + 算子执行的真慢（单因子 ~2 分钟），
  **session aux_cache 修复（b92c7a9）本身有效**，但不能把 120s 算子成本消掉。
  历史 run 没有 @1w 因子，所以从未暴露这一档慢因子；本次模型批量使用 @1w 是新情况。

### 2.2 假超时类：秒级因子被放大 62–500x → 600s 预算到点被掐

铁证链（更新至实验3）：

1. **完整复算否定因子/内核/并发**（`scripts/diag_eval_batch9.py`，实验3，本次新增）：
   用与 run2 相同 panel（3151819×176）、相同 12 路 ThreadPoolExecutor、batch9 的
   **全部 8 个真实表达式**（含此前漏测的 bottom_fractal_dist、max_amount_dist20 两个
   TS_LAST_ARGBOTTOMFRACTAL / TS_ARGMAX 内核）复算：

   | 表达式（run2 11:43:25 批） | 干净进程 12 路并发 | run2 实际 | 放大 |
   |---|---|---|---|
   | vol_spike_cooldown | 6.47s | 417.7s（dsl 401.3s）✅ | ~62x |
   | top_fractal_dist | 11.37s | 601.0s EvalTimeout | ~53x |
   | pb_valuation_rev | 1.22s | 604.8s EvalTimeout | ~500x |
   | bottom_fractal_dist | 11.44s | 无任何结果（冻结） | — |
   | max_amount_dist20 | 10.32s | 无任何结果（冻结） | — |
   | div_yield_ttm / listed_days_old / float_lockup_ratio | 0.6–1.3s | 2.8–12.7s 记忆层拒绝 | 对照 |

   整批 8 个并发总耗时 **11s**。**纯多线程并发最坏只放大 5.0x**（top_fractal
   11374ms vs 单线程 2277ms），远不足以产生 400–600s。

2. **超时时刻 ≈ 恰好 600s**：top_fractal 601.0s、pb_valuation_rev 604.8s 全部 ≈
   `dispatch + 600s`，即 **`asyncio.wait_for(..., timeout=600)` 预算到点被掐**，
   不是"算完了但慢"。该预算**从 run_in_executor 提交时刻起算，包含线程池排队**
   （agentscope_tools.py:545-572）。

3. **信号量未到瓶颈（修正）**：batch9 共 8 个调用，其中 3 个在记忆层被
   MemoryAdvisoryBlock 秒拒（未进入评估），**真算只有 5 个 < `MAX_PARALLEL_EVAL=6`
   默认值**（env_settings.py:11）→ 信号量没有排队。（`_eval_semaphore` 拥挤
   只在单轮真算 >6 时才会发生，本次不是该路径。）

4. **vol_spike dsl_ms=401.3s 是关键实锤**：engine.py 的 dsl 计时只包 eval_factor
   本身（不含排队），401s 意味着**该因子在 run2 进程内真实执行被放大 62x（vs 并发
   实验）**。8 个表达式干净并发实验都 ≤11.4s 却出现 401s → 唯一剩余解释是
   **进程内环境压力**。

5. **为什么那一刻放大**（未被完全定死的尾部）：run2 于 11:26:37 启动（jit_warmup），
   batch9 提交于 11:43:25 —— **仅 17 分钟**内已评估 ~60 个因子（8 批 × 8），每个
   都是 315 万行全量 evaluate（transforms/metrics 全量展开，中间结果数十 GB 级）；
   5 个真算同秒并发展开峰值内存 + 当日 18:00（本地）启动的高 CPU 进程
   （PID 15704/49104，身份未确认）叠加 → 内存/swap/GC 压力把每个算子真实执行
   放大 60–500x。**该时刻的具体资源状态无法从日志回溯**（无内存快照、幽灵进程
   已退出），但因子自身 100% 排除——这是"假超时"最硬的部分。

### 2.3 机制缺陷：超时线程不可杀 + 主循环冻结 → 幽灵线程 → run 冻结

- `asyncio.wait_for` 超时**只取消 future，不杀已提交的线程池线程**；被掐的 eval
  实际还在后台跑（CPU/内存仍占用），成为幽灵线程。
- **新确认的更强问题——主循环冻结**：jsonl 在 11:53:32（pb 超时结果写入）之后
  **停止写任何事件**（11:53:31 research_memory_updated → 11:53:32 tool_results 1
  results，之后直到被杀共 12 分钟无新行）；而 console 的 report_reproduce_judge
  行出现在 20:00:49 / 20:01:07 / 20:05:03（本地，=12:01-12:05 UTC）。bottom/max
  提交于 11:43:27/28，600s 超时点 ≈ 11:53:27/28，与 top（11:53:27）几乎同时，
  但**它们的 EvalTimeout 结果从未写盘** → 主 asyncio 事件循环在 11:53:32 前后已
  无法调度剩余的 wait_for 返回（GC/swap 风暴饿死主线程或 AgentScope 内部协调卡死），
  两个未来得及返回的结果被吞掉。run 于 ~20:05 本地被杀（与先前测量 run 一致）。

### 2.4 本次 vs 历史

| run | 超时数 | max dsl_ms | 因子画像 |
|---|---|---|---|
| 9/24 | 10 | 无记录 | 含简单 RANK/$pb/TS_SINCE_N/CHIP 家族 |
| 9/29 | 26 | 无记录 | 同上，最密 |
| 10/05 baseline | 14 | 无记录 | vwap_wma 家族 |
| run1（10/06 GLM） | 0 | 9993 | 简单因子，无 @1w |
| run2（10/06 V4-Flash） | 2（+2 结果丢失） | 401300 | 首次 @1w 周频 + 8 并发批 |

历史超时**无法验证真/假**：10/06 之前没有 dsl_ms 日志（引擎计时是 10/06 才加的），
所以"以前也超时"与"以前是真超时"都不能确认。本次因为 dsl_ms 首次可用，
才能拆出"真实执行 401s（环境放大）"与"排队/预算被掐 604.8s"两类。

---

## 三、修复建议（按优先级，均需用户确认后实施）

1. **超时预算剔除排队**（最高价值）：`_dispatch_with_timeout` 改成
   "提交后先确认已开始执行，再从执行起点计时"，或把超时判断移进执行内部按
   cost-time 计，日志打印 queue_ms / wait_ms 区分排队与计算 → 假超时直接消失。
2. **预算墙与执行分离**：600s 预算应只覆盖"真正执行"，且执行段本身要有可杀
   机制（如进程级 watchdog 或 numba 编译预热），避免"秒级因子被环境放大"直接
   变成 EvalTimeout。
3. **幽灵线程治理 + 主循环保护**：超时后要么放弃（杀进程/重建 executor），要么
   至少记录 ghost 线程占用；评估执行放进子进程或加内存阈值，避免 swap 风暴饿死
   主事件循环导致结果丢失（bottom/max 这类"无结果"比 EvalTimeout 更危险）。
4. **@1w 算子真慢治本**：周频重采样算子 ~130s 是 pandas 瓶颈，可调研
   向量化/numba 化（真慢，不解决则每轮 @1w 因子固定占 2 分钟）。
5. **run 前环境检查 + 峰值限流**：run 前查内存占用与高 CPU 进程（本次 18:00 起的
   PID 15704/49104 等）；单轮真算并发调低（<6）或按内存预算分桶，避免 5 个
   315 万行全量 evaluate 同秒峰值展开。
6. **记录资源状态**：评估前后打点 RSS/swap（当前 metrics_snapshot 无内存字段），
   下次出问题可直接回溯"那一刻内存多少"，不必再靠推理。

## 四、附带说明

- **②④ 内存修复不受影响**：本报告与 portfolio 缓存/盲测延后链路无关；
  run2 假超时批次是提交路径的评估入口问题，不涉及 submit 段。
- **诊断脚本**（本分支新增）：`scripts/diag_eval_slow_factors.py`、
  `scripts/diag_eval_slow_factors2.py`、`scripts/diag_eval_batch9.py`（实验3：
   batch9 全 8 表达式完整复算）、`scripts/diag_aux_cache_share.py`，
  均可用同一面板复算，保留复现能力。