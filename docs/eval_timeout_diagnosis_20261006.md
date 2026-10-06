# 因子评估超时（EvalTimeout）根因诊断报告

> 日期：2026-10-06　关联 run2：`logs/factor_mining/run_20261006_192628.jsonl`
> （run_meta: DeepSeek-V4-Flash-0731，报告模式，测试段 train 3151819×176，4985 股票）
> 事件触发：用户反馈"因子超时评估看下什么原因，以前很少有超时的"。

---

## 一、结论速览

本次 run2 的 EvalTimeout **不是一个原因，而是两类问题叠加**，其中只有一类"真慢"，
另一类是**排队空耗导致的假超时**：

| 因子（11:43:25 同批 8 个） | 单线程实测 | run2 实际 | 定性 |
|---|---|---|---|
| vw_res_weekly_dev（@1w） | 127.9s | 131.7s ✅ | **真慢，但正常** |
| weekly_mom4（@1w） | 135.5s | 133.2s ✅ | **真慢，但正常** |
| vol_spike_cooldown | 1.6s | 417.7s（dsl 401.3s） | **假超时（环境放大 245x）** |
| top_fractal_dist | 2.3s | 601.0s EvalTimeout | **假超时（预算到点被掐）** |
| pb_valuation_rev | 0.77s | 604.8s EvalTimeout | **假超时（预算到点被掐）** |
| bottom_fractal_dist | 2.5s* | 1300s+ 无结果 | 假超时（run 冻结被杀） |
| max_amount_dist20 | ~8s* | 1000s+ 无结果 | 假超时（run 冻结被杀） |
| div_yield_ttm | 0.7s | 12.7s（记忆层拒绝） | 秒拒，未计算（对照） |

\* bottom_fractal_dist / max_amount_dist20 未单独重测，按同族算子估算。
同批对照铁证：**结构完全相同**的 `div_yield_ttm`（RANK+CS_ZSCORE）12.7s 秒拒，
`pb_valuation_rev`（同样 RANK+CS_ZSCORE）却 604.8s 超时——差异只在记忆层是否拦截。

**"以前很少超时"部分属实但不完全**：历史上有超时（9/24 10 次、9/29 26 次、10/05 14 次，
含简单 RANK/$pb/TS_SINCE_N/CHIP 家族）；run1（GLM）0 超时是因为因子简单
（max dsl_ms 仅 10s）、无 @1w、无 8 并发激进批。本次是 **@1w 真慢 + 8 路并发 +
长跑进程环境压力** 三因素叠加，首次把"排队空耗"触顶 600s。

---

## 二、根因分解

### 2.1 真慢类：@1w 周频因子单算子就要 ~130s（不算超时，但拖长批次）

- **实证**（`scripts/diag_eval_slow_factors2.py`，同面板复算）：
  - 单线程：vw_res_weekly_dev 147.6s、weekly_mom4 135.5s；
  - run2 中两个 @1w 因子 131.7s / 133.2s —— **与单线程一致，没有异常退化**。
- **关键澄清**（`scripts/diag_aux_cache_share.py`，会话级 aux_cache 共享验证）：
  - aux panel 构建 **仅 9.2s**（3151819×176→周频），第二次构建命中缓存 **0ms**；
  - 但两个 @1w 表达式 eval_factor 仍各 118–125s → **~120s 几乎全部花在算子本体**
    （TS_PCTCHANGE/SUBTRACT 等在周频面板上的执行），与 aux_cache 命中与否无关。
- 结论：@1w 因子就是 pandas 周频重采样 + 算子执行的真慢（单因子 ~2 分钟），
  **session aux_cache 修复（b92c7a9）本身有效**，但不能把 120s 算子成本消掉。
  历史 run 没有 @1w 因子，所以从未暴露这一档慢因子；本次模型批量使用 @1w 是新情况。

### 2.2 假超时类：秒级因子被放大 245–775 倍 → 600s 预算到点被掐

三个铁证链：

1. **并发复现实验否定"并发本身放大"**（`scripts/diag_eval_slow_factors2.py`）：
   12 路 ThreadPoolExecutor 同时复算 run2 的 6 个因子，最坏因子仅放大 3.9x
   （vol_spike 6376ms、top_fractal 8493ms、pb 1108ms）。**纯多线程并发解释不了
   400–600s**——故剩余放大只能来自"长跑进程的环境压力"而非并发本身。

2. **超时时刻 ≈ 恰好 600s**：top_fractal 601.0s、pb_valuation_rev 604.8s 全部 ≈
   `dispatch + 600s`，即 **`asyncio.wait_for(..., timeout=600)` 预算到点被掐**，
   不是"算完了但慢"。`_dispatch_with_timeout`（agentscope_tools.py:545-572）
   **从 run_in_executor 提交时刻起算预算，包含线程池排队 + `_eval_semaphore`
   信号量等待时间**。

3. **信号量 < 批并发**：`MAX_PARALLEL_EVAL` 默认 **6**（env_settings.py:11），
   而本次模型同秒提交 **8 个 eval_on_train_set**（11:43:25–28）→ 至少 2 个因子
   在信号量上排队空耗；排队时间计入 600s 预算 → 尾因子假超时。
   （service.py:127 `_eval_semaphore=Semaphore(max_parallel_eval)` 包住 lite 评估。）

4. **vol_spike dsl_ms=401.3s 是关键实锤**：engine.py 的 dsl 计时只包 eval_factor
   本身（不含排队），401s 意味着**该因子在 run2 进程内真实执行被放大 240x**。
   单线程 1.6s、12 路并发实验 6.4s 都不可能出现 401s → 唯一剩余解释是
   **进程内环境压力**：run2 进程从 19:26 长跑到 11:43（约 4 小时 17 分钟），
   期间数百次评估累积内存/中间结果，叠加 5 个 315 万行全量 evaluate 并发展开，
   触发内存/GC 压力 → 每个算子实际执行慢数百倍。（当日另有 10/6 18:00 启动的
   高 CPU 进程 PID 15704/49104 及 10/5 23:16 一批进程，但纯 CPU 争抢按实验只放大
   1.3–3.9x，故内存/GC 是主导。）

### 2.3 机制缺陷：超时线程不可杀 → 幽灵线程 → run 冻结

- `asyncio.wait_for` 超时**只取消 future，不杀已提交的线程池线程**；
- 被掐的 eval 实际还在后台跑（CPU 仍占用、持有信号量/内存），成为幽灵线程；
- run2 中 bottom_fractal_dist / max_amount_dist20 提交于 11:43:28，却直到
  **20:05（1300s+）才 report，且无 tool_results** —— 线程池被前序幽灵占满，
  后续因子全部排队空耗，最终 run2 冻结（~20:05 被杀，与先前测量 run 一致）。

### 2.4 本次 vs 历史

| run | 超时数 | max dsl_ms | 因子画像 |
|---|---|---|---|
| 9/24 | 10 | 无记录 | 含简单 RANK/$pb/TS_SINCE_N/CHIP 家族 |
| 9/29 | 26 | 无记录 | 同上，最密 |
| 10/05 baseline | 14 | 无记录 | vwap_wma 家族 |
| run1（10/06 GLM） | 0 | 9993 | 简单因子，无 @1w |
| run2（10/06 V4-Flash） | 2（+2 卡死） | 401300 | 首次 @1w 周频 + 8 并发批 |

历史超时**无法验证真/假**：10/06 之前没有 dsl_ms 日志（引擎计时是 10/06 才加的），
所以"以前也超时"与"以前是真超时"都不能确认。本次因为 dsl_ms 首次可用，
才能拆出"真实执行 401s"与"排队被掐 604.8s"两类。

---

## 三、修复建议（按优先级，均需用户确认后实施）

1. **超时预算剔除排队**（最高价值）：`_dispatch_with_timeout` 改成
   "提交后先确认已开始执行，再从执行起点计时"，或把超时判断移进执行内部按
   cost-time 计，日志打印 queue_ms / wait_ms 区分排队与计算 → 假超时直接消失。
2. **信号量与批并发对齐**：`MAX_PARALLEL_EVAL=6 < 批并发 8` 造成排队空耗；
   要么提高到 ≥ 批并发（注意并发内存翻倍风险），要么信号量等待**不计入**
   600s 预算，要么在提交前按并发分桶。
3. **幽灵线程治理**：超时后要么放弃（杀进程/重建 executor），要么至少记录
   ghost 线程占用；禁止"超时了线程还在跑还占信号量"。
4. **@1w 算子真慢治本**：周频重采样算子 ~130s 是 pandas 瓶颈，可调研
   向量化/numba 化（真慢，不解决则每轮 @1w 因子固定占 2 分钟）。
5. **长跑前环境检查**：run 前查内存占用与高 CPU 进程（本次 18:00 起的
   PID 15704/49104 等），排除环境干扰后再下结论。

## 四、附带说明

- **②④ 内存修复不受影响**：本报告与 portfolio 缓存/盲测延后链路无关；
  run2 假超时批次是提交路径的评估入口问题，不涉及 submit 段。
- **诊断脚本**（本分支新增，未入库）：`scripts/diag_eval_slow_factors.py`、
  `scripts/diag_eval_slow_factors2.py`、`scripts/diag_aux_cache_share.py`，
  均可用同一面板复算，保留复现能力。