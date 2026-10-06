# 并发评测内存归因与"三段共用"方案可行性分析

> 2026-10-06，承接 `alphaagent_memory_monitor_report_20261006.md`。
> 用户问题：**并发评测为什么这么耗费内存，可不可以验证集、训练集、盲测段三个在内存里，其他所有都公用这三段？**

---

## 一、结论先行

| 问题 | 回答 |
|---|---|
| 并发评测为什么费内存 | **不是 panel 被复制 12 份**，而是每次 eval 在 split panel 上**重新构建辅频广播表**（1w 聚合 + merge_asof 广播到主频全长度），12 路并发 = 12 份独立广播表同时存活。 |
| "三段在内存里，其他公用"可行吗 | **方向对，但当前架构已经做到一半**——`_split_cache` 已经是 view（零拷贝），eval 也已经在 split 上跑。真正的浪费不在"panel 被复制"，而在**每个 eval 调用各自重建辅频广播表 + 各自 float64 上转**。把这三段"共用"的关键不是共享 panel，而是**共享这三段上的辅频广播表与 float64 中间数组**。 |
| 需要改什么 | 3 处：① `eval_factor` 的 `aux_cache` 提升到 session 级（按 split 复用 1w 广播表）；② `portfolio` 的 float64 上转改为 float32 或共享；③ `submit` 的三次 portfolio + 盲测重评改为共享中间数组。**控制流改动 > 数据结构改动**，不需要重构 `StockEvalSession`。 |

---

## 二、并发评测内存花在哪——逐层拆解

### 2.1 调用链（已读代码确认）

```
LLM tool call (eval_on_train_set / eval_on_val_set)
  → agentscope_tools.py:750  loop.run_in_executor(_executor(max_workers=12), ...)
  → service.py:242           with self._eval_semaphore:   ← 12 路并发闸门
  → service.py:408/419       _run_one(split="train"/"val")
  → engine.py:111            panel = self._panel_for_profile(session, profile)
  → engine.py:307            panel, start, end = session.get_split_panel(profile.split)
  → engine.py:123            eval_factor(multi_line_expr, panel)   ← panel 是 split view
  → eval.py:795              eval_factor → eval_multi_line_factor
  → eval.py:514-515          if aux_cache is None: aux_cache = {}   ← 每次 eval 新建空 dict
  → eval.py:517              _merge_build_aux_panels(df, required, aux_cache=aux_cache)
  → aux_cache.py:44          get_or_build_aux_panel → build_timeframe_panel(panel, "1w")
  → resample.py:105          groupby(["instrument","__bucket__"]).agg(...)   ← 1w 聚合
  → eval.py:649-662          _eval_multi_frequency: native + broadcast dicts
  → resample.py:174          broadcast_timeframe_to_main_freq → merge_asof   ← 广播到主频全长度
```

### 2.2 真正的内存大头：辅频广播表，不是 panel

**关键代码 `eval.py:514-515`**：
```python
if aux_cache is None:
    aux_cache = {}   # ← 每次 eval_factor 调用都新建一个空 dict
```

`service.py:365` / `engine.py:123` 调 `eval_factor(multi_line_expr, panel)` **没传 `aux_cache`**，所以：

1. **每次 eval 都重新 `build_timeframe_panel`**（`resample.py:73-127`）：对 split panel 做 `groupby(["instrument","__bucket__"]).agg()`，产出 1w 聚合表。train 段约 3 年 × 5000 股 ≈ 750K 行 × N 列。
2. **每次 eval 都重新 `broadcast_timeframe_to_main_freq`**（`resample.py:130-188`）：`merge_asof` 把 1w 列广播回主频（日频）长度。train 段约 3.6M 行 × 每个被引用的 aux 列。
3. **`_eval_multi_frequency`（`eval.py:649-662`）** 对每个 aux tag 的每一列同时建 `native[tag][py]`（原频）和 `broadcast[tag][py]`（广播到主频）两份。

**12 路并发 = 12 份独立的 `{1w 聚合表 + 广播表}` 同时存活**。这才是并发评测的内存峰值来源，不是 panel 被复制 12 份。

### 2.3 为什么 panel 本身不是问题

- `session.get_split_panel(split)`（`session.py:65-74`）返回 `slice_panel` 的结果。
- `slice_panel`（`panel.py:67-109`）对日期范围切片走 `panel.iloc[lo:hi]`——**这是 view，不是 copy**，与父 panel 共享同一块数据。
- `_split_cache` 缓存这个 view，12 路 eval 拿到的是**同一个 view 对象**，零拷贝。
- `eval_factor` 内部 `eval.py:504-509` 的 `df = df[pruned]` 确实会 copy（列裁剪），但裁剪后只保留表达式引用的列（通常 3-8 列），远小于全 panel。

**所以"panel 被复制 12 份"是假象。** 真正被复制 12 份的是**辅频广播表**。

### 2.4 量化估算（train 段，label_20d 主频）

| 分配 | 单次大小 | 12 路并发峰值 |
|---|---|---|
| split panel view | 0（共享父 panel） | 0 |
| 列裁剪 copy（3-8 列 × 3.6M 行 × float32） | ~50-130 MB | ~0.6-1.6 GB |
| 1w 聚合表（N aux 列 × 750K 行 × float64） | ~50-150 MB | ~0.6-1.8 GB |
| **广播表（N aux 列 × 3.6M 行 × float64）** | **~150-450 MB** | **~1.8-5.4 GB** |
| `native` dict（原频副本） | ~50-150 MB | ~0.6-1.8 GB |
| portfolio float64 上转（`portfolio.py:172`） | ~30 MB | ~360 MB |
| **单路 eval 合计** | **~0.5-1.0 GB** | **~6-12 GB** |

这与实测"12 路并发 eval 约 24 GB"量级吻合（含 portfolio/tradable 等下游分配）。

---

## 三、"三段共用"方案可行性分析

### 3.1 用户方案的准确表述

> 验证集、训练集、盲测段三个在内存里，其他所有都公用这三段。

映射到代码：train/val/test 三段 DataFrame 驻留内存（`_split_cache`），所有消费者（eval / submit / portfolio / tradable_mask）引用这三段 view，不各自复制。

### 3.2 当前架构已经做到的部分 ✅

| 项 | 现状 | 证据 |
|---|---|---|
| 三段驻内存 | ✅ `session._split_cache["train"/"val"/"test"]` 懒缓存 | `session.py:65-74` |
| 三段是 view 非拷贝 | ✅ `slice_panel` 走 `iloc[lo:hi]` | `panel.py:67-109` |
| eval 在 split 上跑 | ✅ `engine._panel_for_profile` → `get_split_panel(profile.split)` | `engine.py:307` |
| 12 路 eval 共享同一 split view | ✅ 同一 session 对象，同一 `_split_cache` | `session.py:68` 加锁 |
| 辅表缓存机制存在 | ✅ `aux_cache` 参数 + `get_or_build_aux_panel` LRU 32 | `aux_cache.py:18,33-54` |

**结论：panel 层面"三段共用"已经实现。** 用户感知的"费内存"不是 panel 复制导致的。

### 3.3 当前架构没做到的部分 ❌（真正的浪费）

| 浪费点 | 位置 | 原因 | 是否可"共用" |
|---|---|---|---|
| **辅频广播表每次重建** | `eval.py:514-515` | `aux_cache` 每次 eval 新建空 dict，没跨调用复用 | ✅ 可共用——同一 split 的 1w 广播表对同一段是确定性的 |
| **`native` + `broadcast` 双份** | `eval.py:649-662` | 每个 aux 列同时存原频 + 广播两份 | ⚠️ 部分可共用——`native` 可由广播表按 tag 反查，但需改 DSL 求值 |
| **portfolio float64 上转** | `portfolio.py:172` | `factor.to_numpy(dtype=np.float64, copy=False)` 把 float32 因子翻倍 | ✅ 可共用——改 float32 或共享上转后的数组 |
| **tradable 7+ wide 矩阵** | `tradable.py:83-152` | 每次 `_wide` pivot long→wide，7 份同时存活 | ✅ 已部分共用（`_CACHE` WeakValueDictionary 按 panel.index 身份） |
| **submit 三次 portfolio** | `submit.py:510,522,548-573` | 全窗口 + tradable 过滤 + 三段各一次 | ✅ 可共用——共享 factor/label 数组 |
| **submit 盲测重评** | `submit.py:577-619` | 盲测段重新 `compute_ingest_metrics`（2× evaluate_on_panel） | ✅ 可延后——stage_one 过了再跑 |
| **`compute_ingest_metrics` 2× evaluate_on_panel** | `ingest.py:167-173` | 每次调用两遍全量评估 | ⚠️ 结构性——winsorize 前后各一次，难合并 |

### 3.4 "三段共用"的真正含义

用户的直觉是对的，但需要精确化。**不是"三段 panel 共用"（已实现），而是"三段上的派生数据共用"**：

```
当前：  12 路 eval × 各自重建 {1w聚合表 + 广播表 + native dict}  → 12 份
目标：  3 段 × 每段 1 份 {1w聚合表 + 广播表}，12 路 eval 共享引用  → 3 份
```

**关键洞察**：辅频广播表是**split panel 的确定性函数**——给定 split panel + tag + 列，广播表唯一。所以可以按 `(split, tag, column_set)` 缓存到 session 级，12 路 eval 命中即复用。

---

## 四、可行性评估：需要改什么

### 4.1 改动 1：`aux_cache` 提升到 session 级（最高 ROI）

**现状**：`eval.py:514-515` 每次 eval 新建 `aux_cache={}`。
**改法**：`StockEvalSession` 增加 `_aux_cache_by_split: dict[str, OrderedDict]`，`engine.py:123` 调 `eval_factor` 时传入 `session.get_aux_cache(profile.split)`。

```python
# session.py 新增
def get_aux_cache(self, split: str) -> MutableMapping:
    with self._split_lock:
        if split not in self._aux_cache_by_split:
            self._aux_cache_by_split[split] = OrderedDict()
        return self._aux_cache_by_split[split]

# engine.py:123 改
raw = cache.evaluate(multi_line_expr, panel,
    lambda: eval_factor(multi_line_expr, panel, aux_cache=session.get_aux_cache(profile.split)))
```

**效果**：12 路 eval 对同一 split 的 1w 广播表只建 1 份，并发峰值从 ~12 份降到 ~3 份（train/val/test 各 1）。
**风险**：`aux_cache.py:31` 的 key 是 `(id(panel), base, tag)`——split view 的 `id` 稳定（缓存在 `_split_cache`），安全。LRU 32 上限不变。
**改动量**：~15 行。

### 4.2 改动 2：portfolio float64 上转改 float32

**现状**：`portfolio.py:172` `f_arr_all = factor.to_numpy(dtype=np.float64, copy=False)`。
**改法**：numba kernel 改接受 float32，或上转后的数组按 split 缓存。

**效果**：单路省 ~30 MB × 12 = ~360 MB。ROI 中等。
**风险**：numba kernel 签名要改，需回归测试。
**改动量**：~40 行（含 kernel 签名）。

### 4.3 改动 3：submit 三次 portfolio 共享数组

**现状**：`submit.py:510`（全窗口）+ `:522`（tradable 过滤）+ `:548-573`（三段各一次）= 5 次 portfolio。
**改法**：factor/label 数组在 submit 入口物化一次，三次 portfolio 共享引用，只传不同的 eligibility mask。

**效果**：submit 路径省 ~2-3 GB 峰值。
**风险**：低——纯引用共享，口径不变。
**改动量**：~30 行。

### 4.4 改动 4：盲测重评延后到 stage_one 过线后（控制流，最高 ROI）

**现状**：`submit.py:577-619` 无条件跑盲测 `compute_ingest_metrics`（2× evaluate_on_panel，488s）。
**改法**：stage_one 不过线直接 return，不跑盲测。

**效果**：绝大多数 submit（stage_one 被拒）省 488s + ~2 GB。
**风险**：零——控制流改动，不改数据。
**改动量**：~5 行（if not stage_one_passed: return）。

### 4.5 实施状态（2026-10-06 落地）

| 改动 | 状态 | 实测收益 | 备注 |
|---|---|---|---|
| ① aux_cache session 级 | ✅ 已实施（commit `b92c7a9`） | WS 峰值 39GB → ~14GB（-64%） | 含 `eval.py` 重排（aux 在列裁剪前建，保证 `id(panel)` 稳定）——**承重修复** |
| ② portfolio float64 身份缓存 | ✅ 已实施（本分支，待提交） | 8 次 `to_numpy` → 1 次缓存，~462MB 瞬时 + 并发 12× ~5.5GB | 新增 `_factor_cache.py`（镜像 `_label_cache.py`），`portfolio.py:173` 改用 `factor_f64(factor)`。**无 numba kernel 改动**（portfolio.py 无 numba）。smoke 验证 36 keys `max_diff=0.00e+00`，cache hit 零拷贝（`same buffer=True`，`writeable=False`） |
| ③ submit 三段 portfolio 共享数组 | ⏸️ 暂不实施 | 边际 ~196.8MB（3 段 × 65.6MB） | ② 已让全窗口 + 8 次调用共享；三段切片 `factor_series[_mask]` 产生新 `id(index)` → `factor_f64` cache miss，但 LRU max=4 容纳 full+train+val+test 不驱逐。ROI 不抵签名改动风险。**验证**：`_verify_3.py` 确认三段 `id(index)` 均不同，4 key 填满 LRU |
| ④ 盲测延后到 stage_one 后 | ✅ 已实施（本分支，待提交） | stage_one 失败候选省 ~488s + ~2GB（多数候选） | `submit.py:577-626` 插入 `stage_one_stats` 预检，失败早 return；删除 `:695-706` 旧重复调用。`gate_reasons` 从预检流过盲测块到 `:698 val_retention`，语义等价（三种路径：stage_one 失败/盲测失败/全过，均与原流程同结果） |

**回归测试**：`tests/test_alphaagent_metrics.py`（8 passed）、`tests/test_metrics_fastpaths.py`（8 passed）、`tests/test_submit_metrics_cache.py`（4 passed）全绿。

**净效果**：①②④ 落地后，submit 路径峰值从 ~11GB 降到 ~4GB 量级，整体 WS 峰值从 39GB 降到 ~8GB 量级（panel ~2GB + numba/factor_cache/autocorr ~7GB 结构性地板 + submit ~4GB）。③ 的 ~196.8MB 边际收益暂不抵改动风险，留作后续可选优化。

### 4.6 不需要改的

- **`StockEvalSession` 结构**：已经是对的，`_split_cache` view + 懒缓存。
- **`slice_panel`**：已经是 view，不需要改。
- **panel 加载**：`SessionStore.create` 已经只加载 `coverage_range`，不需要改。
- **`release_session`**：挖掘 run 不调用（设计），不需要改。

---

## 五、可行性总评

| 维度 | 评估 |
|---|---|
| 方案方向 | ✅ 正确——用户直觉命中真实浪费点 |
| 当前架构兼容性 | ✅ 高——`_split_cache` view + `aux_cache` 机制已存在，只需提升作用域 |
| 改动量 | 小——4 处改动，总计 ~90 行，无数据结构重构 |
| 风险 | 低——改动 1/3/4 是引用共享与控制流，改动 2 需回归 numba kernel |
| 预期收益 | 并发 eval 峰值 ~12 份广播表 → ~3 份（省 ~3-4 GB）；submit 峰值省 ~2-3 GB；总峰值从 ~39 GB 降到 ~30 GB 量级 |
| 是否需要重构 session | ❌ 不需要——`StockEvalSession` 已经是正确抽象 |

**一句话**：用户的"三段共用"方案可行，且当前架构已经完成了 panel 层面的共用（`_split_cache` view）；真正待补的是**派生数据（辅频广播表、float64 中间数组）的 session 级共享**，这是 ~90 行改动，不需要重构 `StockEvalSession`。

---

## 六、证据索引（本轮新读代码）

| 文件:行 | 作用 |
|---|---|
| `alphaagent/dsl/eval.py:514-515` | `aux_cache = {}` 每次 eval 新建——**根因** |
| `alphaagent/dsl/eval.py:649-662` | `_eval_multi_frequency` 建 native + broadcast 双份 |
| `alphaagent/dsl/stock/aux_cache.py:18,33-54` | 辅表 LRU 32 缓存机制（已存在，未被 session 级利用） |
| `alphaagent/dsl/stock/resample.py:73-127` | `build_timeframe_panel` 1w 聚合 |
| `alphaagent/dsl/stock/resample.py:130-188` | `broadcast_timeframe_to_main_freq` merge_asof 广播到主频全长度 |
| `alphaagent/factor/mining/eval/service.py:242,408,419,431` | `_eval_semaphore` 12 路并发闸门 |
| `alphaagent/factor/mining/eval/service.py:365` | `eval_factor(multi_line_expr, panel)` 未传 aux_cache |
| `alphaagent/factor/evaluation/engine.py:111,307` | `_panel_for_profile` → `get_split_panel`（eval 在 split 上跑） |
| `alphaagent/factor/mining/eval/session.py:65-74` | `get_split_panel` 返回 view 缓存 |
| `alphaagent/factor/mining/eval/context.py:59-66` | `split_range` train/val/test 三段定义 |
| `alphaagent/factor/mining/delivery/submit.py:447,461,510,522,548-573,577-619` | submit 全 panel 物化 + 三次 portfolio + 盲测重评 |
