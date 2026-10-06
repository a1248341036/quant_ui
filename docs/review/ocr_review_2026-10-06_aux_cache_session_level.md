# OCR review：`feat/aux-cache-session-level`（2026-10-06，concurrent eval memory 优化）

## 背景与范围

对当前分支 `feat/aux-cache-session-level`（相对 `main`）做 `ocr review`（open-code-review，
skill `open-code-review`），只审代码、排除 docs。

| 项 | 值 |
|---|---|
| 命令 | `ocr review --audience agent --from main --to HEAD --background-file <ctx> --exclude "docs/**" --output <tmp>` |
| 范围 | 7 个代码文件（+185/−26）：`dsl/eval.py`、`factor/metrics/_factor_cache.py`(新)、`factor/metrics/portfolio.py`、`factor/evaluation/engine.py`、`factor/mining/eval/{session,service}.py`、`factor/mining/delivery/submit.py` |
| OCR 结果 | 7 findings / 7 文件（含 2 次 provider HTTP 500 重试后成功），13m20s，session `03711016-9c0b-42b0-a5f6-9cc5fe885791` |
| 原始产物（不入库） | `%TEMP%\ocr_aux_cache_out.txt`、评审背景 `%TEMP%\ocr_bg_aux_cache.md` |
| 复核方式 | OCR 每条结论回源核对（grep + read），另加 3 个独立复现脚本；分支相关 6 个测试文件 61 passed |

被审分支的两个提交：

1. `b92c7a9` aux_cache 会话级共享（12 路并发 eval 共享同一 split 的 1w 辅频聚合表）
2. `1e5fef1` portfolio float64 身份缓存 + 盲测延后到 stage_one 之后

## 结论

**不具备合并条件。** 1 个已实测复现的 **critical 正确性缺陷**（同一 session 内跨因子指标串台，
直接污染 stage_one 换手硬门），另有 **3 个 high**（stage_one 观测口径静默失真、因子落账数据形态变化、
辅频表全列聚合反向开销）。**本次只评审、未改任何代码**（全部 findings 状态 = 未修）。

| 级别 | 条数 |
|---|---|
| Critical | 1 |
| High | 3 |
| Medium | 4 |
| Low | 3 |

---

## Critical

### C-1 `factor_f64` 缓存键不含数据身份 → 同 session 内不同因子串台（实测复现）

**位置**：`alphaagent/factor/metrics/_factor_cache.py:41`（键）→ `alphaagent/factor/metrics/portfolio.py:173`（唯一使用点）

```python
key = (id(index), str(factor.name))   # 只标识「索引对象身份 + 列名」，不标识因子数值
```

**触发链（submit 路径）**：

- `submit.py:475` `metric_series = pd.Series(metric_values, index=panel.index, dtype=np.float32)` —— **不带 name**，`str(None) == "None"`；
- `panel = session.panel`（`submit.py:447`）在**整个 session 内是同一对象**，`panel.index` 因此是同一对象且被 session 强持有；
- 于是同一 session 内**每个候选的键恒为 `(id(panel.index), "None")`**，弱引用校验 `ref() is index` 也必然通过（就是同一个索引对象）。

**实测证据**（`%TEMP%\repro_qp_crosstalk.py`，直接走 `quantile_portfolio_metrics`）：

```
A 全窗口平均单边换手 1.75
B（走缓存，第二个因子）1.75      ← 等于 A，被污染
B（_factor_f64_override 绕过缓存）1.7917  ← B 真值
```

裸 `factor_f64` 层面的同型复现（`%TEMP%\repro_factor_cache.py`）：两个 `name=None`、共享同一 index 对象、
数值不同的 Series，第二次调用返回**第一个**的数组（`[1 2 3 4 5 6]` 而非 `[9 8 7 6 5 4]`）。

**影响**：

- 第 2 个候选起的**全窗口**组合指标（换手 / 年化 / 深度曲线 / 可成交域透镜）全部用上一个因子的值计算；
- `metrics_train["quantile_portfolio"]["avg_daily_side_turnover"]` 正是 `submit.py:587` stage_one 换手硬门的输入
  → **因子 B 可能按因子 A 的换手率被误判通过/拒绝**，且落库/落账指标造假（静默，无日志异常）；
- eval 挖掘路径同因：`engine.py:144` `factor = pd.Series(values, index=panel.index, name=factor_name, ...)`，
  并发/连续评测时 `factor_name` 多为默认 `"expr"` 或 LLM 复用同名 → 同一 split 面板的 `quantile_portfolio` 同样命中旧数组。

**建议修法**：缓存键改用 Series **对象身份** `id(factor)` + `weakref.ref(factor)` 校验（照抄同文件/`_label_cache.py`
既有的弱引用模式；**不能只用裸 `id`** —— `metric_series` 每次 submit 后即释放，裸 id 会 ABA）。
命中率不受影响：同一次 submit 的 `submit.py:510/:529` 与同一次 eval 的多插件调用传的是**同一 Series 对象**；
不同表达式/不同候选天然是新对象，自动 miss。备选：由调用方传入表达式/数值指纹作为键。

---

## High

### H-1 提前拒绝路径打掉了 stage_one 观测口径（bench 台账静默失真）

**位置**：`alphaagent/factor/mining/delivery/submit.py:601-626`（新增提前 return） vs `submit.py:788-795`（`submit.stage_one` 日志）

`submit.stage_one |` 这行日志只在 `submit.py:788` 打印，而它位于提前 return **之后**；gate_reasons 非空时直接
return，该行永不出现。而 `alphaagent/factor/mining/run_metrics.py:61-70` 正是按这一行统计：

- `stage_one_fail` 计数；
- stage_one 失败原因直方图（`stage_one_fails`，从 `fail=[...]` 解析）。

→ 提前拒绝的因子**不再计入 stage_one_fail**，该指标会静默掉到 ~0，失败原因分布清空；bench diff 会把
「口径丢失」读成「改善」。当前 `submit.py:782-795` 走到时 `gate_reasons` 必为空，`passed` 恒 True（correlation 门除外）。

**建议**：提前拒绝分支补打一条兼容的 `submit.stage_one | ... passed=False | fail=[...]`（或改为按
`submit.stage_one_stats` 统计并同步改 run_metrics）。

### H-2 `error_type` 分桶分叉

新路径 `submit.py:623` 写 `"StageOneError"`，既有 stage_one 失败路径 `submit.py:881` 写
`"StageOneDeliveryCheckError"`；`run_metrics.py:230 _bucket_error` 直接透传原始字符串 →
同一类失败在 bench 错误直方图里被拆成两个桶，历史对比断档。

### H-3 仅 stage_one 失败的因子失去原有落账数据（payload 形态变化）

提前拒绝 payload 的 `metrics` 是 `dict(metrics_train)`（train 窗口原始 dict，`submit.py:613`），
而旧路径在 `submit.py:798-856` 生成的是**扁平化 `reported`**：全窗口 ic/icir + `train_*` + `val_*` +
`val_ic_retention` + `val_long_excess` + `test_*`（且 `test_holdout` 为真实盲测报告）。

叠加 `test_holdout: None`（`submit.py:614`）后，`alphaagent/factor/mining/memory/ingestion.py:176-184`
的 `test_ic / test_icir / test_rank_ic / test_ic_retention` setdefault 取到 `None`
→ **这类因子的盲测证据在研究记忆里整体消失**（旧流程盲测先跑，是有的）；`val_ic` 等 val 侧证据同样丢失。
另外 `delivery_check.blind_test = {"passed": False, "skipped": True}` 会把「没跑」计入任何按 `passed`
聚合的消费方（如 `scripts/resubmit_promising_factors.py:244` 的盲测打印）。

> 说明：REJECT/ACCEPT 的**最终判定**确实等价（stage_one 不过，两条顺序都拒），但**可观测产物不等价**。
> `submit.py:603` 注释「payload 结构对齐盲测拒绝 (:620-642)」对不上旧 stage_one 拒绝路径（:879 起）。

---

## Medium

### M-1 辅频表改为「全列」聚合，注释与实现不符（反向开销）

**位置**：`alphaagent/dsl/eval.py:502-520`（构建搬到列裁剪之前）

搬移后 `get_or_build_aux_panel` 收到的是**未裁剪**的完整 panel，`build_timeframe_panel`（未传 `columns`）
会聚合**全部数值列**；而 `alphaagent/dsl/stock/resample.py:35-43` 的 `_select_numeric_columns` 在无 `include`
时就是 `list(panel.columns)`。旧逻辑在裁剪后的 `df[pruned]` 上聚合，aux 只含表达式引用列。

代价有两处：① 会话缓存的 1w 表体积从「引用列」变成「全列」；② `eval.py:657-667` 的 `_eval_multi_frequency`
会对 aux 的**每一列**执行一次 `merge_asof` 广播（`broadcast_timeframe_to_main_freq`），列数膨胀后每次 eval
的 CPU 也膨胀，12 路并发逐路重复 —— 与「内存优化」的目标部分抵消。`eval.py:502-504` 注释「aux panel 只用
表达式 required 的列，裁剪不影响结果」与实现不符。

**建议**：缓存键保持 `id(原始 panel)` 稳定，但把 `build_timeframe_panel` 的 `columns` 收窄为表达式引用列
（给 `get_or_build_aux_panel` 增加 `columns` 透传）；若为跨表达式复用必须全列，则至少量化 aux 表体积与
`_eval_multi_frequency` 广播耗时，写进本分支的收益账。

### M-2 `full` split 的 aux 缓存必然 miss，且最多滞留 32 份全量表

**位置**：`alphaagent/factor/evaluation/engine.py:315-320`（`_panel_for_profile`）+ `engine.py:124-130`（注入）

`split != "full"` 走 `session.get_split_panel()`（`_split_cache` 缓存 → id 稳定，命中）；
但 `split == "full"` 每次都 `slice_panel(session.panel, ...)` 造**全新 DataFrame** → `id(panel)` 每次不同 →
full split（production_delivery / engine_preview，`eval/service.py:371` 显式用 `"full"`）**永远 miss**，
每次评估仍各自重建 1w 聚合表，且 miss 条目按 `(id, tag)` 累积进该 split 的 LRU（`aux_cache.py:18` 上限 32）。

**建议**：full 切片也纳入 `_split_cache` 保证 id 稳定，或缓存键改用 split 名而非 `id(panel)`。

### M-3 session 级 aux 缓存存在 check-then-act 竞态（目标场景在首轮仍会复现）

**位置**：`alphaagent/factor/mining/eval/session.py:97-102`（`get_aux_cache`） + `alphaagent/dsl/stock/aux_cache.py:33-54`

`_aux_cache_lock` 只保护 `OrderedDict` 的**创建与取用**，返回后 12 路并发线程对**同一个** OrderedDict
执行 `key in cache` / `cache[key] = built` / `move_to_end` / `popitem`，全在锁外：首轮并发会**全部 miss 并各自重建**
同一 tag 的 1w 广播表（正是本变更要消除的内存叠加）；插入/淘汰交错时 `len(cache) > 32` 的 `popitem`
也可能多弹/短暂越界（`OrderedDict.popitem` 在空表上抛 `KeyError`，当前规模下概率低但非不可发生）。

**建议**：把「查/建/插/淘汰」整体纳入该 split 的锁内（或提供带锁的包装 cache），并加一条 12 线程同 split
的并发冒烟断言「只构建 1 次」。

### M-4 `aux_cache.py` 的「id 复用防御」是恒真式（本分支把短命缓存变长命缓存，风险升级）

**位置**：`alphaagent/dsl/stock/aux_cache.py:33-42`

键为 `(id(panel), norm_base, norm_tag)`，而校验比的是 `cached.attrs["_aux_panel_id"] == id(panel)` ——
**与键里的 id 是同一个值，比较必然成立**，对 id 复用零防护（注释声称「校验索引身份」）。
该文件原本是 per-call 短命 cache 所以无所谓；本分支把它接到会话级长命缓存后，一旦「旧 panel 被 GC + 新 panel
复用同一 id」，就会**静默返回为另一块 panel 构建的辅表**。

**建议**：照 `_factor_cache.py` / `_label_cache.py` 用 `weakref.ref(panel)` 校验对象身份（对比 `ref() is panel`），
并删掉 `attrs` 里那个恒真的标记。

> 当前实际暴露面有限：split 面板由 `_split_cache` 强持有、`SessionStore.release`（`session.py:178-179`）
> 已同时清 `_aux_cache_by_split`；但 full split 每次重建切片使 id 复用成为常态，且注释的防护是假的，
> 属「迟早踩」的结构性隐患。

---

## Low

### L-1 `_AUX_CACHE_PER_SPLIT_MAX` 死常量

`alphaagent/factor/mining/eval/session.py:23` 定义后全仓无引用（grep 确认）；真实上限由
`alphaagent/dsl/stock/aux_cache.py:18` 的 `_MAX_CACHE_ENTRIES = 32` 施加。同一个 32 两处维护、
本处不生效；`get_aux_cache` docstring 声称的「LRU，上限 32」只是巧合成立。建议删除或从 `aux_cache` 导入。

### L-2 相似度重算漏注入 aux_cache

`alphaagent/factor/mining/delivery/submit.py:154` 的 `cache.evaluate(expr_text, panel, lambda: eval_factor(expr_text, panel))`
是本次唯一没注入 `aux_cache` 的 eval 调用点（另外 3 处都注入了）→ 每个存量候选重复建 1w 表，纯性能问题。

### L-3 注释过期

`alphaagent/factor/mining/delivery/submit.py:628` 注释仍写「盲测终审（stage_one 之前，不通过不进候选池）」，
实际顺序已是 stage_one 预检 → 盲测。

---

## 复现与验证记录

| 项 | 命令 / 产物 | 结果 |
|---|---|---|
| 缓存键冲突（裸层） | `%TEMP%\repro_factor_cache.py` | `B -> [1 2 3 4 5 6]`（真值 `[9 8 7 6 5 4]`）；`_factor_f64_override` 路径返回真值 |
| 指标串台（端到端） | `%TEMP%\repro_qp_crosstalk.py` | A 换手 1.75；B 走缓存 1.75（== A）；B 绕过缓存 1.7917 |
| 现有测试（均绿、但不覆盖新缓存契约） | `pytest tests/test_alphaagent_metrics.py tests/test_metrics_fastpaths.py tests/test_submit_metrics_cache.py tests/test_submit_payload.py tests/test_blind_test.py tests/test_delivery_checker.py -q` | **61 passed** |
| OCR 原始输出 | `%TEMP%\ocr_aux_cache_out.txt` | 7 findings，2 次 provider 500 重试后成功 |

本次评审**全程只读**：未改仓库任何文件（`git status` 与开工前一致，仅原有的 `?? .worktrees/`、`?? blobs/`）。

## 为什么本分支的自验证没抓到 C-1（盲区）

`docs/review/alphaagent_concurrent_eval_memory_analysis_20261006.md:181-183` 的 smoke 只验证了
「**同一**因子 8 次调用命中、`max_diff=0.00e+00`、`same buffer=True`、三段切片 `id(index)` 不同」——
全部是**单因子视角**：既没有「同 session 两个不同因子先后评测」的对照，也没有「匿名/同名 Series + 同 index」
的键冲突用例。`tests/test_submit_metrics_cache.py` 只覆盖 ingest 缓存（按数值指纹为键，是对的），
未覆盖 `factor_f64` 的身份契约。

**建议补两条确定性回归**：

1. `factor_f64`：两个 `name=None`、共享同一 index 对象、数值不同的 Series，必须各自返回自己的数组（含
   `_factor_f64_override` 对照）；
2. 同 session 连续两次 submit（或同一 split 面板上两个不同表达式）的 `quantile_portfolio` 换手/收益，
   必须与「每个因子单独一次干净评估」逐值一致。

## 处置建议（待定，本次不改）

| 方案 | 内容 | 适用 |
|---|---|---|
| A（推荐先做） | 只修 C-1（`factor_f64` 键改 `id(factor)` + weakref）+ 补上述两条回归测试 | 最小止血，避免误判/造假继续产生 |
| B | A + H-1/H-2/H-3（补 `submit.stage_one` 兼容日志、统一 error_type、提前拒绝 payload 对齐旧口径） | 保住 bench 台账与研究记忆可比性 |
| C | 全量（含 M-1 aux 列收窄、M-2 full split 键、M-3 锁范围、M-4 aux 弱引用化、L-1~L-3） | 与本分支「并发内存优化」的目标闭环 |
