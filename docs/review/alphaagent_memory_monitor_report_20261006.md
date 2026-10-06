# AlphaAgent 因子挖掘内存监控报告

**Run**: `9bf3a9ad3df1` | `scripts/run_alphaagent.py --panel cne:// --max-turns 8 --max-tool-calls-per-round 8 --max-tool-workers 12 --max-parallel-eval 12 --population-max 0`
**监控窗口**: 2026-10-05 23:16 → 2026-10-06 00:30 (约 74 分钟)
**后端进程**: PID 43392, FastAPI port 17891
**报告日期**: 2026-10-06

---

## 一、结论摘要

### 1.1 核心诊断

**PM（PagedMemory）在 10GB ↔ 39GB 之间振荡，不是单调泄漏。** 振荡由 **submit（两阶段交付）路径的瞬时分配**驱动，GC 在 submit 阶段间完整回收（T12=11166, T14=10199, T18=9125）。

**39GB 峰值是危险的瞬时尖峰**（接近系统内存上限，有 OOM 风险），但**不是累积性内存泄漏**。

### 1.2 PM 轨迹（实测）

| 采样点 | 时间 | PM (MB) | 事件 |
|--------|------|---------|------|
| T1 | 23:16 | 8826 | run 启动，preflight smoke |
| T6 | 23:35 | 11435 | 正常 eval 呼吸峰 |
| T8 | 23:45 | 8828 | eval 回落 |
| T9 | 23:50 | 10035 | submit.start 前基线 |
| **T10** | **23:55** | **39495** | **submit.precheck 运行中 + 11 并发 eval** |
| T11 | 23:57 | 36676 | precheck 仍在跑 |
| T12 | 00:00 | 11166 | precheck 完成，GC 回收 |
| **T13** | **00:01** | **33525** | **submit.blind_test 运行中 + 并发 eval** |
| T14 | 00:02 | 10199 | blind_test 完成，GC 回收 |
| T15 | 00:08 | 10153 | 正常 eval |
| T16 | 00:15 | 11480 | 正常 eval |
| T17 | 00:22 | 21151 | 高并发 eval 呼吸峰（无 submit） |
| T18 | 00:30 | 9125 | 回到基线 |

### 1.3 39GB 峰值的真正机制

**submit.precheck 与 `--max-parallel-eval 12` 的 eval worker 并发叠加：**

- submit.precheck 独占 ~11GB（tradable wide 矩阵 1.4GB + 2× portfolio float64 0.4GB + ingest metrics + session panel 8.8GB baseline）
- 同时 11-12 个 eval worker 各自跑 `eval_factor` → `_eval_multi_frequency` broadcast dict（每个 ~0.5-2GB float64 broadcast 副本）
- **11GB (submit) + 12 × ~2GB (eval) + 8.8GB (session baseline) ≈ 44GB** — 与 39GB 峰值量级吻合

**证据**：23:50-00:01 窗口内 steps.log 有 **11 次 `eval_on_train_set`/`eval_on_val_set`** 与 2 次 `submit.precheck` 时间重叠。

---

## 二、内存基线构成（~8.8-11GB，设计性保留，非泄漏）

### 2.1 Session panel 全程驻留

- `StockEvalSession.panel`（`eval/session.py`）：full cne:// panel，8.2M 行，train+val+test 全覆盖
- **挖掘主循环从不调用 `release_session`**（`service.py:453-459` docstring 明确："批量挖掘场景的会话由 run 生命周期管理"）
- 这是**设计决策**：避免每次 eval 重新加载 panel。解释 8.8-11GB 基线，**不是 39GB 尖峰来源**

### 2.2 `_split_cache`（视图，非副本）

- `slice_panel`（`panel.py:67-109`）返回 `iloc[lo:hi]` 视图，共享父 panel 数据块
- `_split_cache` 不产生 panel 三倍化；env `ALPHA_PANEL_SLICE_VIEW=0` 可禁用视图路径

### 2.3 Numba JIT 编译码常驻

- 60+ `@njit(cache=True)` 内核（`accel.py`, `dyn_window.py`, `chip_daily/_kernels.py`, `_helpers.py`）
- 磁盘缓存 ~6.5MB，但**进程内每个唯一 (dtype, shape) 签名的编译机器码常驻不释放**
- 结构性 PM 地板，非泄漏

### 2.4 有界缓存（已确认 bounded）

| 缓存 | 位置 | 上界 | 清理 |
|------|------|------|------|
| `FactorValueCache` | `cache.py:130-156` | `_SHARED_MEM_MAX_ENTRIES` + 磁盘 `_FV_MAX_BYTES`/`_FV_MAX_FILES` | LRU + 磁盘驱逐 |
| `_advisory_cache` | `advisory.py:105-106` | `_ADVISORY_CACHE_MAXSIZE=512`, TTL=60s | LRU popitem (advisory.py:305-306) |
| `_batch_history` | `tools/_analysis.py:11-22` | 30 条 | turn=0 清零 (agentscope_run.py:1268) |
| `tradable_mask` `_CACHE` | `tradable.py:114-122` | WeakRef key + LRU | 面板 GC 即失效 |

### 2.5 因子库 memmap

- `zoo.py:43-44` `_open_memmap_1d(path, mode="r")` — 裸 `np.memmap`，页错误驱动，不整载入 RAM
- `read_factor`（`zoo.py:306-323`）每次调用新开 memmap，返回 `np.array(copy=True)` — 无持久 handle 泄漏，依赖 GC
- **本 run 因子库为空**（3 个 npy 目录均 0 文件，`candidate_registry_technical.json` 缺失），正交检查全部走 `empty_factor_libraries` 跳过路径

---

## 三、39GB 尖峰的代码级归因

### 3.1 submit.precheck 分配链（`delivery/submit.py:440-575`）

```
materialize_factor(expr, panel)          # line 461 — full 8.2M-row DSL eval, 107s
  └─ cand_values (float32, 33MB)
values_by_key = pd.Series(cand_values)   # line 467 — Series + MultiIndex
metric_values = _st_mask_values(...)     # line 474 — copy (33MB)
metric_series = pd.Series(..., f32)      # line 475 — another copy (33MB)
_ingest_metrics_cached(train_policy)     # line 492 → compute_ingest_metrics
  └─ evaluate_on_panel × 2 (raw + winsorized)  # ingest.py:167-172
  └─ cross_sectional_winsorize_values
  └─ annualized_long_group_excess_return
quantile_portfolio_metrics(metric_series, panel[label], depth_ks=(5,10,20,50,100))  # line 510
  └─ f_arr_all = factor.to_numpy(f64)    # portfolio.py:172 — float32→float64 全量 (66MB)
  └─ l_arr_all = label_f64(label)        # portfolio.py:173 — label float64 (66MB)
  └─ inst_all = factor.index.get_level_values("instrument")  # portfolio.py:174
  └─ cross_sectional_rank_ic (direction pass)  # portfolio.py:145-150
  └─ 5 depth tiers × per-day pd.qcut
tradable_mask(panel, ...)                # line 522 — ★ 7+ wide 矩阵
  └─ open_w = _wide(panel, "open")       # tradable.py:124 — unstack → ~5000日×5000票 (200MB f64)
  └─ close_w = _wide(panel, "close")     # 200MB
  └─ open_next = open_w.shift(-1)        # 200MB
  └─ per_lot_cost = open_next.astype(f64) * ...  # 200MB
  └─ limit_up = build_limit_flags(...)   # bool wide (25MB)
  └─ limit_up_next = np.vstack(...)      # 25MB
  └─ ok_wide = open_w.notna() & ...      # bool wide (25MB)
  ★ 小计: ~1.4GB wide 矩阵同时存活
quantile_portfolio_metrics(..., eligibility=_mask, depth_only=True)  # line 529 — 第二次 full-panel pass
  └─ 又一组 f_arr_all + l_arr_all float64
portfolio_by_segment (train/val/test)    # line 548-573 — 第三次 × 3 段
  └─ 每段 quantile_portfolio_metrics
```

**单次 precheck 峰值估算**：1.4GB (tradable) + 0.4GB (2× portfolio f64) + 0.2GB (ingest) + 8.8GB (session panel) ≈ **~11GB**

### 3.2 submit.blind_test 分配（`submit.py:577-619`）

```python
test_policy = dataclasses.replace(stage_one_policy, train_start=test_start, val_end=test_end)
test_metrics = _ingest_metrics_cached(test_policy)  # line 584-589
  └─ compute_ingest_metrics(cand_values, panel, test_policy)
     └─ evaluate_on_panel × 2 (raw + winsorized) on test segment
     └─ full backtest, 实测 488s (ms=487800)
```

blind_test 在 test 段重跑完整回测，持有 ~23GB 峰值（T13=33525）。

### 3.3 并发 eval 叠加（`--max-parallel-eval 12`）

`_eval_multi_frequency`（`dsl/eval.py:639-755`）对每个 aux tag/panel 的每列：
- `native[tag][py] = panel[[c]]` — 原频副本
- `broadcast[tag][py] = broadcast_timeframe_to_main_freq(panel[[c]], df_main.index, tag)` — **全主频长度 float64 副本**

12 个并发 eval 各自持有 broadcast dict → **12 × ~2GB = ~24GB** 叠加在 submit 的 11GB 上。

### 3.4 代码库已知的 OOM 历史

`submit.py:443-446` 注释明确记录：
> 旧行为的二次构建曾触发 float64 合并的 8.69GiB OOM

当前代码已重构为复用 session panel（不再二次加载因子库域 panel），但 **float64 upcast 和 triple portfolio pass 仍然存在**。

---

## 四、优化机会（按影响排序）

### ★ P0 — 降低 submit 尖峰（39GB → 目标 <20GB）

#### (A) `tradable_mask` wide 矩阵复用 + float32
**位置**: `metrics/tradable.py:124-152`
**问题**: `open_w` + `close_w` + `open_next` + `per_lot_cost` + `limit_up` + `limit_up_next` + `ok_wide` = 7 个 wide 矩阵同时存活，每个 ~200MB float64 = ~1.4GB
**优化**:
1. `per_lot_cost` 用 float32 计算（价格精度足够）
2. `open_next = open_w.shift(-1)` 后立即释放 `open_w` 引用（用完即 `del`）
3. `limit_up_next` 用 `np.empty` + 切片赋值替代 `np.vstack`（避免临时拼接）
4. 考虑分块计算（按日期段），避免全量 wide 同时物化
**预期收益**: ~1.4GB → ~0.4GB，单次 precheck 峰值降 ~1GB

#### (B) `quantile_portfolio_metrics` float64 降级
**位置**: `metrics/portfolio.py:172-173`
**问题**: `f_arr_all = factor.to_numpy(dtype=np.float64, copy=False)` + `l_arr_all = label_f64(label)` 全程持有 float64 全量数组
**优化**:
1. IC/ICIR 排名计算可用 float32（排名对精度不敏感）
2. `label_f64` 改为按需 float32，仅在年化收益计算处 upcast
3. `cross_sectional_rank_ic` direction pass（line 145-150）可与主 pass 合并，避免额外全面板遍历
**预期收益**: 每次 portfolio pass 降 ~130MB，triple pass 降 ~400MB

#### (C) submit.precheck 三次 portfolio pass 合并
**位置**: `delivery/submit.py:510-573`
**问题**:
1. line 510: `quantile_portfolio_metrics(metric_series, panel[label], depth_ks=(5,10,20,50,100))` — 主口径
2. line 529: `quantile_portfolio_metrics(..., eligibility=_mask, depth_only=True)` — tradable 透镜
3. line 548-573: `portfolio_by_segment` — train/val/test 各一次
**优化**:
1. 主口径与 tradable 透镜共享 `f_arr_all`/`l_arr_all`（传入而非重算）
2. `portfolio_by_segment` 复用主口径的 `metric_series` 切片（按 train/val/test index mask），而非重跑完整 `quantile_portfolio_metrics`
3. 在内存压力下（PM > 阈值）跳过 `depth_only=True` 透镜（仅留档用，不参与门槛）
**预期收益**: 3× portfolio pass → 1× + 2× 轻量切片，降 ~400MB

#### (D) blind_test 延后到 stage_one 通过后
**位置**: `delivery/submit.py:577-619`
**问题**: blind_test 在 stage_one **之前**运行（line 584-589 `_ingest_metrics_cached(test_policy)` 重跑 test 段完整回测，488s），即使 stage_one 会 fail 也已消耗
**实测**: 两个 submit（vwap_wma5/wma15）都在 stage_one `passed=False`（换手率超限），blind_test 的 488s 回测完全浪费
**优化**: 将 blind_test 移到 stage_one `passed=True` 之后，stage_one fail 直接 return，跳过 488s 回测
**预期收益**: stage_one fail 的 submit 不再触发 blind_test 峰值（本 run 2/2 submit 都 fail → 完全避免 T13=33525 峰值）

#### (E) materialize_factor 先在 train 窗口物化
**位置**: `delivery/submit.py:461`
**问题**: `materialize_factor(expr, panel)` 在 full 8.2M-row panel 上物化（107s），但 stage_one 只看 train 窗口
**优化**: 先在 train slice 上物化 + 评估，stage_one 通过后再 full-panel 物化
**预期收益**: stage_one fail 的 submit 不再触发 full-panel materialize（107s + full-panel DSL eval 中间体）

### P1 — 降低 eval 并发峰值

#### (F) `_eval_multi_frequency` broadcast dict 内存感知并发
**位置**: `dsl/eval.py:639-755`
**问题**: 12 并发 eval 各自构建 broadcast dict（每个 aux 列全主频长度 float64 副本），峰值 ~24GB
**优化**:
1. 根据 PM 动态限制并发数（PM > 30GB 时降到 4 并发）
2. broadcast 副本用 float32
3. aux 列裁剪（只 broadcast 实际用到的列，而非全 panel 列）
**预期收益**: 12 × 2GB → 12 × 0.5GB = 降 ~18GB 并发峰值

### P2 — CPU 浪费（非内存，但值得修）

#### (G) `eval_factor` 缺 compile cache
**位置**: `dsl/eval.py:530`
**问题**: `compile_multi_line_factor(...)` 每次调用重新解析 DSL，无编译缓存
**优化**: 按 expr hash 缓存编译结果（LRU 256）
**预期收益**: 重复 expr 的 eval 提速，减少 CPU 但对 PM 影响小

#### (H) `aux_cache` 跨因子不共享
**位置**: `dsl/eval.py:514-515`
**问题**: `if aux_cache is None: aux_cache = {}` — 每次调用新建，1w 聚合不跨因子共享
**优化**: session 级 aux_cache（按 aux tag + panel fingerprint 缓存 1w 聚合结果）
**预期收益**: 减少 1w 重聚合 CPU

### P3 — 已确认非问题（无需改动）

| 项 | 结论 |
|----|------|
| `_split_cache` 三倍化 | ❌ 视图非副本 |
| memmap handle 泄漏 | ❌ GC 回收，无池但无持久泄漏 |
| PM 单调泄漏 | ❌ 振荡，GC 阶段间完整回收 |
| `_advisory_cache` 无界 | ❌ MAXSIZE=512 + TTL=60s + LRU |
| `_batch_history` 无界 | ❌ 上限 30，turn=0 清零 |
| `FactorValueCache` 无界 | ❌ LRU + 磁盘双限 |
| session panel 驻留 | ⚠️ 设计决策（非泄漏），但贡献 8.8GB 基线 |
| numba JIT 编译码常驻 | ⚠️ 结构性，无法释放 |
| 正交检查 float64 upcast | ⚠️ 本 run 因子库空，moot；有库时 valid |

---

## 五、建议优先级

| 优先级 | 优化项 | 预期 PM 降幅 | 实现难度 | 风险 |
|--------|--------|-------------|---------|------|
| **P0-紧急** | (D) blind_test 延后到 stage_one 后 | 避免 T13 峰值 (−23GB) | 低（控制流调整） | 低 |
| **P0** | (E) materialize 先 train 窗口 | 避免 full-panel 物化 | 中 | 中（需保证 full-panel 一致性） |
| **P0** | (A) tradable_mask float32 + 即时释放 | −1GB/submit | 中 | 低（float32 精度足够） |
| **P0** | (C) 三次 portfolio pass 合并 | −0.4GB/submit | 中 | 中（口径一致性需验证） |
| **P1** | (B) portfolio float64 降级 | −0.4GB/submit | 中 | 中（排名精度需验证） |
| **P1** | (F) eval 并发内存感知 | −18GB 并发峰 | 高 | 中（动态限流逻辑） |
| **P2** | (G) compile cache | CPU only | 低 | 低 |
| **P2** | (H) aux_cache 共享 | CPU only | 中 | 中 |

**最高 ROI**：**(D) blind_test 延后** — 仅控制流调整，可完全避免 stage_one fail 时的 488s 回测峰值。本 run 2/2 submit 都在 stage_one fail，若先实施 (D)，T13=33525 峰值不会出现。

---

## 六、监控方法说明

- **PM（PagedMemorySize64）作为泄漏信号**：Windows 在内存压力下会修剪 WorkingSet，WS 不可靠；PM 包含换出页，是稳定的泄漏指标
- **采样方式**：`Get-Process -Id <PID>` 定时采样 PM/WS/VM，不注入进程
- **代码归因**：grep + read 源码定位分配点，结合 steps.log 时间戳对齐 PM 振荡与 submit/eval 事件
- **未做**：未用 `tracemalloc` / `memray` 注入（会干扰运行中挖掘进程）；未做 heap profiling

---

*报告基于 read-only 监控 + 代码审查，未修改任何文件。所有优化建议需在新建分支上实施（AGENTS.md 分支纪律）。*
