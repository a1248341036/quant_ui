# AlphaAgent 面板内存优化方案（2026-11-14）

> 触发问题：**跑挖掘太耗内存**（机器 31.1GB，实测单进程常驻 7.3GB private、加载峰值 14.5GB）。
> 用户授权：**可以牺牲时间换内存**。
> 本方案基于本轮**实测**（非估算）+ 代码级归因，所有数字可在「证据索引」复现。

---

## 一、结论先行

| 项 | 结论 |
|---|---|
| 内存大头在哪 | **不是并发的锅，也不是 submit 的锅**——是 `panel` 本体：7.78M 行 × 230 列加载后**私有内存 7.79GB、RSS 7.31GB**，加载过程峰值 **14.50GB** |
| 为什么 arrow mmap "零拷贝"没生效 | `read_panel_arrow_mmap` 逐列 `combine_chunks().to_numpy(zero_copy_only=True)`，**实测每列产生 29–31MB 私有拷贝**（恰等于 7.78M×4B），230 列累计 **6.83GB**。`zero_copy_only=True` 未报错、`owndata=False`，但 `arr.base` 是 ndarray —— 即 pyarrow 在 IPC 读取路径上**悄悄做了私有拷贝** |
| mmap 到底有没有用 | 有，但只解决**多进程**场景：mmap 页可跨进程共享物理内存。**单进程内仍完整拷贝一份**，所以"跑一个 run 就要 7.3GB" |
| 并发 eval 占多少 | 6 路并发实测仅 **+0.71GB**（历史报告说的"每路 2GB"已被 session 级 `aux_cache` 修掉）。**降并发省不下多少内存**，别把它当主手段 |
| 最优解 | **面板改行主序 2D memmap（`.npy`）**：已实测 `pd.DataFrame(2d_memmap, columns=..., copy=False)` → **单 block、真正零拷贝**（`shares_memory=True`、`iloc` 切片也是 view），且**新增列不再触发整块拷贝**。✅ **已实施（commit `d590cf0`）**：私有内存 **7.890GB → 0.726GB（−90.8%）**，2 进程并行合计 15.79GB → 1.45GB，因子值逐位一致（`max_diff = 0`）。详见**七·B** |
| 立即可用的止损 | 3 条不改代码的配置手段（见第五节），最大可立刻省 6GB |

---

## 二、现状实测（本轮真实测量）

### 2.1 环境

- 机器：31.1GB 物理内存（`Win32_OperatingSystem.TotalVisibleMemorySize`）
- 面板缓存：`artifacts/panel/cache/panel_v11_7fbda4c156e7a3e6_all.arrow`
  - **6.76GB**，7,782,511 行 × 230 数据列（schema 232 含 index 两列），`include_fundamentals=True`，覆盖 2020-01-01 ~ 2026-09-30
- dtype：float32 × 225、int8 × 3（is_trade / not_st / is_st）、object × 2（index 层）

### 2.2 三档实测（psutil `memory_info`：rss / private / peak_wset）

| 场景 | RSS | **Private** | Peak WS |
|---|---|---|---|
| 空进程 baseline | 0.09 | 0.54 | 0.09 |
| 加载 4.63M 行 × 230（2023 起缓存） | 4.43 | **4.97** | **8.60** |
| **加载 7.78M 行 × 230（挖掘实际窗口，2020 起）** | 7.31 | **7.79** | **14.50** |
| 上述 + **6 路并发 eval** | 7.58 | 8.51 | 14.50 |

> 结论：**面板本体 7.79GB 私有**；并发 6 路只再加 0.71GB；加载期峰值是常驻的 1.86 倍（14.50 / 7.79）。

### 2.3 逐阶段隔离（`panel_v11_7fbda4c156e7a3e6_all.arrow`）

| 阶段 | RSS | Private | 说明 |
|---|---|---|---|
| baseline | 0.018 | 0.010 | |
| `pa.memory_map` + `ipc.open_file` | 0.040 | 0.509 | 仅读元数据 |
| `.read_all()` | 0.040 | 0.509 | **不物化数据**（单 record batch） |
| **逐列 `combine_chunks().to_numpy(zero_copy_only=True)`** | **13.35** | **7.297** | ★ **罪魁**：私有 +6.79GB、RSS +13.31GB |
| + 构建 MultiIndex(datetime, instrument) | 14.21 | 8.008 | instrument object 化 + datetime 转换 |
| + `pd.DataFrame(data, index=index, copy=False)` | 14.21 | 8.008 | **不变**（230 blocks，pandas 侧确实零拷贝） |

**逐列斜率**（每 40 列采样，private 增量）：

| 列序号 | private | 增量 / 列 |
|---|---|---|
| 5 | 0.714 | — |
| 39 | 1.707 | **29.1MB** |
| 79 | 2.945 | **31.0MB** |
| 119 | 4.122 | **29.4MB** |
| 159 | 5.299 | **29.4MB** |
| 199 | 6.477 | **29.5MB** |

7,782,511 × 4B = **29.68MB/列** —— 完全吻合。**每列都被完整私有拷贝了一份**。

### 2.4 排除项（已实测否证）

| 假设 | 实测结论 |
|---|---|
| `_enrich_trade_flags` 写 `is_st/not_st` 触发 consolidate 拷贝 | ❌ 否证：把该函数 patch 成 no-op，RSS 仍 7.345GB |
| pandas 把同 dtype 列 `_stack_arrays` 合并成整块 | ❌ 否证：`pd.DataFrame(dict_of_arrays)` → **230 个独立 block，各列 `shares_memory=True`** |
| `apply_universe("all")` 拷贝 | ❌ 否证：源码 `universe.py:108` 对 `all` 原样返回 |
| 缓存命中后 `df.loc[keep]` 窗口裁剪拷贝 | ❌ 否证：`keep` 全 True 时短路，不裁 |
| `read_all()` 物化全表 | ❌ 否证：`read_all` 前后 private 都是 0.509GB |

---

## 三、根因链

```
load_panel_from_cne(include_fundamentals=True)
  └─ _find_cached_panel()                                  (cnequity.py:445)
      └─ read_panel_arrow_mmap(arrow_path)                 (panel_mmap.py:144)
          └─ pa.memory_map + ipc.open_file().read_all()    ← 仅元数据，0.51GB
          └─ for field in table.schema:                    (panel_mmap.py:174-186)
                col = table.column(name).combine_chunks()
                arr = col.to_numpy(zero_copy_only=True)    ← ★ 每列私有拷贝 29.7MB
          └─ pd.DataFrame(data, index=index, copy=False)   ← 零拷贝，但 data 已是私有副本
  └─ _enrich_trade_flags(cached)                           (cnequity.py:976)
```

**核心矛盾**：模块 docstring 声称"数值列 numpy 数组零拷贝指向同一物理页"，但 pyarrow 在
`combine_chunks() → to_numpy()` 这条链上返回了**私有副本**（`zero_copy_only=True` 未抛异常，
因为对 Arrow 而言确实"无需复制类型"，但底层 buffer 与 numpy 兼容对齐时仍发生了 copy）。
结果：**mmap 页 fault（13.35GB RSS 中含 6.76GB mmap 页）+ 每列私有副本（6.79GB）双份并存**，
加载峰值 14.5GB 即由此而来。

**这解释了所有历史现象**：
- "多进程各自 3.6–8GB"：mmap 页共享只对**页**成立，私有副本仍每进程一份；
- "减 worker 省不了多少"：worker 数只影响 eval 临时量（0.7GB 量级），不影响面板本体；
- "列数是唯一杠杆"（2026-10-20 实测 158 列 5.80GB / 19 列 1.37GB）：因为省的是**每列 29.7MB 私有拷贝**。

---

## 四、方案（P0 / P1 / P2）

### P0-1 ★ 面板改「行主序 2D memmap」（本方案核心，收益 ~7GB）

**已验证的关键性质**（`verify_2dmemmap_semantics.py`，500k×230 float32）：

| 性质 | 实测 |
|---|---|
| `pd.DataFrame(2d_memmap, columns=cols, index=idx, copy=False)` | **n_blocks = 1**，`block_shape = [230, 500000]` |
| 整表是否等于 memmap 视图 | `shares_memory(df.to_numpy(), ro) == True` |
| 单列取值 `df[col]` | **view**（`shares_memory == True`） |
| 行切片 `df.iloc[1000:2000]`（`slice_panel` 用的就是这个） | **view** |
| 列裁剪 `df[[a,b,c]]` | 1 block，行为正常 |
| **新增列 `df[new] = ...`** | **2 blocks，原 block 仍共享物理页**（`shares_memory == True`）← 彻底避免 `_enrich_trade_flags` 的整块拷贝隐患 |
| 覆盖已有列 | 其他列仍共享，不整块重排 |
| `groupby(level=0)` / `shift / rolling` | 正常 |

**落盘格式**

```
artifacts/panel/cache/panel_v11_<key>_all.matrix.npy   # float32, shape=(n_rows, n_cols)，C 序
artifacts/panel/cache/panel_v11_<key>_all.matrix.json  # {"columns": [...], "n_rows": ..., "n_cols": ..., "schema": 11, ...}
artifacts/panel/cache/panel_v11_<key>_all.matrix.idx.npz  # datetime(int64 纳秒) + instrument codes(int32) + codes→str 映射
```

- **全部 230 列统一 float32**（含 is_trade/not_st/is_st 以 0.0/1.0 存）——单 block 的前提；
- index 层单独存：`instrument` 用 **int32 codes + 词表**替代 object（object 化 7.78M 行约 0.4–0.6GB 私有，
  每进程独立），`datetime` 存 int64 纳秒，读取时 `pd.DatetimeIndex` 视图化；
- 与现有 `.parquet` / `.arrow` **三格式共存**，新格式优先、失败自动回退（不破坏旧路径）。

**新增模块** `alphaagent/data/adapters/panel_matrix.py`（约 250–350 行）

```python
def write_panel_matrix(base: Path, panel: pd.DataFrame) -> bool: ...
def read_panel_matrix(base: Path, *, columns: Sequence[str] | None = None) -> pd.DataFrame | None: ...
def matrix_column_names(base: Path) -> list[str] | None: ...   # 只读 json，不碰数据
```

- **写入按行分块**避免一次 +6.8GB 临时：每块 20 万行（`np.empty((200_000, 230), f32)` ≈ 184MB），
  `arr[lo:hi, :] = panel.iloc[lo:hi].to_numpy()`；写入后 `flush()` 再原子 `os.replace`；
- **读取**：`np.memmap(mode="r")` → `pd.DataFrame(arr, columns=cols, index=idx, copy=False)`；
  `columns=` 非 None 时返回列子集（列选择本身是 view 语义，零拷贝）；
- **列名与索引从 json/npz 读，不扫数据**（对齐 `arrow_column_names()` 的既有做法）。

**接线点**

| 文件 | 改动 |
|---|---|
| `alphaagent/data/adapters/cnequity.py:445-598` `_find_cached_panel` | 优先试 `.matrix.npy`（`read_panel_matrix`）→ 再 arrow → 再 parquet |
| `cnequity.py:739-741` `_save_cached_panel` | 落盘时追加 `write_panel_matrix()`（失败不影响 parquet 主缓存，与 arrow 同策略） |
| `cnequity.py:379-432` `columns_for_facets` / `_parquet_column_names` | 列名真源扩展到 `.matrix.json` |
| `alphaagent/data/panel.py:368-435` `load_panel` | 同 key `.matrix.npy` 优先（与现有 `.arrow` 优先级并列） |

**预期收益**

| 指标 | 现状 | P0-1 后（估） |
|---|---|---|
| 加载后 private | 7.79GB | **≈0.6–0.9GB**（index + 词表，无列数据） |
| 加载后 RSS | 7.31GB | **≈0.3–1.0GB**（仅被访问列 fault，29.7MB/列） |
| 加载期峰值 WS | 14.50GB | **≈1.5–2.5GB** |
| 典型 eval（5–10 列）后峰值 | 15.2GB | **≈2–3GB** |
| 230 列全触碰（极端） | 14.2GB | **≈6.8GB**（且是**可回收的 mmap 页**，非 private） |

**风险与回退**

- 风险：数据层核心路径，语义必须逐位一致 → **验收要求**：同一批因子在新旧路径下
  `train IC / ICIR / coverage / turnover` 完全一致（`max_diff == 0`）；
- 回退：三格式共存 + `ALPHA_PANEL_MATRIX=0` 环境变量强制回退 arrow；
- 磁盘：新增 6.76GB/缓存组，现有 `_purge_old_cache` 双闸（`_CACHE_MAX_GB` / `_CACHE_MAX_FILES`）
  需把 `.matrix.*` 纳入成对淘汰（**否则会漏删成孤儿**）。

### P0-2 索引层瘦身（可选，收益 ~0.4GB）

- `instrument` object → `CategoricalIndex`（codes + categories）：单进程私有从 ~0.5GB 降到 ~0.06GB；
- 风险：下游若做 `df.index.get_level_values("instrument")` 后按字符串比较/排序，
  Categorical 语义差异需逐点核对；**建议 P0-1 验收通过后再做**。

### P1 并发与临时量（收益有限，别当主手段）

**实测已证明并发不是大头**：6 路并发仅 +0.71GB。因此：

| 措施 | 收益 | 代价 | 建议 |
|---|---|---|---|
| `--max-parallel-eval` 6 → 3 | ~0.3–0.5GB | 吞吐约减半 | 仅在机器吃紧时用 |
| `--max-tool-workers` 8 → 4 | 同上 | 同上 | 可选 |
| session 级 `aux_cache` | 已实现（2026-10-06 `b92c7a9`） | — | 保持 |
| `factor_f64` 身份缓存 | 已实现 | — | 保持 |

### P2 submit 路径（保持既有优化，补两处）

历史已把 submit 峰值从 ~11GB 压到 ~4GB（aux_cache / float64 缓存 / 盲测延后 stage_one）。
本方案补充：

1. **`tradable_mask` wide 矩阵 float32 + 即时释放**（`alphaagent/factor/metrics/tradable.py:124-152`）：
   `open_w/close_w/open_next/per_lot_cost` 等 7 个 wide 矩阵同时存活 ≈1.4GB → float32 + `del` 即时释放 ≈0.4GB；
2. **submit 前后 `gc.collect()`** + 在 `submit.precheck` 入口打一条 RSS 观测点（见下）。

### P3 可观测（必须做，否则下次又靠猜）

当前 `metrics_snapshot` **无 RSS 字段**，全仓无内存打点，导致每次都要临时写探针（本方案的探针就是
`%TEMP%\memprobe_panel.py` / `verify_*.py`）。建议：

- `alphaagent/factor/mining/eval/session.py` 加载完成处、`delivery/submit.py` 各阶段入口，
  记录 `rss_gb / peak_wset_gb`（psutil 已在 venv 中，`psutil 7.2.2`）；
- 写入 `run_meta.json` 的 `memory` 段，前端"运行详情"可直接看。

---

## 五、立即可用的止损（不改代码，今天就能省）

| # | 手段 | 立刻可省 | 代价 |
|---|---|---|---|
| 1 | **`--no-fundamentals`** | 63 列（`funda_*` 52 + `fsi_*` 11）× 29.7MB ≈ **1.87GB** | 失去基本面面；按 `docs` 实测（4.D）该面裸腿 IC ≤0.014，多数场景本就不产出 |
| 2 | **`--focus-facets 价量面`** | 面板降到 ~20–30 列 → 省 **≈6GB** | **硬锁定**探索范围（价量面仅 11 列 + 必需列）；适合明确只挖价量的批量任务 |
| 3 | 单 run 串行，不并行多 run | 每个并行 run 省一整份面板私有拷贝 | 吞吐下降 |
| 4 | 避免 `df.drop(columns=[...])` | 避免一次整块复制 ~6GB | —（`cnequity.py:207-209` 的 `drop(funda_*)` 已改成按需取列，**别退回去**） |

> 说明：`--no-fundamentals` 与 `--focus-facets` 都会**换缓存 key** → 首次会**重建面板**
> （构建路径峰值仍高，见下节），重建完成后第二次起才享受收益。

---

## 六、构建路径（缓存未命中）另需注意

`_finalize_panel` 的 13.7GiB OOM 已由 `8639267`（先 downcast 再 copy）压到 ~6.9GiB 合并峰值。
但**构建路径仍是全量内存动作**（`cnequity.py:907-914` 明确注释：重建路径 `keep_columns` 不生效，
必须写全列缓存）。P0-1 落地后建议追加：

- 构建产物**直接流式写 `.matrix.npy`**（按行分块），避免"先有 7GB DataFrame 再整体转置 +7GB"；
- `_warn_panel_rebuild_cost`（`cnequity.py:761`）的估算公式接入实测系数（29.7MB/列·7.78M 行）。

---

## 七、实施顺序与验收

| 阶段 | 内容 | 预估改动 | 验收标准 |
|---|---|---|---|
| **S1** | P3 可观测打点 | ~40 行 | 跑一个 run 后 `run_meta.json` 出现 `memory.rss_gb/peak_wset_gb` |
| **S2** | P0-1 `panel_matrix.py` + 三处接线 + 单测 | ~350 行 | ① 加载 7.78M×230：**private < 1.5GB、peak WS < 3GB**；② 抽 5 个因子新旧路径 `IC/ICIR/cov/turnover` **max_diff = 0**；③ 强制 arrow 回退可用 |
| **S3** | P2 `tradable` float32 + submit 打点 | ~60 行 | submit 段峰值 < 3.5GB |
| **S4** | P0-2 索引 Categorical（可选） | ~60 行 | 再省 ~0.4GB，且全量单测通过 |

**总预期**：单进程完整 run 峰值 **≈18–20GB → ≈5–7GB**（面板 7.79→0.7、加载峰值 14.5→2、
submit 4→3.5），可在 31.1GB 机器上**安全并行 2–3 个 run**。

---

## 七·B、实施结果（P0-1 已于 2026-11-14 落地，commit `d590cf0`）

### 7B.1 改动清单

| 文件 | 改动 |
|---|---|
| `alphaagent/data/adapters/panel_matrix.py`（新增 380 行） | 行主序 float32 矩阵三件套的写/读/列名/删除 |
| `alphaagent/data/adapters/cnequity.py` | `_find_cached_panel` 三级读取（matrix > arrow > parquet）；`_save_cached_panel` 落盘 matrix（默认开）；`_purge_old_cache` 成对淘汰并计入配额；**覆盖区间预筛修复** |
| `alphaagent/data/panel.py` | `load_panel` 同 key `.matrix.npy` 优先于 `.arrow` |
| `tests/test_panel_matrix.py`（新增 21 项） | 往返 / 零拷贝 / 字符串列 / 回退 / 原子性 |

### 7B.2 实测对比（7.78M 行 × 230 列，`start=2020-01-01`，同条件 A/B）

| 指标 | arrow 路径 | **matrix 路径** | 变化 |
|---|---|---|---|
| 加载耗时 | 10.14s | **3.23s** | **−68%** |
| **私有内存（private / commit）** | 7.890GB | **0.726GB** | **−90.8%** |
| 峰值 WS（peak_wset） | 14.427GB | **7.271GB** | −49.6% |
| pandas block 数 | 230 | **5** | — |
| **2 进程并行私有合计** | 15.788GB | **1.450GB** | **−90.8%** |

> **关于 peak_wset 7.27GB 的诚实说明**：其中约 6.5GB 是**只读文件映射页**（mmap 文件页），
> 属于可回收的 file-backed 页、且跨进程共享；真正的**私有提交内存只有 0.726GB**。
> 系统内存吃紧时这些页会被直接丢弃，不需要换页，这也是 2 进程合计只有 1.45GB 的原因。
> 所以评估内存压力请看 `private`，不要看 `peak_wset`。
> 原方案第七节预估 peak ≈2GB 偏乐观（未预计到索引构建与映射页计入 WS），以实测为准。

### 7B.3 正确性证据

三条代表性因子在两条路径下**逐位相等**（`max_abs_diff = 0.000e+00`，有限值掩码一致）：

| 表达式 | coverage | daily rank IC（两路径相同） |
|---|---|---|
| `CS_ZSCORE(TS_MEAN($ret, 20))` | 0.973384 | −0.07231992 |
| `CS_NEUTRALIZE(TS_MEAN(DIVIDE($amount,$float_cap),20), $industry_sw_l1)` | 0.968473 | −0.07811638 |
| `TS_CORR($adj_close, $volume, 20)` | 0.975367 | −0.06634029 |

### 7B.4 顺带修掉的缺陷（实测意外收益）

`_find_cached_panel` 旧逻辑是**先物化、再判断是否覆盖请求窗口**：扫到一份不匹配的缓存时，
已经付出了完整物化代价才 `continue`。实测（4 组缓存并存、请求 2020 起）会为一份
4.63M 行 × 230 列的 2023 起缓存**白付 4.4GB 私有 + 6.7GB 映射页**，把加载峰值从
~1GB 抬到 11.5GB。

修法：用 `meta.json` 里已有的 `start`/`end` 做**物化前预筛**（缺失则退回原判定）。
这条修复对 arrow 路径同样生效——上表 arrow 列已含该收益。

### 7B.5 开关与迁移

| 环境变量 | 默认 | 说明 |
|---|---|---|
| `ALPHA_PANEL_MATRIX` | `1`（开） | 落盘 `.matrix.npy` 副本；设为 `0` 关闭 |
| `ALPHA_PANEL_ARROW` | `1`（开，沿用旧值） | 建议 matrix 生效后设 `0`：matrix 同样支持跨进程 mmap 共享，而 arrow 每列私有拷贝，二者都开会让单组缓存从 ~9GB 涨到 ~16GB |

- **已有缓存迁移**：不必重建面板。对已有 `.parquet` 跑一次
  `write_panel_matrix(parquet_path, panel)` 即可生成矩阵（本次 6.67GB / 43s）；
- **磁盘**：matrix 6.67GB ≈ arrow 6.76GB，若两者都留则单组约 16GB；
  `_purge_old_cache` 已把 `.matrix.*` 计入 `ALPHA_PANEL_CACHE_MAX_GB` 配额并成对淘汰；
- **回退**：删除 `.matrix.*` 三件套即可自动回落 arrow / parquet，无需改代码。

### 7B.6 尚未做（可选后续）

- P0-2 索引层 CategoricalIndex（再省 ~0.4GB）；
- P2 `tradable_mask` wide 矩阵 float32（submit 段 ~1GB）；
- P3 打点（本轮实测仍靠临时探针，建议按第七节 S1 补 `run_meta.json.memory`）。

---

## 七·C、P0-1b：字符串列移出主矩阵（rss 修复，commit `3f3fc5f`）

### 7C.1 现象

P0-1 落地后 **private 已降到 0.76GB**，但**读取后的 rss（工作集）仍是 7.1GB**——
任务管理器里看进程照样占 7GB，用户感知"没优化"。private 与 rss 的巨大差距说明：
数值列确实是零拷贝视图，但**有东西把整份文件所有页都 fault 进来了**。

### 7C.2 根因：列 stride 小于页大小

面板有 2 个字符串列（`suspend_timing` / `suspend_type`），v1 把它们以类别码
（float32）混在主矩阵里，读取时用 `arr[:, j]` 取列再还原。

关键数字：**列 stride = 230 列 × 4B = 920B，而页大小 4096B**。
按一列跨行遍历 7.95M 行，每读一个元素就落在不同页上（920B < 4KB），
等于**把整份 6.75GB 文件的每一页都摸一遍**。

实测隔离（`verify_colscan_fault.py`）：

| 操作 | rss |
|---|---|
| 仅 `np.memmap` 映射 | 0.029GB |
| 构造 DataFrame（零拷贝） | 0.077GB |
| 读 100 行 × 1 列 | 0.077GB |
| **遍历单列全部行** | **6.947GB** |

private 全程只从 0.537 涨到 0.596GB —— 所以这 6.9GB 是**可回收的文件页**，
但会挤占 page cache、让系统在内存压力下表现变差，数字上也很难看。

### 7C.3 修法（matrix schema v1 → v2）

- 主矩阵**只放数值列**（228 列）；
- 字符串列以 **int32 codes** 存进 `idx.npz` 的 `cat_<name>`，词表进 json；
- 读取时从 npz 还原成 object 数组，作为**独立 block 追加**，全程不碰主矩阵数据页；
- **代价**：矩阵路径列序变成「数值列… + 字符串列」，与 parquet 路径不同。
  下游一律按列名取列（`read_panel_matrix` docstring 已明示）；已核对
  `_find_cached_panel` / `columns_for_facets` / `_prune_panel_columns` / DSL eval
  全部按名操作，不依赖列位置。

### 7C.4 实测（7.95M 行 × 230 列）

| 指标 | v1 | **v2** |
|---|---|---|
| 读取 rss | 7.113GB | **0.302GB（−95.8%）** |
| 读取峰値 WS | 7.414GB | **0.566GB** |
| 读取耗时 | 2.55s | **0.81s** |
| pandas blocks | 5 | 3 |
| private | 0.761GB | 0.761GB（index 部分，未变） |

对照最初 arrow 路径：**private 7.79GB / rss 7.31GB / peak 14.50GB**。
累计降幅：peak **−96%**、private **−90%**、rss **−96%**。

### 7C.5 迁移工具（不必重建面板）

`scripts/convert_panel_matrix.py`：用 `ParquetFile.iter_batches` **流式**转换已有
parquet（两遍扫描建全局词表），逐批写矩阵。

| 方式 | private 峰值 | 耗时 |
|---|---|---|
| 一次性加载 DataFrame 再转（旧） | 15.38GB | 43s |
| **流式 iter_batches（新）** | **2.99GB** | **19.6s** |

用法：`python scripts/convert_panel_matrix.py <cache.parquet> [--force]`
—— schema 升级后只需转一次，不用重建整个面板。

### 7C.6 回归守卫

新增测试 `test_read_does_not_fault_whole_file`：构造 ~48MB 矩阵，断言读取后
rss 增幅 < 文件一半。这是锁住"字符串列又混回主矩阵"这类回归的关键用例
（普通功能测试查不出 rss 浪费）。

### 7C.7 v2 之后端到端仍是 7GB：`_column_all_nan`（commit `dc7bc47`）

修完字符串列后，`read_panel_matrix` 单独调用已是 rss **0.30GB**，但经
`load_panel_from_cne` **端到端仍是 7.04GB**。差异来自缓存命中后的
`_enrich_trade_flags`：

```python
def _column_all_nan(panel, column):
    return bool(pd.isna(panel[column]).all())   # ← 跨行扫单列
```

`pd.isna(panel[col])` 又踩同一个 stride 陷阱（920B < 4KB 页）——扫一列
就把整份 6.75GB 文件 fault 进工作集。

**修法：把"哪些列整列 NaN"在写盘时算好**（写矩阵时数据本就在内存，零额外 IO），
存进 `matrix.json` 的 `all_nan_columns`；读取时挂到 `panel.attrs`，
`_column_all_nan` 优先查表。无 attrs 的普通面板保持原逻辑。

**兼容保护**：早期 v2 缓存没有该字段时**不挂 attrs**（挂空集会让所有列误报
"有值"，导致 `is_trade` 永不被填），退化为原扫描——慢但正确。

端到端实测（`cne://` 2020-01-01 全列，7.95M 行 × 230 列）：

| 指标 | 修复前 | **修复后** |
|---|---|---|
| 读取 rss | 7.035GB | **0.283GB** |
| 读取 peak | 7.043GB | **0.568GB** |
| 加载耗时 | 3.00s | **1.61s** |
| private | 0.735GB | 0.735GB |

### 7C.8 累计效果（相对最初的 arrow 路径）

| 指标 | 最初 arrow | **当前** | 变化 |
|---|---|---|---|
| 加载 peak WS | 14.50GB | **0.57GB** | **−96.1%** |
| 私有内存 private | 7.79GB | **0.735GB** | **−90.6%** |
| 加载后 rss | 7.31GB | **0.283GB** | **−96.1%** |
| 加载耗时 | 10.14s | **1.61s** | **−84.1%** |
| 磁盘/组缓存 | 15.6GB（三格式） | **9.15GB** | **−41%** |
| 2 进程私有合计 | 15.79GB | **1.47GB** | **−90.7%** |

---

## 七·D、P2 / P3 落地（commit `af640d2`）

### 7D.1 P3 内存打点（零依赖）

全仓此前**没有任何内存指标** —— `metrics_snapshot` 无 RSS、`run_meta.json` 无内存，
排查只能每次临时写探针（本轮就写了 6 个）。现收口为：

- 新增 `alphaagent/core/memlog.py`：用 ctypes 直读 `PROCESS_MEMORY_COUNTERS`，
  **零依赖**（`psutil` 不在 requirements，仅作兜底）；非 Windows 返回空 dict，
  永不阻断主流程。
- 口径明确区分：`rss`（含可回收 mmap 页，任务管理器显示的就是它）/ `private`
  （提交内存，**判断压力看这个**）/ `peak`。
- `metrics_snapshot` 增加 `memory` 段 —— 每轮快照可看内存趋势。
- run 收尾把 `peak` 写回 `run_meta.json` 的 `memory` 段 —— 前端任务列表/详情
  能直接看到某个 run 吃了多少内存，不必再猜。

### 7D.2 P2 `tradable` 宽表 float32（收益如实记录：约 70MB）

- `per_lot_cost` 由 float64 改 float32：只用于「一手成本 ≤ 等权预算」阈值比较，
  相对误差 ~1e-7 远小于预算边界；宽表 1540×5160 下少占 ~32MB，
  `astype(copy)` 峰值从 63.6MB 降到 31.8MB。
- `limit_up_next` 用 `empty_like` + 切片替代 `np.vstack`，省一份中间拼接数组。
- 两者算完即 `del`。

> **口径修正**：本方案第四节按「5000 日 × 5000 票」估 tradable 峰值 ~1.4GB，
> 实测面板是 **1540 日 × 5160 票**，7 个 wide 矩阵合计约 **300MB**。
> 所以 P2 真实收益只有 ~70MB，**不是 1GB**。不再作为主要收益项。

### 7D.3 P1 并发（结论：不要当主要手段）

6 路并发 eval 实测只增加 **0.71GB**（session 级 `aux_cache` 已在 2026-10-06 生效）。
降并发只换来 ~0.3-0.5GB，代价是吞吐减半 —— 面板本体才是原来 7.79GB 的来源，
P0 解决后并发更不该背这个锅。

---

## 七·E、端到端验证：窗口裁剪抵消零拷贝（commit `5b7e4f4`）

### 7E.1 现象

matrix v2 全部落地后，用 `harness_mine --full-cols --fundamentals` 跑 8 个因子做
端到端验证，结果**和 arrow 时代一模一样**：

```
面板加载后内存: rss=7.063GB private=7.549GB peak=14.017GB
内存峰值:       rss=7.597GB private=8.238GB peak=14.017GB
```

### 7E.2 根因：`df.loc[布尔数组]` 整块拷贝

`_find_cached_panel` 按请求窗口裁剪缓存时用的是：

```python
keep = pd.Series(True, index=df.index)
keep &= dts >= req_start
keep &= dts <= req_end
if not bool(keep.all()):
    df = df.loc[keep]        # ← 布尔索引 = 整块拷贝 6.75GB
```

**只要调用方传了 `end` 就会被裁一次**。`harness_mine` 会传 `resolve_test_end()`，
于是每次加载都拷贝一份完整面板，memmap 零拷贝彻底白费。

证据：裁剪后 `rows=7,866,010` < 矩阵 `7,949,486`。

> 这也解释了为什么之前 `verify_e2e_v2` 测出来只有 0.28GB ——
> 那次传的是 `start=2020-01-01, end=None`，`keep` 全 True，不触发裁剪。
> **只测「加载」不测「带窗口的加载」，会漏掉这个坑。**

### 7E.3 修法：连续区间用 `iloc` 视图

面板按 `datetime` 排序，请求窗口裁剪本质是「掐头去尾」的**连续区间**：

```python
keep_np = keep.to_numpy(dtype=bool)
lo = int(np.argmax(keep_np))
hi_ex = len(keep_np) - int(np.argmax(keep_np[::-1]))
if hi_ex > lo and keep_np[lo:hi_ex].all():
    df = df.iloc[lo:hi_ex]     # 视图，仍与 mmap 共享物理页
else:
    df = df.loc[keep]          # 零散掩码才回退拷贝
```

### 7E.4 实测（harness_mine 全列 8 因子，train 段）

| 指标 | 修复前 | **修复后** |
|---|---|---|
| 面板加载 | 5s | **1s** |
| 加载后 rss | 7.063GB | **0.383GB** |
| 加载后 private | 7.549GB | **0.87GB** |
| 加载后 peak | 14.017GB | **0.703GB** |
| **全程峰值 private** | **8.238GB** | **1.558GB** |
| 全程峰值 peak_wset | 14.017GB | **3.786GB** |

8 个因子串行评估，全程峰值 private **1.56GB**、peak_wset **3.79GB**。
按此推算 6 路并发挖掘（并发实测只额外 +0.7GB）整体峰值约 2.5–3GB。

### 7E.5 顺带给 `harness_mine` 加内存自报

CLI 跑完会打印：

```
[mine] 面板加载后内存: rss=... private=... peak=...
[mine] 内存峰值: rss=... private=... peak_wset=...（判断压力看 private）
```

以后验证内存优化不必再外部采样（本轮已经踩过"采错进程"的坑：只采到 6MB 的
launcher，而真正干活的是子进程）。

---

## 七·F、全仓内存审计（2026-11-21，系统性扫描结论）

修复收敛后做了一次全仓扫描 + 实测，确认**面板层已无可观优化空间**。

### 7F.1 扫描方法与范围

| 扫描项 | 目标 | 结果 |
|---|---|---|
| **A. 单列全量扫描** | 在行主序矩阵上跨行扫单列会 fault 整文件（stride 920B < 4KB 页） | 仅 `_column_all_nan` 一处，**已改为查 attrs 预计算表**；其余无 |
| **B. float64 上转** | 面板 float32 → float64 会翻倍 | 140+ 处，但**全是 numba kernel 临时数组 / 指标计算的必要精度**；单次 25–64MB 量级，无整表上转 |
| **C. 整块拷贝/拼接** | `df.copy()` / `pd.concat` / `np.vstack` | 无新的**整份面板**拷贝；`submit.py:104`（库合并）、`zoo`（索引列）、`resample.py:179`（几列）均为一次性小量 |
| **D. 缓存上界** | 无界缓存会随 run 增长 | 全部有界：`WorkerPanelStore.max_cached=2`、`max_splits_per_session=8`、`_SLICES_CACHE_MAX=16`、`ops_kit._LAYOUT_CACHE_MAX=4`、`aux_cache._MAX_CACHE_ENTRIES` |
| **E. 多 session 累积** | 每 session 一份 panel | `_cache` 是 LRU(2)，超出即淘汰并清 split 缓存 ✓ |

### 7F.2 并发放大实测（此前从未量过）

挖掘主链路是 `--max-parallel-eval 6`，此前"并发每路 ~2GB"是估算。实测
（train 段 3,151,819 行，8 个因子，DSL 求值层）：

| 阶段 | private | rss | peak_wset |
|---|---|---|---|
| baseline | 0.540GB | 0.029GB | — |
| 加载 full panel | 0.733GB | 0.282GB | 0.571GB |
| 切 train（视图） | 0.733GB | 0.282GB | 0.571GB |
| **串行 8 个** | 1.407GB | 3.178GB | 3.284GB |
| **并发 6 路 8 个** | **1.535GB** | **3.305GB** | **3.693GB** |

**并发放大只有 +0.128GB**（1.407 → 1.535），并发 6 路比串行快 2 倍（3.6s → 1.8s）。
—— 历史"并发每路 2GB"的估算彻底作废。

### 7F.3 rss 3.3GB 的构成（不是浪费）

`rss` 3.3GB 中，面板文件页只有 0.28GB，其余 ~3GB 是**评估真正读过的列所触发的
文件页**——因子计算本来就要读这些数据，属必要开销，且是**可回收的文件页**
（不是私有）。判断压力看 `private`：**1.5GB**。

### 7F.4 其余已确认项

- `WorkerPanelStore` 的 docstring 曾误称"基于 Arrow IPC"，已订正（实际走
  `load_panel_from_cne`/`load_panel`，matrix 优先）；
- 缓存目录中仍有 2 组**非本轮**生成的旧缓存（`41ae6c85` 2021-04~2024-12 的
  4.96GB、`7b41f8f2` 无基本面小窗口 2.07GB），属其他任务的产物，未动。

### 7F.5 结论

| 层 | 状态 |
|---|---|
| 面板加载 | ✅ 已到极限（private 0.73GB，其中绝大多数是索引） |
| 并发放大 | ✅ 实测 +0.128GB，不是瓶颈 |
| 评估计算 | ⚠️ ~0.7GB/批，属必要精度与中间量，改它要动 numba/指标内核，收益 < 0.5GB |
| submit 路径 | 未实测（需真提交）；共享同一 panel + tradable ~300MB，估算 ≤ 1.5GB |
| 缓存/多 session | ✅ 全部有界 |

**建议：面板层停止优化。** 若后续仍嫌内存高，方向应转向「减少评估并发」
（牺牲时间）或「缩小数据面/窗口」，而不是继续抠面板。

---

## 八、证据索引（本轮实测，可复现）

探针脚本（仓库外，未污染仓库）：

| 脚本 | 用途 |
|---|---|
| `%TEMP%\memprobe_panel.py` | 面板加载 / touch / 并发 eval 的 rss/private/peak |
| `%TEMP%\verify_enrich.py` | 否证 `_enrich_trade_flags` 是根因 |
| `%TEMP%\verify_pandas_blocks.py` | 否证 pandas `_stack_arrays`；验证 2D memmap 单 block |
| `%TEMP%\verify_raw_mmap.py` | 逐阶段隔离 arrow 读取（定位 `to_numpy` 为罪魁） |
| `%TEMP%\verify_to_numpy.py` | 逐列量化 +29.7MB/列私有拷贝斜率 |
| `%TEMP%\verify_2dmemmap_semantics.py` | 2D memmap 的 pandas 语义兼容性与零拷贝性质 |

代码定位：

| 文件:行 | 作用 |
|---|---|
| `alphaagent/data/adapters/panel_mmap.py:174-186` | ★ 每列 `combine_chunks().to_numpy()` → 29.7MB 私有拷贝 |
| `alphaagent/data/adapters/panel_mmap.py:188` | `pd.DataFrame(data, index=index, copy=False)`（此步无额外开销） |
| `alphaagent/data/adapters/cnequity.py:445-598` | `_find_cached_panel`：缓存命中与裁剪 |
| `alphaagent/data/adapters/cnequity.py:536-547` | arrow 优先、parquet 回退的物化点 |
| `alphaagent/data/adapters/cnequity.py:686-760` | `_save_cached_panel`：双格式落盘（P0-1 追加第三格式） |
| `alphaagent/data/adapters/cnequity.py:976-1003` | `_enrich_trade_flags`（已否证非根因，但单 block 后写列更安全） |
| `alphaagent/data/panel.py:208-238` | `_finalize_panel` 先 downcast 再 copy（`8639267`） |
| `alphaagent/factor/mining/eval/universe.py:108-110` | `apply_universe("all")` 零开销 |
| `alphaagent/factor/metrics/tradable.py:124-152` | submit 段 7 个 wide 矩阵（P2 目标） |

---

## 九、附录：float16 可行性评估（2026-11-14 实测，**否决整体方案**）

> 起因：用户提问"float32 能变成 16 吗，对精度应该没啥影响吧"。
> 实测结论：**精度确实不是问题，动态范围才是硬墙** —— 面板整体转 fp16 不可行。

### 9.1 范围审计（硬约束）

float16 指数仅 5 bit：正规数范围 `[6.10e-5, 65504]`。

**230 个数值列中 97 列（42%）直接溢出**（`|x| > 65504`），头部溢出列：

| 列 | max | 超 fp16 上限倍数 |
|---|---|---|
| `funda_total_assets` | 7.07e13 | **10.8 亿倍** |
| `fsi_total_assets` | 5.71e13 | 8.7 亿倍 |
| `funda_total_liabilities` | 5.27e13 | 8.0 亿倍 |
| `funda_total_equity` | 2.02e13 | 3.1 亿倍 |
| `tot_cap` | 4.13e12 | 6310 万倍 |
| `float_cap` | 3.27e12 | 4990 万倍 |
| `funda_total_revenue` / `fsi_revenue` | 3.32e12 | 5070 万倍 |
| `pe_ttm` | 5.39e5 | 8.2 倍 |
| `ps_ttm` | 2.17e6 | 33 倍 |

受影响面：**全部 52 个 `funda_*` + 11 个 `fsi_*` + 市值/成交额/估值类**。
溢出即变 `inf`，会直接污染因子值、IC 与换手计算 —— **不是"影响不大"，是结果作废**。

### 9.2 精度审计（不是瓶颈）

对**不溢出**的列做 `float32 → float16 → float32` round-trip（7,782,511 行）：

| 列 | median 相对误差 | p99 相对误差 | **截面 rank 相关** | rolling20 中位相对误差 |
|---|---|---|---|---|
| `label_1d_open_to_open` | 1.72e-4 | 4.29e-4 | **0.9999855** | 7.6e-5 |
| `label_5d_close_to_close` | 1.72e-4 | 4.30e-4 | **0.9999988** | 7.7e-5 |
| `label_20d_close_to_close` | 1.72e-4 | 4.31e-4 | **0.9999999** | 8.4e-5 |
| `dv_ttm` | 1.70e-4 | 4.32e-4 | **0.99999995** | 5.3e-5 |
| `dv_ratio` | 1.16e-4 | 4.24e-4 | 0.9909934 | 5.3e-5 |
| `is_trade` / `not_st` | 0 | 0 | **1.0** | 0 |

**精度结论**：rank 相关 0.99999+、rolling 误差 5e-5 —— **精度完全够**，对 IC 影响可忽略。
`dv_ratio` 的 0.991 是最低值，因该列存在 0 值附近取值（相对误差在近零处被放大），非 fp16 尾数本身问题。

→ **障碍是范围，不是精度。** 这也说明"降精度"这条路在数值意义上本来是对的，
只是被几个大数值列（财务/市值/成交额）一票否决。

### 9.3 收益核算（性价比不成立）

| 手段 | 面板内存 | 省 | 精度风险 | 工程成本 |
|---|---|---|---|---|
| 全 fp32（现状） | 6.67GB | — | — | — |
| 全 fp16 | 3.33GB | 3.33GB | ❌ 42% 列溢出，不可用 | 低 |
| 仅安全列 fp16（白名单） | 4.77GB | **1.90GB** | 低（rank 0.99999+） | 中（白名单维护 + 逐列验证） |
| `--no-fundamentals`（砍 63 列） | — | **1.87GB** | **零** | **零**（参数已存在） |
| **P0 行主序 2D memmap** | — | **≈7GB** | **零** | 中 |

★ 关键换算：**砍 1 列 = 29.7MB = fp16 两列的收益**（2 × 14.85MB）。
砍 `funda_*`（52）+ `fsi_*`（11）= 63 列 → 省 **1.87GB**，
**与"把 126 列转 fp16"收益相同，且零精度风险、零工程成本**。

P0 落地后内存 ≈ 访问列数 × 29.7MB（典型挖掘表达式触碰 5–20 列 ≈ 0.15–0.6GB），
fp16 的边际收益进一步降到 ~0.1–0.3GB —— **越往后越不值得引入白名单复杂度**。

### 9.4 若将来确实要更低精度（均不推荐优先）

- **bfloat16**：指数位同 fp32（不会溢出），但只有 8 bit 尾数（相对精度 3.9e-3，比 fp16 差 8 倍），
  且 numpy 需 `ml_dtypes`、pandas 不原生支持 → 工程成本高、收益同样 3.33GB，**不推荐**；
- **int16 定点 + per-column scale**：无溢出问题（scale 自适应），精度 ≈ `absmax/32767`；
  但要在 DSL 算子层做反量化，改动面远超 P0，**不推荐**。

**结论：float16 不进本方案。要省内存，优先 P0（零精度损失、省 7GB）与砍列（零风险、省 1.87GB）。**

### 9.5 附带发现（数据质量，独立于本方案）

实测澄清统计口径时发现（注意 `np.isfinite` 对 NaN 也返回 False，**inf 与 NaN 必须分开统计**）：

| 列 | NaN 占比 | 真 `+inf` | 真 `-inf` |
|---|---|---|---|
| `ret` | 2.47% | **3** | 0 |
| `dv_ttm` | **33.5%** | 0 | 0 |
| `pe_ttm` | **23.1%** | 0 | 0 |
| `label_20d_close_to_close` | 3.81% | 0 | 0 |
| `amount` / `tot_cap` / `turnover_rate` | 2.39%–2.40% | 0 | 0 |

- `ret` 的 3 个 `+inf` 量极小但**确实存在**，建议单独排查除零来源（可能是复权价为 0 的除权日）；
- `dv_ttm`/`pe_ttm` 的 NaN 属既有已知覆盖问题（非本方案引入），但会拉低 coverage 门判定余量。

---

*本方案基于 2026-11-14 实测。分支：`feat-panel-memory-optimization`。所有改动须在该分支实施并过 OCR review 后合并。*
