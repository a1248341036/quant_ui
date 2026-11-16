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
| 最优解 | **面板改行主序 2D memmap（`.npy`）**：已实测 `pd.DataFrame(2d_memmap, columns=..., copy=False)` → **单 block、真正零拷贝**（`shares_memory=True`、`iloc` 切片也是 view），且**新增列不再触发整块拷贝**。预期私有 7.79GB → **≈0**，常驻 RSS 由"实际触碰的列数 × 29.7MB"决定 |
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

*本方案基于 2026-11-14 实测。分支：`feat-panel-memory-optimization`。所有改动须在该分支实施并过 OCR review 后合并。*
