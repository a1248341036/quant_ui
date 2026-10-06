# OCR review：`feat/submit-split-materialize` 自有工作（S1 分段物化 + A/B 根因修复）

## 背景与范围

对 `feat/submit-split-materialize` **本分支自有的两个提交**做 `ocr review`（open-code-review，
skill `open-code-review`）。范围刻意定为 `e5cec85..b1ae7d0`：`e5cec85` 及更早是本分支合入的
aux-cache 修复（已于 2026-10-06 单独评审并修复），不重复审。

| 项 | 值 |
|---|---|
| 命令 | `ocr review --audience agent --from e5cec85 --to b1ae7d0 --background-file <ctx> --exclude "docs/**" --output <tmp>` |
| 被审提交 | `ee2eb05`（S1 分段物化：`submit.py` +62/−1）、`b1ae7d0`（A `ensure_sorted` 短路 + B `winsorized_ic` 单遍化，5 文件） |
| 范围 | 6 个代码文件（+134/−15）：`data/panel.py`、`factor/align.py`、`factor/ingest.py`、`factor/metrics/{__init__,ic}.py`、`factor/mining/delivery/submit.py` |
| OCR 结果 | **9 findings / 6 文件**，4m40s，session `05029930-27e3-469d-92a7-b469227e64a0`；原始产物 `%TEMP%\ocr_split_materialize_out.txt` |
| 复核方式 | OCR 每条回源核对（read + grep），另写 3 个独立验证脚本实测（见下），**证伪 2 条** |
| 范围外 | 评审后分支又前进了 `03e517c`（prompt schema 修复，不属本次范围） |

## 结论

**verdict：`merge_after_fixes`** —— 三项改动的核心机制经独立实测站得住（B 逐位等价、A 的缓存假设成立、
S1 的边界语义可复现），但存在 1 条需要**拍板**的口径问题（M1）、2 条 medium（可观测性、排序预处理不一致）
与 4 条 low；另有 2 条 OCR 误报已用实测证伪。

| 级别 | 条数 | 是否需先修 |
|---|---|---|
| Medium | 3 | M1 需拍板口径；M2/M3 建议修 |
| Low | 4 | 建议随手修（L1 一行） |
| 误报 | 2 | 无需处理（已完成实测反驳） |

---

## Medium

### M-1（本 reviewer 补充，OCR 未报）S1 改变了「物化值」本身，与库入库路径形成两套口径

**位置**：`submit.py:549`（`materialized = _materialize_split_aware(...)`）→ `:557` `values_by_key`
（落库值）；对照路径 `ingest.prepare_stored_values` → `materialize_to_canonical`（全量求值），被
`alphaagent/factor/ingest.py:288 ingest_factor` 与 `backend/alphaagent_service.py:2130` 使用。

**实测证据**（`%TEMP%\verify_split_semantics.py`，真实 DSL `TS_MEAN($close, 5)`，60 日 × 20 只，
train 1/1–2/9、val 2/12–3/15、test 3/18–4/30）：

```
总行 1200 | 差异行 160            ← 8 个交易日 × 20 只
差异日 = val 段首 4 日（02-12..15）+ test 段首 4 日（03-18..21）   ← 恰为 window-1 = 4
例：02-12  split-aware=8.843170  全量=9.877253
```

**含义**：分段求值让 val/test 段首的 TS 窗口被截断在段内，**数值改变**（并非提交信息所写「由有值变 NaN」——
只有 `min_periods==window` 的算子才会变 NaN；`TS_MEAN` 这类首日仍出值）。因此：

- 同一表达式经 **submit**（candidate 记录 / `values_by_key`）与经 **`ingest_factor`/因子实验室**
  （库 canonical 值）会得到**段界处不同的值**，且两套口径都合法地存在于仓库里；
- 提交信息把该差异描述为「段头 window-1 天值从有值变 NaN」，与实际行为不符（会误导后续排查）；
- submit 的 `values_fp`（sha1 of cand_values）随之改变 → ingest 缓存键、相似度基线样本口径跟着变。

**建议（需拍板）**：① 若认定「提交口径 = 引擎分段口径」是对的，就把该差异写进
`docs/` + `_materialize_split_aware` docstring（含「库入库路径仍是全量口径」这一事实），并修正提交信息描述；
② 若要求「同一表达式全库一致」，则让 `prepare_stored_values`/`materialize_to_canonical` 共用同一物化语义
（或对段界行做一致化处理），否则 promote/rescreen/实验室入口与 submit 会长期分叉。

### M-2 回退路径全静默：优化是否生效在生产完全不可观测

**位置**：`submit.py:97`（`except Exception: return None`）→ `submit.py:550-551` 回退全量

函数内主动 `raise` 的诊断（`TypeError: 因子输出须为 Series`、`ValueError: 三段切片均为空 / 未完全覆盖`）
与 `eval_factor`/`FactorValueCache` 内部异常**全部被吞**，无日志、无计数。若 S1 因表达式类型或会话配置长期
失效（每次都回退），生产端只会「内存峰值回到 22GB」而没有任何信号，也无法确认优化真实生效——
而 S1 的全部价值就在这条路径上。建议在回退点加一条 `warning`/`log_step`
（`submit.py` 已有 `log_step` 惯例，`factor/cache.py` 的失败路径亦有 warning 先例），至少记录异常类型。

### M-3 `winsorized_ic` 未对齐 `evaluate_cs_on_panel` 的排序预处理（潜在口径分叉）

**位置**：`ic.py:304-326`（新函数）对照 `ic.py:215`（`evaluate_cs_on_panel` 入口 `panel = ensure_sorted(panel)`）

`evaluate_cs_on_panel` 先保证面板有序；`winsorized_ic` 没有，直接依赖调用方。实测（`%TEMP%\verify_ocr_findings.py`，
`holding_days=20`）：

```
sorted    old=0.13303844754474523  new=0.13303844754474523  identical=True
unsorted  old=-0.08668568735806184 new=0.08353509289958323  identical=False   ← 符号相反
```

根因：索引非单调整时 `_day_slices` 返回 `None`（实测确认），`cross_sectional_ic` 回退
`groupby(sort=False)` 的「首次出现日序」，`cs_ic_summary` 的 `iloc[::hold]` 便取到不同日点。
当前唯一调用方（`ingest.compute_ingest_metrics`）传入的是 `ensure_sorted` 后的面板，**故未触发**，属潜在风险。
注意修复要**同时**声明前置条件：`values` 必须已按 `panel.index` 行序对齐——否则照抄 `ensure_sorted`
会复现 `evaluate_cs_on_panel` 侧「排序面板 + 未排序 values」的错位（该错位是既有问题，非本分支引入）。

---

## Low

### L-1 新公开入口未进 `__all__`（OCR #1，已实测确认）

`alphaagent/factor/metrics/__init__.py:207` 新增 `winsorized_ic`，但文件底部 `__all__`（35 项，含
`evaluate_on_panel`）**没有它** → `from alphaagent.factor.metrics import *` / 按 `__all__` 枚举 API 的工具拿不到。
建议补入（一行）。

### L-2 函数内 4 个 import 位于 `try` 之外，与 docstring 承诺不符（OCR #6）

`submit.py:63-66` 的 `eval_factor` / `collect_aux_intervals_from_expr` / `align_series_to_panel` /
`MaterializeResult` 在 `try:`（`:68`）之前。docstring 承诺「任一异常 → 调用方回退全量物化」，但符号缺失/改名时
会抛 `ImportError` 击穿 `submit()`，产生旧代码不存在的硬失败点。建议移入 `try` 或修正 docstring 承诺。

### L-3 `ensure_sorted` 把「拷贝隔离」换成「只读约定」，建议写进契约（OCR #3，与我结论一致）

`panel.py:112-124` 已排序时**返回入参本身**；配合 `slice_panel` 的 iloc 视图快路径（`panel.py:88-101`，
明示共享数据块），`compute_ingest_metrics` 拿到的 `eval_panel` 现在是**长生命周期 session 面板的视图**；
改动前 `sort_index()` 无条件拷贝，视图落在一次性副本上。已逐处核对 7 个短路点当前全为只读
（`grep 'inplace=True|panel\[...\]=|eval_panel\[...\]='` 的命中只在 panel 构建与 CNE adapter），
但该安全性属**隐性契约**。建议在 `ensure_sorted` docstring 明确「返回对象可能与入参共享底层数据，调用方不得原地修改」。

### L-4 覆盖校验只比行数，建议改为索引集合级（OCR #5，但「静默漂移」分支已被证伪）

`submit.py:86` 用 `sum(len(p) for p in parts) != len(panel)` 判断三段是否完整覆盖。OCR 担心
「重叠 + 缺口」平衡组合会漏过并静默取首个标签/填 NaN。**实测反驳**：pandas 2.3.3 下含重复标签的
`Series.reindex` 抛 `ValueError: cannot reindex on an axis with duplicate labels`（`verify_ocr_findings.py`），
被 `:97` 兜住 → 安全回退全量，**不存在静默错值通道**。故这是显式化/健壮性建议而非缺陷：
可改为 `merged.index.is_unique and len(merged) == len(panel)`，把隐式守卫变成显式断言。

---

## 误报 / 已证伪

| OCR 结论 | 实测 | 判定 |
|---|---|---|
| `panel.py:0-0`「MultiIndex 的 `is_monotonic_increasing` 并不总是缓存，7.78M 行可能重复 O(n) 扫描」 | 7,800,000 行 MultiIndex：首次 **37.8ms**，第 2/3 次 **0.0ms**；`idx._cache` 键含 `is_monotonic_increasing`（pandas 2.3.3） | **误报**：对象级缓存成立，docstring 主张正确 |
| `submit.py:86-86`「reindex 可能对重复标签取第一次出现、缺口行填 NaN → 因子值静默漂移」 | 重复标签 reindex **抛 ValueError**（见上） | **该分支证伪**（长度守卫的隐含安全性成立，仅建议显式化 → L-4） |

## 已验证无问题（正向结论）

1. **B 的单遍化等价性成立**：`winsorized_ic(values, panel, ...)` ≡ `evaluate_on_panel(cross_sectional_winsorize_values(...))["ic"]`，
   长尾 float32（t 分布 ×1e3）+ 3% NaN 面板、`holding_days=1/20` 均**逐位一致**
   （`%TEMP%\verify_split_materialize.py`）。根因：`cs_ic_summary` 的 `"ic"` 只由 `daily_ic` 决定，
   `daily_rank_ic` 仅影响 `"rank_ic"`（`ic.py:177-201`）；`min_ic_pairs` 默认 5 与
   `evaluate_on_panel(min_ic_pairs=5)` 同值，`_day_slices` 在两侧包装层均注入。
2. **A 的缓存主张成立**：见上表（37.8ms → 0）。
3. **A 的零语义变化成立**：对已排序 MultiIndex 而言 `sort_index()` 是恒等；`align_series_to_panel` 用同一对象
   → `reindex` 结果不变。
4. **S1 的构造完整**：`MaterializeResult` 仅 `values/expr/aux_tags` 三字段（`factor/types.py:110-114`），
   新路径三字段齐备；`align_series_to_panel` 负责行序，拼接顺序无关。
5. **S1 的 gap/overlap 都会安全回退**：缺口 → 行数不等 → 回退；重叠 → 重复标签 `reindex` 抛错 → 回退（M/L-4）。

## 缺失的测试（本分支三项改动**零提交测试**）

`grep` 确认 `tests/` 下无 `_materialize_split_aware` / `ensure_sorted` / `winsorized_ic` / `split_aware`
任何引用；两个提交信息里声称的「小面板真实 DSL 单测」「逐位一致」都不可从仓库复现（OCR #9 只点了
`winsorized_ic` 一条，实际三项全缺）。按项目「口径改动必须同步单测」的约定，建议补：

1. **S1**：小面板 + 真实 DSL，断言 ① 截面表达式逐位等于全量 `materialize_factor`；② TS 表达式差异**恰好**
   落在各段首 window-1 日（把 M-1 的语义变化钉成显式契约）；③ 三段行数和 ≠ len(panel) 时回退全量；
   ④ `eval_factor` 抛错时回退全量且不抛。
2. **A**：`ensure_sorted` 对已排序面板返回**同一对象**（`is` 断言）、对乱序返回排序副本且不改动入参。
3. **B**：`winsorized_ic(...) == evaluate_on_panel(cross_sectional_winsorize_values(...))["ic"]`，
   覆盖 `holding_days=1/20`（并顺带锁住 `min_ic_pairs` 默认值的同步，即 OCR #2 的漂移风险）。

## 处置建议

| 方案 | 内容 | 代价 |
|---|---|---|
| A（推荐，最小） | L1（补 `__all__`）+ L2（import 入 try）+ M-2（回退加一条 log_step）+ 上述 3 条测试 | 小，无口径变化 |
| B | A + M-3（`winsorized_ic` 对齐排序预处理 + 声明 values 行序前置条件）+ L-3/L-4 契约显式化 | 小，行为不变 |
| C | B + M-1 拍板：决定 submit 与库入库是否统一物化口径，并修正提交信息/文档描述 | 需你决策，可能影响既有候选值口径 |

> 本文件记录评审结果与后续修复；评审脚本（`%TEMP%\verify_{split_materialize,split_semantics,ocr_findings}.py`）
> 均只读、不入库。

## 修复落地（2026-10-07，方案 A + B + L-1）

**提交**：`6049928`（分支 `fix/submit-materialize-review`，基于本分支 `4c56e17`）。

| 编号 | 处理 | 落地位置 |
|---|---|---|
| M-1 | **口径显式化**（不改行为）：docstring 记实测差异（`TS_MEAN(5)` 160/1200 行不同、恰为 val/test 段首各 4 日、**非 NaN**），写明 candidate 落库值/`values_fp` 与库入库路径（`prepare_stored_values` 全量口径）在段界**故意不同**及不统一的原因（库路径拿不到 session 分段窗口）；调用点补注释；回归测试把该语义钉死 | `submit.py:45`、`tests/test_split_materialize.py::test_ts_expr_differs_only_at_segment_heads_and_is_not_nan` |
| M-2 | 两条路径都落 `log_step("submit.materialize", name, mode="split"/"full", reason=...)`；4 个函数内 import 移入 `try`（兑现 docstring 的"任一异常即回退"承诺） | `submit.py:45-120` |
| M-3 | `winsorized_ic` 入口**显式拒绝**非单调 panel（不做排序，避免 values 与索引错位），docstring 声明 values 行序前置条件 | `ic.py:304-330`、`tests/test_ensure_sorted_and_winsorized_ic.py::test_winsorized_ic_rejects_unsorted_panel` |
| L-1 | `winsorized_ic` 补入 `__all__` | `metrics/__init__.py:290` |
| L-2 | 随 M-2 一并修（import 入 `try`） | `submit.py` |
| L-3 / L-4 | **未做**（契约显式化建议，非缺陷） | — |
| M-1 的「统一口径」选项 | **未做**：库入库路径无 session 分段窗口，强行统一 = 改库值口径，需单独决策 | — |

**验证**（独立 worktree `.worktrees/submit-materialize-fix`，PYTHONPATH 指向该 worktree）：

- 新增 **11 个用例全绿**：`tests/test_split_materialize.py`（5）+ `tests/test_ensure_sorted_and_winsorized_ic.py`（6，含 hold=1/20 逐位一致、乱序拒绝、`ensure_sorted` 同一对象短路）。
- 相关回归 **16 文件 / 136 用例：134 passed / 2 failed**；2 项均为 `test_depth_curve` 的**既存失败**（基线 `a82c273` 同样失败，非本次引入）。
- 行为差异仅两处：① 新增 `submit.materialize` 步骤日志；② 非单调 panel 传给 `winsorized_ic` 时由「静默分叉」变为 `ValueError`（生产唯一调用方传的是已 `ensure_sorted` 的面板）。
