# 记忆层算子清单真源修复 · 审计与决策记录（2026-10-18）

> 分支：`fix/memory-operator-inventory`（从 `main` 新建）
> 变更：`memory/expressions.py` 算子识别真源化 + `$` 前缀消歧；`docs/adding_dsl_operator.md` 触点 #11；新增 `tests/test_memory_operator_inventory.py`
> 触发：外部（ChatGPT）对「冷门单独通道」的评审结论为「系统性原因 —— 因子全部挖掘完了」。本记录是该结论的复核结论与其被证伪的部分。

---

## 0. 结论

**「冷门空间已 100% 枚举」作为"覆盖"陈述成立；作为"挖完了（无 alpha 可挖）"陈述不成立。**
并且复核过程中定位到一条**真实的系统性缺陷**：记忆/引导层的算子清单对 22/22 个冷门算子
结构性失明 —— 而「冷门算子使用率 <2%」这个反收敛 spec 的立项前提，正是这台失明仪器测出来的。

---

## 1. 冷门通道并不空转

| 指标 | 值 |
|---|---|
| 冷门条目 | 126 = `cold_approved` 13 + `cold_dead` 113（观测 262） |
| 通过率 | 13/126 ≈ **10.3%**（spec §7.3 预估 3-8%） |
| stage | **全部停在 `train`** |
| 入库 | **0**（`candidate_main`/`production_main` 无 `ct_*`、无 `cold_track-approval`） |

死亡证明（DB 实测）：`underpowered` **71（62.8%）** / `no_alpha` 36（31.9%）/ `data_gap` 6（5.3%）。

## 2. 「挖完了」的四点证伪

1. **62.8% 是"测不动"不是"没有"**：`underpowered` 按 spec §7.3 定义即"样本不足、不入永久负证据、
   允许重测"；报告写成"同数据重跑必复现"是循环论证 —— 复现的是无统计功效，不是无 alpha。
2. **覆盖审计自证循环**：R24「22/22 `_COLD_OPS` 零未覆盖」的搜索空间就是这 22 个名字本身。
3. **两条族死在下游门槛**：`KLINE_GEOMETRY`(proxy 0.59)、`VOLUME_CLOCK_VPIN`(0.359) 被
   `crowd_proxy_corr < 0.3` 拦下 —— 正是 spec §9 R2 自己挂的开放风险，属门槛策略而非 alpha 缺失。
4. **6 条 `data_gap`** 是"测不了"（FCF 覆盖断崖 0.43，待数据回填），属未开采。

## 3. 真正的系统性缺陷（本分支修复对象）

### 3.1 位置与实测

`memory/expressions.py` 原第 200 行 `if name in _KNOWN_OPERATORS`：算子识别只认一份**硬编码
57 项白名单**，清单外算子静默丢弃。

```
_KNOWN_OPERATORS: 57
_COLD_OPS:        22
交集:             []      ← 22/22 全缺
```

顺带缺失：`CS_BUCKET` / `CS_GROUP_RANK` / `NEG` / `SIGNAL_BLEND`。
真源一直存在：`dsl.core.operators` 自动收集 **135** 个算子名。

### 3.2 落库影响面

| 事实 | 数值 |
|---|---|
| 含冷门算子的表达式 | **16 324** / 66 829 条（≈24%） |
| 其中 `operator_list_json` 正确记录 | **0** |
| 逐算子 | `CHIP_PEAK_LOC` 3583→0、`CHIP_COM_W_GAP` 3508→0、`CROWD_SHARE` 803→0 … |

`CHIP_PEAK_LOC` 还因撞 `chip_*` 变量前缀规则被**误判为变量**（既丢算子身份又污染变量集）。
`analytics._ops_of` 有正确 regex 兜底，但仅在 `operator_list_json` 为空时触发，而 ingestion
每次写入的都是"非空但已丢冷门算子"的列表 → 兜底永不触发，失明是永久性的。

### 3.3 症状仿真（修复前）

复刻 `_operator_diversity_block` 统计口径，输入 20 个**全部以冷门算子为信号根**的尝试：

```
signal_counter = {'MAX': 20}     ← 冷门算子 0 条
Top3 集中度 = 100.0%            ← 被抬高，触发软引导注入
推荐清单 = TS_TREND_RANK / CROWD_SHARE / TS_LAST_TOPFRACTAL / ...（全是刚用过的）
```

即 **100% 使用冷门算子时，统计器读到的使用次数仍为 0**，并把刚用过的算子当「从未用过」推荐回来。

### 3.4 边界（未受影响的部分）

冷门赛道**判定与硬闸走 AST**（`cold_track.signal_root_op` → `dsl.core.ast` 的
`NORMALIZE_OPS`/`SMOOTHING_OPS`），不经过 `_KNOWN_OPERATORS`。被污染的只有记忆/引导统计层 ——
但那恰是"软引导为何失效"所在的层。

### 3.5 流程根因

`docs/adding_dsl_operator.md` 的触点清单原先没有 `memory/expressions.py`。本分支补为 #11。

## 4. 修复方案

- `_operator_names()`：`DSL 注册表（build_operator_namespace）∪ 静态回落白名单`，进程内缓存。
  注册表不可用时回落旧行为（不报错、不静默变空）。
- **`$` 前缀消歧**：`chip_*`/`crowd_*` 在本仓既作筹码/拥挤面**字段**（`facets.py` 按 `chip_` 识面），
  又作**算子名**。`_FUNC_CALL_RE` 加 `(?<!\$)`、变量循环对 `$` 前缀直接按字段处理 →
  `$chip_peak_loc` 是字段、`CHIP_PEAK_LOC(...)` 才是算子（修复期由 `test_sspm_refactor.py`
  的 motif 断言抓到的真实回归点）。
- 不新增配置项（无阈值/开关），不触碰门槛与判定口径。

## 5. 验证

- 新增 `tests/test_memory_operator_inventory.py`（11 用例）：冷门/套壳算子全覆盖、
  **防漂移守卫**（注册表任何算子都必须被识别）、提取语义、`$` 消歧、回落语义、
  以及**端到端症状回归**（引导块必须看见冷门算子且不再重复推荐已用算子）。
- 记忆/挖掘相关 14 个测试文件批量回归：除 `test_facet_focus_lock.py` /
  `test_facet_focus_scope.py` 的 **3 项在 `main` 基线即失败**（已用 stash 对照确认与本次改动无关）
  之外全部通过。

## 6. 复现

```powershell
# 症状（修复前 0/22，修复后 22/22）
D:\Quant\quant_ui\.venv\Scripts\python.exe -c @"
from alphaagent.factor.mining.memory import expressions as E
from alphaagent.factor.mining.memory.retrieval import _COLD_OPS, _SHELL_OPS
n = E._operator_names()
print('ops:', len(n))
print('cold missing:', sorted({o.upper() for o in _COLD_OPS} - {x.upper() for x in n}))
print('shell missing:', sorted({o.upper() for o in _SHELL_OPS} - {x.upper() for x in n}))
"@

# 全库影响面（只读）
#   select count(*) from memory_entries where upper(expression) like '%CHIP_PEAK_LOC%'      -> 3583
#   select count(*) from memory_entries where upper(operator_list_json) like '%chip_peak_loc%' -> 0（修复前）

# 回归
D:\Quant\quant_ui\.venv\Scripts\python.exe -m pytest tests\test_memory_operator_inventory.py -q
```

## 7. 遗留（未在本分支处理）

1. **存量 `operator_list_json` 未回填**：66 829 行存量仍是"缺冷门算子"的旧口径
   （`memory/backfill.py` 已有重算入口，需另行决策是否批量回填 + 是否重算历史统计）。
2. 71 条 `underpowered` 的冷门条目待修好仪器后重测（spec §7.3 本就允许）。
3. 13 条 `cold_approved` 中 11 条只有 `coverage`、无证据指标，且 `verdict=approved` 与
   `fail_detail=cold_track:no_alpha` 自相矛盾（早于 R23 修复、从未回填）。
4. `store_meta` 无骨架快照签名（spec §5.3 要求）。
