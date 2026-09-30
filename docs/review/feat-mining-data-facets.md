# Review: feat/mining-data-facets

**分支**: `feat/mining-data-facets`（2 commits ahead of main，merge-base `9ea32fe` = main HEAD）
**提交**:
- `d8cfb97` feat(data): 接入 5 个研报要用的数据面插件（行业/公告/研报视角/财报明细/指数成分）
- `7e41de5` feat(mining): 研报课题要用的字段面接入（提示词字段表/别名/宽表列/基本面自动载入）

**范围**: 16 files, +1194/-44。新增 5 个数据面插件、提示词字段表扩展、`$turnover`→`$turnover_rate` 别名、宽表列映射补齐、研报课题基本面联动载入、2 个回归测试。

**结论**: **merge_after_fixes** —— 1 个 REAL defect（提示词承诺了插件不提供的字段，会复现本分支要消除的"不可用字段白报"），修复后可合并。

---

## 变更概览

| 模块 | 文件 | 变更 |
|---|---|---|
| 数据插件 | `adapters/plugins/industry.py` (92L) | 新增。SW 行业 PIT，`industry_sw_l1`/`industry_sw_code` |
| 数据插件 | `adapters/plugins/announcement.py` (156L) | 新增。公告密度（5d/20d）+ 7 类事件打标 + 距最近公告日 |
| 数据插件 | `adapters/plugins/research_views.py` (218L) | 新增。一致预期（ac_*）+ 研报预测（rc_*）双源 PIT 合并 |
| 数据插件 | `adapters/plugins/financial_items.py` (122L) | 新增。财报明细项 PIT（锚点=公告日），pivot 宽表 |
| 数据插件 | `adapters/plugins/index_membership.py` (102L) | 新增。指数成分标志（szcomp/chinext/in_index） |
| 宽表 | `adapters/cnequity.py` (+2) | `_CACHE_SCHEMA_VERSION` 7→8（新列触发缓存失效） |
| 宽表 | `adapters/plugins/stock_daily_wide.py` (+9) | column_map 补 8 列（pre_close/change/pct_chg/up_limit/down_limit/suspend_timing/suspend_type/listed_days） |
| DSL | `dsl/eval.py` (+13) | `_FIELD_ALIASES = {"turnover": "turnover_rate"}` + `_apply_field_aliases` 在编译前改写 |
| 提示词 | `factor/mining/prompt/modules/data_fields.py` (+79) | 3 个新 section（事件面/业绩面/基本面）+ 变量表补行 |
| 后端 | `backend/alphaagent_service.py` (+21) | `--focus-facets` 透传 + `_question_bank_needs_fundamentals()` 自动载入基本面 |
| 测试 | `tests/test_data_plugin_research_facets.py` (89L) | 新增。industry/announcement/financial_items PIT 回归 |
| 测试 | `tests/test_field_aliases_and_fundamentals_auto.py` (58L) | 新增。别名改写 + 题库基本面判定 |
| fixture | 4 个 system_prompt_*.txt | 字段表同步 |

---

## REAL Defects

### D1. research_views 提示词列了 `$ac_target_price`，但插件不产出该列 → "不可用字段"白报

**位置**:
- 提示词：`alphaagent/factor/mining/prompt/modules/data_fields.py` `_RESEARCH_VIEWS_SECTION_MD`（本分支新增，列 `$ac_target_price` "一致目标价（元）"）
- 代码：`alphaagent/data/adapters/plugins/research_views.py:36-38`

**证据**:

插件实际输出列 `_ALL_COLS`（research_views.py:36-38）：
```python
_AC_VALUE_COLS = ["ac_eps_fy", "ac_pe_fy", "ac_rating", "ac_analyst_count"]
_RC_VALUE_COLS = ["rc_eps_y", "rc_pe_y", "rc_roe_y", "rc_tp_wan", "rc_target_price", "rc_rating"]
_ALL_COLS = [*_AC_VALUE_COLS, "ac_days_since", *_RC_VALUE_COLS, "rc_cnt_90d", "rc_days_since"]
```

`_ac_events`（research_views.py:75-86）只产出 `ac_eps_fy / ac_pe_fy / ac_rating / ac_analyst_count`，**没有 `ac_target_price`**。`load()`（research_views.py:212-214）对 `_ALL_COLS` 中缺失列补 `lit(None)`，但 `ac_target_price` 不在 `_ALL_COLS`，所以面板**永远不会出现该列**。

而 `compile_multi_line_factor`（dsl/eval.py:263-278）的字段校验：
```python
known = {str(col).lstrip("$").split("@", 1)[0] for col in columns}
...
unknown = sorted({
    match.group(1) for match in _DOLLAR_REF_RE.finditer(cleaned)
    if match.group(1) not in known
})
if unknown:
    raise MultiLineFactorEvalError(
        f"symbol 阶段失败: 表达式引用了不可用字段: {', '.join('$' + name for name in unknown)}",
        ...
    )
```

`columns` 是面板实际列。面板无 `ac_target_price` → `known` 不含它 → LLM 按提示词写 `$ac_target_price` → raise "不可用字段"。

**影响**: 本分支的动机（见 test_field_aliases_and_fundamentals_auto.py:1-8 注释）正是消除"课题/记忆推荐字段在面板里不存在 → 白报不可用字段 → 白丢一次评估"。research_views 提示词承诺 `$ac_target_price` 却不提供，**在新插件里复现了同一问题**。研报课题若点名"一致目标价"，LLM 会按提示词用 `$ac_target_price`，每次都白报。

**修复建议**: 二选一——
1. 从 `_RESEARCH_VIEWS_SECTION_MD` 删除 `$ac_target_price` 行（一致预期源本就无目标价，目标价在研报侧 `rc_target_price`）。
2. 若确需 ac 侧目标价，在 `_ac_events` 增加该列派生 + 加入 `_AC_VALUE_COLS`/`_ALL_COLS`。

推荐方案 1（语义上目标价属于 rc 侧，提示词已有 `$rc_target_price`）。

---

## 观察项（非 defect，不阻塞合并）

### O1. 测试覆盖缺口：research_views 和 index_membership 无测试

`test_data_plugin_research_facets.py` 只覆盖 industry / announcement / financial_items（L15 import）。research_views（最大、最复杂：双源 join、rating 映射、密度窗口）和 index_membership 无回归测试。research_views 的 `how="full", coalesce=True` 双源合并、`_rc_density` 的 90 日窗口、`_rating_score` 的 `replace_strict` 映射均无测试保护。

建议补 research_views PIT + 双源合并测试（合成 analyst_consensus + report_rc 两源，验证 ac/rc 列同时存在且 PIT 锚点正确）。

### O2. 提示词遗漏 `$rc_tp_wan`

代码 `_RC_VALUE_COLS` 含 `rc_tp_wan`（research_views.py:101，"预测利润总额（万元）"），但 `_RESEARCH_VIEWS_SECTION_MD` 未列。LLM 不会因此报错（少用一个列不是错），但研报课题若点名"预测利润"会少一个可用字段。建议补进提示词。

### O3. `_apply_field_aliases` 在 `_strip_string_literals` 之前执行

dsl/eval.py:262 `_apply_field_aliases` 先跑，L265 才 `_strip_string_literals`。若 DSL 表达式含字符串字面量且字面量内含 `$turnover`（如 `s = "x_$turnover"`），别名改写会误改字符串内容。

实际不可触发：`compile_multi_line_factor` 处理的是数值因子表达式，LLM 不会在因子表达式里写含 `$turnover` 的字符串字面量。且当前只有一个别名（`turnover`→`turnover_rate`）。降级为观察项。若未来别名增多或 DSL 引入字符串算子，建议把 `_apply_field_aliases` 移到 `_strip_string_literals` 之后。

---

## PIT 正确性核查（已验证无 defect）

### shift().over() 语义（announcement.py:124）

`shift(n).over("symbol")` 在 polars 1.43.2 实测：`over("symbol")` 让 shift 在组内生效（A 组 diff=[10,10,10]，B 组=[100,100]）。announcement 的窗口计数 `cum_sum().over("symbol")` - `shift(n).over("symbol").fill_null(0)` 语义正确。测试 test_announcement_counts_and_flags 验证了 5d/20d 累计 + flag 保持。

### pivot(aggregate_function="last") 顺序（financial_items.py:101-102）

实测 polars 1.43.2：`pivot(aggregate_function="last")` 取分组内最后一条（按当前行顺序）。financial_items 先 `sort(["symbol","_pit_date","item_code"])` 再 pivot，同 (symbol,_pit_date) 下不同 item_code 各自一行，无重复聚合到同一 cell 的场景。PIT 锚点=`announce_date`（非 report_period），test_financial_items_pit_anchor_is_announce_date 验证了"未公告前无行"。

### join_asof backward（_pitlib.expand_pit_daily + announcement/research_views）

`expand_pit_daily`（_pitlib.py:155-204）join_asof `strategy="backward"` by symbol，PIT 快照前不可见。announcement 自行实现的 join_asof backward（L147-155）首公告前 `_ann_date` 为 null → `ann_days_since` 为 null（test L68 验证）。research_views `_rc_density` 同理。

### expand_pit_daily 空数据行为

`_pitlib.expand_pit_daily` L178-179 对空事件 raise ValueError。industry/index_membership/financial_items/announcement 各自的 `load()` 在 read_curated 后、expand_pit_daily 前也 raise ValueError（如 research_views.py:187-188）。cnequity 调用方对插件 raise 的容忍度：research_views 采用 try/except 单侧源失败不拖垮另一侧（L176-177, L197-198），两源全空时返回零行列齐全帧（L200-207）避免缓存写入失败。其余 4 插件 raise 后由 cnequity 上层处理（与本分支前既有行为一致，非新引入）。

### 缓存 schema 版本

`_CACHE_SCHEMA_VERSION` 7→8（cnequity.py +2）。新列（ann_*/ac_*/rc_*/fsi_*/idx_*/ind_*）写入宽表，旧缓存面板缺这些列，bump 版本号触发重建。正确。

---

## 提示词与代码字段一致性（除 D1 外已核对）

| 提示词字段 | 代码来源 | 状态 |
|---|---|---|
| `$ann_cnt_5d`/`$ann_cnt_20d` | announcement.py `_CNT_WINDOWS` | ✓ |
| `$ann_flag_*`（7 类） | announcement.py `_FLAG_PATTERNS` | ✓ |
| `$ann_days_since` | announcement.py L154 | ✓ |
| `$ac_eps_fy`/`$ac_pe_fy`/`$ac_rating`/`$ac_analyst_count`/`$ac_days_since` | research_views.py `_AC_VALUE_COLS`+expand_pit_daily since_col | ✓ |
| `$ac_target_price` | **无** | ✗ D1 |
| `$rc_eps_y`/`$rc_pe_y`/`$rc_roe_y`/`$rc_target_price`/`$rc_rating`/`$rc_cnt_90d`/`$rc_days_since` | research_views.py `_RC_VALUE_COLS`+`_rc_density` | ✓ |
| `$rc_tp_wan` | research_views.py L101 | 提示词未列（O2） |
| `$fsi_*`（10 项） | financial_items.py `_ITEM_MAP` | ✓ |
| `$idx_szcomp`/`$idx_chinext`/`$idx_in_index` | index_membership.py `_FLAG_COLS` | ✓ |
| `$industry_sw_l1`/`$industry_sw_code` | industry.py | ✓ |
| `$turnover`→`$turnover_rate` 别名 | dsl/eval.py `_FIELD_ALIASES` | ✓ |
| 宽表补列（up_limit/down_limit/listed_days 等 8 列） | stock_daily_wide.py column_map | ✓ |

---

## 测试有效性核查

### test_data_plugin_research_facets.py（89L）
- L32 `monkeypatch.setattr(_pitlib, "_curated_root", lambda: tmp_path)`：`read_curated` 内部 `root = _curated_root()` 解析模块级名字，monkeypatch 模块属性生效。✓ Hermetic（不依赖本地 CNE 数据）。
- test_industry_is_pit_and_prefix_hierarchical：验证 PIT（快照前丢弃）+ `industry_sw_l1`=前 2 位 + `industry_sw_code`=全码。✓
- test_announcement_counts_and_flags：验证 5d/20d 累计 + flag 20d 窗口保持 + 首公告前 `ann_days_since` NaN。✓ 实测了核心 PIT 逻辑。
- test_financial_items_pit_anchor_is_announce_date：验证 PIT 锚点=公告日（非报告期）+ 未公告前无行。✓

### test_field_aliases_and_fundamentals_auto.py（58L）
- test_turnover_alias_rewrites_to_real_column：`$turnover`→`$turnover_rate`。✓
- test_alias_does_not_touch_real_columns：`$turnover_rate`/`$turnover_rate_f` 不被误改。✓
- test_unknown_field_still_rejected：`$not_a_real_field` 仍 raise "不可用字段"。✓
- test_question_bank_fundamentals_detection：题库含基本面→True，仅价量面→False，raise→False（退回旧行为）。✓

测试有效，确实在测被测逻辑（非纯 monkeypatch 桩）。

---

## 修复清单

| # | 严重度 | 项 | 修复 |
|---|---|---|---|
| D1 | 高 | 提示词 `$ac_target_price` 无对应面板列 | 从 `_RESEARCH_VIEWS_SECTION_MD` 删除 `$ac_target_price` 行（推荐），或在 `_ac_events` 派生该列并加入 `_ALL_COLS` |
| O1 | 低 | research_views/index_membership 无测试 | 补 PIT + 双源合并测试 |
| O2 | 低 | 提示词遗漏 `$rc_tp_wan` | 补进 `_RESEARCH_VIEWS_SECTION_MD` |
| O3 | 低 | alias 改写在 strip string literals 前 | 未来别名增多时调整顺序 |

**合并建议**: 修 D1 后合并。O1/O2/O3 可后续补。
