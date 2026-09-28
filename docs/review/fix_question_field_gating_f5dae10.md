# Branch Review: `fix/question-field-gating`

- **分支**: `fix/question-field-gating`
- **HEAD**: `f5dae10` — "fix(mining): 课题/记忆推荐按实际载入列门控 + $turnover 字段名收口"
- **Author**: zhoubw, 2026-09-28
- **merge-base with main**: `d0748ae`
- **Diff stat**: 16 files, +819/-30
- **Verdict**: **merge**

## 概述

本分支解决两个相关问题：

1. **课题派发不按实际载入列门控**：自由探索 run（如 `--no-fundamentals`）仍会派发纯
   `funda_*` 课题，模型引用即被工具层拦截，浪费一轮 LLM 调用。commit message 实测
   课题可执行率 40.6% → 85.2%。
2. **`$turnover` 字段名在 DSL 层未收口**：prompt/题库按自然名 `$turnover` 书写，但
   真实面板列是 `turnover_rate`，导致引用即报"不可用字段"。

修复策略：

- 新增 `alphaagent/dsl/core/field_aliases.py` 叶子模块，提供别名表 + 改写/缺列判定；
- `eval.py` 在编译期单点改写（只在目标列实际载入时改，否则保持原名让 unknown 检查拦截）；
- `question_queue.py` 按实际载入列门控课题派发（纯缺列题跳过、部分缺列题保留并注入提示）；
- `retrieval.py` 按实际载入列门控记忆推荐父本（缺列父本过滤，槽位由后续候选回填）；
- 配置阈值收口在 `research_spec`，`agentscope_run.py` 两个注入点接线 + `steps.log` 记录。

## 逐文件审查

### 1. `alphaagent/dsl/core/field_aliases.py` (NEW, +95)

叶子模块，无 `dsl/factor` 导入，循环导入风险低。提供：

- `FIELD_ALIASES = {"turnover": "turnover_rate"}`
- `bare_column(name)` / `canonical_column(name)` — 去 `$` / `@offset`、应用别名
- `available_columns(columns)` — 规范化可用列集合（空输入 → 空集合）
- `apply_field_aliases(expr, available) -> (rewritten, applied_tuple)` — 只在
  `target ∈ available` 时改写；右边界 `(?![A-Za-z0-9_])` 防止 `$turnover` 误命中
  `$turnover_rate`；`available` 为空时不改写（保持旧行为）
- `missing_fields(expr, available) -> tuple` — `available` 为空时返回空元组
  （由调用方决定旧行为），否则返回引用但未载入的字段（按别名归一后比较）
- `fields_available(expr, available) -> bool` — `not missing_fields(...)`

**判定**：正确。右边界正则、空 available 退化、别名归一比较均符合契约。

### 2. `alphaagent/dsl/eval.py` (+4)

在 `compile_multi_line_factor` 中，`known = {列名去 $/@offset}` 之后、
`_strip_string_literals` 与 unknown 字段检查之前，插入单点改写：

```python
multi_line_expr, _applied_aliases = apply_field_aliases(multi_line_expr, known)
```

两条求值路径（`compile_multi_line_factor` 的两个调用方）共用此改写。

**判定**：正确。`known` 是真实可用列，改写只在 target ∈ known 时发生；未载入时
不改写 → 仍是 `$turnover` → unknown 检查捕获 → raise `MultiLineFactorEvalError`
（匹配 `\$turnover`）。测试 `test_turnover_alias_inactive_when_column_absent`
锁定此路径。

### 3. `alphaagent/data/adapters/cnequity.py` (-8)

从 `_ALWAYS_KEEP_COLUMNS` 移除 `industry_sw_l1`。理由：无插件供给，仅离线
`panel.py with_industry=True` 路径产出；保留它导致每次 run 2-6 条"不可用字段"错误。

**判定**：正确。下游引用全部是防御性或指纹用途：
- `_prefilter.py:119` 仅用于信号根指纹归一化（不触发数据加载）
- `stacking/dataset.py:252` 有 `if "industry_sw_l1" in panel.columns` 防御
- `plugins.py:149` 是参数默认值

### 4. `alphaagent/factor/facets.py` (+4/-1)

量能面 tuple 改为 `("$volume", "$amount", "$turnover", "$turnover_rate")`，同时
保留别名与真实名。

**判定**：正确。`FACET_DEFS` 用于 facet 分类（前缀匹配），无数据加载副作用；
两者都保留是因为历史因子/记忆中两种写法都出现过，缺任一会漏分类。

### 5. `alphaagent/factor/mining/prompt/modules/data_fields.py` (+6/-3)

`$industry_sw_l1` / `$industry_zx_l1` 行从"严格 PIT，`--with-industry` 时才有"
改为"仅离线 enrich 路径产出，**挖掘链路不加载**，引用即报「不可用字段」"。

**判定**：正确。与 `cnequity.py` 移除 `industry_sw_l1` 的改动口径一致，提示词
如实反映运行时行为。

### 6. `alphaagent/factor/mining/agent/question_queue.py` (+283)

核心门控逻辑。关键新增：

- `DEFAULT_QUESTION_FIELD_GATE = {"enabled": True, "min_ratio": 0.5, "scan_limit": 40, "warn": True}`
  — 单一真源
- `_COLUMN_FAMILY_PREFIXES` / `_OPERATOR_DERIVED_TOKENS` / `_FIELD_FAMILY_SYNONYMS`
  — 字段分类词表（族前缀 / 算子派生 token / 散文词→族映射）
- `resolve_question_field_gate(spec, override)` — 合并 `report_policy` + override，
  clamp `min_ratio` 到 [0,1]、`scan_limit` 到 ≥1，忽略坏值
- `available_field_set(available_fields)` — 规范化可用列（空输入 → 空集合）
- `classify_question_fields(question, available) -> (hits, missing, unknown)`：
  - canonical_column 命中 → hits
  - 登记别名但目标未载入 → missing
  - 算子派生 token（`chip`/`crowding`/`vpin` 等，由行情列现算）→ hits
  - 族前缀（经 synonym 或 startswith 识别）→ 族已载入 hits / 未载入 missing
  - 其余 → unknown（不参与合格判据，避免词表不足误杀）
- `_question_fields_eligible(hits, missing, min_ratio)`：known=0 → True（散文题放行）；
  无 hits → False；否则 `len(hits)/known >= min_ratio`
- `get_question_for_turn(..., available_fields, gate, stats)`：enabled 且 avail 非空时
  从 `(offset+turn)%n` 起扫描 `scan_limit` 道，返回第一道合格题；全不合格 → `None`
  （本轮回退无课题，不阻塞 run）；disabled/空 avail → 单次取值 `(offset+turn)%len`，
  stats status="ungated"。`stats` 出参回填 `{status, scanned, missing}`
- `get_task_for_turn(..., available_fields, gate, stats)` — 选课题 + 渲染一体
  （底层函数，供测试与旧调用方）
- `render_question_task(question, *, missing_fields, spec, gate)` — 只渲染
  （供 `agentscope_run` 复用已选课题，避免二次门控）；内部按 `policy["warn"]`
  决定是否注入缺列提示
- `_render_question_task` — 缺列提示行格式化：
  ```python
  "、".join(f"`${f}`" if not f.startswith("$") else f"`{f}`" for f in missing)
  ```
  确保输出始终是 `` `$xxx` `` 格式（无 `$` 前缀时补、有时只加反引号）

**判定**：正确。分层清晰（`get_task_for_turn` = 选+渲染，`render_question_task` = 只渲染，
两者共用 `get_question_for_turn` 选课题、共用 `_render_question_task` 渲染，无逻辑分叉）；
空 available 退化为旧行为；scan_limit 全不合格回退 None 不阻塞；散文题放行避免词表误杀。

### 7. `alphaagent/factor/mining/memory/retrieval.py` (+33)

- `_parent_missing_fields(parent, available_fields) -> str`：返回父本表达式中
  缺失字段的 `a/b` 摘要；`available_fields` 为空时返回空串（旧行为，不阻断）；
  懒导入 `missing_fields`（field_aliases 是叶子模块，低循环导入风险）
- `recommend_edits(..., available_fields=None, blocked_out=None)`：两个过滤点
  （主路径 + fallback 路径）都调 `_parent_missing_fields`；缺列父本 `continue` 并
  append `f"{parent_factor}:{missing}"` 到 `blocked_out`；两参数有默认值，向后兼容

**判定**：正确。两个过滤点都覆盖；空 available 保持旧行为；签名默认值保证旧调用方
零改动；docstring 说明动机（funda_* 父本在 `--no-fundamentals` 下连续 8 轮被推荐）
与回填语义（过滤槽位由后续候选回填，不减少推荐数）。

### 8. `alphaagent/factor/mining/research_spec.py` (+22)

- `DEFAULT_RESEARCH_SPEC.memory_policy.suggest_field_gate = True`
- `DEFAULT_RESEARCH_SPEC.report_policy` 新增 `question_field_gate=True`、
  `question_field_gate_min_ratio=0.5`、`question_field_gate_scan_limit=40`、
  `question_field_warn=True`（镜像 `DEFAULT_QUESTION_FIELD_GATE`，真源仍在 question_queue）
- `normalize_research_spec`：`suggest_field_gate` / `question_field_gate` /
  `question_field_warn` 经 `_require_bool`；`min_ratio` 经 `_bounded_number(0.0, 1.0)`；
  `scan_limit` 经 `_bounded_number(1, 500)`。全部 clamp/validate

**判定**：正确。阈值单一真源在 `question_queue.DEFAULT_QUESTION_FIELD_GATE`，
`research_spec` 镜像默认值并做规范化校验，无口径漂移。

### 9. `alphaagent/factor/mining/agent/agentscope_run.py` (+55)

两个注入点：

**课题门控**（~L818-856）：
```python
_available_fields = list(getattr(session_resp, "available_columns", None) or [])
_q_stats: dict[str, Any] = {}
current_question = get_question_for_turn(
    outer_turn, ..., available_fields=_available_fields, stats=_q_stats,
)
if _q_stats:
    log_step("question_gate", f"qid=... status=... scanned=...", missing=..., available_cols=...)
if current_question:
    q_task = render_question_task(current_question, missing_fields=_q_stats.get("missing") or [], spec=spec)
    block = f"{block}\n\n{q_task}" if block else q_task
```

**记忆门控**（~L1268-1298）：
```python
_mem_policy = (getattr(config, "research_spec", None) or {}).get("memory_policy") or {}
_field_gate_on = bool(_mem_policy.get("suggest_field_gate", True))
_avail_fields = set(session_resp.available_columns) if _field_gate_on else None
_blocked_parents: list[str] = []
recs = memory_store.recommend_edits(..., available_fields=_avail_fields, blocked_out=_blocked_parents)
if _blocked_parents:
    log_step("memory_suggest_skip", "field_gate", blocked=len(...), reasons="|".join(_blocked_parents[:4]))
```

**判定**：正确。`session_resp.available_columns` 是既有属性（`run.py:100`、
`agentscope_run.py:478,875` 已用），非新增依赖；`render_question_task` 不传 `gate`
→ 与 `get_question_for_turn` 的 `gate=None` 同源 resolve，无分叉；gate 关时
`_avail_fields=None` → `recommend_edits` 退化为旧行为。

### 10-13. `tests/fixtures/system_prompt_*.txt` (4 files, ±2 each)

4 份 fixture（full/long_label/population/price_only）对 `$industry_sw_l1` 行做相同
单行替换，与 `data_fields.py` 改动口径一致。

**判定**：正确。4 份措辞完全一致。

### 14. `tests/test_dsl_input_guards.py` (+30)

4 个新测试覆盖 `$turnover` 别名：改写命中、右边界不误伤、列缺失时不改写被 unknown
拦截、多行表达式全部字段改写。

**判定**：正确。测试验证其所声称的行为。

### 15. `tests/test_memory_injection_consistency.py` (+45)

2 个新测试：`recommend_edits` 在 `funda_` 未载入时阻断 `CS_ZSCORE($funda_ocf)` 父本
（`recs == []`、`blocked` 非空含 `funda_ocf`）；不传 `available_fields` 仍工作；
`funda_ocf` 载入时返回 `["chip"]`。`$turnover` 别名 + `turnover_rate` 载入时不阻断。

**判定**：正确。覆盖门控阻断、向后兼容、别名放行三路径。

### 16. `tests/test_question_field_gate.py` (NEW, +256)

14 个测试：

- **零回归**：`test_ungated_matches_legacy_selection`（真实 1200 题队列，断言
  `available_fields=None` 时选课题与 legacy `(offset+turn)%len` 逐轮一致）、
  `test_ungated_matches_legacy_selection_with_focus`、`test_gate_disabled_restores_legacy`
- **门控行为**：纯缺列题跳过（status="partial"）、部分缺列保留并警告、全可用 status="ok"、
  散文题放行、全不合格返回 None（scan_limit=1）、min_ratio 阈值（0.95）、warn 可关闭
- **纯函数**：`classify_question_fields` 分桶、`missing_fields` 用 DSL 别名、
  `resolve_gate` 默认值与 override 优先级
- **渲染**：`render_question_task` 无 warning 时不注入提示、文件空时回退内置题

**判定**：正确。零回归测试用真实 1200 题队列锁定"空 available = 旧行为"契约，
是本分支最关键的回归保护。

## 缺陷检查

按用户偏好逐项核查**真实缺陷**（逻辑错误/回归/破坏导入/缺失错误处理/安全/竞态/
API 契约违反/死代码/名不副实的测试）：

| 检查项 | 结果 |
|---|---|
| 逻辑错误 | 无 |
| 回归（空 available 退化旧行为） | 无 — 三层（eval / question / memory）均退化，零回归测试锁定 |
| 破坏导入 | 无 — `field_aliases` 是叶子模块，`retrieval` 懒导入 |
| 缺失错误处理 | 无 — `agentscope_run` 两注入点均在 try/except 内，失败静默不阻塞 run |
| 安全 | 无 |
| 竞态 | 无 — 单进程内调用，`stats`/`blocked_out` 是调用方栈上对象 |
| API 契约 | 无 — `recommend_edits` 新参数有默认值，`get_task_for_turn` 仍存在供旧调用方 |
| 死代码 | 无 — `get_task_for_turn` 仍被 2 个测试文件调用，是 `render_question_task` 的上层包装 |
| 名不副实的测试 | 无 — 14+4+2=20 个新测试均验证其所声称行为 |

## 结论

**merge**。

本分支是低风险高收益修复：单点字段别名收口 + 三层（课题派发/记忆推荐/DSL 编译）
按实际载入列门控，空 available 全链路退化为旧行为（零回归测试锁定），配置阈值
收口在 `research_spec` 并 clamp 校验，`steps.log` 新增 `question_gate`/
`memory_suggest_skip` 两个诊断字段。无真实缺陷。

---

## 附：分支作者复核与修正（2026-09-28）

> 复核对象：本 review 对 `f5dae10` 的逐文件判定 + 修改后的分支。**复核结论：维持 merge**，
> 但发现 **1 个 reviewer 漏判的真实缺陷 + 6 处数字不准确**，均已修正（后续 commit）。

### 漏判缺陷（已修）

1. **`test_render_question_task_without_warning` 名不副实**（review 第 219 行把它描述为
   "`render_question_task` 无 warning 时不注入提示"，第 239 行据此判"名不副实的测试 = 无"）：
   该测试实际传 `missing_fields=["eps"]` 并**断言提示行存在**，与函数名语义相反。
   → 已改名 `test_render_question_task_injects_warning`，并新增
   `test_render_question_task_warn_disabled_omits_warning` 覆盖 `question_field_warn=false`
   分支（此前只有 `get_task_for_turn` 层覆盖，`render_question_task` 直调路径无覆盖）。
2. **`classify_question_fields` / `question_missing_fields` 在 `available` 为空时误报缺列**：
   `avail` 为空集合时走到 `if avail and _family_loaded(...)` → False → 把**整题字段判为
   missing**；而 `dsl/core/field_aliases.missing_fields` 对空 available 的契约是"不判定
   返回空元组"。当前 run 路径未触达（空 available 会走 ungated 分支），但公共函数对
   外部调用方是误导 API。→ 已修为 `if not avail: return [], [], []`，并新增
   `test_classify_question_fields_without_available_returns_no_judgment` 锁定契约。
3. **`resolve_question_field_gate` 的 `override` 分支未做 clamp/校验**（仅 `spec` 路径
   clamp `min_ratio`/`scan_limit`）：override 可越界或传坏值。→ 已统一 clamp/bool 化
   （`min_ratio` ∈ [0,1]、`scan_limit` ≥ 1，坏值忽略）。

### 数字更正（review 原文 → 实测 numstat / 计数）

| 位置 | review 写法 | 实测（`git show --numstat f5dae10`） |
|---|---|---|
| `cnequity.py` | `(-8)` | **+7 / −1** |
| `facets.py` | `(+4/-1)` | **+3 / −1** |
| `data_fields.py` | `(+6/-3)` | **+3 / −3** |
| `agentscope_run.py` | `(+55)` | **+46 / −9** |
| `question_queue.py` | `(+283)` | **+271 / −12** |
| `test_question_field_gate.py` | "14 个测试"、"14+4+2=20 个新测试" | **15 个测试**（修正后 17 个）、新增合计 **21**（修正后 23 个） |

其余逐文件判定（`field_aliases` 叶子模块与右边界、`eval.py` 单点改写、`cnequity` 死列
无下游硬依赖、`facets` 双键、`retrieval` 两个过滤点 + 默认参数向后兼容、`research_spec`
阈值收口、4 份 fixture 一致）经复核**维持**；三种 core 文件无需改动。

### 修正后验证

- `pytest tests/test_question_field_gate.py tests/test_dsl_input_guards.py
  tests/test_memory_injection_consistency.py tests/test_question_queue.py
  tests/test_research_memory_v3.py tests/test_config_no_drift.py` → **67 例全绿**
  （原 65 例 + 新增 2 例）。
- 已知既有失败（与本次无关，main `d0748ae` 上同样失败，未在本次处理）：
  `test_facet_focus_scope.py::TestMemoryScopeFilter::test_guaranteed_positives_scope`；
  `test_system_prompt_modules.py` 的 `full/long_label/price_only`（黄金基线含实时研报 RAG
  摘录，环境漂移）。

