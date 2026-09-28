# 评审：`alphaagent_report_knowledge_integration_spec.md`（研报知识接入与四方案对比）

> **审阅对象**：`docs/specs/alphaagent_report_knowledge_integration_spec.md`（Draft v1，2026-09-25，gitignore 工作区参考）
> **审阅基准**：`HEAD=a44f380`（`main`，与 spec 声明一致；`docs/specs/` 属 `.gitignore:70`、`data/` 属 `.gitignore:6`，均已核实）
> **审阅方式**：逐条把 spec 的机制声明与行号引用对照 HEAD 源码/数据核对；只读，未修改任何被审对象
> **审阅日期**：2026-09-25
> **工作区状态**：审阅期间共享工作区含 56 项他人未提交/未跟踪内容，其中 `alphaagent/factor/mining/tools/_dispatch.py` 有 +8/-1 的未提交改动（非本次产物，未触碰）

---

## 0. 总评

方向与分层是对的：把研报定位为"补入口、不复刻全自动"、把研报证据与本平台实测证据硬隔离（§0.4-2）、把"可证伪承诺"留在 LLM（§1 结论 1）——这三条与平台自身已付代价的教训（S3 正交阶梯、D1–D6 同框矛盾、blind-segment 隔离）同源，是可以直接执行的判断。事实底座的可复现度也很高（§0.1 数据计数、§0.3 基建行号，核对基本全过）。

但**当前版本不具备直接进入实施的条件**：

- 5 处设计自相矛盾会让 §8 的对比实验结论失去可解读性（P0），其中 3 处同根：对 prompt 模块装配语义（`_static` / `enabled` / 重建时机）的理解与实现不符；
- 8 处可验证的事实/口径/行号错误（P1），包括两个关键指标的 JSON 路径写错、一个不存在的配置键、一个不存在的面数（"八面"）；
- 10 项缺项与风险（P2），其中 2 项会直接影响 R2 的可用性（聚焦 scrubber 掏空卡片、卡片文件未跟踪导致黄金基线漂移）。

结论：**先修 P0-1 / P0-4（开关与契约），再补 P0-2 / P0-3 / P0-5（接线与对照），然后按 P1 表逐条订正，最后再议 R3 / R4。**

---

## 1. P0 — 会让对比实验结论失效的问题

### P0-1 `_static()` 吞掉 `enabled()`，R1 注册即无条件常驻 —— 与 §10「默认关 → 零影响」直接矛盾

**证据**

- `alphaagent/factor/mining/prompt/modules/__init__.py:64-72`：`_static` 把 base 固定为 `lambda ctx: True`，构造 `PromptModule` 时**完全不读 `module.enabled`**。只有 `_dynamic`（`:75-84`）会取 `getattr(module, "enabled", lambda ctx: True)`。
- spec §3 的 R1 配方是 `_static(_report_mechanisms_mod)` + 定义 `enabled(ctx)`（`asset_type == "stock"`）→ 该 `enabled` 是死代码。
- spec §0.2 关键事实把 `market_mechanisms` 描述成 `_static`，实际注册为 `_dynamic`（`modules/__init__.py:94`）。

**后果**

1. R1 上屏后无法按 `asset_type` 或 `report_policy` 关闭，只能靠 `prompt_policy.excluded_modules`，与 §4.3 的配置面设计冲突；
2. §10「R1/R2/R3 全部落在既有开关体系内，默认关 → 对现状零影响」对 R1 不成立；
3. §8.1 的 `control = 现状` 失真（control 会带上 `report_mechanisms`），所有 arm 的差值都掺入了这一项。

**修法**：R1 改用 `_dynamic`（`market_mechanisms` 本身就是 `_dynamic`）；或让 `_static` 尊重 `module.enabled`（需评估对现有 5 个静态模块的影响）。同时 `control` arm 显式加 `excluded_modules: ["report_mechanisms"]`。

### P0-2 R2 的「跨轮次轮转」在当前装配下不可实现

**证据**

- `alphaagent/factor/mining/agent/agentscope_run.py:818-842`：system prompt **只在 `prompt_phase` 变化时重建**（`if _phase != getattr(agent, "_current_prompt_phase", None)`）。
- `research_spec.py:239`：`prompt_policy.phase_mode` 默认 `"auto"`；`_compute_prompt_phase`（`agentscope_run.py:626-644`）按 turn 比例切 explore/deepen/deliver。
- 结论：每个 run 内 `render()` 至多被调用 **3 次**（每次 phase 切换一次）。
- spec 附录 A 改动文件清单（:383-399）**不含** `agentscope_run.py` / `agent/loop.py`。

**后果**：§4.2 的选片维度「未注入过（跨轮次轮转，避免每轮同一批卡）」拿不到每轮钩子，实际退化为"每个 phase 固定一批卡"。若实现时按该维度设计缓存与记账，会写出永不生效的代码。

**修法**（二选一，需写进 spec）：
- (a) 把该维度改成**跨 run 轮转**（用 `research_memory` 或卡片侧 `last_injected_at` 记账），不引入主循环改动；或
- (b) 明确把"每个 phase 重建"改为"每轮重建"，并把 `agentscope_run.py` 增补进附录 A 与 §9 的触主链路列。

### P0-3 评估 R2 的实验里，R2 的主信号根本不会出现

**证据**：`scripts/run_ablation_study.py:167-178` 的 `run_cmd` **不传 `--focus-facets`**（列表里只有 `--user-message / --max-turns / --research-spec-file / --log-dir / --research-memory-file / --no-fundamentals`），且固定 `--no-fundamentals`。

**后果**

1. `ctx.focus_facets` 恒为空（`prompt/prompts.py:109` 透传自 `--focus-facets`）→ §4.2 权重最高的「facets 交集」维度失效，选片落进未定义的路径；
2. `--no-fundamentals` 使 panel 无 `funda_` 列 → 基本面类卡片即使被选中也不可用，与 research_mode 维度互相矛盾。

**修法**：给 arm 定义加 `focus_facets` 注入字段（或单独一组聚焦 arm），并在 §4.2 明确"focus 为空时的选片回退口径"（例如按 `infer_research_mode` + 最近尝试 family 选片），让实验覆盖该分支。

### P0-4 改 `_LAYER1` 契约会污染所有无卡片 arm

**证据**：`strategy_tracks.py:12-38` 的 `_LAYER1` 是全阶段在场的 `REQUIRED=True` 模块（`strategy_tracks.py:202-205`，`ORDER=20`）。spec §4.4 要把「必须写出 50 字经济直觉」改成「必须引用一张机制卡 ID（或显式声明 `D轨无先验`）」。

**后果**：`control` / `B2_no_mechanisms` / `R1a` 等 arm 没有卡片，却被要求引用不存在的卡片 ID → 控制组行为被同一改动改写，arm 间不可比；与历史 run 的可比性也一并丢失。

**修法**：`_LAYER1` 按卡片可用性（或 `research_spec.report_policy` 开关）**条件渲染两套文本**，把改写与 R2 arm 绑定；同时在 §8.3 声明"新契约上线前后的 arm 结果不跨版本比较"。

### P0-5 `R0_no_report` 是退化对照，度量不到「token 增量效应」

**证据**：spec §4.1 的 `enabled()` 在无卡片时返回 False → 整块不注入；§8.1 的 R0 定义是"卡片路径指向空文件"。两者叠加 ⇒ R0 的 prompt 与 `control` 等价（§10 自己称之为"零风险降级"）。

**后果**：§8.2 想分离的"研报知识的贡献"与"多注入一段文本的贡献（提示效应）"无法分离 —— 而 R0 正是为后者设计的。

**修法**：把 R0 改成 **placebo arm**：注入等字符数的无信息文本（或把真卡片随机打乱到错误 facet），保持长度匹配；原 R0 语义保留为"模块在场但不渲染"的装配自检。

---

## 2. P1 — 事实与口径错误（可逐条复核）

| # | spec 说法 | 实际（HEAD a44f380） |
|---|---|---|
| 1 | §8.2 反指 `funnel.dup_dead_end_rate` | 位于 **`cognition.dup_dead_end_rate`**（`run_metrics.py:437`） |
| 2 | §8.2 用 `gate_failure_reasons` 看「correlation 类失败」 | `gate_failure_reasons` 只收 **engine_gate** 的 `fail=[...]`（`run_metrics.py:74-83`）；相关性失败是 stage_one 的 **`max_cs_corr`**（`delivery_checker.py:158` 规则名 `stage_one_correlation`、`:171` 原因串），落在 **`stage_one_failure_reasons`**（`run_metrics.py:61-70`）。反指正确写法：`stage_one_failure_reasons.max_cs_corr` 占比 + `promotion_status == "offline_orthogonality_blocked"` 计数 |
| 3 | §4.2 打分维度 `research_mode`，取值 `(ctx.research_spec or {}).get("mode")` | `research_spec` **无 `mode` 键**；`PromptContext`（`prompt/prompt_modules.py:31-60`）**无 `research_mode` 字段** → 该维度恒 `None`。需改用 `infer_research_mode(focus_facets)` 或给 context 增字段（属改动清单遗漏项） |
| 4 | §0.1/§4.2/§7.3「`expressions.FACET_DEFS` **八面**」 | `FACET_DEFS` 实为 **14 面**（`alphaagent/factor/facets.py:19-33`：价量/量能/筹码/拥挤/基本面/股东/机构/股东集中/资金/两融/事件/业绩/披露/分红），前端 `focusFacetOptions` 同为 14（`static/src/store/alphaagent.js:88-103`）。且真源在 `alphaagent/factor/facets.py`，`memory/expressions.py:294-304` 只是 re-export |
| 5 | §0.2「`market_mechanisms`（ORDER=55、`_static`、仅 stock 启用）」 | `ORDER=55` ✓，但注册为 **`_dynamic`**（`modules/__init__.py:94`）——见 P0-1 |
| 6 | §4.2/附录 B「按 `_clip` 口径截断（对齐 `memory/retrieval.py:1318-1322`）」 | `_clip` 定义在 **`retrieval.py:1332-1336`**，且是 `context_for` 内**局部闭包**，不可 import → 附录 B 的"直接复用"应改为"复制同一行边界截断语义（`text[:budget].rsplit("\n", 1)[0]`）"；1318-1322 是 `core_budget` 计算 |
| 7 | §1 结论 2「`delivery/submit.py:805-815` 的离线正交门」 | 门在 **`submit.py:840-856`**，`promotion_status="offline_orthogonality_blocked"` 在 **`:852`**（805-815 是 test 段指标写回） |
| 8 | §1 结论 2「hook 由 `agent/agentscope_tools.py:916` 传入」 | `"orthogonality_hook"` 在 **`:944`**（916 是 `submit_factor` 的形参 `parent_factor`） |
| 9 | §5.2 R3a「`agent/agentscope_run.py:399` 附近与 `ov_block` 并列拼接」 | `ov_store.retrieve_lessons(...)` 与拼接在 **`:465-468`**，且是 **run 启动时调一次** |
| 10 | §0.3「`register/unregister_prompt_module` 在 `prompt/modules/__init__.py:106-122`」；「模块级消融 `:48-57`」「阶段过滤 `:53-57`」 | 实为 **`:110-126`**；消融检查 `:50-54`；阶段过滤 `:55-59`（轻微漂移） |

**行号引用精度良好、可保留的部分**：`run_ablation_study.py:47-132 / 143 / 149-165 / 176-177`、`run_metrics.py:402-446`、`ov_store.py:21`（SCOPE 硬编码及隔离注释均核实）、`research_spec.py:179-190 / 383-418`、`strategy_tracks.py:16-38`、`market_mechanisms` ORDER=55、`.gitignore:6 / :70`。

### 事实底座复核（§0.1，全部通过）

| 声明 | 复核结果 |
|---|---|
| `luobo/` 8181 条元数据、2024-09~2026-09、100 PDF | ✓ `reports.jsonl` 8181 行；`publishTime` 2024-09-01 ~ 2026-09-24；`*.pdf` 100 |
| `hibor/` 200 条、近 1 月 | ✓ 200 行；`publishDate` 2026-08-17 ~ 2026-09-24 |
| `nn73/` 1815 PDF、分类多因子 783 / 海外文献 245 / FOF 188 / CTA 159 / 择时 122 / ML 54 / 数据处理 17 / 未分类 247 | ✓ 全部命中，合计 1815 |
| `parsed/` 共 670（luobo 100 + nn73 570） | ✓ 670 = 100 + 570；**570 篇全部落在 `01_多因子与选股体系`**（该域共 783）→ §7.4 的"先跑该域"实际等于"跑全部已解析 nn73"；语料 27.0 MB、平均 41.3 KB（折算蒸馏输入约 6–9M tokens，一次性离线成本可控） |
| `batch_metadata.json` 未生成、解析进程已退出 | ✓ 两目录均无该文件；当前无 `batch_parse` 进程 |
| 解析口径（span ≥ 9pt、单 CJK 行丢弃、pdfplumber 表非空率 > 0.3、resume-safe） | ✓ `%TEMP%\batch_parse_v3.py`（5821 B，2026-09-24 16:53）中 `MIN_SIZE = 9.0`、`RE_SINGLE_CJK`、非空率 0.3、skip existing 均核实 |

### 一处不可复现

**§0.1「hibor 抽样对比：62.5% 为萝卜缺口」** —— 按标题比对（含去标点/括号归一化），hibor 200 条对 luobo 7018 个归一化唯一标题的命中数为 **0/200（缺口 100%）**，无法得到 62.5%。请补比对口径与样本量（若为少量人工抽样请注明，如 5/8 这类小样本会自然产生该数值），否则该论据会误导"hibor 增量价值"的判断。

---

## 3. P2 — 缺项与风险

1. **聚焦 scrubber 会无声掏空卡片（直接影响 R2 主场景）**：`prompt/prompt_modules.py:104-108` 对**每个启用模块**的文本套 `scrub_out_of_scope`，凡含 `$列` 的反引号/围栏片段触及未选面即整段替换为"（本 run 未选「…」数据面，示例略）"（`prompt/scope_filter.py:51-66`）。R2 的 `observable` / `dsl_hint` 若写成反引号代码且跨面（卡片标"价量面"、文本里出现 `$volume`）就会在聚焦场景下被裁掉 —— 而聚焦正是 R2 的主场景。**需在 spec 中硬规定 `_format` 的渲染格式**（字段/算子片段不用反引号，或按 focus 预过滤字段），并补一条装配回归测试。
2. **卡片文件在 `data/`（gitignore）⇒ 黄金基线可能随开发机漂移**：`tests/fixtures/system_prompt_*.txt` 要求逐字节一致（`tests/test_system_prompt_modules.py:80-84`）。若 `enabled()` 读某个硬编码仓库路径、而该文件在本地存在，R2 块就会进 baseline（CI 无该文件则不进），测试结果变成环境依赖。**硬规定**：卡片路径**只**从 `ctx.research_spec.report_policy` 读，缺省即关，绝不落默认仓库路径。
3. **§0.4-3 的论据不成立 / 基准数字需限定**：prompt 模块（system prompt）与 `memory_policy.max_inject_chars=2400` 是两套独立预算，新增 prompt 模块**挤不掉**记忆核心块；真正竞争该预算的只有 R3b（进 `context_for`）。另外「prompt 已 36.9K chars（full）」是**聚焦 run** 的口径：HEAD 全量装配实测为 **explore 44,753 / deepen 48,184 / deliver=full 48,755 字符**（最小 spec + `panel_columns=None` + 算子目录；黄金基线文件 49,756 字符 = 48,755 + CRLF 的 1,001 个 `\r`），纯价量装配 32,553 字符（与 AGENTS.md 记录的 33.4K 同量级）。数字本身不错，但必须写清基准，否则"必须走预算分段"会被误用到 R1/R2。
4. **R3b 把网络检索搬进热路径**：`context_for` 每轮调用，而现状 OV 只在 run 启动调一次（`agentscope_run.py:465`）。spec 应在 §5.2 补延迟预算与降级策略（超时、结果缓存、每 N 轮一次、失败静默）。
5. **配置键不自洽**：§8.1 用 `enable_report_ov_rag`，但 §4.3（`report_policy`）与 §5（`REPORT_SCOPE` + secondary 块）都未定义它归属哪一段。按 §5.2 R3b 应属 `memory_policy` 侧，请统一。
6. **蒸馏脚本必须入库**：附录 A 写"临时或 `scripts/`"，而 `batch_parse_v3.py` 已在 `%TEMP%`（可复现性风险已实证：Temp 被清理即不可复现）。蒸馏器 + schema 校验器 + 抽检脚本应一并进 `scripts/`（走分支提交）。
7. **外部证据的渲染口径缺失**：§7.2 的 `evidence.ic / icir` 与平台门槛（|IC| ≥ 0.02 等）量纲与样本域完全不同，若进 prompt 会诱导 LLM 误判"已达标"。建议给出**渲染字段白名单**：`evidence` 只出"样本域 + 研报自述失效条件"，数值留档不进 prompt，或强制标注"外部口径、不可与平台门槛并置比较"。
8. **provenance 需带内可见**：§0.4-2 只做了"通道隔离"，注入后 LLM 无法区分来源。建议块标题与行为规则层统一标注「【外部研报先验·非本平台实测】」，与 D1/D2 类同框矛盾自查口径对齐。
9. **附录 C 无来源**：RD-Agent 星数、各项目做法、以及"无项目做这条轻量路径"的**全称否定**都缺链接与 as-of 日期，建议补引用或降级为"未见公开实现（截至 2026-09）"。
10. **实验成本未量化**：§8.3 的 3 reps × 8 arms = 24 runs（默认 `--max-turns 5`，`run_ablation_study.py:326`）应在 §9 附预算/时长估算，以便决定是否先砍到 4 arms × 3 reps。

---

## 4. 站得住、建议保留的部分

- **§0.4-2 记忆层语义隔离**（研报证据不得写 `memory_entries` / `memory_cells` / `memory_experience`）：整份 spec 最有价值的一条纪律，且通道设计上确实可隔离。
- **§1 结论 1 / 2**：(d) 预期不可外包、(f) 研报 = 最拥挤层需收窄用法 —— 与 `submit.py:840-856` 离线正交门"评估配额花完才被拦"的代价分析吻合，是正确的工程判断。
- **§7.3 四条硬口径**：`formula_text` 允空不补全、`negatives` 独立成字段、`facets` 对齐、`source.pages` 必填 —— 直接针对"公式/因子表现图在图片里"这一实证事实，是防止蒸馏幻觉的关键约束。
- **§0.3 基建复用清单与行号**（多数精确）、**R0 的意图**（区分知识贡献 vs token 增量）正确，只需按 P0-5 换成 placebo。
- **§9 先落 R1、R4 最后议**的排序合理。

---

## 5. 建议的修订顺序

1. 修 **P0-1**（`_static`/`enabled` + R1 缺省关 + control arm 净化）与 **P0-4**（`_LAYER1` 条件渲染）—— 不改这两条，任何 arm 结果都不可解读。
2. 补 **P0-2 / P0-3**（选片钩子口径 + arm 的 `focus_facets` 注入）与 **P0-5**（placebo arm）。
3. 按 §2 表逐条替换行号与指标路径（`cognition.dup_dead_end_rate`、`stage_one_failure_reasons.max_cs_corr`），把"八面"改为 **14 面**并把 `FACET_DEFS` 指向 `alphaagent/factor/facets.py`。
4. 订正 §0.1 的 hibor 缺口数字（补口径或删除），把 §0.4-3 改写为"上下文预算"论据并写明 36.9K（聚焦 run）/ 48.8K（全量装配）的区别。
5. 补 §4.1 `_format` 渲染格式约束（规避 scrubber）与卡片路径"缺省即关"规定（P2-1 / P2-2）。
6. 之后再议 R3（R3b 热路径预算）与 R4。

---

## 附录 · 本次评审的关键证据索引

```
prompt/prompt_modules.py:31-60        PromptContext 字段（无 research_mode）
prompt/prompt_modules.py:104-108      装配期对每个启用模块套 scrub_out_of_scope
prompt/scope_filter.py:51-66          聚焦投影：含 $列 的越界片段整段替换
prompt/modules/__init__.py:64-72      _static 忽略 module.enabled
prompt/modules/__init__.py:75-84      _dynamic 尊重 module.enabled
prompt/modules/__init__.py:94         market_mechanisms 以 _dynamic 注册
prompt/modules/__init__.py:110-126    register/unregister_prompt_module
prompt/modules/strategy_tracks.py:12-38   _LAYER1（economic intuition 强制）
agent/agentscope_run.py:626-644       _compute_prompt_phase
agent/agentscope_run.py:818-842       system prompt 仅在 phase 变化时重建
agent/agentscope_run.py:465-468       OV 冷路径检索（run 启动一次）
agent/agentscope_tools.py:944         orthogonality_hook 传入点
delivery/submit.py:840-856            离线正交门；852 = offline_orthogonality_blocked
delivery/delivery_checker.py:158/171  stage_one_correlation 规则名 / reason = max_cs_corr
memory/retrieval.py:1332-1336         _clip（context_for 内闭包，不可 import）
memory/ov_store.py:21                 SCOPE 硬编码 + 隔离注释
run_metrics.py:61-70 / 74-83          stage_one / engine_gate 失败原因解析
run_metrics.py:402-445               scorecard（dup_dead_end_rate 在 cognition）
research_spec.py:239                  prompt_policy.phase_mode 默认 "auto"
alphaagent/factor/facets.py:19-33     FACET_DEFS（14 面，真源）
memory/expressions.py:294-304         FACET_DEFS 仅 re-export
static/src/store/alphaagent.js:88-103 focusFacetOptions（14 项）
scripts/run_ablation_study.py:167-178 run_cmd 无 --focus-facets，固定 --no-fundamentals
tests/test_system_prompt_modules.py:80-84  黄金基线逐字节断言
```

> AI 生成
