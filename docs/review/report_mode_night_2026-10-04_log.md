# 研报模式整夜挖掘运行日志（2026-10-03 22:00 → 2026-10-04 07:00）

> 值守方式：goal 续跑轮次驱动、**约 10 分钟一轮**；每轮只读巡检（API + steps.log），
> 发现可修问题当场修（分支 → 测试 → 合入 main）。
> 巡检依据：`C:\Users\zhoubw\.dsh\skills\alphaagent-run-watch\SKILL.md`
> 监控进程：`C:\Users\zhoubw\Desktop\quant\overnight_mining_monitor.v2.py`（PID 1552，23:04:49 起）
> 参数：`--research-mode report --deadline 07:00 --max-turns 8 --stall-warn 15 --stall-kill 25`
> 日志：`C:\Users\zhoubw\Desktop\quant\logs\overnight_monitor_20261003.log`

## Round 0（22:06–23:05）：开工前的环境事故与修复

### 事故：挖掘 LLM 通道整条挂死（22:06 发现，23:0x 用户修复）

| 事实 | 证据 |
|---|---|
| 后端未运行 | `:17891` 无监听（我据此先拉起后端） |
| 挖掘链真实路径挂死 | `load_codex_provider()` + `chat_json()` → `None`，耗时 **545s**（1-token 级请求） |
| 链路 | 挖掘 → cc-switch `127.0.0.1:15721` → EasyCLIProxyAPI/CPA `127.0.0.1:8317` → 上游 codefree/antigravity |
| 分层定位 | CPA `/v1/models` **200（68ms）**；CPA `chat/completions` **超时**（直连同样超时）→ 挂在上游侧，不是 TCP/进程死 |
| 时间线 | cc-switch 代理请求日志**最后一次成功 = 19:43:37**，正好是上一 run `afb3d7707264` 结束时刻；之后 2.5 小时零成功 |
| 已排除 | CPA 进程 22:04:27 已重启过（用户操作）仍不通；okmcode 直连 502；command-goat 用凭据文件 key 401 |
| 恢复 | 23:0x 用户处理后，同一条路径 **2.8s 返回 `{'ok': True}`**，CPA chat 1.5s 200 → 通道恢复 |

**结论**：不是代码问题，是上游账号/反代侧事故；判定顺序（models 通 ≠ 上游通 → 必须打 chat 级请求）已固化在 skill 第 7 条。

### 开工

- 23:04:49 挂起 monitor（report 模式），23:05:32 起 **run `17a35500f1a8`**。
- 23:19:22 `17a35500f1a8` 结束：14.3 分钟、tool_calls=70、`max_turns_reached`、候选 0；
  复现判定 41 / PASS 2、保真度 3 条、**error_type 0 条**。
- 23:19:48 monitor 自动开 **run `93a0d9970c67`**。

## Round 1（23:05–23:40）：边看边改 #1 —— 血统漏传自动补全

### 观察到的真实问题（run `afb3d7707264` 复盘）

全 run 25 条 `tool_failed` 里 **13+ 条**是研报血统门禁拦截，两种形态：
1. **漏传（`当前传入=(空)`）** —— 模型已在正确课题/阶段里干活，只是忘填 `parent_factor` → 白烧一轮；
2. **传了错的非空血统** —— 例如 run `93a0d9970c67` 复现阶段传 `rqffd0ba_style_rotation_anch`（**另一个课题**的因子）、
   或 `rq_addf97_growth_surprise_core`（记忆推荐父本）→ 门禁**拦得对**（防跨课题血统污染），此处不改。

### 修复（commit `c66d766` → merge `cf3cd49`，已合入 main）

- 新增 `_report_lineage_fill()`：**仅**在 `parent_factor` 传空时补全 —— 复现轮补 `reproduce_of:<qid>`、
  发散轮补该课题复现版因子名；**传了非空的错血统一律照旧硬拦**（不猜、不覆盖）。
- 4 个入口（`evaluate_factor` / `eval_on_train_set` / `eval_on_val_set` / `submit_factor`）在门禁前补全，
  补全值同时进入记录参数 → 血统落库一致；补全落 `steps.log` 的 **`report_lineage_autofill`**（可观测）。
- 配置中心三件套：`DEFAULT_RESEARCH_SPEC.report_policy.parent_autofill=True` + `_require_bool` 校验 +
  `lineage_parent_autofill(spec)` 判定回落；**两处研报网关都携带该键**（防「阈值没随网关传」的老坑复发）。
- 测试 +6（复现补全 / 发散补全 / 显式值保留 / 关开关 / 空网关 / 入口挂载一致）；
  127 个 report 相关测试全绿（`test_lineage_gate_coverage` `test_anchor_skip_guard` `test_reproduce_gate`
  `test_reproduce_fidelity` `test_report_structure_gate` `test_method_type_gate` `test_question_quality_filter`
  `test_report_gate_config` `test_anchor_guard_gate_wiring` `test_report_knowledge_modes`
  `test_threshold_config_center` `test_research_spec_overrides` `test_auto_mode_and_freq`）。

### 生效边界（诚实标注）

- `93a0d9970c67`（23:19:48 起）跑的是**修复前**的代码，其 6 次复现门禁拦截全是「非空错血统」形态，
  **本来就不在本次修复范围内**；
- 修复 23:27:41 落盘 + 后端 23:24:24 已重启 → **下一个 run 起**生效（验证锚点：steps.log 出现
  `report_lineage_autofill`，且 `当前传入=(空)` 类拦截消失）。

## Round 2–5（23:40–00:20）：边看边改 #2/#3 + 修复 #1 生产验证

### 生产验证：修复 #1 生效（run `65eedb4b8777`，23:48:20 起）

```
steps.log: report_lineage_autofill = 23 条（phase=diverge qid=RQ_d8fac5 parent=amtcap120_..._rank_med）
该 run 门禁拦截：空传 0 条（其余 1 条为跨课题）→ 「漏传 parent_factor」这一类**已消失**
```
> 注意：这是 23:27:41 的修复；`17a35500f1a8`(23:07) / `93a0d9970c67`(23:21) / `ed5e3414599a`(23:39)
> 三个 run 都在修复前启动，其中前两个各有 2 条「空传」拦截，符合"修复只对之后启动的 run 生效"。

### 观察：门禁拦截的三种形态（今晚 38 条，逐条归类）

| 形态 | 条数 | 判定 | 处置 |
|---|---|---|---|
| ① 漏传（`当前传入=(空)`） | 4 | 模型忘填字段 | ✅ 修复 #1（自动补全） |
| ② 同课题但大小写/下划线不一致 | 17 | **门禁实现与自己的文案口径矛盾** | ✅ 修复 #2（`lineage_key`：忽略大小写与下划线） |
| ③ 跨课题血统（含记忆推荐父本） | 17 | 拦得对（防污染） | ✅ 修复 #3（推荐阶段就过滤掉，不再白烧名额） |

②的典型：课题 `RQ_17eebb` 传 `rq17eebb_rev_timed_turn_slope`（无下划线）、
`RQ_addf97` 传 `rq_addf97_growth_surprise_core`（有下划线）—— 文案写着"或至少包含 `<课题号>`"，
逐字比较必然不匹配。
③的典型：`amtcap120_x_prem20_x_prem10q4_rank` / `peakclear_mom10_pw` 被推给 5 个不同课题，
`rqffd0ba_style_rotation_anch`（RQ_ffd0ba 的复现版）被推给 RQ_addf97。

### 修复清单（本轮新增，均已合入 main）

| commit | 内容 | 测试 |
|---|---|---|
| `6d1dd7f`→`c2fbf09` | 血统比对键 `lineage_key`（忽略大小写+下划线）；复现/发散两处门禁统一用它 | +2 |
| `fb93642`→`ff354f3` | 研报阶段记忆推荐先按课题血统过滤（+ `memory_suggest_lineage_filter` 锚点） | +2 |
| `4daec3e`→`ad82ba5` | 过滤补上"本课题复现版名"（复现版名常不含课题号，只按 qid 会误杀） | +1 |
| `4f3331d`→`fd1a1df` | 复现门禁接受"本课题复现版名"（同一条血统，与发散门禁同尺）+ 文案补该选项 | +1 |
| `50c5461`→`b3b7aa1` | `expected_strong_side` 支持"端点前缀+因子语义"写法（`high_quality`/`high_div`/`high_vpin`） | +1 |

### bench 基线已刷新（00:17）

```
bench baseline --set 65eedb4b8777   # 旧基线 fc9bdcbd505b 自动归档到 artifacts/alphaagent/bench/history/
新配置哈希: spec:39430ac58eb694f6...|mode:report   （= 今晚 23:19 起所有 run 的身份，后续 run 可直接 diff）
说明: 2026-10-04 夜间基线：血统门禁四修后、report 模式、spec hash 39430ac
```
> 旧基线 `0dac03c` 与今晚全部 run 的 `39430ac` 不一致（`parent_autofill` 进入 spec），
> 刷新前 `bench diff` 会判 `INVALID_CONFIG_MISMATCH`。

### 修复 #5 的证据（run 6597fe0e0c84 实测）

```
err=prediction_invalid: expected_strong_side 收到 'high_quality'，合法值: high_factor|low_factor|middle …
err=prediction_invalid: expected_strong_side 收到 'high_div'，合法值: …
err=prediction_invalid: expected_strong_side 收到 'high_vpin'，合法值: …
```
→ 语义就是 `high_factor`，但别名表只覆盖精确写法（`high_decile` 等）。已按"串首端点词 +
整串只含一个端点语义"归一放行；混写（`high_*_low_*`）仍拒绝（不猜）。

> 顺带被仓库自带守卫抓到一处**我自己的注释**问题：`agentscope_run` 外层注释里裸写了嵌套作用域
> 局部量标识符，`tests/test_report_judge_gates.py::test_report_phase_never_referenced_outside_nested_scope`
> 按文本匹配判定违规（注释也算）→ 已改表述，守卫恢复绿。**这正是"多跑测试"的价值**。

> `fb93642` 的过滤初版被 `4daec3e` 立刻修正：证据是 run `65eedb4b8777` 里
> RQ_d8fac5 的复现版名 `amtcap120_x_prem20_x_prem10q4_rank_med` **不含 qid** →
> 只按 qid 过滤会把发散阶段的合法推荐全部丢掉。

### 失败原因分布（今晚 3 个已完成 run，`qp_night_status.py` 口径）

```
14  eval_error: ⛔ 复现门禁（RQ_4ad237 ）     ← 属形态②③，已修
 9  eval_error: memory_blocked_duplicate（同构已评估 3 次）
 6  eval_error: memory_blocked_duplicate（同构已评估 3 次，另一结构）
 4  eval_error: ⛔ 复现门禁（RQ_addf97）
 4  eval_error: ⛔ 复现门禁（RQ_62f045）
 4  eval_error: prediction_invalid: expected_strong_side 收到 None
 3  eval_error: ⛔ 复现门禁（RQ_17eebb）
 2  eval_error: exec 阶段失败: 未定义的名称 'REBETA'（自造算子）
```
`run_summary.failure_counts` 合计：`tool_failed` 32 / `MemoryAdvisoryBlock` 19 /
`MultiLineFactorEvalError` 8 / `ToolArgumentsError` 6。

**本轮判定"不改"的两项（附理由，避免无据改口径）**：
1. `prediction_invalid: expected_strong_side 收到 None`（4 次）—— **不由形态推定强侧**：
   实测样本显示 `monotonic_decreasing + expected_sign=-1` 时模型仍写 `high_factor`
   （强侧=信号最强端，不是收益最高端）→ 形态并不蕴含强侧，自动补全会伪造可证伪声明。
2. `REBETA` 未定义（2 次）—— 模型自造算子，DSL 报错已给最近候选（`REG`），属提示词层面噪声。


## 待观察/待办（随夜更新）

| # | 事项 | 状态 |
|---|---|---|
| 1 | 修复 #1/#2/#3 在**修复后启动**的 run 上同时生效（autofill>0、同课题拦截=0、`memory_suggest_lineage_filter` 出现） | ✅ 已验证：#1 `65eedb4b8777` 起 autofill **135 次**；#2 同课题拦截 **17→0**；#3 过滤 **22 条**；#7 见上表（`eb785f4298b9` 0 报错） |
| 2 | bench 基线刷新：`parent_autofill` 进 spec → spec hash 变，需用**修复后**的 run 重设基线 | ⏳ 待定（挑一个修复后的完整 run：`bench.py baseline --set <run_id>`） |
| 3 | 23:24:12 monitor `GET /runs` 20s 超时 → 触发后端重启（当前 118 run 时 `GET /runs` 实测 1.13s，属**瞬时阻塞**）；若复发要查后端事件循环阻塞点 | ⏳ 观察 |
| 4 | `memory_blocked_duplicate`（同构已评估 3 次）15+ 次/晚：是设计性拦截，但反映模型在同一结构上反复试错 | ⏳ 观察 |
| 5 | skill `alphaagent-run-watch` 引用的 `scripts/patrol_report_mode.py` / `night_summary_report.py` **在本仓不存在** → skill 文档失真，需修 skill 或补脚本 | ⏳ 待办（低优先） |

---

## 整夜汇总（截至 2026-10-04 06:30）

### 一、规模与结果

| 项 | 值 |
|---|---|
| 起止 | 23:04:49 挂 monitor → 07:00 deadline（有效挖掘窗口 **23:05–01:31**，之后上游中断；06:20 恢复后又跑 3 个 run；最后一个 `ff59255e6b38` 06:58 起、跨 deadline 收尾） |
| run 数 | **16**（monitor 退出时自报 16：`17a35500f1a8 … ff59255e6b38`；其中 8 个有 `run_summary`、1 个（`0a4d2bb8d30b`）被看门狗停掉但有 403 events 过程数据、7 个因上游中断只跑到 events=9） |
| 训练评估 | **536** |
| 验证评估 | 12 |
| 复现判定（**真判定，剔除"跳过 phase=diverge"**） | 244 次 / **PASS 10** |
| 保真度 | **10 过 / 18 检** |
| 候选池 / 正式库 | **0 / 0** |
| 提交 | **2+** 次，全部卡在盲测：`blind_test_abs_ic`（65eedb4b8777）、`sign_flip+abs_ic+ic_retention`（eb785f4298b9 `fund_comp_risk2x_rq_abd18b`，test_ic=-0.0018） |
| 边看边改 | **13 项**（12 个仓库修复/交付 + 1 个 monitor 脚本修复；均已测试/验证）+ bench 基线刷新 |

> 与昨夜（13 run、1 候选、11 PASS）相比：**今夜产出为 0 候选**，但有效窗口只有 2.5 小时，
> 且 01:31 起上游断链吃掉 4.8 小时 —— 单夜样本不足以判定"系统变好/变差"。

### 一之二、收尾（07:29:56 monitor 自行退出）

```
[10-04 07:10:11] ff59255e6b38 最后一次写入（turn=0 RQ_45e3f1 复现判定，IC≈0.006~0.008 未过）
[10-04 07:18:40] ⚠ 停滞告警 15 分钟
[10-04 07:28:55] 停滞 26 分钟 ≥25 → 停止 run ff59255e6b38（上游第三次中断：同刻 1-token 探测 ❌ 500）
[10-04 07:29:56] 已到 deadline(07:00)，不再新开 run，监控退出：本次共启动 16 个 run
```
> 上游今夜共三次中断：`~19:43–23:0x`（开工前，用户修复）、`01:31–06:20`（4.8h，7 个 run 失败）、
> `07:10–`（收尾期，导致最后一个 run 被看门狗停掉）。第三次正好落在 deadline 之后，损失可忽略。
> 注：今晚运行的 monitor 是 **23:04 启动的旧版**（内存里没有修复 #9），所以它仍是"到点就杀"；
> 修复 #9 的新逻辑自下次启动起生效。

### 一之三、bench 对比（两把尺子，配置均与基线 `39430ac|mode:report` 一致）

`bench.py diff d27d37aab8c9`（05:57 run）：有效新颖率 0.9674→**1.0**、重复尝试率 0.0326→**0**、
死路遵循率 0.9167→**1.0**；`bench.py diff eb785f4298b9`（06:33 run，本夜最长数据 run）：
数据面分布熵 +0.0898 **[+]**、信号族覆盖 23→**27 [+]**、预测对账覆盖 **1.0**、
评估耗时 P50 9.87→**8.02s [+]**；结构指纹多样率 −0.1377 **[-]**、重复尝试率 +0.0215 **[-]**。
→ **有升有降、单 run 样本不下结论**（与 skill 的"诚实标注样本量"一致）。

### 二、上游两次中断（本夜主要损失来源）

| 时段 | 现象 | 处置 | 结果 |
|---|---|---|---|
| ~19:43–23:0x（开工前） | cc-switch→CPA→上游 chat 全超时；`/v1/models` 200 但 chat 挂 | 用户侧修复；我用 `load_codex_provider()+chat_json()` 复测 2.8s 通 | ✅ 23:05 正常开挖 |
| **01:31–06:20（4.8h）** | `Error code: 500 … CC Switch local proxy failed` → 连接失败；**7 个 run 失败**（`97151a477e65`/`fce211a12d49`/`59f810e5f170`/`d1544f594262`/`14199e00d2d0`/`02eab0703037`，各 events=9；`0a4d2bb8d30b` events=403 中途崩） | 按 skill 第 7 条**不并行起 run**；monitor 的 25 分钟停滞看门狗在 05:54 停掉停滞 run 并重开 | ✅ 06:20 复测 `{'ok': True}`（5s），run `d27d37aab8c9` **自愈续跑**（events 9→136） |

**新发现（建议改进 monitor）**：停滞看门狗"到点直接停 run"，与 skill「上游不可达时应等待自愈」冲突
—— 建议 `--stall-kill` 前先做一次 1-token 上游探测，不可达则只告警不杀。**本夜被看门狗杀掉 2 个 run**：
`0a4d2bb8d30b`（01:57:08 杀，已跑 403 events / ~30 分钟，死前正卡在 turn=5 的首次 LLM 调用）与
`59f810e5f170`（05:54 杀）。→ 该改进已实现为修复 #9。

### 三、7 个修复与实测效果

| # | commit | 内容 | 效果证据 |
|---|---|---|---|
| 1 | `cf3cd49` | 血统漏传 `parent_factor` 自动补全 | `report_lineage_autofill` **135 次**（整夜）；"空传"类拦截 **2 → 0** |
| 2 | `c2fbf09` | 血统比对忽略大小写/下划线 | "同课题被误杀"拦截 **17 → 0** |
| 3 | `ff354f3`+`ad82ba5` | 记忆推荐先按课题血统过滤（含复现版名） | `memory_suggest_lineage_filter` **22 条**被提前丢弃（不再白烧） |
| 4 | `fd1a1df` | 复现门禁接受"本课题复现版名" | 拦截明细里该类消失 |
| 5 | `b3b7aa1` | `expected_strong_side` 支持 `high_quality/high_div` 等前缀写法 | 别名表补全（+1 测试） |
| 6 | `6044e9e` | 复现阶段死路**只提醒不硬拦** | 提出该问题时：`MemoryAdvisoryBlock` 62 次里 **14 次在复现阶段**（RQ_2f2436 单题 11 次）→ 互锁已解；**但生产只部分生效**：run `d27d37aab8c9` 复现阶段仍有 11 次硬拦（被 advisory 缓存陈旧命中）→ 由修复 #10 根治；#6 后的软提醒（`memory.advisory`）4 次仍可见 |
| 7 | `878c827` | DSL 字段别名改写提前到面板裁剪之前 | 修 `$turnover` 误报"不可用字段 `$turnover_rate`"（≥4 次/夜）；**生产验证**：修复后 run `eb785f4298b9` 里出现了同一条先前必挂的表达式 `CROWD_SHARE($turnover, $amount, 20, "high", 0.8)`，该 run `不可用字段` 报错 **0 条** |
| 8 | `15c5d79` | `focus_facets` 失效时显式告警（+ 修长期飘红测试） | 题库 196/196 无 `facets` → 聚焦功能静默失效；现留 WARNING |
| 9 | （仓库外）`overnight_mining_monitor.v2.py` | 停滞 kill 前先做 **1-token 真实上游探测**：不可达则只告警 + 重置计时，不再白丢 run（`--no-stall-probe` 可关） | 本夜 01:31–06:20 上游挂 4.8h、7 个 run 反复被杀/重开；实测探测函数可用（可达 1460ms）｜备份 `…v2.py.bak-20261004-0632`、`py_compile` 通过 |
| 10 | `8027189` | advisory 缓存键加入研报阶段（`_phase_tag()`） | 属**同一问题的第二次尝试**（第一次 #6、第二次 #10、第三次 #12 才对）：#10 之后 run `eb785f4298b9` 复现轮仍 6 次硬拦 → 真因是"网关 phase 被 PASS 翻转"，见 #12 |
| 11 | `scripts/patrol_report_mode.py`（随 #10 提交） | 补上 skill 点名却**缺失**的研报巡检脚本（复现过线率/保真度/门禁三分法/死路按阶段/错误分布，只读） | skill 与实现脱节已消除；输出与手写统计一致 |
| 12 | `eba108f` | 复现豁免改读**轮次阶段** `turn_phase`（`_dispatch` 在复现过线时会把网关 `phase` 翻成 diverge） | **同一问题第三次才修对**：run `eb785f4298b9` 时间线 `06:33:46 STATE reproduce → 06:34:09 REPRO_PASS → 06:34:43 起 6 次复现轮硬拦`；测试回退后失败、打上通过。**生产验证**：修复后启动的 run `ff59255e6b38`（06:58:08）实测 `memory.advisory_block` **0 次** / `memory.advisory`（软提醒）**10 次** —— 死路提醒保留、硬拦归零 |
| 13 | `2c1c4b2`+`b5d2ed3` | `scripts/backfill_question_facets.py` + **题库 facets 回填**（**160/196** 道）——恢复"数据面聚焦"功能 | 之前 `focus_facets` 因题库 196/196 无 facets 而**静默失效**；回填用既有 `FACET_DEFS`/`expr_facets` 确定性推导（不发明词表），缺省干跑、`--apply` 自动备份。验证：`focus=价量面/两融面` 均返回对口题；`披露面`（零覆盖）走 #8 的 WARNING 回退；`test_question_field_gate` 17/17 绿。剩 36 道无面成因已分类：30 道题面本就无可识别要素、4 道被准入门排除（ml_model/graph_deep）、2 道字段不属任何既有面（`$rc_rating`/`ann_flag_buyback`） |

**bench**：基线已从 `fc9bdcbd505b`（spec `0dac03c`）刷新为 **`65eedb4b8777`（spec `39430ac`）**，
与今夜所有 run 身份一致，后续 `bench diff` 可直接判定（旧基线已归档到 `artifacts/alphaagent/bench/history/`）。
**实测验证**（`bench.py diff d27d37aab8c9`）：

```
对比基线: 65eedb4b8777 | 基线配置 spec:39430ac…|mode:report（当前配置 … [一致]）   ← 不再是 INVALID_CONFIG_MISMATCH
有效新颖率 0.9674 → 1.0 [+]   重复尝试率 0.0326 → 0 [-]   死路提示遵循率 0.9167 → 1.0 [+]
评估耗时 P50 9.87s → 18.51s [-] / P95 27.31 → 77.14 [-]   ← 见下"耗时归因"
```
**耗时归因（排除今晚改动）**：分段计时对比（`steps.log` 的 `dsl_ms`/`tf_ms`/`mtc_ms`）：

```
65eedb4b8777(修复前) dsl P50=4514ms  tf P50=1482ms  mtc P50=1445ms
6597fe0e0c84(#1-#4) dsl P50=4504ms  tf P50=1600ms  mtc P50=1544ms
eb785f4298b9(#7/#10) dsl P50=2999ms tf P50=2914ms  mtc P50=1864ms
```
→ **DSL 段没有变慢**（#7 的别名改写开销可忽略），慢在 transform/metric 段，
与本夜改动无关（更可能是逐 run 算子/JIT 组合与机器负载差异）；单 run 样本不足以定论。

### 四、遗留问题（下轮/明天处理）

| # | 问题 | 证据 | 建议 |
|---|---|---|---|
| 1 | run `0a4d2bb8d30b` 中途结束（events=403、无 Traceback、无 run_summary） | **已定位（修正早前"疑似 JIT 崩溃"的猜测）**：monitor 日志 `01:57:08 停滞 26 分钟 ≥25 → 停止 run` —— 是**停查看门狗杀掉的**（上游中断期），exit code 1 且无 Traceback 属预期；不是代码崩溃 | 修复 #9 已覆盖（探测不可达则不杀） |
| 2 | monitor 停滞看门狗与"上游等待自愈"冲突 | 见 §二 | ✅ 已修（修复 #9，仓库外脚本，备份 + `py_compile`；下次启动生效） |
| 3 | ~~`tests/test_question_field_gate.py::test_ungated_matches_legacy_selection_with_focus` 既存失败~~ → **已修**（`15c5d79`）：根因是**题库 196 道整体没有 `facets` 字段**，`focus_facets` 过滤恒空、静默回退全池 | 196/196 无 facets（含昨夜 103 题版本与扩容源文件）；`get_question_for_turn` 实测三种 focus 全返回同一题 | 已加 WARNING 留痕 + 测试钉住"回退全池 + 告警"现实契约；**题库补面标签**已由 #13 回填 156/196（数据文件不在 git，已备份 `.bak-facets-20261004-064554`） |
| 4 | skill 引用的两个脚本在本仓不存在 | `scripts/patrol_report_mode.py` / `night_summary_report.py` | ✅ 已补 `patrol_report_mode.py`（#11）；`night_summary_report.py` 仍缺（低优先） |
| 5 | `memory_blocked_duplicate` 仍是最大错误类（62 次 `MemoryAdvisoryBlock`） | 6 个 run 的 failure_counts | 属设计性拦截；复现阶段已豁免（#12），其余看模型是否换结构 |
| 6 | 每次 run 的 preflight 冒烟含**一次完整盲测**：`submit.blind_test \| __preflight_smoke__` 实测 **89s / 143s**（占总时长 5–10%） | `agent/preflight.py::check_submit_smoke`（刻意端到端，用于暴露索引契约级 bug；本夜确有 1 次 `BlindTestError`） | ⏳ **只记录不改**：这是 deliberate fail-fast，砍掉会丢覆盖；若要提速需先确认盲测段是否可安全短路 |

### 六、本夜最大教训（写给下一个夜班）

**同一个 bug 修了三次才对**：复现轮的死路硬拦（#6 → #10 → #12）。前两次都在"缓存/键"层面找原因，
第三次才回到**时间线证据**：`report_state_machine`(reproduce) → `REPRO_PASS` → 硬拦，一眼看出是
"`_dispatch` 在过线那一刻把网关 `phase` 翻成 diverge，而本轮题面仍是复现"的**状态撕裂**。
教训：**先按时间线取证，再谈机理**；diff 级猜测会连修两轮无效。

**顺带观察（不改）**：同一次翻转也会让"发散门禁"接管本轮剩余调用（要求 `parent_factor` 指向复现版名）。
语义上可接受（过线后本就应该发散），且拦截文案明确；今夜实测未出现该类误拦。

### 五、诚实声明

1. 本夜**候选池与正式库均为 0**；有效挖掘窗口仅 2.5 小时（上游断 4.8 小时）；
2. 所有数字取自各 run 的 `run_summary.json` / `steps.log`，统计脚本口径：门禁拦截三分法（空/同课题/跨课题）、
   死路拦截按阶段归类、autofill/推荐过滤按 `steps.log` 锚点计数；
3. 7 个修复**只对修复后启动的 run 生效**（在跑子进程已加载旧码），效果证据均取自后启动的 run；
4. `0a4d2bb8d30b` 早前被我记为"疑似 JIT 崩溃"，**后经 monitor 日志定位为停滞看门狗所杀**（01:57:08），
   已在 §四 更正 —— 保留这条更正痕迹，避免"看起来像崩溃"的猜测被当成结论。

