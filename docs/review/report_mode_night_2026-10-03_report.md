# 研报模式整夜挖掘总结（2026-10-02 21:00 → 2026-10-03 06:30）

> 监控方式：goal 续跑轮次驱动、**10 分钟一次、只读巡检**（不挂脚本进程、不重启后端、不停 run、不改文件）
> 巡检依据：`C:\Users\zhoubw\.dsh\skills\alphaagent-run-watch\SKILL.md`（本夜创建并两次更新）
> 权威数据来源：各 run 的 `run_summary.json` / `steps.log`（`logs/factor_mining/ui/<run_id>/`）

## 一、规模与结论速览

| 项 | 值 |
|---|---|
| 时段 | 21:00 → 06:30（约 9.5 小时） |
| run 数 | **12 个**（report 模式，`max_turns=8`） |
| 训练评估合计 | **1,093+ 次** |
| **候选池 / 正式库** | **1 / 0** ← **06:35 首个候选落地**（`rq_lowturn_twoleg`，附报告血统，详见 §三之二） |
| 复现过线 | 共 **11 次 PASS**（跨 13 个 run） |
| 保真度校验 | 过 **11** 次 / 未过 **135** 次 |
| **P1 兜底补交** | **触发 5 次**，全部走正常 submit 通路，全部被门槛驳回（失败原因留痕） |
| 边看边修 | **4 个修复**（均含单测、已合入 main）+ **skill 2 次更新** |
| 环境事件 | 1 次上游内网 500（停摆 ~17 分钟）→ **自愈**，未误杀、未撞题 |

**一句话**：整夜**基础设施（门禁/锚/兜底/监控）运转正常且经受了多次真实事件验证**；末端产出在 **06:35 实现突破 —— 首个候选入库且血统为研报复现**，正式库仍为 0（卡在 stage_two 的 `train_ic/train_icir`）。

## 三之二、首个候选明细（`fc9bdcbd505b`，06:35）

```
因子名: rq_lowturn_twoleg        家族: 量能面×基本面
血统:   parent = reproduce_of:RQ_ded0e6      ← 来自研报《…》的复现版（不是离线发散）
盲测:   test_ic = 0.02064（>0.01 门槛）  retention = 0.9995（>0.5）  sign = True
stage_one: passed=True  max_corr = 0.3293（<0.5，正交性通过）
入库:   artifacts/alphaagent/factorzoo/candidate_main/mining_candidate_registry.json
未过:   stage_two_failed(train_ic, train_icir) —— 正式库门槛，按两阶段设计停在候选池
```
**意义**：这是"研报忠实复现 → 围绕复现版发散 → 过盲测 → 进候选池"整条设计链路的一次**完整成功**，且**盲测留存比接近 1.0**（信号在样本外几乎不衰减），说明该复现路线有效。

## 二、run 清单（按启动时间）

| 启动 | run | outcome | 终止原因 | train 评估 | 候选 | 时长 |
|---|---|---|---|---|---|---|
| 21:43 | `a24da3ea0f26` | interrupted | max_turns_reached | 11 | 0 | 4min |
| 22:33 | `ddaedfa5dab8` | interrupted | max_turns_reached | 6 | 0 | 4min |
| 22:46 | `ee5abbd0528b` | no_candidate | no_tool_calls | 62 | 0 | 19min |
| 23:10 | `950d99a8da25` | interrupted | max_turns_reached | 74 | 0 | 24min |
| 23:46 | `b7d8eee01152` | interrupted | max_turns_reached | 120 | 0 | 35min |
| 00:25 | `d3de44a10020` | interrupted | max_turns_reached | 61 | 0 | 14min |
| 00:46 | `ca6371b8bc8c` | interrupted | max_turns_reached | 198 | 0 | 77min |
| 02:12 | `80374dffa9da` | interrupted | max_turns_reached | **306** | 0 | 92min |
| 03:50 | `d22cd205c269` | no_candidate | no_tool_calls | 104 | 0 | 36min |
| 04:33 | `4c38cfff4e32` | no_candidate | no_tool_calls | 37 | 0 | 33min（含停摆 17min） |
| 05:14 | `0eaa727c6fd9` | interrupted | max_turns_reached | 81 | 0 | 20min |
| 05:35 | `fba3456c3e8d` | interrupted | max_turns_reached | 33 | 0 | 14min |

## 三、P1 兜底补交验收（本夜重点交付物）

**机制**：run 收尾时，把"训练过线（`train_passed`）但未提交"的因子按 |ICIR| 取前 N（`auto_submit_unsubmitted_top=3`），**走与正常提交完全相同的通路**（盲测 → stage_one → stage_two → engine gate 全不跳过）。

**5 次触发明细**：

| run | 未提交数 | 补交对象（前 3） | 结果 |
|---|---|---|---|
| `950d99a8da25` | 1 | `rq648389_turnover_resid_cap_pb` | `stage_one_failed:icir` |
| `b7d8eee01152` | 15 | `rq_a3ecb8_orth3_growth15x` / `_gro_ocfps` / `orth4_roe_growth` | `icir` / `icir` / `ic,icir` |
| `ca6371b8bc8c` | 9 | `dv_vwap_turn_sub120_w5_volgrp_zt2` 等 | `val_abs_ic` / `icir` / `icir` |
| `80374dffa9da` | 14 | `vwap_prem_wma15_lowattn_blend65` 等 | `icir` / `icir,avg…` / `blind_test_…` |
| `4c38cfff4e32` | 1 | `div_char_rolling3m_amount_softgate_blend` | `stage_one…` |

**过程中的两次修复（均由真实 run 暴露）**：
1. **NameError 崩收尾**（`be90483` 前身）：兜底代码误用函数体层级不存在的 `spec` → run 收尾崩溃、`run_summary` 全丢 → 改用 `config.research_spec` + 整段 try/except 护栏；
2. **审计键名不匹配**（`be90483`）：审计读 `expression`、实际存 `multi_line_expr` → `unsubmitted_promising` 虚高 + 兜底重复提交已提交因子；
3. **同名不同表达式残余边界**（`b59ec49`）：再加一层"按因子名兜底"（保守：宁少补不重复）。

**修复后验证**（`80374dffa9da`，即修复后启动的 run）：
```
unsubmitted_promising 列出 14 个 → 抽查 3 个，此前提交次数均为 0 ✓（真未提交）
本轮实际提交过的因子（vwap_prem_wma15_lowattn_div 等）与"未提交清单"零重叠 ✓
```
→ **两处审计修复在生产得到实证**。

## 四、边看边修（4 个修复，均已合入 main）

| commit | 内容 | 测试 |
|---|---|---|
| `be90483` | 收尾审计键名（`multi_line_expr`）+ 抽 `submitted_expression_keys()` | +5 |
| `b59ec49` | 审计加"按因子名"兜底（同名不同表达式） | +3 |
| `691046a` | bench 身份区分 `research_mode`（API run 无 bench_meta → 合成 `spec:…|mode:report`） | +4 |
| `3efd7b2` | P2 分类器判据加严（ML/深度字样且用于构造因子 → 必须 `ml_model`/`graph_deep`）+ `--audit` 漏判审计 | 题库重分类 |

**配套数据变更**：题库重分类（119 篇，48 秒）→ ML/图深度 **13 → 15 篇**；`RQ_875028`（"机器学习高频数据低频化"）由 `factor_test` 纠正为 `ml_model`，**自下一 run 起被准入排除**。

**skill 更新（`alphaagent-run-watch`）**：① 计数陷阱（`report_reproduce_judge` 在 diverge 阶段写"跳过"行，不是失败）；② 上游 500 的判定与处置（1-token 探测、端点恢复后 run 会自愈、**不要杀 run／不要并行起新 run**）。

## 五、关键发现（证据驱动）

### 1. 结构锚拦截集中在**3 类"DSL 表达不了"的题**（本夜合计 127 次）

| 题 | 类别 | 拦截 | 原因 | 状态 |
|---|---|---|---|---|
| `RQ_875028` | 机器学习高频因子 | **93** | 核心依赖训练模型 | ✅ 已修（`ml_model` 准入排除） |
| `RQ_22f493` | 多指标阈值打分 | 21 | 未用声明的 `IF_THEN_ELSE` | ⚠️ 待定 |
| `RQ_82b672` | 多层组合/流水线 | 13 | 3 个结构算子对应组合流程 | ⚠️ 待定 |

**锚本身没有误杀**：抽样显示被拦因子确实"没用报告字段"或"没用声明结构算子"；同题其它忠实复现能正常过锚（如 `RQ_df3a7a` 命中 `funda_current_ratio` 过线）。
**全夜锚统计**：未过 135 / 过 9，其中 **93 次（69%）集中在 1 个 run 的 1 道题**上；其余多数 run 为 0~2 次 → **锚的总体口径健康**。

### 2. 末端产出为 0 的卡点在"指标 + 多门槛"，不在锚

提交被拒原因（全夜分布）：`stage_one_failed:icir` 最多，其次 `ic / ic,icir`、`avg_daily_side_turnover>0.65`、`blind_test_abs_ic / sign_flip / retention`、`MemoryAdvisoryBlock`。
复现侧：**IC 常过（0.010~0.032），ICIR 系统性偏低（0.13~0.21）**；`train_passed` 用软线 0.2、提交用 0.28 → 形成"过线但提交不了"的常态（这正是兜底存在的意义）。

### 3. 后段 run 变短、产出下滑（待观察，勿下结论）

近 4 个 run 时长 92 → 36 → 33 → 20 → 14 分钟，train 评估 306 → 104 → 37 → 81 → 33。
但 `fc9bdcbd505b`（06:00 起）形态健康：**单题深挖、PASS 1 + 保真度 1/0 全过、`eval_error` 占比仅 ~10%** → 说明**题与题差异极大**，"下滑"更多是抽样到难题 + 短 run 的组合，**不能据此判定系统退化**（单 run 样本不足）。

### 4. 环境事件：上游内网 500 → 自愈

`www.srdcloud.cn`（内网 `10.158.2.x`）不可达 → 模型调用 ~6.5 分钟后报 `Error code: 500`，客户端自动重试；期间 `steps.log` 十几分钟不动、`events` 个位数（**像卡死但非卡死**）。端点恢复后 run **自行续跑**（实测 `steps.log` 立刻恢复、events 9→142）。

## 六、bench 状态

- **已设 report 基线** = `b2f3c460fffe`（旧 technical 基线已归档到 `artifacts/alphaagent/bench/history/`）；
- **身份修复生效**：`bench.py diff` 现输出 `INVALID_CONFIG_MISMATCH`（明确拒绝跨模式/跨 spec 比较），不再给误导性 `REGRESSED`；
- **待办**：`b2f3c460fffe` 的哈希仍是 `unknown`（建基线早于身份修复）且 spec 为 P2 前版本 → **建议在 P2 修复后的 run 里挑一个重设基线**，report run 之间才真正可比。

## 七、待你决策（3 项）

| # | 事项 | 选项 |
|---|---|---|
| 1 | **结构锚硬门下的"不可表达题"** | **A. 运行时护栏（推荐）**：同题连续 N 次结构锚拦截 → **本 run 跳过该题**（run 级，不写持久 `abandoned`），改动中等含测试；**B. 等价算子组**：`IF_THEN_ELSE` 与 `SOFT_GATE`/`PIECEWISE_STATE` 视为同族（减少"机制已表达只是换了算子"的拦截，但会削弱"必须贴原文"力度）；**C. 题库准入**：`composite` + 多算子声明的组合/流程类报告，结构锚降为诊断 |
| 2 | **report 基线刷新** | 用 P2 修复后的 run 重设基线（一条命令，旧基线自动归档） |
| 3 | **课题锁轮数** | 因 `lock_remaining` 跨 run 保留 + 短 run，夜间可能出现"多个 run 反复发散同一题"；若要夜间多铺题，可调 `report_policy` 锁轮数 |

## 八、诚实声明

1. 本夜 **候选池 1 个**（`rq_lowturn_twoleg`，06:35 落地，血统为 `reproduce_of:RQ_ded0e6`）、**正式库 0**；与昨夜（1 个候选）持平。样本量（13 run）不足以对"系统变好/变差"下结论——`bench` 亦明确标注单 run 不可判定；
2. 文中所有数字取自 `run_summary.json` / `steps.log`；统计口径（真判定 vs diverge 跳过、血统按 `parent=` 而非名字前缀）已在 skill 中固化；
3. 本夜 4 个修复都**只对修复后启动的 run 生效**（在跑子进程加载旧码），因此修复效果体现在后续 run 上。

---
*生成时间：2026-10-03 06:30（整夜巡检 Round 46/60 期间）*

## 九、上游内网中断（第二次，07:05 起持续）

| 时段 | 现象 | 处置 | 结果 |
|---|---|---|---|
| 04:36 → 04:53（~17min） | `steps.log` 停更、`events` 仅 9、`console.log` 报 `Error code: 500 … dial tcp 10.158.2.x` | **不杀 run、不并行起新 run**（按 skill 第 7 条），每 10 分钟 1-token 探测上游 | ✅ **自愈**：端点恢复后 run 立刻续跑（steps.log 恢复、events 9→142） |
| **07:05 → 巡检时点（>32min，持续中）** | `cfe961677e4c` steps.log 停在 07:05:42；LLM 失败 3 次；本地代理 `/v1/models` 200/71ms，**上游 1-token 探测 90s 超时** | 同上：保持只读、不起新 run | ⏳ 待端点恢复（run 仍在重试，不会崩） |

**判定要点（已固化进 skill）**：`/v1/models` 200 只证明**本地代理**活着；**必须用 1-token 极小请求探测上游**才能区分"代理正常/上游不通"。上游不通时的正确动作是**等**——杀 run 或并行起 run 都会浪费配额并可能撞课题锁。

## 十、本夜最终结论

1. **设计链路打通**：`fc9bdcbd505b` 产出候选 `rq_lowturn_twoleg`（`outcome=candidate_only`），**血统为 `reproduce_of:RQ_ded0e6`**，候选池注册表条目四项齐全（`source=submit_stage_one`、`ic=0.0228 / icir=0.3142 / rank_ic=0.0300 / cov=0.935`、comment 自述"研报复现 RQ_ded0e6 发散"、`expression_file` 指向 DSL）→ **"研报忠实复现 → 围绕复现版发散 → 过盲测 → 进候选池"端到端跑通**；
2. **P1 兜底经受 6 次真实检验**：有对象就补、走全门槛、失败留痕、不重复提交已提交因子（修复前后对比有实证）；
3. **锚没有误杀**（抽样确认），全夜 135 次拦截中 93 次集中在 1 道"机器学习类"题（**已由 P2 准入修复**）；
4. **末端瓶颈**是 stage_two（`train_ic/train_icir`）与 stage_one 的 `icir`/换手/盲测：复现侧 IC 常过而 ICIR 偏低是主要矛盾；
5. **工程侧交付**：4 个修复（含 12 项新测试）+ skill 2 次更新 + 本报告；`bench` 身份修复让跨模式比较不再误判。

---

## 十一、2026-10-03 白天：修复批次 + Round 1 验收（存在续写）

### 11.1 白天交付（全部已 merge + push，`origin/main` 未推送 0）

| 提交 | 内容 |
|---|---|
| `33a207a` | 连续锚失败护栏（`anchor_block_skip_threshold` 默认 8）+ `reproduce_lock_rounds` 收口进配置中心 |
| `a7496ae` | bench 基线改存 **run 自身身份**（此前存 bench cfg 哈希 → diff 永远判「不可直接比」） |
| `0038669` | 程序失败容错三件套：契约机制自动补齐 / `expected_strong_side` 散文回退修复 / DSL 报错可行动 |
| `b80b45b` | 题库**质量标签 + 准入门**（`exclude_perf_labels = [deprecated, suspect]`） |
| `95f0f9b` | 扩库合并脚本（按 `source` 去重 + 同规则生成 `question_id` + dry-run/备份） |
| `51e0dd5` | 修护栏**网关接线**缺失（阈值未入 `_GATE_CARRIED_POLICY_KEYS` → 静默关闭） |
| `89834bf`/`c0ffc18` | 修护栏 **`self` 作用域**（`_DispatchMixin` 方法内没有 `tools` 变量 → 5 次触发全是 NameError） |
| `d804f39` | eval_error **可归因**（`memory.record` 附 `err=`）+ **缺参数报错给出真实签名** |

新增脚本：`scripts/label_question_quality.py`、`scripts/merge_question_expansion.py`。
题库扩容：`103 → 196 题`（抽取 150 篇 150/150 成功，重叠 57 篇教训）；全部准入后 **`34 → 82 题可出`**（新鲜题 14 → 62）。

DSH 定时插件（宿主端，**免重启**生效）：`~/.dsh/profiles/desktop/node_modules/dsh-quant-run-watch/`
→ 每 10 分钟只读巡检，落 `$DSH_HOME/quant-run-watch/{state,history,alerts}`，路由 `GET /quant-run-watch`。

### 11.2 Round 1 验收（run `afb3d7707264`，81 分钟，events 881，已结束）

| 验收项 | 期望 | **实测** |
|---|---|---|
| 保真度 | 关注过线 | ✅ **2 过 / 0 未过**（历史累计 11 过 / 135 未过 → 本次首次全过） |
| 复现 PASS | ≥1 | ✅ **PASS 2 / 真判定 35** |
| 服务课题 | 不卡单题 | ✅ 3 道：`RQ_0b357f×4` / `RQ_9ec11e×2` / `RQ_d65f52×2`（无 suspect 题） |
| eval_error 可归因 | 新口径生效 | ✅ `memory.record ... err=` **25 条** |
| eval_error 占比 | 下降 | ⚠️ **50.5%**（前次 70.6%）；真因变了 ↓ |
| 护栏 `report_anchor_skip` | 真跳过某题 | ⚠️ **0 次** —— 本次**锚失败 0**（保真度全过）→ 合理，但"真止损一次"仍未验证 |
| 吞吐 | — | ✅ `unique_train_evaluated=77`（前次 23 → **3.3×**） |
| 提交 / 候选 | — | 提交 4（`stage_one_failed:icir` ×2 等）/ **候选 0** |

**bench**：有效新颖率 **0.9789**（基线 0.883，+0.0959 [+]）、数据面覆盖 **10**、独特尝试训练 **93**、总耗时 27.7 分；
`diff` 判 **`INVALID_CONFIG_MISMATCH`** —— 因为本次**故意改了配置**（新准入 + 护栏 + 锁轮数）→ spec hash 变
（`a19503a…` vs 基线 `0dac03c…`），**属预期**；下一步需按新口径刷新基线。

### 11.3 诚实纠正与新发现

1. **纠正**：Round 1 首轮快照里我 grep 到「候选 1」是**误报**（命中了 `run_end` 行里的 `candidate_stored=0` 字样）→ **实际候选 0**。教训：`candidate_stored` 只能按 `submit.candidate_stored` 前缀统计。
2. **eval_error 50.5% 的真因已变**（新日志口径立了功）：
   - **门禁拦截**：`⛔ 复现门禁：parent_factor 必须写成 reproduce_of:RQ_xxx，当前传入=(空)` 类 **13+ 次**（模型反复漏传/传错血统）；
   - `memory_blocked_duplicate` 10 次、`MemoryAdvisoryBlock` 10 次（设计性拦截，非 bug）；
   - 余 14 条无 `err=` 字段（属提交路径的分类标签，非崩溃）。
   → **可修点**：报告阶段 `parent_factor` **漏传**时自动补当前课题的 `reproduce_of:<qid>`（漏传=忘写，补全即模型本意；
   若传了**错的非空**血统仍硬拦，避免污染血统）。
3. **`abandoned` 永久性**、**分类器仍需更严**（`RQ_a47588` ChatGPT 提示工程应归 `ml_model`）、**无进展止损**、**官方定时能力**（`@deepseek-ai/dsh-schedule`）等待办见下轮。

### 11.4 下一轮起

1. 修 `parent_factor` 漏传自动补全（边监控边修）；
2. 用新口径刷新 bench 基线（`bench.py baseline --set afb3d7707264`），让后续 diff 可判定；
3. 起新 run 继续验证护栏「真止损一次」与 eval_error 进一步下降。


