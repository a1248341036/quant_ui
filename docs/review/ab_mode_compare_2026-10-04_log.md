# A/B 模式对照：自挖掘(technical) vs 研报复现(report) —— 2026-10-04

> 目标（用户 2026-10-04 定）：**同条件**对比"挖掘模式"与"研报模式"，**每种各跑 2 小时**，
> 巡检节奏 10 分钟一次。目的：给"复现研报是不是不如自己挖"这个问题一个**可裁决**的数据。

## 一、协议（固定不变项）

| 项 | 取值 |
|---|---|
| 后端 / 数据 | `http://127.0.0.1:17891`，panel=`cne://`，model=`DeepSeek-V4-Flash-0731` |
| 训练 / 验证 | train `2020-01-01~2022-12-31` / val `2023-01-01~2024-12-31`，label=`label_1d_open_to_open` |
| 轮数 | `--max-turns 8`（两段一致） |
| 数据面 | **自由探索**（两段都不传 `--focus-facets`） |
| 代码 | `main @ 99b823a`（含 10-04 夜间 13 项修复）；每段启动时 monitor 会重启后端以加载磁盘最新代码 |
| 用户消息 | 默认消息（两段相同） |
| 监控 | `overnight_mining_monitor.v2.py`，`--poll 60`、`--stall-warn 15`、`--stall-kill 25`、`--restart-backend` |
| 唯一变量 | `--research-mode technical` vs `--research-mode report` |

## 二、两段安排（顺序执行，各 2 小时）

| 段 | 模式 | 起 | 止 | monitor PID | run |
|---|---|---|---|---|---|
| **A** | `technical`（自挖掘） | 10:24:07 | **12:31:46 退出**（块长 2h07m） | 67092 | `97a7861e91a7`（75.3min，完整） / `8e52e39bab47`（51.6min，**在 2h 窗口边界被我手动停掉**） |
| **B** | `report`（研报复现） | 12:32:56 | 计划 14:33（块长 2h） | 25044 | `2db9102658a7` 12:35:45 起（已确认 `research_mode=report`） |

**用于裁决的等长窗口（各 2h，均从该段第一个 run 起算）**：
- A：`2026-10-04 10:24:44 → 12:24:44`
- B：`2026-10-04 12:35:45 → 14:35:45`

**段 A 窗口内实测（逐行时间戳口径）**：`runs=2`、`evaluate 行 747`（val 行 8）、`submit_result 行 2`、
**候选落地 0**；其中完整 run#1 的权威 summary：`train=313 / val=6 / candidate=0 / usage_calls=119 / 75.3min`。
关键过程证据：`verdict=train_passed` 累计 48 个（run#1）；兜底补交 `vw_res_pure_zscore__autorescue`
**过盲测**（test_ic=0.01999 / retention=0.6463 / sign=True）却**卡 stage_one**
（`cs_autocorr` + `avg_daily_side_turnover=1.67 > 0.65`）；run#2 收尾同类拒绝
（`vwap_premium_size_neutral__autorescue` turnover=1.644）。

**运维插曲（值得记）**：我在 2h 边界用 API 停掉 run#2 后，段 B 启动时**后端把该 run 重新标成 running**
（子进程其实已死、API 仍显示 events=2243）→ 段 B 的 monitor 把它当成"自己的活动 run"等了 2 分钟并计入一次快速失败；
**再停一次**该 run 状态才变 `stopped`，段 B 随即在 12:35:45 起了真正的 report run。

> 顺序执行意味着两段处于不同时段（上游/机器负载可能不同）；**若要排除时段效应，需再跑一次反向顺序**。
> 本页会如实标注这一点，不做"单次即定论"的结论。

## 三、对比口径（预先定死，避免事后挑指标）

主指标（产出效率）：
1. **候选池入库数 / 训练评估数**（两段各除以自己的评估数）；
2. **正式库入库数**；
3. **每小时候选数**（按各段实际墙钟时长归一，排除中断时间）。

辅指标（过程质量，bench.py 已有）：`有效新颖率`、`重复尝试率`、`数据面覆盖数`、`结构指纹多样率`、
`信号族覆盖数`、`评估耗时 P50/P95`、`总消耗 Token`。

报告模式专属（只用于解释成本，不参与胜负）：复现判定数 / PASS / 保真度过检率。

口径提醒：两段的 `research_mode` 不同 → **spec 哈希不同 → `bench.py diff` 会判"配置不一致"**，
所以本对照**逐指标比较**，不用 `bench diff` 的总体判定。

## 四、运行记录（随轮更新）

- 10:24:07 段 A 启动（technical，deadline 12:30）；10:24:42 后端就绪；10:24:44 起 run `97a7861e91a7`。
  `run_start` 确认：`mode=technical`、train/val/label/panel 与昨夜 report run **完全一致**（唯一差异是 mode）。
- 10:28:06 monitor `GET /runs` 20s 超时 → 触发后端重启（**与昨夜 23:24 同款瞬时阻塞**）；
  run `97a7861e91a7` **未受影响**（events 3→176 继续增长）→ 又一条"后端重启不杀活动 run"的实证。
- 10:36 轮询：run `97a7861e91a7` events=176、last=assistant_tool_call（正常推进）。

- 10:36–11:38 轮询：run `97a7861e91a7` 持续推进（events 176→2879）；技术模式**评估吞吐明显高于 report**
  （20 分钟 130 次 → 60 分钟 374 次），且 `verdict=train_passed` 累计 **48 个**。
- 11:31–11:38 收尾 + 兜底：`auto_submit_rescue` 依次补交高潜因子，其中
  `vw_res_pure_zscore__autorescue` **过了盲测**（test_ic=0.01999 / retention=0.6463 / sign=True）
  但**卡在 stage_one**：`cs_autocorr` + `avg_daily_side_turnover=1.67 > 0.65`
  （weekly 调仓下日单边换手过高 → 实盘不可交付）→ 诚实被拒。
- **11:40:04 段 A run#1 结束**：`97a7861e91a7`，**75.3 分钟**，train=**313**、val=6、tool_calls=436、
  `max_turns_reached`、**候选 0**；monitor 随即起 run#2 `8e52e39bab47`（11:40:04）。
  对照昨夜 report run：平均 68 次评估 / 27 分钟 → 技术模式评估吞吐约 **4.2 次/分** vs report **2.5 次/分**。

### 对比脚本（已就绪，待两段跑完出表）
`qp_ab_compare.py <A标签> <A起> <A止> <B标签> <B起> <B止>`：按模式聚合 run_meta/run_summary/steps.log，
产出 `runs / train / val / 候选 / 正式库 / tokens / calls / 复现判定 / PASS / 保真` +
`候选每百评估`、`候选每 run`，并把 `scorecard.json`（bench 台账，schema_version=4）**整棵展开**逐指标均值。

---
*本文件是本次 A/B 的过程记录 + 结果汇总载体；结论见下节。*

## 六、反向一轮（段 C/D，2026-10-04 下午）——为排除"时段效应"

> 口径：窗口内**逐行时间戳**统计（`steps.log`），不去重；两段规则完全一致。
> 复现判定/PASS/保真度**已剔除"跳过"行**（非研报模式也会写 judge 跳过行——skill 固化的计数陷阱）。

| 指标 | **technical（自挖掘）** | **report（研报复现）** |
|---|---|---|
| 窗口 | 10:24:44 → 12:24:44 | 12:35:45 → 14:35:45 |
| run 数 | 2 | 4 |
| **评估行数（train+val）** | **747** | **499** |
| 其中 val 行 | 8 | 8 |
| **提交次数** | 2 | 3 |
| **候选池落地** | **0** | **0** |
| 正式库落地 | 0 | 0 |
| 真复现判定 | 0（该模式无此阶段） | **252** |
| 复现 PASS | — | **4**（1.6%） |
| 保真度过检 | — | 4 / 30 |
| 候选 / 百次评估 | 0.00 | 0.00 |
| LLM 调用 / tokens（summary 口径） | 119 次 / 6.45M（run#2 无 summary，**实际更高**） | 129 次 / 8.01M |

**结论：等长 2 小时里，两种模式候选产出都是 0。**

### 5.1 自挖掘（technical）过程

- 完整 run `97a7861e91a7`：**75.3 分钟、train=313、val=6、tool_calls=436**、`max_turns_reached`、候选 0；
  `verdict=train_passed` 累计 **48 个**。
- 交付卡点：`vw_res_pure_zscore__autorescue` **过了盲测**（test_ic=0.01999 / retention=0.6463 / sign=True）
  但 stage_one 卡在 **`avg_daily_side_turnover=1.67 > 0.65`** + `cs_autocorr`；run#2 同类
  （`vwap_premium_size_neutral__autorescue` turnover=1.644）。
  → 印证历史教训：**1d 快速信号配 weekly 调仓 → 换手超标**。

### 5.2 研报复现（report）过程

- 4 个 run（42.4 / 20.7 / 43.5 分钟 + 1 个跨窗），252 次真判定 → **4 次 PASS**（1.6%，与昨夜 4.3% 同量级）。
- **指标过线但被形态/保真锚拦下**是主要损耗：run `2db9102658a7` 有 7 条 `metrics=过`，
  却因 `shape=contradicted` 或 `off_reference:shared_specific_fields=0<1` 未过；
  例如 `rq_roe_z12_probe` IC=0.0380 / ICIR=0.1637 仍被判"非忠实复现"。
- 提交卡点：`rq_pead_es60_div_attn20` **过盲测**（test_ic=0.01563 / retention=0.9004 / sign=True）
  但 stage_one `ic=0.01736 / icir=0.2317`（候选线 0.02/0.28）→ `stage_one_failed:ic,icir`。

### 5.4 完整块口径（两段各自跑完的整块，非仅 2h 窗口）

| 指标（整块） | **technical** | **report** |
|---|---|---|
| run 数 / 块长 | 2 / 2h07m（10:24:07–12:31:46） | 4 / 2h15m（12:32:56–14:48:26） |
| 评估行 | **759** | 552 |
| val 行 | 8 | 10 |
| `verdict=train_passed` | **62** | 0（该模式无此判定） |
| 提交次数 | 2 | 3 |
| **候选落地** | **0** | **0** |
| 真复现判定 / PASS | 0 / — | **297 / 4（1.3%）** |
| 保真度过检 | — | 4 / 30 |
| **`memory.advisory_block`（死路硬拦）** | **198** | 34 |
| 有 summary 的 run | 1（run#2 被手动停） | 4 |
| summary 口径 train | 313（被低估） | 452 |

**新发现（值得单列）**：整块里 technical 触发**死路硬拦 198 次**（≈99 次/run），report 只有 34 次（≈8.5 次/run）。
即自由探索会**反复提出结构上已判定为死路的式子**，每次都被记忆守卫拦下——拦下省了评估时间，但**白花一次 LLM 调用与一轮上下文**；
研报模式因为有课题与复现协议约束，这类碰撞少一个量级。这条差异与"谁产出更多"无关，但它解释了
**技术模式的 LLM 调用为何没有转成更多有效评估**（759 评估行里有 198 次是被拦的调用）。

---

### 5.5 诚实边界（怎么读这个结果）

1. **样本极小**：各 2 小时、各 2–4 个 run、候选各 0 → **不能判定谁更好**，只能说"两段都没出候选、卡点不同"。
2. **时段不同**（A 上午 / B 下午，顺序执行）→ 要排除时段效应需**反向再跑一次**（report 2h → technical 2h）。
3. **吞吐差异可信**：自挖掘评估行数多 ~50%（747 vs 499）；其 run 更长（75.3 vs 42.4 分钟）。
   研报模式把约一半评估花在复现判定（252 次）与保真校验（30 次）——**设计成本，不是 bug**。
4. **两侧门槛不同**：report 多一道"复现门 + 保真锚"；但候选/正式库门槛与 technical 相同
   → "复现 PASS"≠候选；technical 的 `train_passed` 也≠候选。
5. 段 A run#2 在 2h 边界被**手动停掉**（无 `run_summary`）→ summary 口径下 A 的 train/tokens 被低估；
   等长窗口表（747 vs 499）用 steps.log 逐行口径，不受影响。
6. 若要提高研报模式产出，观察到的两个卡点是：**慢信号 IC 被 1d label 稀释**（|ICIR| 够而 |IC| 不够）
   与**换手率口径**（weekly 调仓 vs 1d 信号）——是否调口径属你的决策，本次未改。

---

## 六、反向一轮（段 C/D）——为排除"时段效应"

第一轮顺序是 technical→report（上午/下午），本轮**反过来**：report→technical。其余参数与第一轮**完全一致**
（同一 monitor、`--max-turns 8`、自由探索、`--restart-backend`、`--poll 60`、`--stall-warn 15`、`--stall-kill 25`）。

| 段 | 模式 | 起 | deadline | monitor PID | 首个 run | 对比窗口 |
|---|---|---|---|---|---|---|
| **C** | `report` | 14:52:46 | `2026-10-04T16:55` | 74028 | `2a2291fbeb4b`（14:53:33） | 14:53:33 → 16:53:33 |
| **D** | `technical` | 待段 C 退出后起 | 起 +2h | 待起 | 待起 | 首个 run 起算 → +2h |

**本轮相对第一轮的两处流程修正**：
1. **不再手动停 run**：第一轮我在 2h 边界用 API 停掉 technical run#2，后端把它仍标成 `running`，
   段 B 启动时白等 2 分钟。本轮改为**让 monitor 的 deadline 自然收尾**；等长比较仍用"首个 run 起算的
   2 小时窗口"逐行统计，不受块长影响。
2. 段间衔接：**必须等段 C 的 monitor 进程退出**再起段 D，避免两个 monitor 抢管 run。

**结果（口径与第一轮一致：等长窗口逐行统计 + 完整块口径 + 卡点证据 + 死路硬拦计数）**：见下（跑完后补）。

### 四段总表（等长 2 小时窗口，逐行时间戳口径，不去重）

| 段 | 模式 | 窗口 | 评估行 | val 行 | 提交次数 | **候选落地** | 复现判定 / PASS | 保真度过检 | 死路硬拦 |
|---|---|---|---|---|---|---|---|---|---|
| A | technical（第一轮上午） | 10:24:44–12:24:44 | **747** | 8 | 2 | **0** | — | — | 198 |
| B | report（第一轮下午） | 12:35:45–14:35:45 | 499 | 8 | 3 | 0 | 252 / 4 | 4 / 30 | 34 |
| C | report（反向腿下午） | 14:53:33–16:53:33 | 423 | 4 | 1 | 0 | 132 / 3 | 3 / 9 | — |
| D | **technical（反向腿傍晚）** | 17:23:40–19:23:40 | 322 | 6 | **13** | **1** | — | — | 103 |

**结论（本次四段）**：
1. **候选：technical 1 / report 0**（两个方向各跑一次；report 两段都是 0）。今天 A/B 的唯一候选来自
   反向腿的 technical 段：`turn120_prem20_twoleg_softgate_indneut`（19:06:19 `candidate_stored=True`，
   stage_two 未过 → 按两阶段设计停在候选池）。
2. **report 两段高度一致**（499 / 423 评估行、4 / 3 次复现 PASS、0 候选）——被"复现协议"约束、方差小；
   **technical 两段差异极大**（747 vs 322 评估行、2 vs 13 次提交、0 vs 1 候选），产出更依赖当轮选的路子。
3. **卡点位置不同**：technical 卡在**交付门**（本日 `avg_daily_side_turnover=1.48/1.64/1.67 > 0.65`、
   `stage_one_failed:icir`、`max_cs_corr`）——提交多、命中少；report 卡在**提交之前**
   （复现门 + 形态对账 + 保真锚），故提交少（1–3 次/2h）。
4. **样本仍极小**（四段 × 2h、候选合计 1 个）→ **不足以判定优劣**；只能说本次同条件反向对照下
   technical **1 : 0** 领先，且把更多尝试推到了交付门口。
5. 段 D 的 run `2601a1d04e3e` 在窗口结束后仍在跑（turn 2/8），其 `run_summary` 待 run 结束；
   monitor 会在它结束后按 deadline 退出。

### 段 C 运行记录（含上游第 4 次中断）

- 14:53:33 起 run `2a2291fbeb4b`（report，36.9 分钟）：143 次评估、真判定 94、**PASS 2**、提交 1
  （`rq_flow_turnover_div` 卡 `blind_test_failed:blind_test_abs_ic,blind_test_ic_retention`）、候选 0。
- 15:30:29 起 run `c85e85f9347b`（report）：前 25 分钟 137 次评估 / 真判定 15 / **PASS 1**；
  turn=0 时该题 `RQ_d3c20b` 已处于 **diverge 阶段**（跨 run 的复现锁：`reproduce_lock_rounds=3`），
  故前 50 条 judge 全是"跳过 phase=diverge"，没有复现判定——**研报模式的阶段状态跨 run 保留**，
  这会让部分 report run 实际在做"带课题锚的发散"而非复现。
- **15:56 起上游第 4 次中断**：`steps.log` 停在 15:52（turn=5）、events 冻在 902；
  16:08 monitor 报停滞告警；16:12 用 1-token 探测确认 ❌（5.5s 返回 500）。
  → 段 C 的后 ~57 分钟窗口被这次中断吃掉（诚实计入，不剔除也不补跑）；16:18 上游自愈，
  monitor（旧版无 #9 修复）在 16:18:40 杀掉停滞 run 并在 16:19:43 起了 run#3。

### 段 C 窗口结果（14:53:33 → 16:53:33，逐行口径）

| 指标 | 值 |
|---|---|
| run 数 | 4（`2a2291fbeb4b` / `c85e85f9347b` / `c5fc229c3a34` / `902d998ce433`） |
| 评估行 | **423** |
| val 行 | 4 |
| 提交次数 | 1（`rq_flow_turnover_div` 卡 `blind_test_failed:abs_ic+ic_retention`） |
| 候选落地 | **0** |
| 真复现判定 / PASS | **132 / 3** |
| 保真度过检 | 3 / 9 |

### 段 D 运行记录（technical，反向腿）

- 17:16:43 起 monitor（PID 76684；实进程 78404），deadline 19:18；17:23:40 起 run `2601a1d04e3e`（`mode=technical` ✓）。
- 启动时又踩到**陈旧 run 复活**：段 C 被 stop 的 `902d998ce433` 在后端重启（17:22:38）后被重新标成 `running`，
  段 D 的 monitor 白等了 ~1 分钟 → **再 stop 一次**才清掉，随后才起 run。这是**同一个"stop 只标记、重启后复活"**的坑第 2 次出现。
- 段 D 对比窗口：**17:23:40 → 19:23:40**。
- **段 D 产出今天 A/B 四段里唯一的候选**（19:06:19）：
  `turn120_prem20_twoleg_softgate_indneut`（家族 价量面×量能面×基本面，父本 `v19_turn120_prem_blend06_volgate`）
  `submit.candidate_stored=True`；stage_two `passed=False`（`train_ic/train_icir/max_cs_corr`）
  → 按两阶段设计**停在候选池**（`verdict=candidate_approved`）。同日其它三次提交被拒原因：
  `avg_daily_side_turnover=1.48 > 0.65`、`stage_one_failed:icir`、`stage_one_failed:max_cs_corr`。
- **反向腿期间的运维修复（`1fca702` + `d23deef`）**：
  1. `GET /runs` 超时真因与修复（`1fca702`）：`list_runs()` 对每个 run 都把整份 `run_*.jsonl`
     读一遍数行数 —— 143 个 run 时**冷调用 15.7s**（monitor 阈值 20s → 误判"后端挂了"并重启）。
     加 `(path, size, mtime_ns, tail)` LRU 缓存后 **15.7s → 0.23s（68×）**，只有正在写的 run 会 miss。
  2. "已 stop 的 run 被后端重启复活"（`d23deef`）：`stop_run()` 终止分支不落盘 + `_load_run_from_disk()`
     不读 `run_meta.status`（只看"日志 15 分钟内新鲜"）→ 恢复即复活。现在 run_meta 终态优先于启发式。
     两条修复都有"回退即失败"的单测，测试 +7；**均于下次后端重启生效**。
- **17:16–17:22 后端被外部关闭（用户误关 PowerShell 窗口）**：monitor 在 17:17:00 拉起后端后
  5 分钟未就绪并告警（17:22:03），随即 17:22:09 再次拉起、17:22:38 就绪 → **monitor 自愈**。
  该次后端重启也是"陈旧 run 复活"的直接诱因（见上条）。核验结果：17891 与 8787 均在听，
  monitor 存活，段 D run `2601a1d04e3e` 正常推进（events 3→172）。

---

## 七、由本次对照引出的口径决定：label 与调仓频率**强制一致**（2026-10-04，用户定调）

**直接触发**：本日 4 次提交被拒里，`avg_daily_side_turnover=1.48/1.64/1.67 > 0.65`
（`rebalance_freq=weekly`）这一条最刺眼。追下去发现**两把尺子混用**：
`avg_daily_side_turnover` 是"信号日频抖动"（与调仓频率无关），而门槛按"声明的交付频率"
取档（weekly 0.65）—— 又因为 `holding_days = label 天数 = 1`，组合其实**每天在换**
（段 D 候选实测 `avg_rebalance_side_turnover == avg_daily_side_turnover = 0.2788`、
`n_rebalances = 1615`）→ "周频摊薄成本"的折算前提不成立。

**用户定调**：「调仓频率是多少，label 就应该多少」→ 实现见 commit `09c06c0`
（合入 `67221a6`），文档见 `docs/alphaagent_architecture.md`「强制一致（2026-10-04）」节：

- 规则：`engine_gate.freq` 持有期 == label 持有期（**daily↔1d、weekly↔5d、monthly↔20d**），
  三处 fail-closed 收口（构建期 / API 入口 / `start_run`）；
- 档位：`technical`、`report` 由 `label_1d + weekly` 收成 **`label_1d + daily`**；
- 换手门改用**调仓口径** `avg_rebalance_side_turnover`（daily↔1d 时与日频相同）；
- **代价（接受）**：1d 档换手硬门 0.65 → **0.50**、engine_gate 改按 daily 真实回测（更严）；
  **门槛数值未动**（`min_abs_ic` 等仍按 1d 标定）——要切 5d/20d label 必须同批重标定门槛。

**对后续 A/B 的影响**：下一轮对照起，technical/report 的换手硬门与交付回测节奏都变成日频，
"周频折让"消失 → 快信号必须靠结构变慢过门；这也让"研报 vs 自挖掘"的比较更接近
"同一把尺子下的产出率"。
