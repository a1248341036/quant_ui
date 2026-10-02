# 昨夜研报模式挖掘 · 总结报告（2026-10-02 01:30 – 08:05）

> 报告对象：**昨夜实际跑的 run**（改了什么 / 有哪些问题 / 指标如何）
> 数据来源：各 run 的 `run_summary.json` / `scorecard.json` / `steps.log`（权威值，非估算）
> 代码分支：`feat/question-structure-gate`（已合 main）/ `feat/question-schema-v2`（抽取与工具）
> 采集脚本输出：`logs/night_runs_metrics_20261002.json`

---

## 摘要（三句话）

1. **夜里共跑 7 个 run**（5 个 report 模式 + 2 个 technical），只有**最后 1 个 run 产出候选入库**
   （`b2f3c460fffe`：327 次训练评估 → 1 个候选，0 正式库）。
2. **改动 18 个提交**：把"复现目标"从*那条公式*改成*报告结构*，加**血统门禁**，并修掉
   **两个会让整类课题永久卡死的缺陷**（血统门禁覆盖缺口、参照无表达式导致复现永久失败）。
3. **指标方向明确但样本很小**：修复后段**8 个达标因子的血统内占比 100%**（全部衍生自研报复现版），
   而修复前段的 22 个达标因子**100% 是离线同根灌水**（记忆建议的 `mix_*`）。

---

## 一、夜间 run 全景（权威指标）

| run | 模式 | 起止 | 时长 | 课题 | 复现判定/过线 | 保真度/过线 | 训练评估 | 达标 | 因子 | **血统内** | 候选/正式 | 终态 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `d9fea015dca4` | report | 01:30–01:34 | 4.6m | 0 | 12 / 0 | 0 / 0 | 11 | 0 | 0 | — | — / — | 题库缺 qid，作废 |
| `a898c626b991` | report | 01:35–02:45 | 69.8m | 4 | 186 / **2** | 2 / 2 | 183 | **22** | 239 | **60%** | 0 / 0 | interrupted |
| `baeb4b983d2f` | **technical** | 02:51–03:58 | 67.3m | 0 | 144 / 0 | 0 / 0 | 137 | 4 | 47 | 6% | 0 / 0 | interrupted |
| `ad280ca9edb0` | report | 02:57–03:30 | 32.6m | 2 | 139 / 1 | 10 / 1 | 137 | 0 | 188 | **98%** | — / — | LLM 掉线卡死，被手动停 |
| `d6fb37459d9c` | **technical** | 05:36–05:43 | 7.5m | 0 | 40 / 0 | 0 / 0 | 37 | 2 | 36 | 0% | — / — | 监控参数缺省致模式错 |
| `5ce845512076` | report | 05:47–06:19 | 32.4m | 2 | 111 / **0** | 16 / **0** | 109 | 0 | 167 | 98% | — / — | 卡在已修 bug 上，被手动停 |
| **`b2f3c460fffe`** | **report** | 06:22–08:05 | **102.9m** | 3 | 321 / **3** | 4 / **3** | **306** | **8** | 455 | **98%** | **1 / 0** | **candidate_only**（max_turns） |

**关键落差**：
- **只有 1 个 run 产出候选**，且必须三个条件同时满足：report 模式 + 全套配置 + 两个缺陷已修；
- `run_summary` 里 `stage_one_yield_pct = 0.0`、`gate_survival_pct = 0.0`（所有 run）——
  即"过复现/达标"与"能提交入库"之间还有很大落差（细节见第四节）；
- 两个 technical run 是**监控参数缺省**造成的（见第四节 B1），与本设计无关。

---

## 二、指标：达标因子与复现明细

### 2.1 修复前段 `a898c626b991` —— 达标 22 个，**全部离线同根**

| 因子 | \|IC\| | \|ICIR\| | 血统 | 父本 |
|---|---|---|---|---|
| `rq8b4cdd_vwap_er90_nosmooth_v1` | 0.0259 | 0.416 | **离线** | `mix_ovlead_wma5_kg90_totcap_m15` |
| `rq8_vpd_neg_fcap_tri_603010` | 0.0336 | 0.369 | **离线** | 同上 |
| `rq8_vpd_neg_fcap_tri_553015` | 0.0333 | 0.364 | **离线** | 同上 |
| `rq8_vpd_neg_fcap_quin_3525201208` | 0.0324 | 0.352 | **离线** | 同上 |
| …（22/22 同一父本） | | | | |

→ 父本来自 **`memory_suggest`（记忆建议）**，不是研报复现版：**研报血统被绕过，22 个"达标"实为 1 个机制的变体**。
复现侧另有 2 条 PASS、2 条保真度过线；**32 个训练过线因子未提交**；候选 0。

### 2.2 修复后段 `b2f3c460fffe` —— 达标 8 个，**100% 在研报血统内**

| 因子 | \|IC\| | \|ICIR\| | 血统 | 父本 |
|---|---|---|---|---|
| `rq7810a8_v19e_vwap_prem_igroup` | 0.0232 | 0.382 | **血统内** | `rq7810a8_gnn_dual_edge_v5b`（复现版） |
| `rq7810a8_v18b_vwap_premlevel_turnstate` | 0.0329 | 0.360 | **血统内** | 同上 |
| `rq7810a8_v18c_vwap_premlevel_pure` | 0.0313 | 0.340 | **血统内** | 同上 |
| `rq7810a8_v18e_holder_conc_prem` | 0.0299 | 0.337 | **血统内** | 同上 |
| `rq7810a8_v17f_vwap_prem5_turnw` | 0.0267 | 0.302 | **血统内** | 同上 |
| `rq7810a8_v17a_vwap_prem5` | 0.0262 | 0.299 | **血统内** | 同上 |

**复现过线 3 条**（全部 `shape=confirmed` + `signal=ok`）：
```
qid=RQ_7810a8  rq7810a8_gnn_dual_edge_v5b  ic=+0.0142 icir=+0.10
qid=RQ_367653  rq367653_fastdecay_v2       ic=+0.0174 icir=+0.1714
qid=RQ_367653  rq367653_composite_v1       ic=+0.0158 icir=+0.1772
```
**保真度过线 3 条**（`fj` 0.10~0.40、`oj` 0.24~0.29），其中 `ad280ca9edb0` 那条
`shared_specific=['pb']` 说明**真锚（报告特有字段）也能正常工作**，不只是退化路径。

### 2.3 两个"卡住"的 report run（缺陷的直接代价）

| run | 现象 | 原因 |
|---|---|---|
| `5ce845512076` | 111 条复现判定 **0 PASS**、16 条保真度 **0 过线** | **参照记录无表达式** → 字段锚永远判 `shared_fields=0<1`（已于 06:10 修复，该 run 子进程是修复前代码） |
| `ad280ca9edb0` | 137 次评估、达标 0，32.6 分钟被停 | 上游 LLM 掉线 → run 卡死重试（监控当时未开 `--stall-kill-minutes`） |

---

## 三、改了哪些（18 个提交，全部已合 main，未 push）

### 3.1 设计层（6 项）

| # | 改动 | 配置项 | 依据（实测） | 影响 |
|---|---|---|---|---|
| 1 | **结构准入**：只派发"报告提出了可复现结构"的课题 | `report_policy.require_report_structure` | 26,707 次评估：1 算子达标 1.4% vs 3+ 字段 11.2% | 挡掉必然无效的通用单因子题 |
| 2 | **题面 B 版**：复现目标=结构+要素清单，公式降为 `reference_formula` | `render_reproduce_task` | A/B 17 篇：给公式 → 15/17 照抄；只给描述 → 65% 自行复现且字段重叠同 90% | 不再让模型盲抄抽错的公式 |
| 3 | **结构硬锚**：题面声明的 `structure_ops` 须命中 | `check_reproduce_fidelity` | 巡检发现"题面要求 5 因子合成，模型换成上下行波动分解" | 换结构会被拦 |
| 4 | **弱声明兜底**：无结构算子且声明字段 ≤1 时，要求 ≥2 算子 + ≥2 字段 | `reproduce_fidelity.min_ops_floor/min_fields_floor` | 标定：59% 结构题未声明 `structure_ops` | 挡一行式 |
| 5 | **血统双门禁**：复现轮须 `reproduce_of:<qid>`；发散轮 parent 须是本课题复现版 | `reproduce_of_required / diverge_parent_required` | 前段 22 个达标因子全挂离线 `mix_*` | **血统内 60% → 98%** |
| 6 | **抽取口径**：`spec_requirements` 必须描述"结构"要素（而非参考公式要素） | 抽取 prompt | 标定 59% 缺 `structure_ops` | prod2 覆盖 42% → **63%** |

### 3.2 缺陷修复（5 项，都靠现场证据挖出）

| # | 缺陷 | 证据 | 修复效果 |
|---|---|---|---|
| 1 | **血统门禁覆盖缺口**：只装在 `evaluate_factor`/`submit_factor`，批量主路径 `eval_on_train_set`（108 次/run）没装 | 34/154（**22%**）因子挂其它课题血统 | 血统内 **78% → 95%**（验收 run） |
| 2 | **参照记录无表达式 → 复现永久失败** | RQ_13e253：报告 13 条记录只有"联发科:营收"这类指标名 → `ref_fields=∅` → 判 `0<1` 永远失败，该 run 111 条判定 **0 PASS** | 回放 **0/66 → 11/12 通过**；末段 run 复现 **3 条 PASS** |
| 3 | **v2 题库缺 `question_id`** | 状态机 `qid=-`，run 异常退出（`d9fea015dca4`） | 补 `RQ_<source哈希>` 稳定 ID |
| 4 | 汇总相位判定靠日志字符串 | 新 run 未产生校验时全被标"硬锚前" | 改读 run 冻结 spec |
| 5 | 汇总把 technical run 算进研报 A/B | 普通模式 spec 同样含 `report_policy` 键 | 改读 `run_meta.research_mode` |

### 3.3 锚的校准纪律（重要经验）

同一晚连续加强锚 → 用 **106 条真实表达式回放**发现**误杀 live 已过线案例** → 两次放松：
**硬门只保留「声明的 `structure_ops` 命中」+「弱声明复杂度地板」**，其余（命中声明字段、字段总数）**降为诊断**。
终态：**拦截 8/106（7.5%）、误杀 0**。工具 `scripts/replay_anchor_on_run.py`（**改锚必跑**）。

---

## 四、有哪些问题

### A. 设计/实现问题（已修，5 个 → 见 3.2）

### B. 运维陷阱（3 个，本夜真正的时间杀手）

| # | 陷阱 | 后果（实测） |
|---|---|---|
| 1 | 监控 `--research-mode` **缺省空** → 后端自定 `technical` | 监控自起的 2 个 run（`baeb4b983d2f`、`d6fb37459d9c`）**根本不是研报模式**，设计完全没生效 |
| 2 | 监控 `--stall-kill-minutes` **缺省 0**（只告警不自杀） | 上游 LLM 掉线 → run 卡死重试 **122 分钟**，整夜空转近 1.6 小时 |
| 3 | ~~**后端冻结 spec**~~ **（2026-10-02 复核推翻）** | 原判"代码合并后新配置不生效"**不成立**：实测 `build_run_research_spec` / `load_research_spec` 都会把代码默认值**逐键深合并**（含嵌套键）。错因：我拿 run 目录里冻结的 `research_spec.json` 反推运行时而它只是**记录**；已把该结论与配套的 `default_missing_keys()` 自检一并撤销 |

> 修正后只剩 **2 个真陷阱**（都属"参数没显式给 → 静默走默认且不报错"）。
> 补充的正确约束：**代码改动只对"合并之后启动的 run"生效**（run 是独立子进程）。

### C. 未解决 / 待确认（本夜暴露，尚未动手）

1. **`max_turns` 是绑定约束**：末段 run 跑满轮次时仍有 **30 个训练过线因子未提交**
   （前段 32 个）→ 瓶颈不在"产不出来"，在"轮次预算/提交窗口"。
2. **tool_calls 失败率 36%**（末段 186/518）：包含血统门禁与交互契约的正常拦截，
   但"有效调用占比"未量化。
3. **`stage_one_yield_pct / gate_survival_pct` 全为 0.0**（所有 run）：达标因子离"能提交入库"还有落差，
   提交被拒原因集中在 `stage_one_failed:icir`、`blind_test_failed:blind_test_abs_ic`，
   **尚未逐条归因**。
4. **`5ebbd6956b17`**：监控日志列出该 run 但**没有留下目录**（疑秒失败），**未核实原因**。
5. **产出量 vs 深度**：结构题面让单 run 墙钟从 ~28 分钟涨到 **102.9 分钟**（同轮次对比），
   一晚能跑的 run 数减少 —— 需你决定取哪一边。

---

## 五、结论与下一步

### 5.1 结论（严格按证据）

- ✅ **"复现研报结构、再在结构上发散"这条路径已经打通**：末段 run 的 8 个达标因子
  **全部衍生自研报复现版**（修复前段的 22 个则全部是离线同根灌水）；
- ⚠️ **样本极小**：末段只有 **1 个课题（3 个 turn 满轮次）**，且其一前序 run 因掉线报废。
  **不能据此声称"产出质量已提升"**，只能说"方向正确、血统可控"；
- ❌ **入库仍为 0**（1 个候选池、0 个正式库），达标因子与可提交之间仍有结构性落差（第四节 C3）。

### 5.2 下一夜启动参数（照抄，规避两个默认值陷阱）

```powershell
python C:\Users\zhoubw\Desktop\quant\overnight_mining_monitor.v2.py `
  --deadline 07:00 --research-mode report `
  --stall-warn-minutes 15 --stall-kill-minutes 25
# 改过配置中心 → 先重启后端（或依赖监控 --restart-backend）
```

### 5.3 待你拍板的四件事

1. **产出量 vs 深度**：是否精简结构题面（当前 5.4K 字符）或提高 `max_turns`；
2. **提交窗口**：是否让"训练过线但未提交"的因子在 run 末尾自动补交（本夜两段共 62 个）；
3. **composite 弱锚**（占结构题 39%）：本仓无因子合成一阶算子，是否给 DSL 加 `SIGNAL_BLEND`；
4. **题库规模**：目前 **119/1898 篇**（结构题 67 道），是否跑全量抽取（约 5–8 小时，可后台）。

### 5.4 回滚与推送

- 题库备份：`research_questions.jsonl.bak-20261002-024527` → `scripts/switch_question_bank.py --restore`；
- **未 push**：`main` 领先 origin **18 个提交**；`question-schema-v2` 另 **15 个提交**（无远端分支）。

---

## 附录：如何复现本报告的数据

```powershell
# 逐 run 权威指标（含 funnel/tool_calls/headline）
logs/factor_mining/ui/<run_id>/run_summary.json  +  scorecard.json
# 分段 + 同根度汇总（自动区分 硬锚前/部分生效/全套生效/非研报模式）
.venv\Scripts\python.exe scripts\night_summary_report.py --since 2026-10-02T01:30
# 巡检（复现过线 / 结构锚 / 母本强度 / 错误分布）
.venv\Scripts\python.exe scripts\patrol_report_mode.py
# 锚校准（改锚必跑：口径 judged；判读见脚本内说明）
.venv\Scripts\python.exe scripts\replay_anchor_on_run.py --run b2f3c460fffe
```

---

## 十、合并前 OCR 评审（open-code-review，两轮）

**评审对象**：`origin/main..main`（未推送的全部改动，10 个代码文件 / 614 行插入）。
命令：`ocr review --audience agent --from origin/main --to main --background-file logs/ocr_background_20261002.md`。

### 第一轮：6 条 findings（全部核实成立，已修）

| # | 严重度 | 位置 | 问题 | 处置 |
|---|---|---|---|---|
| 1 | medium | `alphaagent_factor_mining.py` | **我加的 spec 自检恒不触发**（`load_research_spec` 已把默认值合并回来） | **连带其错误前提一并删除**（见第五节更正） |
| 2 | medium | `question_queue.py` | **fail-open**：无 `structure_ops` 的中等声明题（fields=2/min_fields=2）**没有任何硬门** → 一行式可绕过 | 地板改为"只要无 `structure_ops` 就施加" |
| 3 | low | `question_queue.py` | 混合题库（部分行无新字段）会把历史题**静默整批丢弃** | 改为逐行兜底：只丢显式 `False` |
| 4 | low | `agentscope_tools.py` | `submit_factor` 仍是内联门禁副本 → 口径漂移风险 | 改用共享助手；覆盖测试断言提升到 ≥4 处且文案唯一 |
| 5 | medium | `question_queue.py` | `_sr` 未做 `isinstance` 防御 → 非 dict 时 `AttributeError` 崩溃 | 补防御 |
| 6 | low | `question_queue.py` | 题面装配处同样缺防御（无 try 包裹） | 补防御 |

### 第二轮（复核修复后状态）：12 条（去重后 9 条，全部成立，已修）

| 类型 | 问题 | 处置 |
|---|---|---|
| **真缺陷** | `primary` 本身未防非 dict（判定侧被宽泛 except 吞掉 → **该课题复现被静默误杀**；题面侧直接中断） | 判定/装配两侧都先归一 `primary` |
| **真缺陷** | `structure_ops`/`fields` 若被抽成**字符串** → `for o in "CS_GROUP_RANK"` 按**单字符**迭代 → 命中集恒空 → **忠实复现被判 `structure_ops_missing=C,S,_…` 硬拦** | 统一归一为 list（字符串整体视为单元素） |
| 静默 fail-open | 绕过 `normalize_research_spec` 的调用方会让三条新开关**静默失效** | 键缺失时**显式告警** |
| **工具口径** | 校准工具 `replay_anchor_on_run.py`：只判前 40 条却用全量做分母、qid join 静默失败、同名多表达式只留首个、异常裸崩、坏行静默、硬编码盘符、死代码 | **整脚本重写**（口径 `--scope judged`、覆盖率自检、异常计数、路径可移植） |

### 由 OCR 促成的两处自我更正（重要）

1. **"trap 3：后端冻结 spec"不成立**（第一轮 #1 牵出）—— 实测 `build_run_research_spec` /
   `load_research_spec` 都会**逐键深合并**代码默认值（含嵌套键）。我拿 run 目录里冻结的
   `research_spec.json` 反推运行时行为是错的：那只是**记录**。相应纪律已在第五节更正。
2. **"锚拦截 8/106（7.5%）、误杀 0"是坏工具的产物**（第二轮工具类 findings 牵出）——
   旧工具只判前 40 条且多数因 join 失败被静默跳过。**重写后的正确口径与数字**：

   | run | 真实复现判定条数 | 拦截率 | live 曾过线但被当前锚拦截 |
   |---|---|---|---|
   | `b2f3c460fffe`（修复后段） | 38 | **6/38 = 16%** | **0** |
   | `a898c626b991`（前段） | 24 | 2/24 = 8% | 1（原因 `structure_ops_missing=IF_THEN_ELSE` → **模型换了结构，锚的本职，非过严**） |
   | `5ce845512076`（卡死段） | 84 | 2/84 = 2% | 0 |

   **结论不变**：当前锚对"真实复现"的拦截率约 2~16%，且逐条查因后**没有发现真过严**；
   但原报告里的 7.5% 这个具体数字作废。
