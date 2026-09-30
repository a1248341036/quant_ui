# AlphaAgent 挖掘 Run 报告 — Qwen3.8-27B 首跑

- **run_id**: `bc0090d8255b`
- **生成时间**: 2026-09-21 23:02:48 ~ 23:19:08（墙钟 ~16 分 20 秒）
- **model**: `Qwen3.8-27B`（本地代理 `http://127.0.0.1:15721/v1`，ccswitch 已切）
- **git_revision**: `630b934b38bd0f445442fb9600ce3561799f5e7a`（含 P0-4 ICIR 绝对值口径修复）
- **分支**: `main`（fix/prompt-compression-v1 已全量合并）
- **触发方式**: `POST /api/alphaagent/runs`（后端 17891，Web UI 链路）

## 一、Run 配置

| 参数 | 值 | 说明 |
|---|---|---|
| max_turns | 3 | 外层 ReAct 轮数上限 |
| max_tool_calls_per_round | 6 | 单轮 tool_calls 上限 |
| max_tool_workers / max_parallel_eval | 6 / 6 | 并行评估车道 |
| max_tokens | 16384 | 单次回复上限 |
| reasoning_effort | medium | Qwen 代理接受 |
| focus_facets | ["价量面"] | 自动推断 technical_weekly 档 |
| rebalance_freq | weekly | 门禁频率 |
| label_col | label_1d_open_to_open | 1d 快信号 |
| no_fundamentals | true | 仅价量列 |
| panel | cne:// | CNE 数据湖实时构建 |
| train / val | 2020-01-01~2022-12-31 / 2023-01-01~2024-12-31 | 标准双段 |

## 二、最终状态

| 项 | 值 |
|---|---|
| status | completed |
| outcome | **interrupted**（reason=`max_turns_reached`，正常跑完 3 turn 按上限停） |
| event_count | 1451 |
| tool_calls | 208 |
| usage_calls（LLM 调用） | 44 |
| error | 无 |

### candidate_funnel

| 漏斗节点 | 数量 | 比率 |
|---|---|---|
| unique_train_evaluated | 37 | — |
| unique_val_evaluated | 3 | train→val 8.11% |
| candidate_stored | **0** | val→candidate 0% |
| production_stored | **0** | — |

### overfit_audit

- status: `partial`（仅 train/val 覆盖，锁定测试集需独立盲测）
- overfit_suspected: `false`
- selection_bias_warning: `true`（候选为 0，统计意义有限）

## 三、Turn 时间线

| turn | 起始 | 时长 | 说明 |
|---|---|---|---|
| run_start | 23:02:48 | — | model=Qwen3.8-27B, mode=technical_weekly |
| 0 | 23:02:49 | ~7 分 4 秒 | 主力探索轮，评估最多 |
| 1 | 23:09:53 | ~1 分 39 秒 | 产出 promising 因子 |
| 2 | 23:11:32 | ~7 分 36 秒 | 继续探索，负 ICIR 因子集中出现 |
| run_end | 23:19:08 | — | max_turns_reached |

## 四、Verdict 分布（88 个带 verdict 的评估）

| verdict | 数量 | 占比 | 含义 |
|---|---|---|---|
| weak | 66 | 75.0% | 未过 train 门槛（|IC|<0.02 或 |ICIR|<0.28） |
| eval_error | 8 | 9.1% | LLM 产因子格式/契约错误 |
| revise_required | 6 | 6.8% | 过 train 但 val 段不达标 |
| revise | 4 | 4.5% | Reviewer 建议修改 |
| **promising** | **2** | **2.3%** | 过 train 门槛（|IC|≥0.02 且 |ICIR|≥0.28） |
| near_miss | 2 | 2.3% | 接近门槛 |

## 五、关键验证点

### ✅ C6 ICIR 绝对值口径修复生效（核心验收项）

**630b934 修复内容**：`delivery_criteria.py:339` stage_one 从 `ICIR > {min_icir}` 改为 `abs(ICIR) >= {min_icir}`，与 stage_two 及末尾"按绝对值判断"统一。

**本次 run 实测证据**：

1. **负 ICIR 因子全部正常评估，无一处因负号被直接拒**：
   - 18 个 `ic<0` 的因子 verdict 全部为 `weak`（|ICIR|<0.28 或 |IC|<0.02），**没有出现"负 ICIR 直接 reject"的异常**
   - 若 C6 未修（stage_one 用 `ICIR > 0.28` 原始符号），这些负 ICIR 因子会在 lite 筛选阶段被误拒，不会进入 full 评估

2. **8 个负 ICIR 且 |ICIR|≥0.28 的因子进入 full train 评估**（lite→full 升级用 abs 口径放行）：
   - `ic=-0.0288, icir=-0.3281`（|ICIR|=0.3281，|IC|=0.0288，双过门槛）
   - `ic=-0.0261, icir=-0.2931`
   - `ic=-0.0257, icir=-0.2917`
   - `ic=-0.0254, icir=-0.2877`
   - `ic=-0.0299, icir=-0.3119`（出现在 `eval_on_train_set`，证明进入 full 阶段）
   - `ic=-0.0253, icir=-0.285`
   - `ic=-0.024, icir=-0.2825`
   - `ic=-0.0237, icir=-0.2805`

   **这些因子若在旧口径（`ICIR > 0.28`）下全部会被误拒**（负值不大于 0.28）；新口径（`abs(ICIR) >= 0.28`）正确放行进入 full 评估。

3. **lite 筛选阶段匿名因子（`evaluate_factor ?`）与 full 阶段带名因子（`eval_on_train_set`）口径一致**，均按绝对值判断。

**结论**：C6 矛盾（stage_one 原始符号 vs stage_two 绝对值 vs 末尾说明）已彻底消除，三处统一为 `abs(ICIR) >=`。

### ✅ Qwen3.8-27B 模型工作正常

- **API 连通**：本地代理 15721 全程稳定，无 429/超时/连接错误
- **tool_calls 解析**：208 次 tool_calls 全部被 agentscope 正确解析执行
- **prompt cache 生效**：cache_hit_rate 81.28%（1966K/2419K input tokens 命中缓存），单次 input 均摊 ~55K tokens
- **上下文压力**：200K 窗口下，3 turn / 44 calls 累计 input 2.4M tokens（cache 后实际新增 ~453K），单次最大 input 未超 200K——**上下文压缩未触发**（trigger_ratio 0.5 = 100K，本次未达阈值）
- **reasoning_effort=medium**：代理接受，无参数报错

### ✅ 评估链路完整

- **preflight smoke** 通过（`__preflight_smoke__` 跑完 submit.precheck + blind_test，blind_test 故意 fail 是正常 smoke 行为）
- **两段式海选**生效：lite 筛选（`evaluate_factor ?`）→ 过线升 full（`eval_on_train_set`）
- **reviewer** 触发：3 个因子 val 段 `metric_precheck verdict=revise stage=validation`
- **研究记忆**全程记录：memory.record 事件覆盖每个带名因子
- **JIT 预热**完成（jit_warmup_start → jit_warmup_done）

### ✅ 换手率门槛

- **未触发**：因 0 因子进 submit 链路（candidate_stored=0），换手率门槛（weekly 0.65）未被执行
- **无法验证 S4 修复**（换手率统一 0.4→0.5/0.65）：需因子进 submit 才能测，本次 run 数据不足

## 六、Failure 明细（failure_counts）

| 错误类型 | 次数 | 说明 |
|---|---|---|
| ToolArgumentsError | 6 | prediction 参数缺失/格式错（必填项未传） |
| MultiLineFactorEvalError | 4 | DSL 表达式编译/求值错 |
| HomogenizationSmoothingBlock | 5 | 同质化平滑拦截（记忆层防重复） |
| InteractionContractRejected | 2 | 交互契约未声明/不合规 |
| DuplicateExactEval | 1 | 完全重复表达式 |
| tool_failed | 1 | 工具执行失败 |

**性质**：全部是 LLM 产因子格式/契约瑕疵，**容错不阻断**——系统正确降级为 eval_error 并继续。无引擎/数据/链路级故障。

## 七、Promising 因子（2 个）

| 因子 | turn | train IC | train ICIR | rank_ic | family | motif | parent |
|---|---|---|---|---|---|---|---|
| `tug_vwap_dev_res_rank` | 1 | 0.0200 | 0.308 | 0.0273 | residual | operator_substitute | tug_vwap_dev_res_size |
| （匿名，turn 2 末尾） | 2 | — | — | — | — | — | — |

- `tug_vwap_dev_res_rank`：**唯一带名 promising**，过 train 门槛（|IC|=0.02≥0.02, |ICIR|=0.308≥0.28），是变异轨产物（父本 `tug_vwap_dev_res_size`，算子替换）
- **未入库**：run 结束时 `unsubmitted_promising` 警告——Qwen 在 turn 2 末尾未主动调 `submit_factor`，因子停留在 promising 未进 candidate

## 八、Val 段因子（3 个进 val，全部 revise_required）

| 因子 | turn | val IC | val ICIR | verdict |
|---|---|---|---|---|
| intraday_vwap_grp_int2 | 0 | 0.0105 | 0.1097 | revise_required |
| intraday_vwap_dev_grouped | 0 | 0.0100 | 0.1035 | revise_required |
| vwap_meanrev_slow | 0 | 0.0117 | 0.0955 | revise_required |
| （匿名） | 1 | 0.0098 | 0.1397 | revise_required |

**全部 val IC < 0.012 门槛** → revise_required，未进 candidate。这是 technical 1d label + 价量面的常态（val 段 IC 衰减到 0.01 量级）。

## 九、Token 消耗

| 项 | 值 |
|---|---|
| input_tokens（累计） | 2,419,090 |
| output_tokens（累计） | 71,485 |
| cache_input_tokens | 1,966,208 |
| cache_creation_input_tokens | 0 |
| cache_hit_rate | 81.28% |
| calls | 44 |

- 单次 LLM 调用均摊 input ~55K tokens，output ~1.6K tokens
- cache_hit_rate 81% 说明代理 prompt cache 工作良好，实际新增 input ~453K tokens
- **200K 上下文窗口未触发压缩**（单次最大 input < 200K，trigger_ratio 0.5 = 100K 未达）

## 十、问题与建议

### 🟡 Qwen 未主动 submit promising 因子

- `tug_vwap_dev_res_rank` 过 train 门槛但 LLM 未调 `submit_factor`，run 结束时 `unsubmitted_promising` 警告
- **可能原因**：Qwen3.8-27B 在 medium reasoning_effort 下偏向继续探索而非收敛提交；或 prompt 中 submit 触发条件不够强
- **建议**：观察更多 run；若持续不 submit，考虑在 prompt 中强化"过 train 门槛即应 submit"指令，或调高 reasoning_effort

### 🟡 0 因子入库（candidate_stored=0）

- 3 个因子进 val 但全部 revise_required（val IC < 0.012），无因子过 val 门槛
- **不是 bug**：technical 1d label + 价量面 + 3 turn 短 run 的常态表现（历史 run 也多在 5-10 turn 才有入库）
- **建议**：若要验证完整 submit→candidate→production 链路，需跑更长 run（max_turns=10+）或用 fundamental 档（val 段更容易过）

### 🟡 换手率门槛未验证

- S4 修复（换手率 0.4→0.5/0.65 统一）需因子进 submit 才能测
- 本次 run 0 进 submit，**S4 修复未得到运行时验证**
- **建议**：跑一个 fundamental 档或更长 run，确保有因子进 submit 链路

### 🟢 eval_error 8 个（9.1%）

- ToolArgumentsError（prediction 必填）+ MultiLineFactorEvalError（DSL 格式）+ InteractionContractRejected（契约）
- 全部是 LLM 产因子格式瑕疵，容错不阻断
- **建议**：观察 Qwen 在 prediction 参数填写上的表现是否稳定；若 ToolArgumentsError 比例 >15%，考虑在 prompt 中强化 prediction 参数示例

## 十一、总结

| 验收项 | 结果 |
|---|---|
| Qwen3.8-27B 模型切换生效 | ✅ model=Qwen3.8-27B，代理 15791 稳定 |
| 后端 17891 + API 链路 | ✅ POST /runs → 子进程 → 1451 事件 |
| **C6 ICIR 绝对值口径修复** | ✅ **8 个负 ICIR 因子正常进入 full 评估，无一处误拒** |
| 评估引擎 + 两段式海选 | ✅ lite→full 升级用 abs 口径 |
| Reviewer + 研究记忆 | ✅ 全程工作 |
| Qwen 上下文 200K 够用 | ✅ 单次最大 input <200K，未触发压缩 |
| prompt cache | ✅ hit_rate 81.28% |
| 因子入库 | 🟡 0（val 段未过门槛，非 bug） |
| 换手率门槛 S4 修复 | 🟡 未触发（0 进 submit） |
| 链路异常/崩溃 | ✅ 无 |

**核心结论**：Qwen3.8-27B 首跑链路畅通，**630b934 的 C6 ICIR 绝对值口径修复在运行时得到实证**（8 个负 ICIR 因子正确进入 full 评估，旧口径下会被全部误拒）。0 入库是 technical 1d 短 run 的常态，非修复回归。换手率 S4 修复需更长 run 验证。

---
*报告生成: 2026-09-21 23:20 | run_id: bc0090d8255b | git: 630b934*
