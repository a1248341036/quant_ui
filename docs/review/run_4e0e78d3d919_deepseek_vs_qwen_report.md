# DeepSeek vs Qwen 挖掘 Run 对比报告

- **对比对象**: `4e0e78d3d919`（DeepSeek-V4-Flash-0731，2026-09-22）vs `bc0090d8255b`（Qwen3.8-27B，2026-09-21）
- **配置**: 完全一致（价量面 / weekly / label_1d_open_to_open / max_turns=3 / 6 并发 / 2020-2022 train / 2023-2024 val），**唯一变量 = 模型**
- **代码版本**: DeepSeek run 用 `307a08a`（含 prompt 审计遗留修复），Qwen run 用 `630b934`
- **产出**: `docs/review/run_4e0e78d3d919_deepseek_vs_qwen_report.md`

## 一、核心指标对比

| 项 | Qwen3.8-27B | DeepSeek-V4-Flash | 倍数 |
|---|---|---|---|
| 墙钟 | 17.9 min | 33.2 min | 1.9× |
| LLM 调用 / tool_calls | 44 / 208 | 48 / 252 | — |
| **train 评估数** | 37 | **219** | **5.9×** |
| val 评估数 | 3 | 5 | 1.7× |
| **promising** | 2 | **30**（15 个未提交） | **15×** |
| near_miss | 2 | 32 | 16× |
| weak | 66 | 400 | 6× |
| eval_error | 8 | 32 | 4× |
| **candidate_stored** | **0** | **0** | — |
| prediction confirmed 率 | 12.1% | **23.0%** | 1.9× |
| dup_dead_end 率 | 4.4% | 11.7% | 2.7× |
| output tokens | 71K | **429K** | 6× |
| cache 命中率 | 81.3% | 75.4% | — |
| 有效尝试率 | 90.9% | 93.7% | — |
| tool 失败数 | 19（9.1%） | 16（6.3%） | 更优 |

## 二、结论

### ✅ 1. DeepSeek 探索能力显著强于 Qwen

- **评估吞吐 5.9 倍**（219 vs 37）：DeepSeek 每轮并发槽位用得满，Qwen 大量槽位闲置
- **过线因子 15 倍**（30 promising vs 2），质量也更高（最好 `vwap_dev_rev_raw` train IC=0.0303/ICIR=0.3166）
- **预测对账 confirmed 率 23% vs 12.1%**：经济直觉可证伪质量更高
- 出现真实 OOS 数据：4 个 train/val 配对，median 保留比 0.623，无过拟合嫌疑（Qwen 0 配对）

### 🟡 2. 0 入库根因收敛：**两模型都不调 submit_factor**

- DeepSeek 15 个 promising 未提交、Qwen 1 个未提交——**submit 链路（非 preflight）两个 run 均为 0 条调用**
- val 段最好 `tug_vwap_dev_res_size_w10_int` val IC=0.0128 已过 0.012 绝对门，但 LLM 没有发起 submit
- **这排除了"模型能力"解释**——问题在机制侧，候选原因：
  1. **max_turns=3 截断**：turn 2 末尾被 max_turns_reached 停掉，模型还在探索未进入交付心态（prompt rule 9"结束前检查"没机会执行）
  2. deliver 阶段（turn 2）prompt 已全量注入交付规则，但模型探索惯性大
  3. 建议：验证性跑 max_turns≥5，或在 turn 2 强注入"必须 submit"控制消息（`/messages` 注入）

### 🟡 3. DeepSeek 同族扎堆更严重

- dup_dead_end 11.7%（vs 4.4%）、HomogenizationSmoothingBlock 10 次（vs 5）
- 15 个未提交 promising 中 **vwap 系 8 个**（vwap_dev_rev×5 + tug_vwap×3）、overnight 系 4 个——饱和度警告未被有效重视

### ✅ 4. 修复后的 prompt 生效且无异常

- git `307a08a`：stage_two `abs(ICIR) >=` 统一 + 交互清单去重 + PHASES 文档同步（22 测试全过）
- run 全程无链路异常，failure 仅 16 次（6.3%），Reviewer 正常工作（1 次 reject `vwap_dev_acc5`：novelty=low + val 未过线）
- 修复前审计发现的重复/矛盾在本次渲染中已消除

## 三、下一步建议

1. **max_turns 5-8 复跑**（DeepSeek）：验证 0 入库是否为 turn 截断所致——这是当前唯一挡在"评估→入库"之间的环节
2. 同族饱和硬约束：DeepSeek 扎堆明显，可考虑把 memory 的族饱和度警告升为 evaluate_factor 硬拦截（P3 项）
3. 若复跑仍 0 submit：在 deliver 阶段 prompt 增加"turn 内必须至少 submit 一次（或说明不 submit 的证据）"的硬性收尾指令

---
*报告生成: 2026-09-22 | run_id: 4e0e78d3d919 vs bc0090d8255b | git: 307a08a*
