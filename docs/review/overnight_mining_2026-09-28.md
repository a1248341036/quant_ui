# 整夜挖掘复盘 · 2026-09-28 22:58 → 2026-09-29 07:00

> 数据来源：`logs/factor_mining/ui/<run>/steps.log` + `run_*.jsonl`、`C:\Users\zhoubw\Desktop\quant\logs\overnight_monitor_20260928.log`、`~/.cc-switch/cc-switch.db`（只读）。金额/口径均为日志原值。

## 一、总览

| 指标 | 值 |
|---|---|
| 有效挖掘窗口 | **00:00 → 07:00（7h）**；22:59–00:00 损失（主动停 2 run + CC Switch 502 + 30min 退避） |
| run 数 | **10**（9 结束 + 1 在跑） |
| 评估次数 | **2,296** |
| unique 训练评估 / 验证评估 | **1,445 / 24** |
| promising | **92** |
| submit_result | **21（全部 not stored）** |
| **candidate / production** | **0 / 0** |
| MemoryAdvisoryBlock | **614（占评估 27%）** |
| HomogenizationSmoothingBlock | 9 |
| ToolArgumentsError | 133 |
| 墙钟 / token | 6.4h ｜ 输入 30.86M ｜ 输出 1.92M |

对比历史：09-23/24 6 run/2,930 evals；09-26 5 run/3,552 unique train/83 promising。今晚 unique train 偏低（1,445），但 **promising/eval 效率最高**（6.4% vs 09-26 的 2.3%）。

## 二、逐 run

| run | 时长(min) | turns | evals | promising | memBlk | submit | u_train | val | 结局 |
|---|---|---|---|---|---|---|---|---|---|
| `01a8013ff978` | 47.2 | 2 | 354 | 16 | 112 | 3 | 229 | 4 | **error**（毒 tool_call） |
| `cf4cf7f39980` | 43.8 | 10 | 250 | 6 | 60 | 1 | 166 | 5 | max_turns |
| `7629dadc0a80` | 43.3 | 10 | 264 | 20 | 87 | 1 | 173 | 2 | max_turns |
| `40c29c2c79b4` | 46.8 | 10 | 300 | 10 | 80 | 3 | 206 | **0** | max_turns |
| `4c8cfb615f76` | 60.6 | 10 | 294 | 18 | 79 | 5 | 207 | **0** | max_turns |
| `a269d29ed6ed` | 26.5 | 5 | 125 | 6 | 29 | 1 | 111 | 2 | no_candidate（模型停手） |
| `e54a43c4d4e2` | 14.6 | 4 | 90 | 0 | 21 | 0 | 54 | 3 | **error**（毒 tool_call） |
| `e50a94f03489` | 24.2 | 9 | 156 | 0 | 37 | 0 | 125 | 3 | no_candidate（模型停手） |
| `060525a45ffd` | 46.8 | 9 | 236 | 14 | 44 | 3 | 174 | 5 | **error**（毒 tool_call） |
| `7b89d852bfc4` | 32.2+ | 4+ | 227+ | 2 | 65 | 4 | — | — | running（07:00 时） |

## 三、头号杀手：毒 tool_call（杀死 3/9 个 run）

**证据（唯一出现畸形 JSON 的 run = 唯一出现上游 5xx 的 run = 唯一死亡的 run）**

| run | ToolArgumentsError | JSONDecodeError | 上游 500/502 | 结局 |
|---|---|---|---|---|
| `01a8013ff978` | 14 | **1** | **10** | 死亡 |
| `cf4cf7f39980` | 9 | 0 | 0 | 正常 |
| `7629dadc0a80` | 10 | 0 | 0 | 正常 |
| `40c29c2c79b4` | 16 | 0 | 0 | 正常 |
| `e54a43c4d4e2` | 12 | **1** | **10** | 死亡 |
| `060525a45ffd` | 9 | **1** | **10** | 死亡 |

链路：模型生成畸形工具参数（如 `turn_delta_rev_soft_winsor` 的 `interaction` 契约混入多余引号）→ 该 `tool_call` 留在消息历史 → 之后每次重放都被上游判 `400 Bad Request` → CC Switch 包装成 `500` → 框架 5s 间隔重试 10 次同一 payload → 会话终止。

单次代价：run 1 损失 47 分钟 + 8 个已过线因子未提交；3 次合计直接损失约 **1.5 小时墙钟**。

**修法**：① 工具参数 JSON 解析失败时**不把该 tool_call 写入历史**；② 重试前做"毒 payload 熔断"（payload 未变且连续 4xx/5xx → 截断到最后 N 轮或丢该条消息），不要同 payload 重试 10 次。

## 四、提交链路：21 次提交、0 入库

- **最好因子 `vwap_bias_dualres_v5_longamt_ctrl`（run 5）全门通过**：blind_test `test_ic=0.0186 / sign=True`；stage_one_stats `ic=0.02451 / icir=0.2921 / autocorr=0.8827 / turnover=0.5281`；val_retention `val_ic=0.02275 / retention=0.9281`；**stage_one passed=True（max_corr=0.4283，耗时 489s）**
- 最终被**离线正交门**拦下：`skipped=offline_orthogonality_failed:correlation_threshold`
  - `_ORTHO_MAX_CORR = 0.7`（`agentscope_tools.py:294`），比较集合 = **正式库 + 候选池 + registry 候选**；而 stage_one 的 `max_corr=0.4283` 只对**正式库** → 真凶是候选池里的 vwap 近亲 ≥0.7
  - **日志不可用**：`delivery/submit.py:851` 的 reason 是兜底字面量 `"correlation_threshold"`（`agentscope_tools.py:486` 置 `passed=False` 时未写 error/skipped_reason），既无实际相关值也无对手因子 → **建议**把 `max_abs_corr` + `blocked_factor_id`/`similar_factors[0]` 写进 reason 与 steps.log
- 其余提交失败画像：`stage_one_failed:icir,avg_daily_side_turnover`（如 `vwap_extreme_softgate_wma10`）、`blind_test_failed:blind_test_abs_ic`（`vwap_grouprank_wma10`）、`HomogenizationSmoothingBlock` 秒拒、`offline_orthogonality_failed`
- 成本：一次**完整**提交链路 5–8 分钟（stage_one 单步 489s、blind_test 130s、val 148s）→ 21 次尝试中多数是秒级熔断，完整链路约 4 次 ≈ 半小时墙钟

## 五、promising 未提交（每 run 都有）

- run 1：**8 个**（`turnover_rev20_capind_double(+0.0203)`、`turnover_rev20_wma(+0.0219)`、`turnover_rev10_wma_capind(+0.0232)`、`ovintra_blend60(+0.0204)`、`turn_delta_rev_soft_size(+0.0227)` 等）
- run 2：3 个（`vwap_bias_volratio_gate(+0.0235)` 等）
- **修法**：run 结束前对训练过线因子做一次强制提交/验证，或在过线当轮立即提交。

## 六、家族收敛与"越跑越薄"

- 92 个 promising 绝大多数是 **vwap_bias 族**（`vwap_bias_resid_*`、`vwap_bias_dualres_*`、`vwap_grouprank_*`…），与 09-26 复盘"避开库内已有 vwap 骨架"的结论相反；
- 凌晨段 run 6/7/8 的 promising 掉到 6/0/0，submit 0；`MemoryAdvisoryBlock` 稳定在 25–30%；
- 结合 run 4/5 的 **val=0**（各 10+ 训练过线却零验证），判断是"反复试同族变体 → 被死路记忆拦 → 无新族开拓"，**不是门槛收紧**。
- **修法**：对已饱和族加严或强制 D 轨配额（现提示"至少一半 D 轨"执行不佳）；把"语料/死路"提示从"禁止同构"升级为"点名未开拓的 3 个族"。

## 七、平台侧（非模型）问题

1. **CC Switch 转发态卡死**：23:15 前所有 LLM 调用 502（`error sending request`，稳定 ~2.05s），EasyCLI 8317 侧 GET/POST/stream 全 200/401 正常 → 重启 CC Switch（PID 81744→27812）后立即恢复 200。**建议**：整夜脚本启动前加 `127.0.0.1:15721` 探活 + 失败自动重启 CC Switch；并在 CC Switch 里打开 failover（当前 `enableFailoverToggle: false`）。
2. **首次面板重建**：schema v7→v8（新增 5 个数据面插件 + 基本面自动载入）首次 11 分钟，之后命中缓存 2.5 分钟；面板 2.0GB parquet / 5.3GB arrow，进程 RSS 10.9GB，可用内存最低瞬间 2GB（瞬时分配尖峰，非持续压力）。
3. **RAG 质量**：OpenViking 研报检索原命中的是 PDF 页面图片语义索引（`page1_img7.png` 的图注），本次已修（`ov_store._flatten_hits` 剔除图片条目 + 多取候选），修复后注入 414→816 / 1154→1194 字符且来源全为 `research_reports/*.md` 正文。
4. **黄金基线**：`report_rag` 渲染时实时调 OV → 4 份 prompt 基线实为 OV 快照（OV 一起来就假失败）；已改为静态冻结（`knowledge_mode=off`）。

## 八、修复清单（按优先级）

| # | 项 | 位置 | 预期收益 |
|---|---|---|---|
| P0 | 工具参数 JSON 解析失败不入历史 + 毒 payload 熔断 | `agent/agentscope_run.py` / `tools/_dispatch.py` | 今晚可挽回 3 个 run ≈ 1.5h 与 8 个过线因子 |
| P1 | run 结束前强制提交/验证过线因子 | `agentscope_run.py`（run_end 钩子） | 减少 train→val 断流损失 |
| P1 | 正交门失败写入实际相关值 + 对手因子 | `delivery/submit.py:851`、`agent/agentscope_tools.py:486` | 让模型能据此改正而非盲试 |
| P2 | 家族配额/饱和度：点名未开拓族，替代纯"禁止同构" | `prompt/modules/*`、`memory/advisory` | 打破 vwap 收敛 |
| P2 | 整夜脚本加 CC Switch/EasyCLI 探活 | `Desktop\quant\overnight_mining_monitor.py` | 避免整夜第一次调用即 502 |
