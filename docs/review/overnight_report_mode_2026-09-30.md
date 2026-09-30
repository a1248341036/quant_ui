# 研报模式整夜运行报告（2026-09-29 晚 → 09-30 07:00）

- 目标：达成「复现研报 → 围绕研报发散」闭环，并整夜运行验证
- 分支：`feat/report-mode`（**未 push、未合 main**）；main 干净（`76d609c`）
- 运行方式：`overnight_mining_monitor.py --deadline 07:00 --no-restart-backend --research-mode report`
- 环境：后端 17891 携带 MinerU 语料开关（`ALPHA_REPORT_RAG_SOURCE=local`、`ALPHA_REPORT_CORPUS=parsed_mineru`）

## 一、结论

**闭环已达成并在真实 run 中验证**（目标达成项，附实测证据）：

| 环节 | 证据 |
|---|---|
| 复现题面 | `report_reproduce_task turn=0 qid=RQ_003 chars=2506 card=mc_1229 evidence=748`（机制卡 + 研报原文 + 硬约束） |
| 复现判定（指标通道） | `RQ_712 → rq712_vwap_soft_gate_sz`，`ic=0.0342 icir=0.3597 cov=0.932` → `reproduce_ok`，锁 3 轮 |
| 复现判定（形态通道，spec 主判据） | `RQ_002 → rq002_mom60_grouprank_ind`，`shape=confirmed | ic=0.0094 icir=-0.0853 cov=0.844` → `reproduce_ok` |
| 锁定发散（跨轮同题） | `turn=0 reproduce → turn=1 diverge`，状态机 `lock 3→2→1`，qid 全程不变 |
| 单维变异 | `rq712_vwap_soft_gate_sz_w20`（window）、`_sizeonly`（neutralize）、`_ma20/_ema`（operator）、`_grp`（interaction）、`_rank`（operator） |
| 父本声明 | jsonl 实测 `"parent_factor": "rq712_vwap_soft_gate_sz", "edit_note": "重建父本基线…"` |
| 阶段化注入 | `phase=reproduce` 时 **无** `report_rag_dynamic`；`phase=diverge` 时注入 `chars=1090`（MinerU 语料） |
| 复现门禁 | 模型遵守 `parent=reproduce_of:RQ_xxx`；不合规调用被拒 |

**未达成项（如实）**：严格锚定复现版的单维变体**没有产出 candidate**。本夜 run `794bfef83938` 的 8 个 promising **全部来自模型脱稿自由探索**（`turn20_sg065_52wdist_vol2`、`turn8_sg52r_vol3_f` 等，无 `rq002_` 前缀）；锚定变体最高 `|ic|=0.0243 / |icir|=0.1943`（promising 线为 0.02/0.28）。

## 二、本夜交付的代码（分支 `feat/report-mode`，23 个提交）

| commit | 内容 |
|---|---|
| `4d96590` | 研报 RAG 可切本地 MinerU 语料（`ALPHA_REPORT_RAG_SOURCE/ALPHA_REPORT_CORPUS`）+ 检索片段改「全文最相关段落」+ MinerU 批量解析与运维脚本 |
| `a139124` | **Phase 0**：新增 `report`（研报复现）模式 + `report_policy_overrides` + 阶段化知识注入（其他模式零影响） |
| `3cfa991` | **Phase 1**：复现题面 `render_reproduce_task` + `reproduce_of` 硬门禁 + 题库文本清洗（去表格残渣） |
| `07c26e5` | **Phase 2**：课题状态机（复现 1 轮 + 锁定 N 轮发散 + 跨 run 累积） |
| `6fc080e` | 修 `report_flow_enabled` 局部遮蔽（曾致题面/状态机/RAG 整块被静默跳过）+ `expected_shape` 主型别名（曾致 36/36 评估全废） |
| `da8e8ac`/`db8d5de` | 修门禁返回值形状（`ToolChunk(content=[TextBlock(...)])`）与残留语法错误 |
| `e1edb4d` | **复现判定加形态对账通道**，并前置到 promising 门槛之前（此前判定根本走不到） |
| `0c5a383` | 复现题面接入机制卡 + 研报原文；判定改「信号存在档」（`reproduce_min_abs_ic=0.010`/`reproduce_min_icir=0.10`） |
| `e5fb657`/`15bfdf4` | 发散轮补题面（父本 + 单维约束）+ 修上次补丁中止导致的未注入 |
| `829331e` | 复现失败可重试一次（`reproduce_max_attempts=2`） |
| `b56fce3` | 发散题面补目标线（promising 线/换手/撞库）+ 维度轮换建议 |
| `f36a8f9` | 发散题面展示父本实测基线（必须超越） |
| `c328228` | 机制卡匹配阈值 2.0→1.0（实测前 40 题 40/40 匹配上） |
| `b441bc6` | 弱母本优先结构杠杆（gate_shape/interaction） |
| `8f0f02d` | 同构空间耗尽即止损（防整轮空转；实测上一轮白烧 40 分钟） |
| `6bc1e04`+`b8eef9e` | 可选发散轮父本硬门禁 `diverge_parent_required`（**默认关**） |

## 三、Phase 4：机制卡扩容（已完成）

```
[done] 新增 335 张卡 | 跳过 670 篇（报告未描述可复现机制）| 失败 8 篇 | 耗时 33.5 分钟
[stats] confidence high=19 medium=219 low=97 | 公式回证成功=176/335
[validate] 输出文件共 810 张卡，校验错误 0 条
```
卡片样例（`mc_1360`，东方证券《成长风格登顶，pb roe 排序差因子表现出色》）：含 `who_wrong/why_persists/observable` 三段式机制 + fields/params + 页锚点来源。

## 四、夜间资源与脚本

- `scripts/batch_mineru_parse.py`：MinerU 全量解析驱动（断点续跑、并发、`--min-year/--order recent`、`CREATE_NO_WINDOW` 防黑框、doclib 图片定位符转注释）
- `scripts/mineru_progress.ps1` / `scripts/resume_mineru_parse.ps1`：进度速查与一键续跑（幂等）
- MinerU 语料：`data/research_reports/parsed_mineru/`（2020+ 全覆盖；`parsed/` v1 与 `parsed_v2/` 保留可回滚）

## 五、遗留与建议（等你决定）

1. **取舍开关**：严格「围绕研报发散」（`diverge_parent_required=true`，产出更低但纪律纯）vs 允许脱稿（当前默认，产出 promising 但脱离研报血统）vs 两模式时间片交替。**建议 C：report 模式 2 : 自由探索 1**，兼顾闭环验证与候选产出。
2. **候选转化**：本夜提交的因子均未过盲测（如 `amp60_resid_size` retention 0.152 < 0.5），与研报模式无关，是策略层既有卡点。
3. **复盘事项**：我今晚两次把坏文件提交进历史（`da8e8ac` 语法坏 → `db8d5de` 修；`6bc1e04` 缩进坏 → `b8eef9e` 修），tip 均干净；建议合并前 squash。
4. 机制卡可继续扩容（目标「每课题 ≥1 张」，当前 810 张覆盖约 810 篇报告）。
