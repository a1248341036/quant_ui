# OCR review 卡死根因与实操 SOP（2026-10-20 实测）

> 触发：一个 8 文件 / 306 行的功能分支，OCR 连续 3 次在 20 分钟内零产出被停。
> 本文记录 6 次对照实验的数据、定位到的真实约束，以及能稳定出结果的用法。
> 相关纪律已写入 `AGENTS.md`「Git 分支纪律」段的 OCR review 条。

## 一、结论先行

**OCR 能不能跑完，取决于「单组文件体积」，不是总 diff 大小，也不是模型档位。**

可用渠道里不存在"窗口 ≥30 万 + 支持 tool-call + 够快"的组合；而 OCR 的 plan 阶段
在特定条件下会吃掉 16–30 万 token，且那两个阈值**不可配**。

## 二、实测数据（同一批改动：8 文件 / +306/-19）

| # | provider / model | 窗口 | 范围 | 耗时 | tokens | 结果 |
|---|---|---|---|---|---|---|
| 1 | easycli / DeepSeek-V4-Flash-0731 | 200k | 6 文件 | **20 min 被停** | — | 0 产出 |
| 2 | ark-glm / glm-5-3-flash | 大 | 13 文件（含他人合并） | 6m14s | 340k | 0 评论，预算耗尽 |
| 3 | ark-glm / glm-5-3-flash | 大 | 11 文件 | 5m11s | 304k | 三组全超时 |
| 4 | easycli / **0731** | 200k | 8 文件 | 7m37s | 317k | **7/7 失败，HTTP 500（爆窗）** |
| 5 | easycli / gemini-3.8-flash-high | 1M | 8 文件 | 10s | 0 | **7/7 失败，HTTP 400（渠道不支持 tool-call）** |
| 6 | command-goat / deepseek-v4.1-flash | 1M | 8 文件 | 4s | 0 | **7/7 失败，HTTP 400（同上）** |
| 7 | easycli / 0731（多文件组变小后） | 200k | 7 文件 | 7m25s | 563k | **EXIT=0，1 条 finding** ✅ |

第 7 次唯一的改动是"把大文件从组里摘出去"，结果从"全挂"变成"部分成功并出结论"。

## 三、根因链条

1. **plan 阶段**在以下任一条件触发（二进制里的 `PLAN_MODE_LINE_THRESHOLD=50` /
   `PLAN_MODE_GROUP_LINE_THRESHOLD=100`）：
   - 组内最大改动文件 ≥50 行；
   - 或组内 2+ 文件改动合计 ≥100 行。
2. 该阶段要把整个 diff 读进去做规划，**一次可吃 16–30 万 token**。
3. `--exclude` 只影响 core review 的**分组**，**不影响 plan 阶段的输入量**
   ——实测：排除到剩 1 个文件时 token 仍是 167k（与剩 6 文件相同）。
4. 于是 200k 窗口模型必然爆（HTTP 500），大窗模型则慢到超时或预算耗尽。

**两个阈值不可配**：
- `PLAN_MODE_LINE_THRESHOLD` 设成环境变量**不生效**（日志仍有 `Review planning`）。
- 二进制里搜到的 `compact_threshold` / `context_window` 是 **OpenAI Go SDK 的字段名**
  （`BetaResponseNewParams.CompactThreshold`）与 API 错误码
  （`model_context_window_exceeded`），**不是 OCR 的配置项**；`ocr config` 无压缩相关键。

OCR 确有 `Context compaction` 阶段（日志可见），但它只在"已接近墙边"时触发，
且那次直接 `cancelled`——属于兜底失败而非可用压缩。

## 四、可用 SOP

### 1. 先看分组（1 秒、零 token）

```powershell
ocr delegate preview --from main --to <branch>
```

输出会列出：可评审文件数 / 总增删行数 / **被默认排除的文件**（如 `tests/**` 命中
`default_path`）。先看有没有大文件（改动 ≥50 行），它是爆窗的来源。

### 2. 小文件批量跑（稳定出结论）

```powershell
ocr review --audience agent `
  --effort low `                 # 2 轮 → 1 轮（滚雪球主因）
  --timeout 7 `                  # 硬上限，别用默认 15 × retry
  --max-tokens 60000 `           # 单组超限直接跳过，不撞墙重试
  --exclude "**/大文件1.py,**/大文件2.py" `
  --from main --to <branch> --output <file>
```

### 3. 大文件另跑或直接人工审

单文件改动 ≥50 行时，它自己就会触发 plan 阶段。要么把它拆成多个 <50 行的小 commit
（每个 commit 单独 review），要么由宿主 agent 直接逐行审。

### 4. 止损

超 20 分钟仍无产出 → 停（`Stop-Process` 掉 node 进程）→ 转人工逐行评审，不反复重试。

## 五、价值定位（不要因为卡就弃用）

2026-10-20 这次，OCR 在部分成功的那一轮抓到一个**宿主 agent 人工评审漏掉的回归**：

> `session.py` 的 parquet 按需取列分支只并了 `label_col`，没并系统必需列 →
> `--panel <parquet> --universe top300cap` 时 `apply_universe` 抛
> `ValueError: 股票池 top300cap 需要面板列 'tot_cap'`，而改动前是正常的。

所以正确姿势不是"能省就省"，而是**喂对粒度**：

| 改动规模 | 谁审 |
|---|---|
| ≤2 文件 / ≤150 行 | OCR（上面的参数），作为独立视角 |
| 大改动 / 跨模块 | 宿主 agent 自审为主，OCR 只跑小文件部分作补充 |
| 涉及门槛 / 口径 / 金额 | 两者都做，且必须用户点头 |
