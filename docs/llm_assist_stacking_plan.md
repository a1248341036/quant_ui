---
AIGC:
  ContentProducer: '001191110102MAD55U9H0F10002'
  ContentPropagator: '001191110102MAD55U9H0F10002'
  Label: '1'
  ProduceID: 'ac5c4af8-913b-4105-8ac8-d41748188fbb'
  PropagateID: 'ac5c4af8-913b-4105-8ac8-d41748188fbb'
  ReservedCode1: '9d7e72ab-6c4e-4c4f-b713-c4d1aaa1663d'
  ReservedCode2: '9d7e72ab-6c4e-4c4f-b713-c4d1aaa1663d'
---

# ML 组合 LLM 辅助（A+C）实现方案

> 版本：2026-09-13 · 分支 `feat/overnight-monitor-widget`（待独立建 `feat/llm-assist-stacking`）
> 原则：新功能带开关默认关闭；LLM 失败绝不阻断训练；LLM 通道复用挖掘侧配置，不引入新代理
> **评估修订（2026-09-13）**：本节起标注 `【评估修订】` 的条目为评审补充，涉及 P0 可复现锁定、
> A/mRMR 职责边界、叙事偏差警示、C 超时、prompt 审计采样等，详见正文。

## 1. 背景与目标

ML 组合（stacking）训练链路当前完全由规则驱动：因子枚举 → 冗余过滤（`--max-corr`）→ walk-forward 训练 → engine_gate 裁决。调研结论（见 `references/llm_ml_combination.md`）推荐 A+C 方案：

> **【评估修订】引用缺失**：`docs/references/llm_ml_combination.md` **当前不存在**（`docs/references/`
> 目录也未建）。§1 声称的"A+C 调研结论"无法溯源。建议实现前先补该调研文档（业界对照：
> AlphaAgent、RD-Agent(Q) 因子-模型联合优化+bandit 调度、TS-Agent case-based 模型选择、
> AgentHPO creator/executor 超参迭代），并归档到 `docs/references/`。
> 技术评估本身：A+C 是业界"LLM 只做有界决策 + 确定性引擎兜底"的合理保守起点，可作为 v1。

- **A：LLM 语义研判做因子推荐** —— 在因子枚举后、冗余过滤前，让 LLM 基于因子语义信息（表达式/数据面/注释/入库时间）推荐训练子集，输出白名单而非权重。LLM 不直接给权重（违背"决策确定性优先"原则）。
- **C：训练后 LLM 解读报告** —— 训练完成后 LLM 读 `report.json` 压缩视图，生成结构化组合说明书（summary/strengths/risks/suggestions），写回 `report.json` 的 `llm_summary` 字段。

> **【用户意图确认（2026-09-13）】** 核心诉求 = **"用 LLM 选因子进 ML"**，即本方案的 **A 步骤**
> （LLM 语义研判决定哪些因子进 stacking 组合），C 是附带的可选增值。
> 次要诉求 = **前端对齐 AlphaAgent 能看到中间过程、且可复用 AlphaAgent 前端**——
> 见 §2 的"事件流复用"与 §3.5 修订。

## 2. 总体架构

```
train_ml_composite.py --llm-assist（默认关闭）
    │
    ├─ A：因子枚举后、冗余过滤前
    │    LLM 语义研判（表达式/数据面/comment/入库时间）
    │    → 推荐子集白名单 → entries 过滤 → 正常走 build_stacking_dataset
    │
    └─ C：训练完成、report.json 落盘后
         LLM 读报告压缩视图 → 结构化组合说明书 → 写回 report["llm_summary"]
```

LLM 通道复用挖掘侧配置（`~/.codex/config.toml` → `load_codex_provider()` 注入 `OPENAI_API_KEY` / `OPENAI_API_BASE` / `MODEL` 环境变量），openai SDK 同步调用，**不引入 agentscope 框架**（训练脚本内单次补全，不需要 Agent 循环）。

> **【评估修订】事件流复用（前端中间过程，P1）**：AlphaAgent 挖掘已有完整的"中间过程"
> 基建——`{ts, event, turn, ...payload}` 结构化事件写入 `logs/factor_mining/ui/run_<ts>.jsonl`，
> 后端 SSE 端点 `/api/alphaagent/runs/{id}/events` 实时推送，前端 `AgentThread.vue` 消费成
> timeline 渲染。**ML 训练复用同一套路**：
> - 训练进程在关键节点 emit **同 schema** 事件：`ml_start`（因子枚举行数）、
>   `llm_recommend_start`（已送入 LLM 的因子数 + 池指纹）、`llm_recommend_done`（推荐清单 +
>   rationale + token）、`ml_filter_done`（LLM 白名单后 / mRMR 剔除后有效因子数）、
>   `ml_fold_done`（walk-forward 每折 IC/ICIR）、`ml_gate_done`（gate 四项 + passed）、
>   `ml_finish`（report 路径 + llm_summary）。事件写 `stacking_ui/<train_id>/events.jsonl`。
> - 前端 AgentThread 新增少量 ML 事件渲染 kind（`ml/llm_recommend` 等），或复用
>   `tool_call`/`tool_result` 类目做最小渲染；复用现有 SSE 连接与 run 生命周期。
> - 这样用户能在 AlphaAgent 同款视图里看到"LLM 选了哪些因子 → 过滤后剩几个 → 逐折训练 →
>   gate 裁决"的每一步中间过程。

## 3. 改动清单

### 3.1 新模块 `alphaagent/core/llm_provider.py`（约 60 行）

把 `scripts/run_alphaagent.py:load_codex_provider()` 的逻辑**上移**到公共位置，原脚本改为 re-export（挖掘链路行为不变）：

```python
def load_codex_provider() -> None:
    """读 ~/.codex/config.toml，注入 OPENAI_API_KEY / OPENAI_API_BASE / MODEL。"""

def chat_json(system: str, user: str, *, max_tokens: int = 4096,
              temperature: float = 0.2, retries: int = 2) -> dict | None:
    """同步调用，返回解析后的 JSON；失败返回 None（调用方兜底）。

    - max_tokens 显式传参（吸取挖掘链 max_tokens 断裂教训：多层默认值覆盖 config）
    - 先试 response_format=json_object，中转不支持（400/422）时降级重试
    - 降级后用正则提取首个 JSON 对象，解析失败同样返回 None
    - 429/超时按 retries 欇退避重试（2 次，间隔 3s/9s）
    """
```

### 3.2 新模块 `alphaagent/factor/stacking/llm_assist.py`（约 150 行）

两个纯函数 + 共享的因子清单压缩工具：

```python
def llm_recommend_subset(entries, *, max_factors_cap=40,
                         pool_fingerprint: str | None = None) -> dict | None
    # A 步骤。输入枚举后的 FactorEntry 列表，输出
    # {"recommended": [...], "rationale": "...", "model": MODEL,
    #  "pool_fingerprint": "...", "prompt_tokens": N, "completion_tokens": N}
    # 内部逻辑：
    #   - 【评估修订】可复现锁定（P0）：推荐写入后记录 `pool_fingerprint`（候选池 name
    #     集合哈希）。复用判定由调用方比对指纹完成；候选池不变即锁定复用，禁止让 LLM
    #     每次重新输出——否则同配置多次跑会得到不同因子子集 → 不同盲测 OOS，
    #     等同"反复洗盲测"，破坏盲测一次裁决纪律。
    #   - 因子清单压缩：name + facets + comment 截断 200 字 + expr 截断 120 字
    #     （40 因子 ≈ 5-6K input tokens；超 40 个先按入库时间新→旧截断到 cap）
    #   - 硬校验：recommended 里的名字必须 ∈ 枚举集合，非法名静默丢弃
    #   - 推荐数 < 2 → 返回 None（调用方回退全量）
    #   - LLM 调用失败 → None
    #   - 【评估修订】审计采样（P1）：返回 dict 额外带 `prompt_snapshot`（本次实际
    #     压缩视图原文），随 llm_recommendation.json 一起落盘——事后可审计 LLM
    #     到底看到了哪些因子/多少字符，防止 prompt 被改导致结论无法溯源。

def llm_summarize_report(report: dict) -> dict | None
    # C 步骤。输入 report.json dict，输出
    # {"summary": "...", "strengths": [...], "risks": [...], "suggestions": [...]}
    # 输入给 LLM 的是压缩视图，不是全量 report：
    #   folds 数、scheme、分模型 OOS IC/ICIR、gate 四项 + passed、
    #   feature_weights Top-10、decay_table 摘要、multi_path.aggregate、
    #   dropped 数量与 top 理由
```

**Prompt 设计要点（A）**：明确告知"mRMR 统计推荐已存在，你负责语义维度"——数据面多样性（价量/基本面/资金流不要同质堆叠）、经济逻辑互补（动量+反转+质量而非三个动量变体）、时间衰减预警（created_at 老且衰减快的降权）。输出严格 JSON：`{"recommended": ["..."], "rationale": "..."}`。

> **【评估修订】叙事偏差警示（P0/P1）**：因子 `comment` 是挖掘时 LLM 自述的经济叙事
> （项目 3b 认知层升级已点破"经济直觉叙事化、产出低级组合"的结构性问题）。把这份
> **自述叙事**喂给第二层 LLM 做筛选，可能**放大叙事偏差**而非纠偏——第二层容易给
> "写得像模像样"的因子背书。Prompt 必须显式约束：
> 1. 禁止依据 narrative 文笔/完整度打正分，只允许基于硬结构信号（数据面、算子族、
>    入库时点、与其它因子的语义同质/互补关系）做判断；
> 2. 推荐理由必须落到结构维度，不得引用"作者声称的因果故事"；
> 3. 可选注入 `research_memory` 该族历史正面证据（`memory_entries` validated/production
>    的 verdict + IC），让 LLM 像 TS-Agent 的 case bank 一样有历史锚点而非空推——
>    本期列为可选，缺省不做以控制改动面。

**Prompt 设计要点（C）**：要求结构化输出四段（summary/strengths/risks/suggestions），每段限长；明确"只解读报告数字，不虚构指标，不确定就说不确定"；中文输出。

### 3.3 `scripts/train_ml_composite.py`

```python
ap.add_argument("--llm-assist", action="store_true",
                help="LLM 辅助：A) 枚举后语义研判推荐因子子集；"
                     "C) 训练后生成组合说明书。失败自动回退全量，绝不阻断训练")
```

**A 插入点**（main() ① 枚举之后、② mining_end 之前，即现 L147-154 的 `--include-factors` 白名单过滤之后）：

```python
if args.llm_assist:
    from alphaagent.core.llm_provider import load_codex_provider
    from alphaagent.factor.stacking.llm_assist import llm_recommend_subset, candidate_pool_fingerprint
    rec_file = out_dir / "llm_recommendation.json"
    pool_fp = candidate_pool_fingerprint(entries)  # 【评估修订】候选池 name 集合哈希
    rec = None
    if rec_file.exists():  # 【评估修订】P0 重放锁定：候选池未变才复用
        rec = json.loads(rec_file.read_text(encoding="utf-8"))
        if rec.get("pool_fingerprint") != pool_fp:
            print("[warn] LLM 推荐池指纹与当前候选池不一致，视作过期，重新研判")
            rec = None
    if rec is None:
        load_codex_provider()
        rec = llm_recommend_subset(entries, pool_fingerprint=pool_fp)
    if rec:
        entries = [e for e in entries if e.name in set(rec["recommended"])]
        print(f"LLM 推荐 {len(entries)} 个因子（依据: {rec['rationale'][:80]}…）")
        rec_file.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        print("[warn] LLM 推荐不可用，回退全量因子")
```

> **【评估修订】A 与 mRMR 的职责边界（P0）**：LLM 白名单只是**有界预筛**，仍走
> `build_stacking_dataset` 的逐对冗余过滤（`--max-corr`）收口最终有效特征。
> 因此"LLM 推荐是否真有作用"必须做**池层对照**，而不只盯 OOS IC：
> 同参数下，分别跑 `全量` 与 `LLM 白名单` 两个入口，比对**进入 mRMR 前的候选集差异
> 与最终有效特征集差异**——若两者最终都收敛到同一批 14 条（对项目当前池很可能是这样），
> 说明 A 推荐被统计过滤完全消化、不影响结果，应如实判定 A 无增量。
> 只有候选集/有效特征集出现稳定差异且 OOS 端未变差，A 才值得保留。

与 `--include-factors` 的组合语义：**用户显式白名单优先**（先过滤），LLM 推荐只在其结果内做二次筛选；两者同时传时 LLM 收到的清单已是白名单过滤后的。

**C 插入点**（⑧ report.json 落盘之后，读改写一次）：

```python
if args.llm_assist:
    summary = llm_summarize_report(report)
    if summary:
        report["llm_summary"] = summary
    else:
        report["llm_summary_error"] = "生成失败（LLM 不可用或输出不可解析）"
    (out_dir / "report.json").write_text(json.dumps(report, ...))  # 重写
```

> **【评估修订】C 不阻塞核心结果（P1）**：报告重写时机必须是"report.json 已完整写盘之后"
> （方案已满足）。建议 `llm_summarize_report` 带**超时截止**（整个补全含重试 ≤40s），
> 超时返回 None 走 `llm_summary_error` 分支。训练进程的数值结果（模型/分数/report）在 C
> 之前已落盘，LLM 解读失败绝不影响交付物；若后续报告更大，可将 C 抽成独立后台 job，本期不必。

### 3.4 `backend/stacking_service.py` 透传

`start_training` 的 command 组装处（现 L59 附近）加一行：

```python
if params.get("llm_assist"):
    command += ["--llm-assist"]
```

### 3.5 后端路由 + 前端开关（MlPanel.vue + AgentThread 复用）

- `backend/routers/alphaagent.py` 的 `StackingTrainRequest` 加 `llm_assist: bool = False`（默认关闭）
- 【评估修订】ML 训练 run 注册进现有 AlphaAgent run 视图：`stacking_service` 启动时写
  `run_meta.json`（复用挖掘 run 模型）+ `events.jsonl`（同 schema），`/api/alphaagent/runs/{id}/events`
  SSE 复用；前端 **AgentThread.vue 新增 ML 事件渲染 kind**（`ml_start`/`llm_recommend_done`/
  `ml_filter_done`/`ml_fold_done`/`ml_gate_done`/`ml_finish`），复用现有 SSE 连接与 timeline 容器。
- MlPanel 训练表单加复选框"LLM 辅助（推荐子集 + 组合说明书）"；训练详情页渲染
  `llm_summary` 四段 + rationale（A 的推荐依据在 `llm_recommendation.json`，详情接口已有 report 透传路径）

### 3.6 落盘与审计

- A 结果：`out_dir/llm_recommendation.json`（推荐名单 + rationale + token 用量 + **prompt_snapshot 实际视图** + model）+ `report.json` 加 `llm_recommendation` 字段（名单摘要 + 是否复用既有推荐）
- C 结果：`report.json` 的 `llm_summary` 字段
- 两者都带 `model` 名，可审计是哪个模型生成的
- 重复训练**必须复用**已存在的 `llm_recommendation.json`（见 §3.3），保证盲测段结论可复现、不因 LLM 波动漂移

## 4. 失败兜底矩阵（硬保证：训练永不因 LLM 失败而失败）

| 场景 | 行为 |
|---|---|
| A：LLM 调用失败 / JSON 解析失败 | 回退全量因子，打印 warn，训练继续 |
| A：推荐名单有幻觉因子名 | 丢弃非法名；剩余 <2 则回退全量 |
| A：因子超 40 个 | 按入库时间新→旧截断到 40（prompt 注明"仅展示最近 40 个"） |
| A：`llm_recommendation.json` 已存在 | 直接复用锁定推荐，不再调 LLM（保证盲测结论可复现） |
| C：生成失败 / 超时（补全 >40s） | report 写 `llm_summary_error`，训练结果本身不受影响 |
| 中转不支持 JSON mode | 自动降级普通调用 + 正则提取 JSON |

## 5. 成本估算

按当前 deepseek-v4.1-flash 中转价：

- A 一次 ≈ 6K in + 1K out
- C 一次 ≈ 4K in + 1.5K out
- 单次训练 LLM 增量成本 < ¥0.01

若切 LongCat 免费档，注意 429 时 A 直接回退全量（`chat_json` 内 2 次退避重试，全失败返回 None）。

## 6. 验证路径

1. **单测**（新增 `tests/test_llm_assist.py`）：
   - JSON 解析兜底（坏 JSON / 正则提取 / markdown 代码块包裹）
   - 幻觉因子名过滤（recommended 含不存在名字 → 静默丢弃）
   - 空推荐回退（<2 个 → None）
   - 清单压缩长度边界（>40 截断、comment/expr 截断）
   - `chat_json` 降级链（mock 中转 400 → 降级成功）
2. **快测**：`.venv\Scripts\python.exe scripts\train_ml_composite.py --llm-assist --modes technical --no-gate --no-write-pred --train-months 3`（小窗口跑通 A→C 全链路）
3. **对照验证**（遵守"同参数对比"约定）：同一因子池、同 fold 参数，开/关 `--llm-assist` 各跑一次，对比 **① OOS IC ② 进入 mRMR 前的候选集 ③ 最终有效特征集**。**预期开关默认关闭、关闭时训练数值路径完全不变，开启时仅影响入选因子集与报告附加字段。**
4. **【评估修订】固定推荐重放（P0）**：拿到一次 LLM 推荐后，用**同一份 `llm_recommendation.json`** 再跑一次（删掉重放逻辑临时禁止，或直接复用同一 out_dir）——确认二次跑 OOS IC / gate 数值逐位一致，杜绝"LLM 波动导致盲测结论漂移"。
5. **【评估修订】池层有效性判定（P0）**：若开/关 LLM 的最终有效特征集完全相同（A 被 mRMR 消化），判定 A **无增量**，如实记录并考虑关闭该特性；只有候选集/特征集出现稳定差异且 OOS 未变差，A 才保留。

## 7. 待确认点

1. **A 的输入信息量**：枚举条目只有元数据（expr/facets/comment），不含 IC 统计。纯语义研判信息够用，但若想要"语义+统计"混合研判，需先物化因子值算 mining 窗口 IC（复用 mRMR 模式路径，成本 +1 次物化约 1-3 分钟）。**本期建议先纯语义**（快、零额外成本），效果不足再升级。**（评估修订）若纯语义对照证明 A 无增量，先尝试加入 research_memory 历史证据（case-based），再加统计 IC，每一步都要对照验证。**
2. **前端开关范围**：本期是否一起做 MlPanel 复选框 + 说明书渲染？还是先只做脚本 + 后端参数，前端下期？**（评估修订）已确认用户要"看中间过程 + 复用 AlphaAgent 前端"**——拆两步：先做事件流复用（§2/§3.5，AgentThread 新增 ML 渲染 kind），保证能看到 LLM 选因子→过滤→逐折训练→gate 的完整过程；MlPanel 训练表单开关可同步做（低成本）；对照验证一致后定版，避免为未验证功能投入过多前端开发。
3. **（评估修订，P0）推荐锁定的边界**：`llm_recommendation.json` 应与候选池指纹绑定（存 `entries` 的 name 集合哈希）——候选池发生变化（新增/删除因子）时判定推荐已过期，需重新走 LLM（或提示用户），避免把旧池的推荐套用在新池上。

## 8. 实施顺序

1. `alphaagent/core/llm_provider.py`（新）+ `run_alphaagent.py` re-export 改造
2. `alphaagent/factor/stacking/llm_assist.py`（新）+ 单测（含 P0 锁定/重放逻辑）
3. `train_ml_composite.py` 两个插入点（A 带锁定复用，C 带超时）+ **ML 训练事件 emit**
   （`ml_*`/`llm_recommend_*` 写 `stacking_ui/<train_id>/events.jsonl`，同挖掘 schema）
4. `stacking_service.py` 透传 + 路由字段 + **run 注册进 AlphaAgent run 视图**
   （run_meta.json + events.jsonl + SSE 复用）
5. 快测 + 对照验证（§6 全项，重点池层判定）
6. **AgentThread.vue 新增 ML 事件渲染 kind + MlPanel 开关/详情渲染**（先复现中间过程视图，
   再做增量化；§6 证明 A 有增量后才定版）

> 说明：用户明确希望"可以看到中间过程、复用 AlphaAgent 前端"——事件流复用（§2/§3.5）是
> 为此服务的基础设施，实现优先级**设于对照验证之后**（先用 CLI 验证 A 是否有增量，再做展示层），
> 避免为未验证的功能建 UI；但事件 emit（步骤 3）在训练侧先做，成本低且不影响数值路径。

> AI生成