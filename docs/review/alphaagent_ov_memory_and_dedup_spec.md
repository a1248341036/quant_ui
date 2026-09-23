# AlphaAgent 长期记忆接入 OpenViking + 重复评估治理方案

> 状态：设计稿（待用户确认后实施）
> 分支：`docs/alphaagent-ov-memory-and-dedup-spec`
> 日期：2026-09-23
> 背景：用户反馈"记忆效果不满意，重复评估因子非常多"，要求接入 OpenViking 作 AlphaAgent 专属长期记忆库

---

## 0. 问题诊断（现状基线）

### 0.1 重复评估规模

963 次评估中 **46% 是重复**（443 次）：
- `duplicate_known_dead_end`：320 次（33%）——同结构指纹历史 ≥2 次失败，仍重复评估
- `duplicate_prior_result`：123 次（13%）——同结构指纹历史有正向 verdict，仍重复评估

### 0.2 根因

| 问题 | 位置 | 现状 |
|------|------|------|
| `hard_block_duplicates` 默认 `False` | `research_spec.py:194` | 死路重复只提醒不拦截，LLM 可无视 |
| `current_run_id` 没传给 `advisory_for` | `_dispatch.py:374-378` | `curr_run_passed` 豁免永远不生效（边缘 bug） |
| 族级饱和度不门控 `recommend_edits` | `retrieval.py:618-658`（D2） | vwap 族饱和度 ~1.00 仍被推荐 |
| 跨 session 策略记忆缺失 | 无 | 换 session 后族级趋势/模型偏好/换手率教训丢失 |

### 0.3 现有研究记忆能力边界

SQLite WAL + FTS5（`artifacts/alphaagent/research_memory.db`，31827 entries）：
- ✅ 毫秒级结构指纹精确去重（`_structure_fingerprint` AST 归一化）
- ✅ 表达式级正/负证据检索
- ❌ 无法语义模糊检索（"vwap 族为什么饱和"这类自然语言问题查不了）
- ❌ 跨 session 策略教训无沉淀（每次 run 从零开始）

---

## 1. 方案总览

三层并行治理，互不依赖，可分批落地：

| 层 | 名称 | 目标 | 改动面 | 风险 |
|----|------|------|--------|------|
| **L1** | 硬拦截止血 | 砍 33% 死路重复评估 | 配置文件 2 行 | 低（可回滚） |
| **L2** | OpenViking 长期记忆 | 跨 session 策略记忆语义检索 + 写回 | 新增 `ov_store.py` + 接入点 | 中（新代码，需测试） |
| **L3** | 重复治理代码修复 | 修 `current_run_id` bug + D2 族级饱和门控 | dispatch + retrieval 代码 | 中（改热路径） |

**落地顺序**：L1（立即）→ L2（核心价值）→ L3（深度治理）

---

## 2. L1：硬拦截止血（配置层，零代码）

### 2.1 改动

`artifacts/alphaagent/research_specs/technical.json` 和 `fundamental.json` 各加一行：

```json
{
  "memory_policy": {
    "hard_block_duplicates": true,
    ...其他保持不变
  }
}
```

### 2.2 拦截链路（已验证完整闭环）

```
research_spec.json memory_policy.hard_block_duplicates=true
  → agentscope_run.py:359 传给 ResearchMemoryStore(hard_block_duplicates=True)
  → dispatch._memory_gate:384  if getattr(self.memory_store, "hard_block_duplicates", False):
  → advisory_for 返回 duplicate_known_dead_end + exempt_from_block=False
  → dispatch:386-393  构造 blocked = {"ok": False, "error": "memory_blocked_duplicate:...", "error_type": "MemoryAdvisoryBlock", "memory_advisory": advisory}
  → dispatch:613/699/733/807  if gate.get("ok") is False: return gate  ← 评估/提交直接返回，不执行
```

**4 个调用点全部硬拦**：`evaluate_factor` / `eval_on_train_set` / `eval_on_val_set` / `submit_factor`。

### 2.3 LLM 收到的拦截信息（已确认非静默）

拦截 result 经 `_result_tool_chunk`（`agentscope_tools.py:480-482`）包成 ToolChunk JSON 文本块返回 LLM，包含：
- `ok: false` ——评估未执行
- `error: "memory_blocked_duplicate: 该表达式结构与历史死路相同：同构已评估 N 次...（reason）"` ——死路原因
- `error_type: "MemoryAdvisoryBlock"`
- `memory_advisory: {advisories: [{kind, message, exempt_from_block}]}` ——历史证据

**System prompt 已有行为引导**（`prompt/modules/strategy_tracks.py:85`）：
> 工具返回的 `memory_advisory` 是研究记忆的硬提醒（同结构死路 / 编辑方向被否决）：**命中后必须换方向**；无视提醒重复提交只会积累更多负证据。

LLM 收到拦截后会看到原因 + 证据 + "必须换方向"引导，不是静默丢弃。

### 2.4 拦截条件（不会误杀）

`duplicate_known_dead_end` 触发（`advisory.py:159`）：
- 同结构指纹 **≥2 个失败条目** 或 **≥2 次累计尝试**
- 且该指纹**全历史无正向 verdict**（`has_positive=False`，第 149 行全历史检查）

**安全保证**：只要该指纹**任何时候**有过 promising/入库，根本不触发死路判定（`not has_positive` 条件不满足）。硬拦只杀"从没出过信号、还反复试"的纯死路。

### 2.5 已知边缘 bug（L3 修，不阻塞 L1）

`_dispatch.py:374-378` 调 `advisory_for` 时没传 `current_run_id`，导致 `curr_run_passed` 永远 `False`，`exempt_from_block` 永远 `False`。

**实际影响有限**：`has_positive` 是全历史检查，只要该指纹历史有过正向 verdict 就不触发死路。bug 只在一种边缘情况触发——同结构指纹历史**只有失败**，但当前 run 内刚出了 promising（还没写入记忆库时）。这种情况下开硬拦会误杀当前 run 的 promising 变体。

**L1 缓解**：这种边缘情况概率低（需要"历史纯失败 + 当前 run 刚过线 + 还没写库"三重条件同时成立）。L3 修 bug 后彻底消除。

### 2.6 回滚

删掉两个 json 里的 `hard_block_duplicates` 行即恢复默认 `False`，无需改代码。

---

## 3. L2：OpenViking 长期记忆接入（核心价值）

### 3.1 定位：两层记忆分工

| 层 | 存储 | 用途 | 检索方式 | 延迟 | 规模 |
|----|------|------|----------|------|------|
| **热路径** | SQLite `research_memory.db` | 表达式级精确去重 + 正负证据 | 结构指纹精确匹配 + FTS5 | 毫秒级 | 31827 条 |
| **冷路径** | OpenViking `viking://resources/alphaagent/` | 跨 session 策略教训 + 族级趋势 + 模型偏好 | 语义向量检索（bge-small-zh 512 维） | 百毫秒级 | 数十条 |

**互补不重叠**：SQLite 回答"这个表达式结构试过没"，OpenViking 回答"vwap 族为什么不该再挖"。

### 3.2 隔离方案（已验证）

OpenViking 无 per-agent ACL，**代码层硬编码 `target_uri="viking://resources/alphaagent/"` 实现隔离**：

| 操作 | 隔离方式 | 验证结果 |
|------|---------|---------|
| `search` | `target_uri="viking://resources/alphaagent/"` | ✅ 只返回专属目录，0 泄漏个人记忆 |
| `find` | `target_uri="viking://resources/alphaagent/"` | ✅ 同上 |
| `read` | 只传 `viking://resources/alphaagent/...` URI | ✅ 代码硬编码 |
| `write` | 只写 `viking://resources/alphaagent/...` URI | ✅ 代码硬编码 |
| `glob` | `uri="viking://resources/alphaagent/"` | ✅ 只匹配专属目录 |

**AlphaAgent 所有 OpenViking 调用都硬编码 `viking://resources/alphaagent/` 前缀**，根本不碰 `viking://user/default/memories/`（个人记忆）或其他项目资源。

### 3.3 SDK 接入（已验证连通）

```python
from openviking_sdk.client import SyncHTTPClient

ov_client = SyncHTTPClient(
    url="http://127.0.0.1:1933",  # 本地 OpenViking
    account="default",
    user="default",
)
ov_client.initialize()  # 必须调

# 检索（硬编码 target_uri）
results = ov_client.search(
    query="vwap 族饱和度 换手率教训",
    target_uri="viking://resources/alphaagent/",
    limit=5,
)

# 写入
ov_client.write(
    uri="viking://resources/alphaagent/run_summaries/2026-09-23.md",
    content="...",
)
```

**依赖状态**：`openviking` 0.4.21 + `openviking-sdk` 0.1.12 已装到 `.venv`，`openai` 已回滚到 3.3.1（SDK 的 search/find/read/write 不依赖 openai 2.54）。

### 3.4 库骨架结构

`viking://resources/alphaagent/` 下：

```
viking://resources/alphaagent/
├── strategy_lessons.md          # 跨 session 策略教训（换手率/1d label/盲测段等结构性认知）
├── dead_families.md             # 已饱和/已证伪的信号族清单（vwap/volume momentum 等）
├── yield_patterns.md            # 已验证有效的产出模式（什么结构 + 什么面 + 什么调仓频率能出因子）
├── model_behaviors/
│   ├── deepseek.md              # DeepSeek-V4-Flash 行为偏好（同族扎堆、dup_dead_end 11.7%）
│   └── gemini.md                # Gemini-3.8-flash 行为偏好
└── run_summaries/               # 每次 run 结束写回的摘要（按日期）
    ├── 2026-09-22-deepseek.md
    └── ...
```

### 3.5 接入点设计

#### 3.5.1 run 启动时：语义检索注入 system prompt

**位置**：`agentscope_run.py` 的 `build_system_prompt` 调用前（约 line 380）

**逻辑**：
1. 根据 run 的 `focus_facets` + `research_mode` 构造检索 query（如"价量面 technical 模式 vwap 饱和度教训"）
2. 调 `ov_client.search(target_uri="viking://resources/alphaagent/", limit=5)`
3. 把检索到的教训摘要拼成"跨 session 策略记忆"块，追加到 `extra_instructions`
4. 失败静默（OpenViking 挂了不影响挖掘）

**注入内容示例**：
```
### 跨 session 策略记忆（来自 OpenViking 长期记忆库）

**已饱和信号族（勿重复探索）**：
- vwap 族：3012 条评估/976 正向（32%），饱和度 ~1.00，同结构变体无新增信息
- volume momentum 族：913 条/21 正向（2.3%），过线率极低

**结构性教训**：
- 1d label + weekly 调仓换手率必爆 (>0.50)，engine_gate 会拦截统计有效但换手高的因子
- 盲测段 2025 起锁定，频繁重测会烧掉诚实样本外

**当前模型偏好（DeepSeek-V4-Flash）**：
- 探索吞吐高但同族扎堆更严重（dup_dead_end 11.7%），需主动引导跨族开拓
```

#### 3.5.2 run 结束时：写回摘要

**位置**：`agentscope_run.py` 的 run 结束清理段（`run_end` 日志写入后）

**逻辑**：
1. 从 `run_metrics` 提取本次 run 的关键统计（评估数/重复率/各族过线率/入库数/换手率拒绝比例）
2. 从 SQLite 记忆库提取本次 run 新增的饱和族 + 新死路族
3. 拼成 markdown 摘要，调 `ov_client.write(uri="viking://resources/alphaagent/run_summaries/<date>-<model>.md")`
4. 如果发现新的结构性教训（如新饱和族），追加到 `dead_families.md` / `strategy_lessons.md`
5. 失败静默

**写回内容示例**：
```markdown
# Run 摘要 2026-09-23 DeepSeek-V4-Flash

## 关键统计
- 评估 219 次，重复 33%（dup_dead_end 11.7%）
- promising 30 个，入库 0 个
- 换手率拒绝占提交的 68%

## 新饱和族
- tug_vwap_dev_res_size_w10_int 族：本 run 评估 12 次，0 过线，建议加入 dead_families

## 结构性发现
- vwap_dev_rev_raw：train IC=0.0303 但 val 衰减严重，1d 快信号 weekly 调仓换手率 69%
```

#### 3.5.3 新增模块：`alphaagent/factor/mining/memory/ov_store.py`

封装所有 OpenViking 调用，硬编码 scope，对外暴露 3 个函数：

```python
class OVStore:
    """OpenViking 长期记忆接入层。所有调用硬编码 viking://resources/alphaagent/ scope。"""
    
    SCOPE = "viking://resources/alphaagent/"
    
    def __init__(self, url="http://127.0.0.1:1933"):
        self._client = None  # 懒初始化，失败静默
    
    def retrieve_lessons(self, focus_facets, research_mode, limit=5) -> str:
        """run 启动时检索跨 session 策略记忆，返回注入用文本块。失败返回空串。"""
    
    def write_run_summary(self, run_id, model_name, run_metrics, new_dead_families) -> bool:
        """run 结束时写回摘要。失败静默。"""
    
    def update_dead_families(self, families: list[str]) -> bool:
        """追加新饱和族到 dead_families.md。失败静默。"""
```

**设计原则**：
- 所有方法失败静默（OpenViking 不可用时挖掘照常跑）
- 懒初始化（首次调用才建 client，避免无 OpenViking 环境启动报错）
- 硬编码 SCOPE，不暴露 target_uri 参数（防误用泄漏 scope）

### 3.6 配置开关

`research_spec.py` 的 `memory_policy` 新增：

```python
"enable_ov_long_term_memory": True,   # OpenViking 长期记忆接入（False = 完全关闭，走纯 SQLite）
"ov_endpoint": "http://127.0.0.1:1933",  # OpenViking HTTP 端点
"ov_inject_max_chars": 1200,          # 注入块预算
```

`technical.json` / `fundamental.json` 可覆盖。默认 `True` 但失败静默，无 OpenViking 环境自动降级到纯 SQLite。

---

## 4. L3：重复治理代码修复

### 4.1 修 `current_run_id` 没传的 bug

**问题**：`_dispatch.py:374-378` 调 `advisory_for` 没传 `current_run_id`，导致 `curr_run_passed` 豁免永远不生效。

**修复**：
1. `FactorEvalTools.__init__` 加 `run_id: str | None = None` 参数
2. `agentscope_run.py:373` 创建 `FactorEvalTools` 时传 `run_id=log_dir.name`
3. `_dispatch.py:374` 调 `advisory_for` 时传 `current_run_id=self.run_id`

**影响**：当前 run 内已过线的 promising 因子，即使同结构指纹历史有失败，也不会被硬拦（`exempt_from_block=True`）。

### 4.2 修 D2：`recommend_edits` 加族级饱和度门控

**问题**：`retrieval.py:618-658` 的 `recommend_edits` 主路径只查 APV veto，不查族级饱和度。vwap 族饱和度 ~1.00 仍被推荐。

**修复**：在 `recommend_edits` 排序前加族级饱和度过滤——饱和度 >0.4 的族不进推荐清单（或降权到末位）。

**饱和度计算**（已有，`retrieval.py` 饱和度函数）：
```python
saturation = min(1, min(n_promising, 8)/40 + n_candidate/5 + n_validated/3)
```

**改动**：`recommend_edits` 返回前对每个候选查其 family 的饱和度，>0.4 的过滤掉或标 `saturated=True` 降权。

### 4.3 （可选）`duplicate_prior_result` 加冷却期

**问题**：同结构指纹历史有正向 verdict，仍重复评估（123 次）。当前 `duplicate_prior_result` 永不拦截。

**方案**：加冷却期——同结构指纹在当前 run 内已评估过且结果为 promising，N 轮内不重复评估（避免"promising 换名重测"）。冷却期外允许重测（可能参数变了）。

**风险**：可能误杀合法迭代（同结构不同参数）。建议默认不拦，仅提醒，等 L1/L2 落地后观察数据再决定。

---

## 5. 实施计划

### 5.1 分批落地

| 批次 | 内容 | 改动 | 验收 |
|------|------|------|------|
| **批次 1**（立即） | L1 硬拦截止血 | 改 2 个 json 配置 | 下次 run dup_dead_end 拦截率 0→33%，总评估数降 33% |
| **批次 2**（核心） | L2 OpenViking 接入 | 新增 `ov_store.py` + `agentscope_run.py` 接入点 + `research_spec.py` 配置 | run 启动注入跨 session 教训块，run 结束 `viking://resources/alphaagent/run_summaries/` 有新文件 |
| **批次 3**（深度） | L3 代码修复 | `FactorEvalTools` 加 run_id + `recommend_edits` 加饱和度门控 | 边缘 bug 消除，vwap 族推荐占比 34%→<20% |

### 5.2 分支纪律

- 批次 1：配置改动，走 `fix/hard-block-duplicates` 分支
- 批次 2：新代码，走 `feat/alphaagent-ov-memory` 分支
- 批次 3：热路径改动，走 `fix/dedup-governance` 分支

### 5.3 测试

- 批次 1：跑一次 test run，确认 `memory.advisory_block` 日志出现、评估数下降
- 批次 2：
  - 单元测试 `ov_store.py`（mock SyncHTTPClient，验证 scope 硬编码 + 失败静默）
  - 集成测试：启动 run，确认 system prompt 含"跨 session 策略记忆"块，run 结束后 OpenViking 有新文件
  - 隔离测试：确认 AlphaAgent search 不返回 `viking://user/default/memories/`
- 批次 3：跑 test run，确认 `curr_run_passed` 豁免生效，vwap 族推荐占比下降

---

## 6. 风险与取舍

### 6.1 L1 硬拦截风险

- **误杀风险**：边缘 bug（4.1）可能在"历史纯失败 + 当前 run 刚过线 + 还没写库"时误杀。概率低，L3 修后消除。
- **LLM 行为风险**：LLM 收到拦截后可能"学会"避开整族（过度泛化）。缓解：拦截 message 含具体死路原因，引导"换方向"而非"放弃族"。
- **回滚成本**：删 2 行配置即恢复，零代码回滚。

### 6.2 L2 OpenViking 风险

- **依赖风险**：OpenViking 挂了影响挖掘。缓解：所有调用失败静默，降级到纯 SQLite。
- **依赖污染**：`openviking-sdk` 装时升级 openai。已回滚，SDK 的 search/find/read/write 不依赖 openai 2.54。但**新环境装 openviking-sdk 时需手动 pin openai==3.3.1**。
- **隔离风险**：代码层硬编码 scope，无 ACL。缓解：`ov_store.py` 不暴露 target_uri 参数，所有调用走硬编码 SCOPE 常量。
- **性能风险**：语义检索百毫秒级，run 启动一次 + run 结束一次，不影响热路径。

### 6.3 L3 代码修复风险

- **`current_run_id` 修复**：改 `FactorEvalTools` 签名，向后兼容（默认 None，老调用方不受影响）。
- **D2 饱和度门控**：改 `recommend_edits` 排序逻辑，可能影响推荐多样性。建议先过滤 >0.4，不过滤 0.2~0.4（保留轻度饱和族的迭代空间）。

---

## 7. 验收指标

| 指标 | 现状 | 目标 |
|------|------|------|
| 重复评估占比 | 46%（443/963） | <15% |
| `dup_dead_end` 拦截率 | 0%（全提醒） | 100%（硬拦） |
| vwap 族评估占比 | 34% | <20% |
| `recommend_edits` 推荐族饱和度 >0.4 占比 | ~高 | 0% |
| 跨 session 策略记忆注入 | 无 | 每次 run 启动有注入块 |
| run 摘要写回 OpenViking | 无 | 每次 run 结束有摘要文件 |

---

## 附录 A：OpenViking SDK 调用验证记录

```
client = SyncHTTPClient(url="http://127.0.0.1:1933", account="default", user="default")
client.initialize()

# write
client.write(uri="viking://resources/alphaagent/strategy_lessons.md", content="...")
# → {'uri': 'viking://resources/alphaagent/strategy_lessons.md', 'mode': 'create', 'written_bytes': 147, ...}

# search 限定 scope（0 泄漏）
client.search(query="换手率 教训", target_uri="viking://resources/alphaagent/", limit=5)
# → 只返回 viking://resources/alphaagent/ 下条目

# search 不限定 scope（会泄漏个人记忆）
client.search(query="缚龙策 小说", limit=5)
# → 返回 viking://user/default/memories/ 下条目（泄漏！）

# 结论：必须在代码层硬编码 target_uri
```

## 附录 B：硬拦截 LLM 反馈验证

拦截 result 结构：
```json
{
  "ok": false,
  "error": "memory_blocked_duplicate: 该表达式结构与历史死路相同：同构已评估 3 次，最新 xxx（已累计 3 次评估未通过）",
  "error_type": "MemoryAdvisoryBlock",
  "memory_advisory": {
    "advisories": [
      {
        "kind": "duplicate_known_dead_end",
        "message": "该表达式结构与历史死路相同...",
        "exempt_from_block": false
      }
    ]
  }
}
```

System prompt 引导（`strategy_tracks.py:85`）：
> 工具返回的 `memory_advisory` 是研究记忆的硬提醒：**命中后必须换方向**

LLM 收到：原因 + 历史证据 + 行为引导，非静默丢弃。
