# AlphaAgent 长期记忆接入 OpenViking + 重复评估治理方案

> 状态：设计稿 v2.1（评审修订版，待用户确认后实施）
> 分支：`docs/alphaagent-ov-memory-and-dedup-spec`
> 日期：2026-09-23
> 背景：用户反馈"记忆效果不满意，重复评估因子非常多"，要求接入 OpenViking 作 AlphaAgent 专属长期记忆库
>
> v2 修订摘要（相对 v1）：
> - **D2 已修复**：`retrieval.py:650-654` 已有族级饱和度 >0.4 门控 + D3 零产出族门控，§4.2 删除，L3 批次缩为只修 `current_run_id` bug。
> - **L1 配置路径修正**：`artifacts/alphaagent/research_specs/*.json` 被 gitignore（不进版本控制），原"改 json + 走分支 + git 回滚"方案不成立。改为改 `research_spec.py` 默认值 + 提供 env 开关 + 脚本化回滚。
> - **`current_run_id` 修复范围修正**：CLI 路径 `agent/run.py:87` 的 `FactorEvalTools` 未接 `memory_store`（`_memory_gate` 直接短路），无需传 `run_id`。只改 Web 路径 `agentscope_run.py:373`。
> - **L1 增补 `duplicate_prior_result` 轻量提醒**（同 run 内已评估过的同构 → 提示 submit 或变异，不硬拦）。
> - **L2 `run_summaries/` 增长策略 + append 语义 + 隔离测试断言**补强。
> - **验收指标**里 D2 相关两条改为"实测确认"。
>
> v2.1 修订摘要（相对 v2）：
> - **§2.1 改动 C 修正**：`duplicate_prior_result` 分支已有完整 `action_advice`（"建议 submit 或变异"），原 v2 追加与之语义重复。改为只追加"本 run 内已测过同构"事实前缀，不重复 action_advice。`exempt_from_block=True` 保持不变（永不硬拦）。
> - **§3.4 与 §3.5.2 矛盾修正**：v2 §3.4 写"单文件 `run_summaries.md` append"，§3.5.2 写"独立文件 + compaction"，两处矛盾。统一为"独立文件 `run_summaries/<date>-<model>-<run_id>.md` + 定期 compaction 合并到 `run_summaries.md` 主文件"，目录树同步更新。

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
| ~~族级饱和度不门控 `recommend_edits`~~ | ~~`retrieval.py:618-658`（D2）~~ | **已修复**（`retrieval.py:650-654`，饱和度 >0.4 过滤 + 零产出族过滤 + 回退路径同口径） |
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
| **L1** | 硬拦截止血 + prior_result 提醒 | 砍 33% 死路重复 + 提示 13% prior_result 重复 | `research_spec.py` 默认值 + env 开关 | 低（可回滚） |
| **L2** | OpenViking 长期记忆 | 跨 session 策略记忆语义检索 + 写回 | 新增 `ov_store.py` + 接入点 | 中（新代码，需测试） |
| **L3** | 重复治理代码修复 | 修 `current_run_id` bug（D2 已修复，不再列入） | dispatch + agentscope_run 代码 | 中（改热路径） |

**落地顺序**：L1（立即）→ L2（核心价值）→ L3（深度治理）

---

## 2. L1：硬拦截止血 + prior_result 提醒（配置层）

### 2.1 改动

> **v2 修正**：`artifacts/alphaagent/research_specs/*.json` 被 gitignore（`git check-ignore` 确认），不进版本控制。原方案"改 json + 走分支 + git 回滚"不成立——改 json 既无法走分支纪律也无法通过 git 回滚。改为：

**改动 A**：`alphaagent/factor/mining/research_spec.py:194` 默认值 `False` → `True`：

```python
# research_spec.py:194 附近
"hard_block_duplicates": True,   # v2：默认硬拦死路重复（原 False 只提醒）
```

**改动 B**：新增 env 开关 `ALPHA_MEMORY_HARD_BLOCK_DUPLICATES` 覆盖默认值（`agentscope_run.py` 加载 `memory_policy` 时读取），便于不改代码紧急回滚：

```python
# agentscope_run.py 加载 memory_policy 后
import os
_hard_block_env = os.environ.get("ALPHA_MEMORY_HARD_BLOCK_DUPLICATES")
if _hard_block_env is not None:
    memory_policy["hard_block_duplicates"] = _hard_block_env.lower() in ("1", "true", "yes")
```

**改动 C**：`duplicate_prior_result` 增补"本 run 内已测过同构"事实前缀（不硬拦，`exempt_from_block=True` 不变）。`advisory.py:217-229` 的 `duplicate_prior_result` 分支已有完整 `action_advice`（"建议直接 submit 走入库门槛；或以其为父本做显式变异"），无需重复。仅追加同 run 内已评估的事实判断前缀：

```python
# advisory.py duplicate_prior_result findings.append 内，message 拼接前
curr_run_evaluated = False
if current_run_id:
    curr_run_evaluated = bool(conn.execute(
        "SELECT 1 FROM memory_entries WHERE structure_fingerprint=? AND last_run_id=? LIMIT 1",
        (fingerprint, str(current_run_id)),
    ).fetchone())
run_prefix = "本 run 内已测过同构，" if curr_run_evaluated else ""
# message 改为：f"{run_prefix}该表达式结构与历史条目重复：..."
```

> 此改动依赖 L3 的 `current_run_id` 传递才能生效。L1 先落地时（`current_run_id` 仍为 None）该前缀不触发（`run_prefix=""`），等 L3 修完后自动生效，无额外成本。`exempt_from_block=True` 保持不变——`duplicate_prior_result` 永不硬拦，只提醒。

### 2.2 拦截链路（已验证完整闭环）

```
research_spec.py memory_policy.hard_block_duplicates=True（默认）
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

### 2.5 已知边缘窗口（L3 修后收窄但不消除）

`_dispatch.py:374-378` 调 `advisory_for` 时没传 `current_run_id`，导致 `curr_run_passed` 永远 `False`，`exempt_from_block` 永远 `False`。

**时序语义澄清**（v2）：即使 L3 修了 `current_run_id`，`curr_run_passed` 查的是 `memory_entries` 表里 `last_run_id = current_run_id` 的正向条目——而**当前 run 内刚过线的 promising 要等 `submit_factor` 或评估落库后才写入 `memory_entries`**。在写入前的那几次同构重评，`curr_run_passed` 一定是 `False`。所以 L3 修 bug 后豁免窗口从"永远不豁免"收窄到"**写入前不豁免**"，但不完全消除。

**L1 缓解**：在拦截 message 里追加引导"若本 run 内已测出过信号，请直接 submit 走入库门槛"（见 §2.1 改动 C 的 message 已含此意），避免 LLM 被硬拦后放弃真正有希望的变体。该边缘窗口需要"历史纯失败 + 当前 run 刚过线 + 还没写库"三重条件同时成立，概率低。

### 2.6 回滚

**v2 修正**：json 不进 git，无法通过 git 回滚。回滚方式：
- **紧急回滚**：设 env `ALPHA_MEMORY_HARD_BLOCK_DUPLICATES=0` 重启 run，无需改代码。
- **持久回滚**：改回 `research_spec.py:194` 默认值为 `False`（走 git 分支）。

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

**v2 隔离加固**：`ov_store.py` 不暴露原始 `ov_client`，只暴露业务方法（`retrieve_lessons` / `write_run_summary` / `update_dead_families`）。**新增单元测试断言"`ov_store.py` 所有出口的 OpenViking 调用 target_uri 都以 `viking://resources/alphaagent/` 开头"**，防回归（见 §5.3）。

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

**v2 依赖 pin**：把 `openai==3.3.1` 写进 `requirements-alphaagent.txt`，不靠手动 pin。新环境 `pip install -r requirements-alphaagent.txt` 不会误升级 openai。

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
├── run_summaries.md             # compaction 后的滚动主文件（近 30 条摘要，定期合并）
└── run_summaries/               # 每次 run 独立摘要文件（按 date-model-run_id 命名，无并发冲突）
    ├── 2026-09-23-deepseek-<run_id>.md
    └── ...
```

**v2 增长策略**：`run_summaries/` 目录存每次 run 的独立摘要文件（文件名含 run_id 保证唯一，多 run 并行写无冲突）。**定期 compaction** 任务读 `run_summaries/` 下全部文件，按时间排序取近 30 条，合并写入 `run_summaries.md` 主文件供语义检索；旧独立文件保留作归档（不删除，避免丢数据）。`dead_families.md` / `strategy_lessons.md` / `yield_patterns.md` 是结构化主文件，由 run 结束时的 `update_dead_families` 等方法维护，不随 run 摘要滚动。

### 3.5 接入点设计

#### 3.5.1 run 启动时：语义检索注入 system prompt

**位置**：`agentscope_run.py` 的 `build_system_prompt` 调用前（约 line 380）

**逻辑**：
1. 根据 run 的 `focus_facets` + `research_mode` 构造检索 query（如"价量面 technical 模式 vwap 饱和度教训"）
2. 调 `ov_store.retrieve_lessons(focus_facets, research_mode, limit=5)`
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

**v2 数字来源澄清**：示例中的具体数字（"3012 条评估/976 正向"）由 `retrieve_lessons` 实时查 SQLite 记忆库生成（不存 OpenViking 快照，避免过时）。run 启动时多一次 SQL 查询，延迟可接受（毫秒级）。

#### 3.5.2 run 结束时：写回摘要

**位置**：`agentscope_run.py` 的 run 结束清理段（`run_end` 日志写入后）

**逻辑**：
1. 从 `run_metrics` 提取本次 run 的关键统计（评估数/重复率/各族过线率/入库数/换手率拒绝比例）
2. 从 SQLite 记忆库提取本次 run 新增的饱和族 + 新死路族
3. 拼成 markdown 摘要，调 `ov_store.write_run_summary(run_id, model_name, run_metrics, new_dead_families)`
4. 如果发现新的结构性教训（如新饱和族），调 `ov_store.update_dead_families(families)` 追加到 `dead_families.md`
5. 失败静默

**v2 并发策略**：多 run 并行结束时可能同时写 `run_summaries.md` / `dead_families.md`。采用**独立文件 + 定期 compaction** 两段式避免并发覆盖：
- `write_run_summary` 写独立文件 `viking://resources/alphaagent/run_summaries/<date>-<model>-<run_id>.md`（`mode="create"`，文件名含 run_id 保证唯一，无并发冲突）。
- **compaction 任务**（手动触发或每 N 次 run 后）：读 `run_summaries/` 下全部文件，按时间排序取近 30 条，合并写入 `run_summaries.md` 主文件，旧独立文件保留作归档（不删除，避免丢数据）。
- `dead_families.md` / `strategy_lessons.md` 的 `update_*` 方法用 read-modify-write（先 read 主文件 → 追加新条目 → write 回去），并发冲突时后写者覆盖先写者（教训条目幂等，重复追加同一族信息无实质损失，可接受最终一致性）。

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
        """run 启动时检索跨 session 策略记忆，返回注入用文本块。失败返回空串。
        数字部分实时查 SQLite 记忆库生成（不存 OpenViking 快照）。"""
    
    def write_run_summary(self, run_id, model_name, run_metrics, new_dead_families) -> bool:
        """run 结束时写回摘要到独立文件 run_summaries/<date>-<model>-<run_id>.md。
        失败静默。"""
    
    def update_dead_families(self, families: list[str]) -> bool:
        """read-modify-write 追加新饱和族到 dead_families.md。失败静默。"""
```

**设计原则**：
- 所有方法失败静默（OpenViking 不可用时挖掘照常跑）
- 懒初始化（首次调用才建 client，避免无 OpenViking 环境启动报错）
- 硬编码 SCOPE，不暴露 target_uri 参数（防误用泄漏 scope）
- **不暴露原始 `ov_client`**，只暴露业务方法

### 3.6 配置开关

`research_spec.py` 的 `memory_policy` 新增：

```python
"enable_ov_long_term_memory": True,   # OpenViking 长期记忆接入（False = 完全关闭，走纯 SQLite）
"ov_endpoint": "http://127.0.0.1:1933",  # OpenViking HTTP 端点
"ov_inject_max_chars": 2400,          # v2：注入块预算，对齐 SQLite max_inject_chars（原 1200 偏小）
```

`technical.json` / `fundamental.json` 可覆盖（注意 json 被 gitignore，覆盖只在本地生效，不进版本控制；默认值在 `research_spec.py` 里走 git 分支管理）。默认 `True` 但失败静默，无 OpenViking 环境自动降级到纯 SQLite。

---

## 4. L3：重复治理代码修复

### 4.1 修 `current_run_id` 没传的 bug

**问题**：`_dispatch.py:374-378` 调 `advisory_for` 没传 `current_run_id`，导致 `curr_run_passed` 豁免永远不生效。

**修复**：
1. `FactorEvalTools.__init__` 加 `run_id: str | None = None` 参数
2. `agentscope_run.py:373` 创建 `FactorEvalTools` 时传 `run_id=log_dir.name`
3. `_dispatch.py:374` 调 `advisory_for` 时传 `current_run_id=self.run_id`

**v2 范围修正**：CLI 路径 `agent/run.py:87` 的 `FactorEvalTools` 调用**未接 `memory_store`**（line 87-94 只有 service/session/submit/focus/cognition/operator，无 `memory_store=`），`_memory_gate` 第一行 `if self.memory_store is None: return None` 直接短路，无需传 `run_id`。**只改 Web 路径 `agentscope_run.py:373`**。

**影响**：当前 run 内已过线的 promising 因子，即使同结构指纹历史有失败，也不会被硬拦（`exempt_from_block=True`）。但受 §2.5 时序窗口限制，豁免仅在 promising 写入 `memory_entries` 后生效。

### 4.2 ~~D2：`recommend_edits` 加族级饱和度门控~~

**v2 删除**：D2 已修复。`retrieval.py:650-652` 已有 `saturation > 0.4` 门控，`:654` 有 D3 零产出族门控，回退路径 `:677-687` 同口径（`saturation_score <= 0.4` + `n_promising > 0` + 排除 `zero_yield_fams`）。本节无需实施。

### 4.3 （可选）`duplicate_prior_result` 冷却期

**问题**：同结构指纹历史有正向 verdict，仍重复评估（123 次）。当前 `duplicate_prior_result` 永不拦截。

**v2 调整**：L1 已增补轻量提醒（§2.1 改动 C，同 run 内已评估过 → 提示 submit 或变异，不硬拦）。冷却期硬拦方案暂缓——可能误杀合法迭代（同结构不同参数），等 L1/L2 落地后观察数据再决定是否升级为硬拦。

---

## 5. 实施计划

### 5.1 分批落地

| 批次 | 内容 | 改动 | 验收 |
|------|------|------|------|
| **批次 1**（立即） | L1 硬拦 + prior_result 提醒 | `research_spec.py` 默认值 + env 开关 + `advisory.py` message | 下次 run dup_dead_end 拦截率 0→33%，总评估数降 33% |
| **批次 2**（核心） | L2 OpenViking 接入 | 新增 `ov_store.py` + `agentscope_run.py` 接入点 + `research_spec.py` 配置 + `requirements-alphaagent.txt` pin | run 启动注入跨 session 教训块，run 结束 `viking://resources/alphaagent/run_summaries/` 有新文件 |
| **批次 3**（深度） | L3 修 `current_run_id` bug | `FactorEvalTools` 加 run_id + `agentscope_run.py` 传参 + `_dispatch.py` 传 `current_run_id` | 边缘 bug 消除，`curr_run_passed` 豁免生效 |

### 5.2 分支纪律

- 批次 1：`research_spec.py` / `advisory.py` 代码改动，走 `fix/hard-block-duplicates` 分支（json 不进 git，不涉及）。
- 批次 2：新代码 + requirements，走 `feat/alphaagent-ov-memory` 分支。
- 批次 3：热路径改动，走 `fix/dedup-current-run-id` 分支。

### 5.3 测试

- 批次 1：
  - 跑一次 test run，确认 `memory.advisory_block` 日志出现、评估数下降。
  - **具体测试用例**：从历史 320 次 `dup_dead_end` 里挑一个表达式（如某 vwap 死路指纹），硬编码进测试，确认硬拦触发 + LLM 收到 `MemoryAdvisoryBlock` result。
  - env 回滚测试：设 `ALPHA_MEMORY_HARD_BLOCK_DUPLICATES=0` 确认降级回提醒。
- 批次 2：
  - 单元测试 `ov_store.py`（mock SyncHTTPClient，验证 scope 硬编码 + 失败静默）。
  - **v2 隔离断言测试**：对 `ov_store.py` 所有出口（`retrieve_lessons` / `write_run_summary` / `update_dead_families`）断言"底层 OpenViking 调用的 target_uri 都以 `viking://resources/alphaagent/` 开头"，防回归。
  - 集成测试：启动 run，确认 system prompt 含"跨 session 策略记忆"块，run 结束后 OpenViking 有新文件。
  - 隔离测试：确认 AlphaAgent search 不返回 `viking://user/default/memories/`。
- 批次 3：
  - 跑 test run，确认 `curr_run_passed` 豁免生效。
  - **测试步骤**：先写一条同 run 的正向 `memory_entries`（`last_run_id = 当前 run_id`，verdict=promising），再触发同指纹评估，确认 `exempt_from_block=True` 不硬拦。

---

## 6. 风险与取舍

### 6.1 L1 硬拦截风险

- **误杀风险**：边缘窗口（§2.5）在"历史纯失败 + 当前 run 刚过线 + 还没写库"时误杀。概率低，L3 修后收窄到"写入前窗口"。L1 拦截 message 含"若本 run 已测出信号请直接 submit"引导缓解。
- **LLM 行为风险**：LLM 收到拦截后可能"学会"避开整族（过度泛化）。缓解：拦截 message 含具体死路原因，引导"换方向"而非"放弃族"。
- **回滚成本**：env 开关紧急回滚（无需改代码）或改 `research_spec.py` 默认值持久回滚（走 git 分支）。

### 6.2 L2 OpenViking 风险

- **依赖风险**：OpenViking 挂了影响挖掘。缓解：所有调用失败静默，降级到纯 SQLite。
- **依赖污染**：`openviking-sdk` 装时升级 openai。v2 已要求 pin `openai==3.3.1` 进 `requirements-alphaagent.txt`。
- **隔离风险**：代码层硬编码 scope，无 ACL。缓解：`ov_store.py` 不暴露 `ov_client` + 隔离断言测试防回归。
- **性能风险**：语义检索百毫秒级，run 启动一次 + run 结束一次，不影响热路径。
- **v2 并发风险**：多 run 并行写 `run_summaries/`。缓解：独立文件 + run_id 唯一命名 + 定期 compaction，`dead_families.md` 用 read-modify-write 接受最终一致性。

### 6.3 L3 代码修复风险

- **`current_run_id` 修复**：改 `FactorEvalTools` 签名，向后兼容（默认 None，老调用方不受影响）。CLI 路径无需改（未接 memory_store）。

---

## 7. 验收指标

| 指标 | 现状 | 目标 |
|------|------|------|
| 重复评估占比 | 46%（443/963） | <20%（v2 放宽：L1 砍 33% 死路，剩 13% prior_result 仅提醒不拦，理论下限 ≈13%+噪声） |
| `dup_dead_end` 拦截率 | 0%（全提醒） | 100%（硬拦） |
| vwap 族评估占比 | 34% | **v2：实测确认**（D2 已修复，可能已达标；若未达标需另查根因） |
| `recommend_edits` 推荐族饱和度 >0.4 占比 | ~高 | **v2：实测确认**（D2 已修复，可能已为 0%） |
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

## 附录 C：v2 评审事实核对记录

| spec v1 断言 | 核对结果 |
|---|---|
| `hard_block_duplicates` 默认 `False`（`research_spec.py:194`） | ✅ 属实 |
| `_dispatch.py:374-378` 调 `advisory_for` 没传 `current_run_id` | ✅ 属实，bug 真实存在 |
| `FactorEvalTools.__init__` 无 `run_id` 参数 | ✅ 属实 |
| 4 个调用点硬拦（`_dispatch.py:613/699/733/807`） | ✅ 属实 |
| `strategy_tracks.py:85` 有"命中后必须换方向"引导 | ✅ 属实 |
| `_result_tool_chunk` 包装拦截 result 返回 LLM | ✅ 属实 |
| `research_memory.db` 31827 entries | ✅ 属实（实测一致） |
| D2：`recommend_edits` 不查族级饱和度 | ❌ **已修复**（`retrieval.py:650-654` + 回退路径 `:677-687`） |
| OpenViking 接入代码不存在 | ✅ 属实（L2 是全新代码） |
| `artifacts/alphaagent/research_specs/*.json` 可走 git 分支 | ❌ **被 gitignore**，不进版本控制（`git check-ignore` 确认） |
| CLI 路径 `agent/run.py:87` 需传 `run_id` | ❌ **未接 memory_store**，`_memory_gate` 直接短路，无需传 |
