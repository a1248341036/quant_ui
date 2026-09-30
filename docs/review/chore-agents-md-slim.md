# Review: commit 6d42da9 — AGENTS.md 瘦身

- **分支**: `chore/agents-md-slim`
- **提交**: `6d42da9` — "docs: 瘦身 AGENTS.md，详细架构迁移至 docs/alphaagent_architecture.md"
- **类型**: doc-only（无代码改动）
- **Diffstat**: `AGENTS.md` 471→91 行（452 changed，主体为删除）；`docs/alphaagent_architecture.md` +341 新增。合计 +377/-416。
- **评审日期**: 2026-09-28

## 结论

**merge** — 瘦身干净，强制指令零丢失，迁移内容完整，交叉引用在 commit 内自洽。无 REAL defect。

## 评审范围与方法

按 `branch_review_workflow` 纪律：仅查 REAL defect（doc-only 提交的口径 = 静默删除强制/mandatory 内容、断链的交叉引用、事实错误、被弱化的安全指令）；cosmetic condensation 不算缺陷。逐节对比三份文件：

1. 原 `AGENTS.md`（commit 前，471 行，经 `git show <parent>:AGENTS.md` 取得）
2. 瘦身后 `AGENTS.md`（commit 内，91 行）
3. 新增 `docs/alphaagent_architecture.md`（commit 内，341 行）

## 强制指令保留核对（核心检查）

| 强制段 | 原始位置 | 瘦身后位置 | 核对结果 |
|---|---|---|---|
| 多 Agent 共享工作区纪律 — 引言段 | orig L14 | slim L8 | 逐字一致 ✅ |
| 规则 1 前端构建只用独立 worktree | orig L16 | slim L10 | 逐字一致 ✅ |
| 规则 2 禁止跨分支覆盖工作区文件 | orig L17 | slim L11 | 逐字一致 ✅ |
| 规则 3 动共享工作区前先 git status 快照 | orig L18 | slim L12 | 逐字一致 ✅ |
| 代码搜索必须先带筛选（强制）— 全段 | orig（开发效率节） | slim L14-22 | 逐字一致 ✅ |
| Git 分支纪律（强制）— 全段 | orig（Git 分支纪律节） | slim L24-30 | 逐字一致 ✅ |
| 端口约定 17891/8787、8000 废弃 | orig（常用命令后注） | slim L81 | 保留 ✅ |

**3 条共享工作区规则 + 搜索筛选 + Git 分支纪律** 三组强制指令全部逐字保留，无弱化。

## 唯一删除的非规则内容

orig L20 的「背景（2026-09-11）」事故叙述段被删除：

> 背景（2026-09-11）：为重建含主干功能的前端 bundle，曾在共享工作区直接执行 `git checkout feat/alphaagent-metrics -- MetricsPanel.vue alphaagent.css`，覆盖了并行 agent 的未提交改动……

**判定：可接受，非缺陷**。该段是规则 1、2 的**背景说明**，不是规则本身；3 条规则已完整保留并自含可执行措辞（"禁止把共享工作区的构建产物直接当作线上 bundle"、"禁止在共享工作区执行 `git checkout <branch> -- <path>`"）。删除历史叙述不削弱强制力。如团队希望保留事故记忆，建议迁入 arch doc 或 commit message，但不构成 review 阻断项。

## 迁移内容完整性核对

arch doc 章节头与原始 AGENTS.md 对应：

| 原始章节 | 原始行 | arch doc 对应 | 核对 |
|---|---|---|---|
| 定位 | L24-26 | arch L6-8 | 一致 ✅ |
| steps.log 分步日志（原嵌在目录结构注） | L82-86 | arch L10-18（独立成节） | 一致 ✅ |
| 核心流程 1-9 + 3b + 6b | L94-285 | arch L20-（### 1~9 + 3b + 6b） | 章节齐全 ✅ |
| 评估档位自动推断 + 调仓频率 | L286 | arch（评估档位节） | 一致 ✅ |
| REST API 全表 | L302 | arch（REST API 节） | 一致 ✅ |
| 关键设计决策 | L339 | arch（关键设计决策节） | 一致 ✅ |
| 聚宽策略兼容层（JQ Shim） | L364 | arch（JQ Shim 节） | 一致 ✅ |
| 数据接入方式（datalake 插件体系） | L387 | arch（数据接入方式节，L335-341） | 一致 ✅ |

body 抽查（定位段、steps.log 段、JQ Shim 数据接入规则 1-4 末段）与原始逐字一致，无静默删改。

## 交叉引用核对

slim `AGENTS.md` 引用的路径，在 commit `6d42da9` 内或工作区是否存在：

| 引用 | slim 位置 | commit 内 | 工作区（当前 HEAD） | 判定 |
|---|---|---|---|---|
| `docs/alphaagent_architecture.md` | L3, L85 | ✅ `git cat-file -e 6d42da9:docs/alphaagent_architecture.md` 命中 | ❌（当前在 `fix/margin-dividend-plugin-load`，未 merge） | merge 后自洽 ✅ |
| `scripts/start_backend_with_sync.ps1` | L60 | — | ✅ 存在 | ✅ |
| `scripts/rescreen_candidate_pool.py` | L72 | — | ✅ 存在 | ✅ |
| `scripts/blind_test_factors.py` | L75 | — | ✅ 存在 | ✅ |
| `scripts/promote_candidates.py` | L69 | — | ✅ 存在 | ✅ |
| `scripts/run_alphaagent.py` | L66 | — | ✅ 存在 | ✅ |
| `requirements-alphaagent.txt` | L91 | — | ✅ 存在 | ✅ |
| `docs/specs/alphaagent_mine_precheck_spec.md` | L86 | — | ✅ 存在 | ✅ |
| `delivery_criteria.DeliveryCriteria`（代码符号指针） | L87 | — | 代码符号，非路径 | ✅（保留为"门槛唯一真源"指针，与原 AGENTS.md §5 口径一致） |

**注**：`docs/alphaagent_architecture.md` 在当前工作区不存在是**预期**——该 commit 在 `chore/agents-md-slim` 分支，尚未 merge 到当前检出的 `fix/margin-dividend-plugin-load`。merge 后引用即生效，非断链。

## AIGC frontmatter 删除

orig L1-10 的 YAML frontmatter（`AIGC:` 块）被删除。这是 commit message 声明的"去除 AIGC frontmatter 噪声"，符合瘦身意图，无副作用。

## 风险提示（非阻断）

1. **背景段删除**：如上所述，2026-09-11 事故叙述被删。规则本身保留，但事故记忆不再随 AGENTS.md 流转。若团队重视该教训的可追溯性，建议补迁入 arch doc 顶部或保留在 git 历史（commit message 未含此段）。**不阻断 merge**。
2. **arch doc 单点依赖**：slim AGENTS.md L85 把"改 AlphaAgent 模块前先读它"指向 arch doc。arch doc 与 AGENTS.md 在同一 commit、同一分支，merge 后二者同进同退，无分裂风险。但未来若有人只改 AGENTS.md 不同步 arch doc，可能产生漂移——这是文档治理问题，非本 commit 缺陷。

## 最终判定

**merge**。瘦身达成声明目标（去 AIGC frontmatter + 迁移详细架构至 arch doc + 保留指令与速查），强制指令零丢失，迁移内容完整，交叉引用在 commit 内自洽。可独立 merge。
