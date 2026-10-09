# quant_ui 项目指南

> 2026-09-28 瘦身：详细架构/流程/设计决策已迁移至 `docs/alphaagent_architecture.md`，
> 本文件只保留 agent 必须遵守的指令与高频速查。**改 AlphaAgent 模块前先读那份文档。**

## 多 Agent 共享工作区纪律（强制，永久生效）

本仓库常态是多 agent 共用同一工作区（D:\Quant\quant_ui），分支被并行 agent 频繁切换。分支纪律（新建分支→提交→merge）保护的是**已提交内容**，对工作区中他人**未提交**的改动零保护。以下三条禁止绕过：

1. **前端构建只用独立 worktree**：`npm run build` 的产物内容 = 当前检出分支。构建前 `git worktree add` 临时挂载包含目标提交的 worktree，在其中构建，再把 `static/dist` 拷回。禁止把共享工作区的构建产物直接当作线上 bundle——共享工作区检出哪个分支取决于"碰巧在哪个 agent 手里"，会出现"功能已提交已推送、页面却看不到"。
2. **禁止跨分支覆盖工作区文件**：`git checkout <branch> -- <path>` 会无警告、无条件覆盖工作区文件，禁止在共享工作区执行。需要其他分支的文件时，用 `git show <branch>:<path>` 导出到临时位置，或走临时 worktree。
3. **动共享工作区前先 `git status` 快照**：记录他人未提交文件清单；提交只 stage 自己的文件；恢复/清理/覆盖类操作不得触碰他人未提交文件。

## 代码搜索必须先带筛选（强制）

本仓库包含大量**大型数据产物**：`artifacts/`（JSON/parquet/npy 因子库、面板缓存、报告）、`logs/`（JSONL 轨迹）、`.venv/`、`static/node_modules/`、`CNEquity/`、`sentiment-mvp/` 等。**全量搜索会扫描这些目录，动辄数百毫秒甚至更久，且污染结果**。这是用户明确要求的硬性规则，任何搜索/匹配都必须先过滤。

- **搜索（grep/glob）必须带 `include` 参数**（如 `*.py` / `*.vue` / `*.ts` / `*.js` / `*.md`），只搜源码文件；禁止不带 include 的全仓 grep。
- **能限定 `path` 时就限定到源码目录**：`alphaagent/`、`backend/`、`core/`、`scripts/`、`static/src/`、`tests/`、`docs/`。
- 禁止无筛选扫描：`artifacts/`、`logs/`、`.venv/`、`node_modules/`、`CNEquity/`（除非明确要查数据产物/第三方库本身）。
- 查 DSL 算子/因子评估逻辑 → `alphaagent/dsl/`、`alphaagent/factor/`；查回测引擎 → `core/`；查前端 → `static/src/`；查 API → `backend/`。
- 文件读取（glob/read）同样先确认目标是否在源码目录；数据产物目录不得直接列目录全文。

## 统一配置中心纪律（强制，永久生效）

**所有可调阈值 / 开关 / 门槛数值必须收口到统一配置中心，禁止在业务逻辑里硬编码魔法数。**

- **唯一真源位置**：
  - `alphaagent/factor/mining/research_spec.py` 的 `DEFAULT_RESEARCH_SPEC`
    （研究/研报模式的门槛：`evaluation_policy` / `delivery_policy` / `report_policy` …）；
  - `alphaagent/factor/mining/memory/constants.py`（记忆层阈值/枚举，如 verdict 与 near-miss 线）；
  - `alphaagent/factor/mining/delivery/delivery_criteria.py`（候选/正式库交付门槛）。
  判定代码只允许 `.get(key, <默认>)` 回落，默认值必须与配置中心一致。
- **新增配置三件套缺一不可**：
  1. `DEFAULT_RESEARCH_SPEC` 里的默认值 **+ 说明性注释**：含义、单位、为什么取这个值、
     调大/调小的后果、影响的模式范围（研报模式专属的必须写明"仅研报模式"）；
  2. 校验函数（`build_run_research_spec` 内）里的类型与**范围约束**（`_bounded_number` /
     `_require_bool`），防止越界值静默生效；
  3. 判定处的 `.get(key, 默认)` 回落 + 单测覆盖（默认值、覆盖生效、越界被拒）。
- **判定侧读取路径**：优先随 **run gate** 传递（`agentscope_run._gate_state` →
  `set_run_gate` → `_dispatch._effective_report_policy`）。**禁止依赖工具对象上未赋值的属性**
  ——`self.report_policy` 全仓从未被赋值（恒为 `None`），2026-10-01 因此导致研报复现门槛
  被静默跳过（0 条 `report_fidelity_check` 日志），并让 `reproduce_min_*` 长期吃硬编码默认。
- **改阈值 = 改口径**：必须同步更新相关单测与 `docs/`，并在提交信息里写明"旧值 → 新值 + 理由"。

## Git 分支纪律（强制）

- **任何改动（代码/文档/配置/脚本）必须先从 `main` 新建分支再动手**（`feat/*`、`fix/*`、`chore/*`），**禁止直接在 `main` 上修改或提交**。
- **文档与开发分支放在一起，不单独建分支**：spec / review / 方案设计等文档直接提交在对应特性的开发分支上，严禁单独拉 `docs/*` 分支，避免产生无代码的孤岛分支。
- 会话开始时若发现当前在 `main`，先建分支再改动；一串相关改动共用一个主题分支，分支名体现内容。
- **合并前 OCR review（强制，2026-10-09 起）**：任何功能/修复分支合并回 `main` 之前，必须先跑
  `ocr review --audience agent --from main --to <branch> --background "<功能背景>" --output <file>`
  （用 `open-code-review` skill），critical/high 发现必须修复，medium 择实锤处理，low 过滤误报；
  修复与单测随分支提交。review 结果（发现数/修复情况/误报）在向用户请求合并时一并汇报。
- 分支完成后交由用户决定合并时机；不主动 push / merge 到 main。
- 多会话并行工作时注意：切分支前确认工作区干净；他人/其他会话的在途未提交改动不得卷入自己的提交（`git add` 精确到文件，禁用盲目 `git add -A`）。

## 目录结构（速查）

```
alphaagent/
├── core/           # 共享类型、路径常量、配置加载
├── data/           # 数据层：panel 构建 + CNE 数据湖适配器（插件化）
│   └── adapters/   # cnequity.py 入口 / registry.py 注册中心 / plugins/ 数据源插件
├── dsl/            # 因子表达式 DSL
│   ├── core/       # parser.py 解析器 + operators.py 算子库 + accel.py Numba JIT + chip_daily.py 筹码
│   └── stock/      # 股票专用：增量计算、resample、缓存
├── factor/         # 因子管理层
│   ├── evaluation/ # 冻结 profile 评估引擎 + 插件 transforms/metrics/rules
│   ├── mining/     # LLM 挖掘主循环（loop/tools/prompts/config/session/service/submit/research_spec/research_memory）
│   └── zoo/        # FactorZoo：因子库存储 + 相似度索引
└── __init__.py

backend/            # FastAPI 服务层 + routers/alphaagent.py
scripts/            # run_alphaagent.py / promote_candidates.py / dedup_candidate_factors.py 等
static/src/views/AlphaAgent.vue  # 前端页面：研究 / 因子实验室 / 因子库
artifacts/alphaagent/            # factorzoo（candidate_main/production_main 统一大库）/ research_specs / research_memory.db
logs/factor_mining/ui/           # 每次 Web run 的 JSONL 轨迹 + run_meta.json + steps.log（按 run_id 分目录）
```

## 常用命令

```powershell
# 启动后端（推荐：一键启动 Quant UI 后端 17891 + CNE dashboard 8787）
# 桌面快捷方式「Quant UI 启动.lnk」即运行本脚本
powershell -NoProfile -ExecutionPolicy Bypass -File scripts\start_backend_with_sync.ps1

# 仅后端（若只需 API 不想起 CNE dashboard）
.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 17891

# CLI 直接启动挖掘（默认 cne:// panel + production_main 统一大库）
.venv\Scripts\python.exe scripts\run_alphaagent.py

# 候选池重放晋升（把候选池已有因子重新走一遍修复后的两阶段链路）
.venv\Scripts\python.exe scripts\promote_candidates.py

# 存量候选池按当前门槛重筛（缺省 dry-run，--apply 执行；门槛复用 DeliveryChecker，与提交路径零口径漂移）
.venv\Scripts\python.exe scripts\rescreen_candidate_pool.py --data-root . --apply

# 盲测段因子重测（默认 2026-01-01 起；盲测段是唯一诚实样本外，此命令是其正式用途）
.venv\Scripts\python.exe scripts\blind_test_factors.py

# 前端 dev
cd static && npm run dev
```

> **端口约定**：Quant UI 后端固定 **17891**；CNE dashboard **8787**。历史上 AGENTS.md 曾写 8000，已废弃 —— 一律以 `scripts/start_backend_with_sync.ps1` 为准。

## 整夜挖掘监控（`overnight_mining_monitor.v2.py`）

脚本在仓库外：`C:\Users\zhoubw\Desktop\quant\overnight_mining_monitor.v2.py`。

**后端一律无窗口启动**（用户 2026-10-04 定调）：该脚本自 2026-10-04 起**缺省**用
`CREATE_NO_WINDOW` 拉起 `scripts/start_backend_with_sync.ps1`，并把 launcher/uvicorn/CNE 输出
重定向到 `C:\Users\zhoubw\Desktop\quant\logs\backend_console_<YYYYMMDD>.log`。

- 显式开关：`--backend-hidden`（缺省，等价于不传）；需要可见窗口时用 `--backend-console`。
- **为什么**（2026-10-04 实测两次）：原先 `CREATE_NEW_CONSOLE` 会留一个独立控制台窗口，
  误关该窗口 = 后端被杀 → monitor 要空等约 5 分钟才发现并重拉；且**后端重启会把已 stop 的
  run 复活成 `running`**，害得下一段 monitor 白等（需再 stop 一次才清掉）。无窗口启动同时
  消除了"窗口被误关"这一整类事故。
- 标准调用：

```powershell
D:\Quant\quant_ui\.venv\Scripts\python.exe C:\Users\zhoubw\Desktop\quant\overnight_mining_monitor.v2.py `
  --repo D:\Quant\quant_ui --research-mode report --deadline 07:00 --max-turns 8 --backend-hidden
```

## 关键文档指针

- `docs/alphaagent_architecture.md` — 核心流程（启动链路/挖掘循环/三层约束/认知层升级/评估引擎/入库两阶段/候选库管理/因子中台/研究记忆）、评估档位推断、REST API 全表、关键设计决策、聚宽 JQ Shim 兼容层。**改这些模块前先读它。**
- `docs/specs/` — 各功能 spec（如 `alphaagent_mine_precheck_spec.md`、`alphaagent_prompt_contradictions_repair_spec_v2.md` 等）。
- 门槛数值唯一真源在代码：`delivery_criteria.DeliveryCriteria`（由 research_spec 注入），prompt 渲染（`to_prompt_text()`）同源，杜绝口径漂移。

## 依赖

见 `requirements-alphaagent.txt`：agentscope, openai, numba, jieba。