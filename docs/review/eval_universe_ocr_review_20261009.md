# OCR Review: feat/eval-universe（8371b61）

**分支**: `feat/eval-universe`（1 commit ahead of main，merge-base `06db552` = main HEAD）
**提交**: `8371b61` feat(eval): 股票池（universe）评估维度——规则池市值 TopN/名次段 + 指数池 idx_szcomp/idx_chinext（10 files, +399/−26）

**方法**: OCR（open-code-review）commit 模式审查 + 人工逐条证据核验 + 补充自查两条线（submit/交付链路的 universe 感知、FactorValueCache 跨池复用正确性）。

**结论**: **merge_after_fixes** —— 1 个 REAL defect（D1，一行修复），修后可合并。

---

## REAL Defects

### D1.（OCR 发现，已核实）dry_run_delivery 的相关墙在全市场池上计算，并顶替池会话

**位置**: `alphaagent/mcp_server/tools.py:591` → `:597` → `:561`

**证据链**:

1. `dry_run_delivery` :591 用指定池开会话：
   ```python
   svc, sid = ctx.session(mode, fundamentals=fundamentals, universe=universe)
   ```
2. :597 调相似度**未透传 universe**（默认 `"all"`）：
   ```python
   sim = library_similarity(ctx, expr, mode).get("similarity")
   ```
3. `library_similarity` :561 内部 `ctx.session(mode, universe=universe)`；`ServerContext.session()` 的会话缓存键已含 universe（本提交新增），`"all"` ≠ 池名 → `_release_locked()` 并**以全市场面板重建会话**。

**影响**:
- `stage_one_correlation` / `stage_two` 判定用的 `sim` 在**全市场**计算，与同一响应里**池内**的 train/val 指标口径相悖——直接违背 `library_similarity` 自己的 docstring（tools.py:551「相关性也应在同一股票池内比较（池内比池内）」）；
- 池会话被顶替丢弃，下一次池内调用需重建面板（分钟级浪费）。

**修复**（一行）:
```python
sim = library_similarity(ctx, expr, mode, universe=universe).get("similarity")
```

### D2.（OCR 发现，已核实）响应回显原始 universe 入参，未经规范化

**位置**: `alphaagent/mcp_server/tools.py` eval_batch（:519-520 附近 `"universe": universe`）、eval_val（:539 附近）；另 dry_run_delivery 响应（:606-618）**完全没有** universe 字段。

**证据**: `ServerContext.session()` 内部经 `parse_universe`（strip + lower）规范化后才作会话键/元数据；工具函数返回的 `universe` 却是调用方原始入参。传 `"TOP300CAP "`（带空格/大写）时，响应里的池名与实际评估所用规范化名对不上，按名称记账/复现会错位。

**修复**: 三个工具函数入口统一 `universe = parse_universe(universe)["name"]`（早校验 + 规范化一并完成），响应用规范化名回显；dry_run_delivery 响应补 `"universe": universe`。

### D3.（自查）submit/交付链路不感知 universe——计划内 P2 延后，但本次提交后口径漂移风险升级

**位置**: `alphaagent/mcp_server/tools.py:641`（submit_factor）、`:673`（memory_record）

**证据**:
- `submit_factor` 开会话 `ctx.session(mode, fundamentals=fundamentals)` —— 无 universe → 整条交付链路（stage_one → 盲测 → stage_two → engine_gate）**永远在全市场面板**上重评；
- `evaluated_universe` 全仓仅存在于两处 docstring（`eval/context.py:57`、`eval/universe.py:10`），代码零落地——「候选元数据按池记账」的承诺未实现；
- 设计记录（2026-10-08 股票池 universe 参数设计）明确 P2 =「evaluated_universe 进候选/提交元数据 + engine_gate 按同池回测」，属**计划内延后**，非意外遗漏。

**但 D1 修复后出现新组合**: dry_run 全池内 vs submit 全市场——同一因子两套门可能给出相反结论（实测池子敏感性：同一批因子全市场 vs Top300 IC 可反号 +0.0092 → −0.0132），而当前工具描述未作任何警示。

**最小修复（二选一，不阻塞合并）**:
1. `submit_factor` 增加 `universe` 参数透传 session（交付门随池跑）；元数据记账仍留 P2；
2. 暂不透传，但在 catalog 的 dry_run/submit 描述与 submit 返回 `note` 写明「交付门固定全市场口径，池内预检仅供研究参考」。

### D4.（自查，test）测试恒真断言，零保护力

**位置**: `tests/test_eval_universe.py:64`

```python
assert "szcomp" not in available_universes() or "szcomp" in available_universes()
```

`A or ¬A` 恒真——断言什么都没测。**修复**: 改为 `assert "szcomp" in available_universes() and "chinext" in available_universes()`。

---

## 观察项（非 defect，不阻塞）

### O1. 规则池对 `tot_cap=NaN` 行为未文档化

`universe.py` `apply_universe`: `rank(ascending=False, method="first")` 对 NaN 产 NaN → keep False → **静默剔除**；指数池 `fillna(0)` 同理。行为合理（无法排名/非成分即不入池），但 docstring 与测试均未说明。建议 docstring 补一句。

### O2. FactorValueCache 跨池复用——已排查，安全 ✅

担忧：CS_ZSCORE/CS_RANK 等截面算子在池内与全市场结果不同，全局缓存若按表达式复用会串池。核实：缓存键 = `_cache_key(expr, panel_fp)`，`_panel_fingerprint`（`alphaagent/factor/cache.py:85-122`）含**行数 + datetime 范围 + 内容抽样哈希**——掩码后面板行数必不同 → 指纹必不同 → 键必不同（cache.py:71 注释「内容寻址 ⇒ 跨会话安全」成立）。`engine_preview` 亦用 `session.panel`（掩码后）跑（`eval/service.py:358-375`），池内口径一致。

### O3. 测试环境说明

worktree 内 `tests/test_alphaagent_mcp_server.py` 20 测 = 17 过 / 2 skipped / 2 failed；2 个失败（`test_list_fields_and_describe_operator`、`test_list_runs_and_run_summary`）读的是不进 git 的 `artifacts/panel/cache` 与 `logs/factor_mining` 产物，worktree 不存在所致，两个函数本 diff 未触及——**环境性失败，非本分支引入**。

---

## OCR 原始输出（2 条，均已核验为真，无误报）

> **[bug · medium] tools.py:591-592** — dry_run_delivery 在创建了指定 universe 的会话后，:597 仍以 `library_similarity(ctx, expr, mode)` 调用（未透传 universe，默认 "all"）。由于新的会话缓存键含 universe，当所评池子 != all 时，`ctx.session(mode, universe="all")` 会 `_release_locked()` 并以全市场面板重建会话：result 一侧（stage_one_correlation / stage_two 相关墙）实际在全市场池计算，与本函数 train/val 的池子不一致，违背 library_similarity 文档中"池内比池内"的语义；同时会替换掉当前按池建立的会话。
>
> **[maintainability · low] tools.py:520** — 返回的 `universe` 用的是调用方原始入参，而 session() 内部经 parse_universe 规范化后才作为会话键/元数据。若客户端传非规范写法（如 "TOP300CAP "），响应里的 universe 与实际评估所用池名不一致，按名称记账/复现可能对不上。同问题也存在于 eval_val。

---

## 修复清单

| # | 严重度 | 项 | 修复 | 成本 |
|---|---|---|---|---|
| D1 | 中 | dry_run 相关墙全市场计算 + 会话顶替 | :597 透传 `universe=universe` | 一行 |
| D2 | 低 | 响应 universe 未规范化 / dry_run 缺字段 | 入口 `parse_universe` 规范化 + dry_run 响应补字段 | ~6 行 |
| D3 | 中（计划内延后，风险升级） | submit/交付链路不感知 universe | 透传参数 或 明示口径差异（二选一） | 小 |
| D4 | 低 | 测试恒真断言 | 改为双向 `in` 断言 | 一行 |
| O1 | 低 | NaN tot_cap 剔除未文档化 | docstring 补句 | 一句 |

**合并建议**: D1 必修（一行）后合并；D2/D4 顺手；D3 择一处理或明确记入 P2 待办。
