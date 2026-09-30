# Review: `fix/decile-collapse-root-fix`（对照 main `165892b`）

- 审查对象：`fix/decile-collapse-root-fix` @ `1d8eeeb`（2026-09-27，"根治分箱塌缩：常数簇标定修正 + SOFT_GATE 连续门控 + 探索期短路"）
- 审查基线：`main` @ `165892b`（本会话合入 R1/R2 + 三通道开关后的 HEAD）
- 方法：`git diff <branch> main`（双向整树 name-status + 逐文件 ± 行计数）→ 只读，未改文件、未跑测试
- **结论：`superseded`（内容已全部在 main，且 main 严格更新）**

```
is_subset_of: main
can_merge_independently: false     # 无内容可合；合并只会引入重复文档 + 回退 prompt 基线
conflicts_with: tests/fixtures/system_prompt_{full,long_label,population,price_only}.txt
```

## 一、决定性证据：分支的实作文件与 main **逐字节相同**

`git diff 1d8eeeb main` 的 name-status 里**完全没有出现**下列文件（既非 A 也非 M 也非 D ⇒ 两棵树内容一致）：

| 文件 | 分支 vs main |
|---|---|
| `alphaagent/factor/mining/tools/_precheck.py` | 0 增 0 删（含 `_MAX_CONSTANT_CLUSTER = 0.30` / `_BLOCK_ENV` / `constant_cluster_share` / `blocked`） |
| `alphaagent/factor/mining/eval/service.py` | 0 / 0（含 `_COLLAPSE_GATE_LIMIT = 0.30` + `_collapse_prescreen_enabled()` + lite 后 collapse 短路） |
| `alphaagent/factor/mining/interactions.py` | 0 / 0（`"soft_gate": {"SOFT_GATE"}` + `softgate/soft_gated` 别名） |
| `alphaagent/dsl/core/operators.py` | 0 / 0（`def SOFT_GATE(...)`） |
| `alphaagent/factor/mining/agent/agentscope_tools.py` | 0 / 0（precheck 硬拦 + 逃生阀注释） |
| `alphaagent/factor/mining/diagnostics/__init__.py` | 0 / 0 |
| `alphaagent/factor/mining/prompt/modules/behavior_rules.py` | 0 / 0（rule 14 同一版本） |
| `tests/test_decile_collapse_guard.py` | 0 / 0（main 侧 19 个用例） |
| `tests/test_mining_skills.py` | 0 / 0 |
| `docs/specs/alphaagent_decile_collapse_root_fix_spec.md` | 0 / 0（两棵树同路径同内容） |

即：**这条分支的"代码 + 测试 + spec"三件套，一行不差地已经在 main 里。**

## 二、这些内容是怎么进 main 的（provenance）

```
git log -S 在 main 上定位：
  _MAX_CONSTANT_CLUSTER → afdac00  feat(mining): 落地分箱塌缩根治与按轮动态课题 RAG 检索深度闭环
  def SOFT_GATE         → afdac00
  _COLLAPSE_GATE_LIMIT  → 05416c7  test(factor): 补充分箱塌缩根治与连续加权算子测试用例
```

同一批改动在 2026-09-27 走了两条路径：`afdac00`（随后经 `feat/report-r4-r3-hybrid` 线）合入 main，
`1d8eeeb` 留在本地分支未合 —— 属**重复落地**，不是"待合的新功能"。

## 三、分支唯一独有的东西（都不该合）

| 项 | 内容 | 判断 |
|---|---|---|
| `docs/reports/alphaagent_decile_collapse_root_fix_spec.md` | 与 main 的 `docs/specs/…` 同内容的**重复副本**（整树 diff 里唯一的 `D` 项） | 不合；main 的 `docs/specs/` 是正式位置 |
| 4 份 prompt 黄金基线 | 分支的版本**早于** main 的后续演进（本仓库 prompt 已多轮变更） | **合了会回退**：实测 `git diff 1d8eeeb main -- tests/fixtures/system_prompt_full.txt` = main 独有 60 行 / 分支独有 1 行，`price_only` = 22 / 1 |

## 四、REAL defects

**无。** 分支自身改动（常数簇标定、`SOFT_GATE`、探索期短路、逃生阀）均已随 `afdac00`/`05416c7` 进入 main，
且 main 侧实现比该分支后续还多演进了一步（`_precheck.py:198` 的"中等风险带" `elif mid > _MAX_CONSTANT_CLUSTER * 0.8 → blocked=False`，分支与 main 同为该版本）。因此不存在"分支修好了、main 还坏着"的缺陷。

带来的唯一实际风险是**流程风险**：这条分支停在 33 个提交之前，若有人按"名称像 bugfix"直接 merge，会把 4 份 prompt 基线回退且引入重复文档。

## 五、建议

1. **删除** `fix/decile-collapse-root-fix`（内容已在 main；先卸掉它的 worktree `D:/Quant/quant_ui/.worktrees/collapse-root-fix`，否则 `git branch -d` 会被拒）。
2. 收尾动作制度化（避免再次出现）：`git merge` 后立刻 `git branch -d`，或定期跑一次"`git diff --name-status <branch> main` 为空 ⇒ 可删"的体检；本次那 ~120 个零独有提交分支即按此清理。
3. 若确实想保留这份 spec 的 `docs/reports/` 位置，只需单文件摘录（`git show 1d8eeeb:docs/reports/...` 导出），不必合分支。
