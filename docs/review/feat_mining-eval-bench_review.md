# 分支评审：feat/mining-eval-bench

**分支**: `feat/mining-eval-bench`
**提交**: `aab513d` + `234d6b0`（2 commits, +2804/-24, 16 files）
**merge-base**: `71a2c12`（main 当前 `dc82326`，新增研报解析提交，与本分支无冲突）
**评审日期**: 2026-09-25
**评审模式**: 静态代码审查（未运行测试，未修改文件）

## 概述

本分支实现 AlphaAgent 挖掘评测 Bench 模块，提供：
- 离线解析 run 目录（`run_*.jsonl` / `steps.log` / `run_meta.json` / `run_manifest.json`）→ A-F 六组扩展指标（`extended_metrics.py`, 731 行）
- 冻结配置管理与 config_hash（`config.py`）
- 基线快照、scorecard 对比判定（`baseline.py` / `compare.py`）
- 评测台账与逐 run 趋势追踪（`ledger.py` / `trend.py`）
- 评测执行器（`runner.py`，派发 `run_alphaagent` 子进程）
- CLI 统一入口 `scripts/bench.py` + 历史回填 `scripts/backfill_run_metrics.py`
- 后端 `metrics_overview` 增强（v4 聚合 + 时间排序 + 缓存读取）
- 前端 `MetricsPanel.vue` 增强（v4 聚合卡片 + 趋势图 + 扩展列）
- 测试 `tests/test_bench.py`（304 行，5 大模块覆盖）+ `test_run_scorecard.py` schema 升级

## 评审结论

**merge**

无阻断缺陷。逻辑正确，导入完整，契约一致，测试覆盖充分。发现 1 个 P1 + 4 个 P2 + 2 个 P3，均不阻断合并，建议后续迭代处理。

## 发现清单

### P1 — schema 版本与内容可能不一致

**文件**: `alphaagent/factor/mining/run_metrics.py:416-440`
**代码**:
```python
ext: dict[str, Any] = {}
try:
    from alphaagent.factor.mining.bench.extended_metrics import compute_extended_metrics
    ext = compute_extended_metrics(run_id, run_path, base_metrics=m)
except Exception:
    pass
# ...
scorecard: dict[str, Any] = {
    "run_id": run_id,
    "created_at": time_meta.get("created_at") or utc_now_iso(),
    "schema_version": 4,  # 无条件 v4
    "time_meta": time_meta,
    "headline": ext.get("headline") or {},
    # ... exploration/dynamics/quality/cost/process/integrity 全部 ext.get(...) or {}
}
```
**问题**: `compute_extended_metrics` 被包在 `try/except` 中，失败时 `ext={}`，所有 v4 字段为空 dict，但 `schema_version` 无条件设为 4。下游 `compare.py` 的 `_extract_metric` 从空 dict 取值返回 None，对比时该指标跳过——功能上不致命，但 schema 声明与实际内容不一致。
**建议**: 失败时降级 `schema_version` 为 3，或在 scorecard 中加 `ext_metrics_ok: bool` 标记。

### P2 — config_hash 遗漏影响执行行为的字段

**文件**: `alphaagent/factor/mining/bench/config.py:49-54`
**代码**:
```python
HASH_FIELDS = [
    "panel", "train_start", "train_end", "val_start", "val_end", "label_col",
    "model", "temperature", "reasoning_effort", "max_tokens", "max_turns",
    "max_tool_calls_per_round", "max_parallel_eval", "focus_facets",
    "research_spec_file", "user_message", "no_reviewer",
]
```
**问题**: `max_tool_workers`（默认 8，tool_calls 分发并发）和 `min_tool_call_rounds`（默认 3，挖掘轮数下限）影响执行行为但不进 config_hash。用户改这两项后，config_hash 不变，对比基线时配置"看起来一致"但并行度/轮数不同。
**建议**: 将 `max_tool_workers` 和 `min_tool_call_rounds` 加入 `HASH_FIELDS`。

### P2 — saturation_epsilon 死配置

**文件**: `alphaagent/factor/mining/bench/config.py:45` + `extended_metrics.py:543`
**代码**:
```python
# config.py
"saturation_epsilon": 0.001,

# extended_metrics.py:543 (硬编码，未读配置)
if (ic_next - ic_now) < 0.001 and (ic_next2 - ic_now) < 0.001 and (ic_final - ic_now) < 0.002:
```
**问题**: `saturation_epsilon` 存在配置中但 `extended_metrics.py` 硬编码 0.001/0.002，未读取此配置项。
**建议**: 要么在 `extended_metrics.py` 中读取 `saturation_epsilon`，要么从配置中移除。

### P2 — trend.py 全量即时重算性能隐患

**文件**: `alphaagent/factor/mining/bench/trend.py:30-60`
**代码**:
```python
for d in UI_ROOT.iterdir():
    # ...
    if not sc:
        try:
            from alphaagent.factor.mining.run_metrics import compute_run_metrics
            from alphaagent.factor.mining.bench.extended_metrics import compute_extended_metrics
            bm = compute_run_metrics(d.name, d)
            ext = compute_extended_metrics(d.name, d, base_metrics=bm)
            # ...
```
**问题**: 对无 `scorecard.json` 的目录全量即时重算 `compute_extended_metrics`（需解析 JSONL 轨迹）。`UI_ROOT` 下可能有数百个 run 目录，每次 `bench trend` 触发全量重算后再排序截取 `last` 条。仅 CLI 手动触发，非 Web 热路径，但用户体验差。
**建议**: 先按 `run_meta.json` 的 `created_at` 排序，只对 `last` 条目重算；或优先用 `backfill` 预生成 scorecard。

### P2 — max_parallel_eval=0 边界值跳过

**文件**: `alphaagent/factor/mining/bench/runner.py:121`
**代码**:
```python
if cfg.get("max_parallel_eval"):
    cmd.extend(["--max-parallel-eval", str(cfg["max_parallel_eval"])])
```
**问题**: `max_parallel_eval=0` 是 falsy，`--max-parallel-eval 0` 不会传给子进程。语义上 0 = 禁用并行（退回串行），但实际会使用子进程默认值（6）。
**建议**: 改为 `if cfg.get("max_parallel_eval") is not None:`，与 `temperature` 同口径。

### P3 — datetime.utcnow() 弃用

**文件**: `alphaagent/factor/mining/bench/ledger.py:68` + `baseline.py:45`
**问题**: `datetime.utcnow()` 在 Python 3.12 弃用（DeprecationWarning）。项目已有 `alphaagent.core.timeutil.utc_now_iso`。
**建议**: 统一用 `utc_now_iso()` 或 `datetime.now(timezone.utc)`。

### P3 — --detach 模式文档不完整

**文件**: `alphaagent/factor/mining/bench/runner.py:151-155` + `scripts/bench.py:226`
**问题**: `--detach` 模式启动子进程后立即返回，不生成 scorecard/台账/对比。用户需后续手动 `bench score <run_id>` + `bench diff <run_id>`。帮助文本"后台运行"未说明此点。
**建议**: 在 `--detach` 帮助文本中补充"后台完成后需手动 `bench score` + `bench diff`"。

## 已验证无问题的点

- `extended_metrics.py` 单次遍历 + `expr_cache` 设计稳健，p90/p95/p10 索引边界安全（`int(len*0.9)` 在 len=1/10/11 均不越界）
- `compare.py` 的 `overall` 判定 5 分支均赋字符串值，`runner.py:207` 的 `"INVALID" in ov` 不会 TypeError
- `config.py` 的 `set_config_value` 类型强转正确（bool/int/float/None 分支）
- `runner.py` 失败分支正确生成失败 scorecard + 记录台账（exit_code=2）
- `baseline.py` 归档旧基线逻辑正确（copy2 + history 目录）
- `ledger.py` 的 `total_tokens` / `summary` 字段契约与 `run_metrics.py:444/448` 一致
- `trend.py` 的 Δ 计算按 config_hash 分段，不跨配置段比较
- 后端 `metrics_overview` 的 `_dir_sort_key` 三级回落（run_meta → scorecard → mtime）比原字典序更准确
- 前端 `formattedRuns` 反转算 Δ 再反转回倒序，逻辑正确
- 测试覆盖 config/extended_metrics/compare(IMPROVED/REGRESSED/INVALID)/ledger/trend 五大模块
- 所有导入路径已验证：`structure_fingerprint` / `classify_family_ex` / `expression_ops` / `canonical_hash` / `FACET_DEFS` / `_iter_run_events` / `compute_run_metrics` / `generate_scorecard` / `atomic_write_text` / `utc_now_iso`
- `parents[4]` 路径计算正确（`bench/` → `mining/` → `factor/` → `alphaagent/` → 仓库根）
- 统一大库后 `candidate_registry_path("technical")` 实际指向 `candidate_main`，功能正确
