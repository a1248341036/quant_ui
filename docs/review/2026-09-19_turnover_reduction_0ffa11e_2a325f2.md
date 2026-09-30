# Review: 0ffa11e + 2a325f2 — 因子降换手方法体系（P0 + P1）

- **0ffa11e** `feat(mining): 因子降换手方法体系 P0-prereq + P0`（2026-09-19 21:07）
  - `TurnoverDiagnostic` 结构化归因输出（`worst_col_*` + `col_autocorrs` 进 `extra_fields`）
  - `behavior_rules.py` rule 2 整体重写（归因驱动强制流程 + ρ_f 量级锚 + 信号积分 + @1w + WMA 替 EMA）
  - `market_mechanisms.py` 追加 4d 节（机制-换手连接）
  - 黄金基线 4 份同步重生成
- **2a325f2** `feat(mining): 因子降换手 P1 — buffer zone + no-trade band`（2026-09-19 21:44）
  - `quantile_portfolio_metrics` 新增 `buffer_ratio` / `no_trade_band` 参数（默认 0.0 向后兼容）
  - `EngineGateCriteria` 默认 `buffer_ratio=0.5` + `no_trade_band=0.15`
  - `run_backtest` / `BacktestConfig` / `build_targets` / `simulate.py` 透传 buffer/band
  - `submit.py` 合并 `EngineGateCriteria` 默认值

两 commit 同属"因子降换手"体系，0ffa11e 是 P0（prompt + 诊断层），2a325f2 是 P1（metrics + engine_gate 层）。
分支从 `8850648` 分出，**不含 e2e10cb/058b37f 的 vwap 单位归一化修复**。

---

## 0ffa11e — 评级：🟢 P0 降换手方法体系，逻辑正确，有一处测试桩契约不一致

### ✅ 正确的部分

1. **`TurnoverDiagnostic` 结构化归因**（`alphaagent/factor/mining/diagnostics/__init__.py`）
   - `worst_col_name` / `worst_col_autocorr` / `worst_col_tier` / `col_autocorrs` 进 `extra_fields`
   - 去掉 `<0.6` 输出门槛，所有区间（high_freq / mid / slow）都输出归因
   - tier 分档（`<0.6` high_freq / `[0.6,0.85)` mid / `≥0.85` slow）与 prompt rule 2 归因驱动流程对齐
   - `session=None` 时四字段保持 `None` / `{}`，`test_turnover_attribution_no_session` 覆盖

2. **`behavior_rules.py` rule 2 重写**
   - 归因驱动强制流程：① 读 `worst_col_tier` → ② 按 tier 选路径 → ③ 再设计表达式，禁止跳过加 EMA
   - ρ_f 量级锚：≥0.85 才算慢，<0.6 必配合降换手手段
   - 信号积分范式 `λ·F+(1-λ)·DELAY(F,1)` + ρ_f 提升近似 `ρ_f' ≈ λ·ρ_f + (1-λ)·1` + λ 上界反推 `λ ≤ (0.85-ρ_f)/(1-ρ_f)`
   - @1w 周线衍生路径、WMA 短窗替代 EMA（N=5~10）、RANK vs TS_RANK 区分
   - 与诊断层 tier 分档呼应

3. **`market_mechanisms.py` 4d 节**：选机制 = 选换手档位，换手超标先问信号根持续性而非加平滑。与 rule 2 呼应。

4. **黄金基线 4 份同步更新**：`system_prompt_full/price_only/population/long_label.txt`，
   `test_system_prompt_modules.py` 验证逐字节一致。

5. **测试覆盖**：`test_turnover_attribution_structured_all_ranges`（3 个 tier）+
   `test_turnover_attribution_no_session`（无 session 边界）。断言与代码逻辑一致。

### 🟡 测试桩与生产实现契约不一致（不影响当前测试通过，但隐患）

`test_turnover_attribution_structured_all_ranges` 里的 `FakeSession.get_column_autocorr` 返回
`self._corrs.get(col)`，对不存在的 col 返回 **None**。但生产代码
`alphaagent/factor/mining/eval/session.py:get_column_autocorr` 永远不返回 None
（有 `default_val=0.5` 兜底，所有路径都返回 float）。

0ffa11e 新增的 `if ctx.session.get_column_autocorr(c) is not None` 检查：
- 在 FakeSession 下会过滤 None（测试验证了这个分支）
- 在生产代码下永远不会过滤（**死代码**——`get_column_autocorr` 永远不返回 None）

**风险**：测试验证了一个生产代码永远不会触发的分支。如果未来有人移除 `if ... is not None`
检查（认为是死代码），测试仍会通过（因为测试 expr 都只含一个 `$col`，None 过滤不触发）。
测试与生产行为脱节。

**建议**：FakeSession 的 `get_column_autocorr` 对不存在的 col 应返回 0.5（与生产 `default_val`
一致），而非 None；或移除生产代码里的 `if ... is not None` 检查。

### 🟡 次要：`col_autocorrs` 字典推导双重调用

```python
col_autocorrs = {
    c: float(ctx.session.get_column_autocorr(c))
    for c in set(cols)
    if ctx.session.get_column_autocorr(c) is not None
}
```

对每个 `c` 调用了两次 `get_column_autocorr(c)`。虽然第二次是缓存命中
（`_column_autocorr_cache`），成本可忽略，但代码风格上不优雅。可改成先取值再过滤：

```python
col_autocorrs = {}
for c in set(cols):
    v = ctx.session.get_column_autocorr(c)
    if v is not None:
        col_autocorrs[c] = float(v)
```

---

## 2a325f2 — 评级：🔴 buffer zone 在引擎层（`selection.py:build_targets`）实现错误，完全无效

`portfolio.py` 的 buffer 实现正确，band 逻辑正确，但 **`selection.py:build_targets` 的
buffer zone 实现有逻辑错误，导致引擎回测层的 buffer zone 完全无效**——成员集合与无 buffer
时完全一样，不降低换手。

### 🔴 核心 bug：`selection.py:build_targets` buffer zone 无效

`core/selection.py` 的 buffer 逻辑：

```python
chosen = self.rank_select(gated, g_scores, ..., policy.top_n, ...)  # top-N
if buffer_keep and buffer_ratio > 0.0 and chosen:
    long_n = len(chosen)
    m_buf = int(round(long_n * float(buffer_ratio)))
    if m_buf > 0:
        order = np.argsort(g_scores, kind="mergesort")
        if not policy.ascending:
            order = order[::-1]
        ordered_all = [int(gated[o]) for o in order]
        buf_zone = set(ordered_all[:long_n + m_buf])  # top-(N+M)
        kept = [k for k in chosen if k in buffer_keep and k in buf_zone]
        # ↑ chosen 是 top-N，chosen 里的元素都在 top-N ⊂ top-(N+M) = buf_zone
        #   所以 k in buf_zone 恒 True，kept = top-N ∩ 老持仓
        deficit = long_n - len(kept)
        if deficit > 0:
            refill = [k for k in chosen if k not in kept]  # top-N 里不在 kept 的
            chosen = kept + refill[:deficit]
        else:
            chosen = kept[:long_n]
```

**问题**：`chosen` 是 `rank_select` 返回的 top-N。`buf_zone` 是 top-(N+M)。`chosen` 里的
元素都在 top-N ⊂ top-(N+M)，所以 `k in buf_zone` 恒 True。`kept` = top-N ∩ 老持仓
（这些本来就在 top-N 里，buffer 与否都会被选）。

**buffer zone 的意图**（commit message + docstring）："老持仓在 top-(N+M) 内则保留，
空缺从严格 top-N 补足"。意思是老持仓里排名 N+1~N+M 的票（在 buffer zone 但不在 top-N）
应该被**保留**，然后从严格 top-N 补足空缺。但当前实现只保留了 top-N ∩ 老持仓，
**没有把 top-(N+1~M) ∩ 老持仓拉进 chosen**。

**结果**：`chosen = kept + refill[:deficit]`，`kept` 和 `refill` 都来自 top-N，
`len(chosen) = len(kept) + deficit = long_n`。`chosen` 还是 top-N 的 long_n 个票——
**与无 buffer 时成员完全一样**（只是顺序可能不同）。buffer zone 完全无效，不降低换手。

**正确实现**应该是：

```python
if buffer_keep and buffer_ratio > 0.0 and chosen:
    long_n = len(chosen)
    m_buf = int(round(long_n * float(buffer_ratio)))
    if m_buf > 0:
        order = np.argsort(g_scores, kind="mergesort")
        if not policy.ascending:
            order = order[::-1]
        ordered_all = [int(gated[o]) for o in order]
        buf_zone = set(ordered_all[:long_n + m_buf])
        # 老持仓在 top-(N+M) 内的都保留（按全序排序，高分优先）
        old_in_buffer = [k for k in ordered_all if k in buffer_keep and k in buf_zone]
        deficit = long_n - len(old_in_buffer)
        if deficit > 0:
            # 从严格 top-N 补足不在 old_in_buffer 的
            refill = [k for k in ordered_all[:long_n] if k not in old_in_buffer]
            chosen = old_in_buffer + refill[:deficit]
        else:
            chosen = old_in_buffer[:long_n]
```

**影响范围**：
- `run_backtest` → `simulate.py` `use_cash=True` 分支（默认路径，`cash_mode=True` +
  `long_short=False`）→ `build_targets`：buffer zone **无效**
- `simulate.py` `use_cash=False` 分支：用 `rank_select` 直接选，**完全没有 buffer zone**
  （只有 band）
- `portfolio.py` `quantile_portfolio_metrics`：buffer 实现**正确**
  （`kept = nav_prev_members & (members | buf_names)`），但只影响信息性字段
  `avg_rebalance_side_turnover_buffered_banded` 和 `net` 序列（`top_group_*`），不影响引擎回测

**实际影响**：`EngineGateCriteria` 默认 `buffer_ratio=0.5`，engine_gate 回测时 buffer zone
不起作用，换手不会因 buffer 降低。只有 band（`no_trade_band=0.15`）有效。engine_gate 的
降换手效果**只有 band 一半**，buffer 那一半是空的。

### 🟡 测试无法检测 buffer 无效

`test_buffer_band_reduces_turnover`（`tests/test_engine_gate_buffer_band.py`）断言
`buf_turn <= raw_turn + 1e-9`（允许持平）。buffer 无效时换手持平或因 band 降低，断言仍通过。
**没有测试验证"buffer 单独（`no_trade_band=0.0`）能降换手"**——如果有，它会失败。

`test_buffer_zone_reduces_rebalance_turnover`（`tests/test_buffer_band.py`）测的是
`quantile_portfolio_metrics`（`portfolio.py`），那里的 buffer 实现正确，所以通过。但它
不覆盖 `selection.py` 的 `build_targets`。

**建议**：加一个 `test_buffer_only_reduces_engine_turnover`
（`run_backtest(buffer_ratio=0.5, no_trade_band=0.0)`），断言 `buf_turn < raw_turn`
（严格小于，不允许持平）。这个测试会暴露 `build_targets` 的 buffer 无效 bug。

### 🟡 `portfolio.py` 与 `selection.py` buffer 口径不一致

- `portfolio.py`：buffer = 次高组全体（`buf_bin = long_bin - 1`，`buf_names` = 次高组所有票）
- `selection.py`：buffer = top-(N+M) 精确排名（`ordered_all[:long_n + m_buf]`）

两者服务不同层（`portfolio.py` 影响信息性字段 + `top_group_*` 净值；`selection.py` 影响引擎
回测），口径不一致不会直接冲突，但会让"因子层 buffer 降换手"和"引擎层 buffer 降换手"的
数值不可比。如果 spec 要求两层口径一致，需要统一。

### 🟡 `submit.py` 与 `run_engine_gate` 默认值合并重复

`submit.py` 和 `run_engine_gate` 都做了 `EngineGateCriteria` 默认值合并
（`buffer_ratio` / `no_trade_band` 字段不存在时回落默认）。两层合并是幂等的（不会冲突），
但属于冗余代码，维护时容易遗漏一处。

### ✅ 正确的部分

1. **`portfolio.py` buffer + band 实现**：`kept = nav_prev_members & (members | buf_names)`
   正确保留老持仓 ∩ (top + buffer)；band 逻辑 `changed_ratio <= no_trade_band` 时跳过调仓，
   换手归零。

2. **`avg_rebalance_side_turnover` 保持原口径**：用 `nav_turnover_sum_raw`（无 buffer/band），
   因子层门槛（stage_one/two）继续用原口径。buffer+band 后换手单独输出
   `avg_rebalance_side_turnover_buffered_banded`。两层分离正确。

3. **默认参数向后兼容**：`buffer_ratio=0.0, no_trade_band=0.0` 时行为与无参数完全一致
   （`test_default_params_byte_identical_to_no_params` 验证 `LEGACY_KEYS` 逐位一致，含
   `top_group_*`）。

4. **band 逻辑正确**：`use_cash=True` 和 `use_cash=False` 两个分支都实现了 band 跳过逻辑，
   `new_hold = hold.copy()` 保留老持仓，`turn = abs(new_hold - hold).sum()/2 = 0`。

5. **`EngineGateCriteria` 默认值 + 透传**：`buffer_ratio=0.5, no_trade_band=0.15` 默认启用，
   `run_engine_gate` 透传到 `run_backtest`，旧 spec 文件无新字段时回落默认。

6. **显式关闭有效**：`policy.get("buffer_ratio", 0.0) or 0.0` 对显式传 0.0 的处理正确
   （`0.0 or 0.0 = 0.0`）。

---

## 跨分支冲突风险

0ffa11e + 2a325f2 分支从 `8850648` 分出，**不含 e2e10cb/058b37f 的 vwap 单位归一化修复**。

`tests/test_engine_gate_buffer_band.py:test_run_engine_gate_passes_buffer_band` 里：

```python
# amount 千元 → 元（与 alpha_panel_to_engine_frame stock 口径一致）
mi_panel["amount"] = mi_panel["amount"] * 1000.0
```

这个测试假设 panel amount 是千元，与 058b37f 后的"panel amount 是元"矛盾——
**合并到含 058b37f 的主干后，这个测试会失败**（amount 被二次放大 ×1000）。
合并时需要调整测试，去掉手动 ×1000。

---

## 验证建议

```powershell
# 0ffa11e 测试
.venv\Scripts\python.exe -m pytest tests/test_feedback_consistency.py -v -k turnover_attribution
.venv\Scripts\python.exe -m pytest tests/test_system_prompt_modules.py -v

# 2a325f2 测试（当前测试无法检测 buffer 无效，都会通过）
.venv\Scripts\python.exe -m pytest tests/test_buffer_band.py -v
.venv\Scripts\python.exe -m pytest tests/test_engine_gate_buffer_band.py -v

# 建议补的测试（会暴露 build_targets buffer 无效 bug）
# tests/test_engine_gate_buffer_band.py 加：
# def test_buffer_only_reduces_engine_turnover(panel):
#     res_raw = run_backtest(panel=panel, codes=CODES, factor="mom20", ascending=False,
#         start=START, end=END, capital=1_000_000, top_n=2, freq="monthly")
#     res_buf = run_backtest(panel=panel, codes=CODES, factor="mom20", ascending=False,
#         start=START, end=END, capital=1_000_000, top_n=2, freq="monthly",
#         buffer_ratio=0.5, no_trade_band=0.0)  # 只开 buffer，不开 band
#     raw_turn = float(res_raw["trades"]["turnover"].sum())
#     buf_turn = float(res_buf["trades"]["turnover"].sum())
#     assert buf_turn < raw_turn - 1e-9, \
#         f"buffer alone should reduce turnover: buf={buf_turn} >= raw={raw_turn}"
```

---

## 总结

| commit | 评级 | 核心问题 |
|---|---|---|
| 0ffa11e | 🟢 | P0 降换手方法体系，逻辑正确。测试桩 FakeSession 与生产 `get_column_autocorr` 契约不一致（None vs 0.5），`if ... is not None` 是死代码。不影响当前正确性。 |
| 2a325f2 | 🔴 | **`selection.py:build_targets` buffer zone 实现错误，完全无效**——`kept = [k for k in chosen if k in buffer_keep and k in buf_zone]` 只保留 top-N ∩ 老持仓（`chosen` 是 top-N，`k in buf_zone` 恒 True），没有把 top-(N+1~M) ∩ 老持仓拉回来。引擎回测层 buffer zone 不降换手，只有 band 有效。测试 `test_buffer_band_reduces_turnover` 断言 `<=`（允许持平）无法检测。 |

**2a325f2 需要修复**：`selection.py:build_targets` 的 buffer 逻辑应改为遍历 `ordered_all`
（top-(N+M) 全序）保留老持仓，而非遍历 `chosen`（top-N）。修复后补一个
`test_buffer_only_reduces_engine_turnover`（`no_trade_band=0.0`，断言严格 `<`）。

**跨分支冲突**：合并到含 058b37f 的主干时，`test_run_engine_gate_passes_buffer_band`
里的 `mi_panel["amount"] * 1000.0` 需要去掉（panel amount 已是元）。
