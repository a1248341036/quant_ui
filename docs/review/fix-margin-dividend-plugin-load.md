# Review: fix/margin-dividend-plugin-load

**分支**: `fix/margin-dividend-plugin-load`
**单提交**: `79f4934` — "fix(data): 修两个数据插件加载失败（融资融券日分区守卫 / 分红 date_ranges）"
**合并基**: `d0748ae342c06e906f4ca00cd2dec273f9667f25` (main)
**改动**: 4 文件 +170/-7
**评审日期**: 2026-09-28
**评审结论**: **merge**

---

## 改动概览

修复两个 CNE curated 数据插件加载失败，根因都是 polars 1.43 API / 文件数守卫与日分区数据集不兼容：

| 文件 | 改动 | 作用 |
|---|---|---|
| `_pitlib.py` | +56 | `read_curated` 新增 `start/end` 关键字参数；日分区数据集按窗口裁剪；超 `_MAX_EAGER_FILES` 时窗口内改 `scan_parquet` 流式读取 |
| `margin.py` | +2/-1 | `load()` 把窗口透传给 `read_curated("margin_trading", start=start, end=end)` |
| `dividend.py` | +5 | 除息倒计时窗口 `pl.date_range(eager=True)` → `pl.date_ranges`（复数，元素级 API） |
| `tests/test_data_plugin_windows.py` | +114 (新) | 5 个回归测试覆盖裁剪/流式/守卫/空窗口/分红展开 |

---

## 逐文件评审

### 1. `_pitlib.py` — 核心改动，逻辑正确

**新增常量与辅助函数** (`_pitlib.py:71-83`):

```python
_MAX_EAGER_FILES = 512
_PARTITION_DIR_RE = re.compile(r"^[a-z_]+=(\d{4}-\d{2}-\d{2})$")

def _partition_date(path: Path) -> datetime.date | None:
    for part in reversed(path.parts[:-1]):
        m = _PARTITION_DIR_RE.match(part)
        if m:
            return datetime.date.fromisoformat(m.group(1))
    return None
```

- 正则 `^[a-z_]+=(\d{4}-\d{2}-\d{2})$` 精确匹配 hive 日期分区目录名（`trade_date=2024-01-02`），对非日期分区（`symbol=XXX`）不匹配 → 返回 `None`，行为安全。
- `reversed(path.parts[:-1])` 从叶子向根找最近一层日期分区，正确处理嵌套分区（多级 hive）。
- `path.parts[:-1]` 排除文件名本身，正确。

**`read_curated` 窗口裁剪** (`_pitlib.py:86-127`):

```python
windowed = False
if start or end:
    s, e = parse_window(start, end)
    dated = [(f, _partition_date(f)) for f in files]
    if any(d is not None for _, d in dated):
        kept = [f for f, d in dated if d is None or s <= d <= e]
        if not kept:
            known = sorted(d for _, d in dated if d is not None)
            raise ValueError(f"CNE curated {dataset} 在窗口 {s}~{e} 内无分区数据"
                             f"（分区范围 {known[0]} ~ {known[-1]}）")
        files = sorted(kept)
        windowed = True

if len(files) > _MAX_EAGER_FILES:
    if not windowed:
        raise ValueError(f"CNE curated {dataset} parquet 文件数异常（{len(files)}），拒绝全量读取")
    return pl.scan_parquet([str(f) for f in files]).collect()
return pl.concat([pl.read_parquet(f) for f in files], how="vertical")
```

逐路径核查：

1. **无窗口** (`start=end=None`): `windowed=False`，超 `_MAX_EAGER_FILES` 仍抛 `文件数异常` —— **旧守卫行为完全保留**，无回归。其余 4 个 `read_curated` 调用方（`disclosure.py:55` / `top_holders.py:53` / `express.py:58` / `institutional.py:43,98` / `fundamental.py:147`）均不传窗口，向后兼容。
2. **有窗口 + 有日分区**: 裁剪到 `[s, e]`；空结果显式抛 `无分区数据`（带分区范围提示，可诊断）。
3. **有窗口 + 无日分区**: `any(d is not None)` 为假 → 不裁剪、`windowed=False`。若文件数超限仍走旧守卫拒绝 —— 正确（无分区信息无法安全裁剪，不能盲目放行）。
4. **有窗口 + 混合分区**: 无日期分区的文件 `d is None` → 无条件保留；有日期分区的按窗口裁剪。安全默认（无日期文件可能含任意日期，交给下游 `trade_date` 行级过滤处理，`margin.py:64` 正是这样做的）。
5. **`windowed=True` + 文件数 ≤ 上限**: 落到 `pl.concat` eager 路径 —— 小窗口仍 eager，正确。
6. **`windowed=True` + 文件数 > 上限**: `scan_parquet([...]).collect()` 流式读取 —— 正确，避免上千小文件 concat。

**无缺陷。** 边界覆盖完整，错误信息可诊断，向后兼容。

### 2. `margin.py` — 窗口透传，正确

`margin.py:63`:

```python
raw = _pitlib.read_curated("margin_trading", start=start, end=end)
```

`load()` 原本就接收 `start/end`，只是没往下传。透传后 `read_curated` 先做分区级裁剪，`margin.py` 自身的 `(pl.col("trade_date") >= s) & (pl.col("trade_date") <= e)` 行级过滤仍保留 —— 双层过滤，分区级粗筛 + 行级精筛，正确且无冗余风险（分区级只保留命中分区的文件，行级再切到精确窗口边界）。

**无缺陷。**

### 3. `dividend.py` — polars API 修复，正确

`dividend.py:120-126`:

```python
pl.date_ranges(
    pl.col("imp_ann_date"),
    pl.col("ex_date") - datetime.timedelta(days=1),
    interval="1d",
).alias("_days")
```

- **根因正确**: polars 1.43（本仓库 `polars 1.43.2`）中 `pl.date_range(..., eager=True)` 在 `with_columns` 内会对空 frame 立即求值，列解析失败抛 `ColumnNotFoundError`；`pl.date_ranges`（复数）才是 `with_columns` 内的元素级表达式 API。
- **边界正确**: `ex_date - 1day` 使区间为 `[imp_ann_date, ex_date)`，除息日当天不在展开范围内 → `div_days_to_ex` 为 NaN（测试 `test_dividend_load_expands_ex_window` 断言 2024-03-05 NaN，符合）。
- **`_left = ex_date - _days`**: 公告日 `_left = ex_date - imp_ann_date`（测试 imp_ann 2024-03-01 / ex 2024-03-05 → `div_days_to_ex=4`，符合）。
- **上游守卫**: `windows.is_empty()` 分支兜底 `pl.lit(None, dtype=pl.Int32)`，空事件不进 `with_columns`，双保险。

**无缺陷。** 注释清楚解释了 API 差异，便于后续维护。

### 4. `tests/test_data_plugin_windows.py` — 测试测其所声称

5 个测试逐个核查：

| 测试 | 断言 | 评 |
|---|---|---|
| `test_read_curated_prunes_date_partitions` | 3 分区（2024-01-02/03, 2030-01-02），窗口 2024-01 → height 4，仅 2024 日期 | 覆盖裁剪主路径 ✓ |
| `test_read_curated_streams_when_window_exceeds_eager_cap` | monkeypatch `_MAX_EAGER_FILES=2`，窗口留 3 天 → height 6，断言走 scan 路径 | 覆盖流式回退 ✓ |
| `test_read_curated_without_window_keeps_guard` | 无窗口 + `_MAX_EAGER_FILES=2` + 3 文件 → 抛 `文件数异常` | 覆盖旧守卫保留 ✓ |
| `test_read_curated_empty_window_is_explicit` | 窗口 2030 + 仅 2024 数据 → 抛 `无分区数据` | 覆盖空窗口显式错误 ✓ |
| `test_dividend_load_expands_ex_window` | monkeypatch `dividend._DIVIDEND_PG`，单事件（imp_ann 2024-03-01, ex 2024-03-05, cash 0.5）→ 公告日 `div_days_since_ann=0` / `div_days_to_ex=4`；2024-03-04 `div_days_to_ex=1`；2024-03-05 NaN；`div_cash_div≈0.5`；`trade_date.min()==2024-03-01` | 覆盖分红展开边界 ✓ |

- 测试用 `_write_partition(root, dataset, day, rows=2)` helper 写真实 hive 分区 parquet，不是 mock —— 测的是真实文件系统路径，可信。
- monkeypatch `_MAX_EAGER_FILES` / `_DIVIDEND_PG` 是标准 pytest 手法，隔离干净。
- 每个测试断言与所声称行为一一对应，无"测 A 却断言 B"的错位。

**无"测试不测其所声称"的缺陷。**

---

## 缺陷汇总

**REAL defects: 0**

未发现逻辑错误、回归、破坏性导入、缺失错误处理、安全问题、竞态、API 契约违例或死代码。

---

## 优点

1. **根因修复而非症状掩盖**: `_pitlib.read_curated` 的文件数守卫原本对日分区数据集（融资融券 2607 文件、估值 2608 文件）整包拒绝，导致面板缺 `mgn_*` 列且哨兵校验拒绝落盘 —— 本分支从分区级裁剪 + 流式读取两层解决，而非简单调大 `_MAX_EAGER_FILES`。
2. **向后兼容**: 新参数全为关键字可选，4 个不传窗口的调用方零改动、零行为变化。
3. **错误信息可诊断**: 空窗口抛 `无分区数据` 并附分区范围，比裸 `FileNotFoundError` 或静默空帧好。
4. **注释解释 API 陷阱**: `dividend.py:118-119` 注明 `pl.date_ranges`（复数）vs `pl.date_range + eager=True` 的差异，防后续维护者回退。
5. **测试覆盖完整**: 5 测试覆盖裁剪/流式/守卫/空窗口/分红展开五条路径，且用真实 parquet 而非 mock。

---

## 结论

**merge**。改动最小、根因准确、向后兼容、测试可信，无 REAL defects。可合并回 main。
