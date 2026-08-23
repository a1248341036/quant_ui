# AlphaAgent DSL Core

股票因子表达式 DSL 核心：负责把文本表达式编译成可执行 Python 代码，并在 `(datetime, instrument)` 面板上求值。提供 Numba / 纯 Python 加速后端。

## 职责范围

- **表达式解析**：把多行 DSL（如 `x = DELTA($close,1); TS_MEAN(x, 10)`）编译成 Python 代码。
- **时序算子**：`TS_*`、`DELTA`、`SLOPE` 等，按 **instrument** 分组计算。
- **截面算子**：`RANK`、`CS_ZSCORE`、`CS_DEMEAN`、`CS_WINSORIZE`、`CS_BUCKET`、`CS_NEUTRALIZE`，按 **datetime** 截面计算。
- **动态窗口**：按品种分组的时间序列滚动（dynamic window），支持变长窗口。
- **混频支持**：主频日 panel + 辅周期（`@1d` / `@1w`），自动广播对齐。

## 设计：三层后端

| 后端 | 位置 | 触发条件 | 特点 |
|---|---|---|---|
| C++ | `fam_accel.cpp` + `setup.py` | `FUTURE_ALPHA_MINER_ACCEL_BACKEND=cxx` 或自动检测到 `aqra.dsl._shared.dsl_core._fam_accel` | OpenMP 并行，`double` 内部计算，`float32` 输入输出 |
| Numba | `accel.py` / `dyn_window.py` 中 `@njit` 函数 | C++ 未安装或强制 `numba` | JIT 编译，纯 `float32` |
| 纯 Python | 同上函数的回退分支 | 无 Numba 无 C++ | 解释执行，兼容性好 |

默认自动选择：C++ > Numba > Python。当前 `aqra/dsl/accel.py` 会尝试导入本目录下的 `_fam_accel` 扩展并注入后端。

## 精度策略

- 算子内部默认使用 `float32`（中间数组、输出数组）。
- C++ 扩展通过 `pybind11::forcecast` 接受 `float32` 输入，内部按 `double` 计算，输出再 cast 回 `float32`。
- 行情 / label 加载后自动 downcast 数值列到 `float32`（见 `aqra/api/data_loader.py::_downcast_floats`）。
- 因此 **Python/Numba 路径与 C++ 路径的 float32 输出在 f32 精度下一致**（详见一致性测试）。

## 文件与代码位置

### 算子层（`operators.py`）

面向 DSL 的算子入口，例如 `TS_MEAN`、`DELTA`、`EMA`、`WMA`、`TS_CORR`、`CHIP` 等。

| 功能 | 入口 | 位置 |
|---|---|---|
| 一元滚动算子（TS_MEAN/STD/SUM/MIN/MAX/MEDIAN/SKEW/KURT/PROD） | `TS_*` 系列 | `operators.py` |
| 时序秩（窗口内） | `TS_RANK` | `operators.py` |
| 截面秩 / 标准化 | `RANK`、`CS_ZSCORE`、`CS_DEMEAN`、`CS_WINSORIZE`、`CS_BUCKET`、`CS_NEUTRALIZE` | `operators.py` |
| 滞后/差分/涨跌幅 | `DELTA`、`TS_SHIFT`、`TS_PCTCHANGE` | `operators.py` |
| EMA / WMA | `EMA`、`WMA` | `operators.py:852` 起 |
| 相关/协方差 | `TS_CORR`、`TS_COV` | `operators.py` |
| 筹码分布 | `CHIP_*` 系列（日频 uniform/cyq/tri） | `operators.py` / `chip_daily.py` |
| 局部极值 | `TS_ARGMAX`、`TS_ARGMIN`、`TS_LOCAL_PEAK` 等 | `operators.py` |
| 自变量引用解析 | `dollar_ref_to_pyname` | `parser.py` |

### 加速后端（`accel.py`）

底层向量化算子，按后端自动分发。

| 功能 | 入口 | 位置 |
|---|---|---|
| 固定窗滚动聚合 | `roll_fixed` | `accel.py:129` |
| EMA | `ema` | `accel.py:385` |
| WMA | `wma` | `accel.py:408` |
| 滞后/差分/涨跌幅 | `shift_fixed` / `delta` / `pctchange` | `accel.py:358` / `454` / `490` |
| 局部极值位置/值 | `arg_local_extreme` / `local_extreme_value` | `accel.py:564` / `594` |
| 滚动协方差/相关系数 | `roll_cov_fixed` / `roll_corr_fixed` | `accel.py:913` / `928` |
| 滚动分位数 | `roll_quantile_fixed` | `accel.py:986` |
| 互信息 | `roll_mutual_info_lag_fixed` | `accel.py:1616` |
| 效率比率 | `roll_efficiency_ratio_fixed` | `accel.py` |
| 排列熵 | `roll_permutation_entropy_fixed` | `accel.py:2512` |
| 筹码指标 | `roll_chip_metric_fixed` / `chip_daily.py` | `accel.py` / `chip_daily.py` |
| C++ 后端可用性探测 | `accel_available` | `accel.py:72` |

### 动态窗口（`dyn_window.py`）

按品种（instrument）分组做时间序列运算，再写回原始索引。

| 功能 | 入口 | 位置 |
|---|---|---|
| 通用动态窗口聚合 | `roll_dynamic` | `dyn_window.py` |
| 动态滞后 | `delay_dynamic` | `dyn_window.py` |
| 动态 ARG 极值 | `arg_extreme_dynamic` | `dyn_window.py:451` |

### 其他核心文件

| 文件 | 作用 |
|---|---|
| `parser.py` | DSL 文本 → Python 代码；`$col` / `$col@freq` 解析 |
| `ops_kit.py` | 面板分组工具：`gb_instrument`、`gb_datetime`、`per_*` 包装 |
| `resample.py` | 辅周期面板构建、主频对齐广播 |
| `intervals.py` | 周期归一化（`1m` / `5m` / `1h` / `1d` 等） |
| `errors.py` | 结构化异常 `MultiLineFactorEvalError` |
| `fam_accel.cpp` | C++ 加速核源码，与 `DSL_CORE` 子仓库一起提交 |

## 目录结构

```text
aqra/dsl/_shared/dsl_core/
  __init__.py
  operators.py          # DSL 算子入口
  accel.py              # 加速后端分发 + Numba 实现
  dyn_window.py         # 动态窗口（按 instrument 分组）
  parser.py             # 表达式解析
  ops_kit.py            # 小工具
  resample.py           # 混频对齐
  chip_daily.py         # 日频筹码分布内核（uniform/cyq/tri）
  intervals.py          # 周期归一化
  errors.py             # 异常类型
  fam_accel.cpp         # C++ 加速核源码
  .gitignore            # 忽略 *.so / build /
```

## 快速开始

```python
import numpy as np
import pandas as pd
from aqra.dsl.evaluator import eval_multi_line_factor

idx = pd.MultiIndex.from_product(
    [pd.date_range("2024-01-01", periods=100, freq="min"), ["A", "B"]],
    names=["datetime", "instrument"],
)
panel = pd.DataFrame(
    {
        "open":  np.random.rand(len(idx)).astype(np.float32),
        "close": np.random.rand(len(idx)).astype(np.float32),
    },
    index=idx,
)

expr = """
x = DELTA($close, 1)
TS_MEAN(x, 10)
"""
result = eval_multi_line_factor(expr, panel)
```

切换后端：

```python
import os
os.environ["FUTURE_ALPHA_MINER_ACCEL_BACKEND"] = "cxx"  # 或 "numba" / "python"
# 然后重新导入 aqra.dsl.accel
```

## 编译 C++ 扩展

在项目根目录执行：

```bash
uv run python setup.py build_ext --inplace
```

编译产物会生成在：

```text
aqra/dsl/_shared/dsl_core/_fam_accel.cpython-311-x86_64-linux-gnu.so
```

跳过 C++（纯 Python/Numba）：

```bash
AQRA_SKIP_CXX=1 uv run python setup.py build_ext --inplace
```

## 一致性测试

验证 C++ 与 Numba 路径在 `float32` 输入下输出一致：

```bash
uv run python scripts/test_dsl_core_f32_consistency.py
```

输出包括：

- 30+ 个 `accel` 直接函数的最大绝对误差、相对最大误差、Pearson 相关。
- 20 个常见 DSL 表达式的双后端对比。

结果会保存到：

```text
data/test_artifacts/dsl_core_f32_consistency.csv
```

## 日频筹码算子（`CHIP_*`）

默认 **CYQ 换手衰减**（`method='cyq'`、`nbins=64`），标准 6 参：

```text
CHIP_PEAK_LOC($adj_close, $adj_low, $adj_high, $volume, window, $float_cap)
```

| 位置 | 含义 |
|------|------|
| 1–3 | `$adj_close` / `$adj_low` / `$adj_high` |
| 4 | `$volume` |
| 5 | 窗口（交易日，推荐 20~120） |
| 6 | `$float_cap`（CYQ 换手率分母） |
| 7 | 可选 `nbins`（默认 64） |
| 8+ | 可选 `method`（仅 `'tri'`/`'uniform'`）；`'tri'` 时第 6 参改传 `$vwap` |

内核见 `chip_daily.py`。

## 当前限制

- C++ 核内部仍是 `double` 计算，只是把输入输出 cast 成 `float32`；内存占用没有真正减半。
- 动态窗口算子（`dyn_window.py`）目前走 Numba/Python 路径，未接入 C++ 扩展。
- 部分复杂算子（chip、permutation entropy）在 C++ 与 Numba 路径上可能存在 `float32` 舍入差异，但相关系数为 1.0。
- C++ 扩展依赖 `pybind11` 和系统编译器（GCC + OpenMP），Windows/macOS 需要单独调整编译参数。
