# -*- coding: utf-8 -*-
"""模块 06 · multi_period：多周期语法与筹码算子写法。原文精确切片。"""

RAW = """### 多周期：`$field@<周期>`

仅支持 **`@1d`** 与 **`@1w`**（W-FRI 周线，严格无前视 backward 广播）。

**行作用域规则：**

| 当行引用 | 计算面板 | `TS_*(x, N)` 中 N |
|---|---|---|
| 仅同一种 `@周期` | 该辅频面板 | 该频 N 根 bar |
| 仅主频列 | 日频面板 | N 个交易日 |
| 主频 + `@周期` 混合 | 日频；`@` 列先广播 | N 个交易日 |

**要「真正 N 根周线」的滚动统计**，须单独写纯 `@1w` 行得到中间变量，再与主频列组合：

```text
ma_w = TS_MEAN($adj_close@1w, 4)
SUBTRACT($adj_close, ma_w)
```

混频 `TS_MEAN($col@1w, N)`：**在 broadcast 后的日频 index 上 rolling**，N = 日 bar 数。

**截面算子示例**（逐日跨股票）：

```text
# 市值中性动量（10 档等频分组后组内去均值）
raw = TS_MEAN($ret, 20)
CS_NEUTRALIZE(raw, CS_BUCKET(LOG($float_cap), 10))

# 截面秩
RANK(CS_ZSCORE($amount))
```

**日频筹码算子**（默认 CYQ 换手衰减；6 参即可，勿写 `method` / 旧两参写法）：

```text
# 标准写法（close, low, high, volume, window, float_cap）
peak = CHIP_PEAK_LOC($adj_close, $adj_low, $adj_high, $volume, 60, $float_cap)
entropy = CHIP_ENTROPY($adj_close, $adj_low, $adj_high, $volume, 30, $float_cap)
com_gap = CHIP_COM_W_GAP($adj_close, $adj_low, $adj_high, $volume, 40, $float_cap)

# 可选：第 7 参 nbins（默认 64）；第 8 参 method（仅 tri/uniform 时）
tri_gap = CHIP_COM_W_GAP($adj_close, $adj_low, $adj_high, $volume, 40, $vwap, 64, 'tri')
```

"""

NAME = "multi_period"
TITLE = "多周期与筹码算子"
ORDER = 60
REQUIRED = False
SEP_BEFORE = "\n\n"
# 分阶段注入：探索阶段不注入（用基础算子即可），深耕/交付阶段注入
PHASES = frozenset({"deepen", "deliver", "full"})


def render(ctx) -> str:  # noqa: ANN001
    return RAW
