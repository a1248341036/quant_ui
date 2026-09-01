# -*- coding: utf-8 -*-
"""模块 10 · ic_robustness：IC 方向、月度稳健性与十分组判读。原文精确切片。"""

RAW = """### IC 方向、月度稳健性与十分组

- 研究阶段可分析正、负 IC；负 IC 和负 ICIR 均为有效信号，两阶段池以 `abs(IC)` 和 `abs(ICIR)` 判断，负方向因子无需手动取反。
- `summary.cs_pearson_autocorr` 继续展示并用于研究判断，但不参与当前两阶段硬门槛。
- **`ic > 0`**：`mean_monthly_ic` 宜为正；`share_months_ic_positive`（终端「月IC+」）须 **> 0.7**。
- **`ic < 0`**：`mean_monthly_ic` 宜为负；`share_months_ic_positive` 须 **< 0.3**。
- **十分组 `decile_mean_label`**（全样本等频，D1=因子最低）：
  - `ic > 0`：宜 **D10.mean_label > D1.mean_label**（因子越高、label 越高）
  - `ic < 0`：宜 **D1.mean_label > D10.mean_label**
  - D1≈D10 或顺序与 IC 符号相反 → 分位无区分，不宜作保留级
"""

NAME = "ic_robustness"
TITLE = "IC 方向与月度稳健性判读"
ORDER = 100
REQUIRED = False
SEP_BEFORE = "\n\n---\n\n"


def render(ctx) -> str:  # noqa: ANN001
    return RAW
