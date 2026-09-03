# -*- coding: utf-8 -*-
"""模块 10 · ic_robustness：IC 方向、月度稳健性与十分组形态学判读。原文精确切片。"""

RAW = """### IC 方向、月度稳健性与十分组形态学

- 研究阶段可分析正、负 IC；负 IC 和负 ICIR 均为有效信号，两阶段池以 `abs(IC)` 和 `abs(ICIR)` 判断，负方向因子无需手动取反。
- `summary.cs_pearson_autocorr` 继续展示并用于研究判断，但不参与当前两阶段硬门槛。
- **`ic > 0`**：`mean_monthly_ic` 宜为正；`share_months_ic_positive`（终端「月IC+」）须 **> 0.7**。
- **`ic < 0`**：`mean_monthly_ic` 宜为负；`share_months_ic_positive` 须 **< 0.3**。
- **十分组形态学 `decile_mean_label`**（全样本等频，D1=因子最低、D10=因子最高）。
  每次 evaluate 结果都带 `prediction_check`（你提交的 prediction 与实际形态的自动对账），
  判读顺序：**先看对账，再看门槛**——形态与你预期不符 = 机制假设错误，调参数救不了错误的结构：

  1. **单调性**：`ic > 0` 宜 D10>D1 且逐组递增；`ic < 0` 宜反向。D1≈D10 或顺序与 IC 符号相反 → 分位无区分，不作保留级。
  2. **alpha 集中端**：看哪几组最强。常见陷阱是 **倒 U 型**：IC 正号由"涨最多的组未来最差"（空头端）驱动，
     而非"跌最多的组反弹"（多头端）。例：反转因子 D1(涨最多)=-0.8%/20d、D5-D7 最强 +1.0%、D10(跌最多)仅 +0.4%——
     IC=+0.04 但纯多头买入 D10 年化超额为负。
  3. **纯多头可交易性（A 股硬约束）**：策略只能做多。若 alpha 集中在低因子端或中间组，
     因子对多头组合**没有变现路径**——除非你把它当剔除器（负向筛掉最差组）使用。
     判定标准：持仓端（IC>0 时 D10 / IC<0 时 D1）的 mean_label 必须优于全样本均值，且超额能覆盖换手成本。
  4. **门控消融 `ablation_check`**：表达式含 GATED_SIGNAL / IF_THEN_ELSE / PIECEWISE_STATE / CS_GROUP_RANK
     且契约给了 base_expr 时自动返回。`conditioning_destroyed_value` / `conditioning_flipped_signal`
     = 条件化正在摧毁基信号（实测：年线门控把 20d 反转 IC 从 +0.039 变 -0.005）——机制假设错误，删门控或反转条件方向；
     `conditioning_added_value` 才是门控有机制性增量的证据。收到 `ablation_hint` 说明没传 base_expr，补上重跑。
  5. **预测被证伪（prediction_check.verdict=contradicted）后**：禁止只换窗口/参数重试同结构；
     要么给出解释新形态的替代机制重新提交，要么放弃该方向。
"""

NAME = "ic_robustness"
TITLE = "IC 方向与十分组形态学"
ORDER = 100
REQUIRED = False
SEP_BEFORE = "\n\n---\n\n"


def render(ctx) -> str:  # noqa: ANN001
    return RAW
