# -*- coding: utf-8 -*-
"""模块 · rotation_guide：行业/风格轮动因子构建引导（P1，默认关闭）。

背景：因子上报时最常见的轮动伪命题是「全市场 IC 高 = 行业内有效」——事实
上全市场高 IC 可能完全来自行业贝塔（行业风格暴露），行业内选股能力为零。
P1 评估端会同时披露行业诊断 metrics（industry_within_rank_ic /
industry_exposure_r2 / industry_ic_dispersion），本模块只负责在 prompt 端
引导 LLM 把「行业内 alpha」与「条件 alpha / 轮动信号」二选一显式声明清楚，
避免同一个因子含糊地同时声称两种语义。

启用条件：``research_spec.rotation_policy.enabled=True``（默认 False → 本模块
不渲染，黄金基线逐字节不变）。
"""

from alphaagent.factor.mining.prompt.prompt_modules import PromptContext

NAME = "rotation_guide"
TITLE = "行业/风格轮动因子构建引导"
ORDER = 82  # neutralization_guide(80) 之后、tool_contracts(90) 之前
REQUIRED = False
SEP_BEFORE = "\n\n---\n\n"

RAW = """### 行业/风格轮动因子构建引导（rotation_guide）

**1. 先二选一声明你的因子语义，不允许含糊。**

- **「行业内 alpha」**：因子在**每个行业内部**选股仍有效，不押注行业涨跌。
  实现首选组内变换（以下任选，行业码 `$industry_sw_l1` 已是离散组号，直接作
  group 键，**勿再套 `CS_BUCKET`**）：
  - `CS_GROUP_RANK(信号, $industry_sw_l1)`：行业内截面百分位秩，输出天然 0~1；
  - `CS_NEUTRALIZE(信号, $industry_sw_l1)`：行业内去均值，之后可接 `CS_ZSCORE`
    或 `RANK` 标准化。
  典型链路：`CS_ZSCORE(CS_NEUTRALIZE($ret_5d, $industry_sw_l1))`。
- **「条件 alpha / 轮动信号」**：因子只在特定市场/行业景气状态启用，是行业配置
  型或状态依赖型信号。实现：
  - 首选 `SOFT_GATE(signal, state, strength=0.5)`（连续加权，不塌缩）或
    `PIECEWISE_STATE(signal, state, low_q=0.2, high_q=0.8, ...)`；
  - `GATED_SIGNAL` 也能表达开关语义，但 constant 簇会塌缩十分位分箱，新因子慎用；
  - **必须声明门控载体与阈值**：state 是哪一列（如 `$industry_sw_l1` 涨跌、
    `$funda_roe`、波动率状态），激活方向与阈值是多少，为什么此时该风格有效。

**2. 红线（评估端会同时披露行业暴露诊断，不要自相矛盾）。**

- 一只因子**不允许同时声称「行业内 alpha」与「行业配置」**。全市场 IC 高 +
  行业内 IC 低 + 行业暴露 R² 高 ⇒ 因子本质是行业贝塔，不是行业内选股；
- 纯行业动量/行业配置型因子单独作候选时**必须自认行业暴露**，并在 comment 里
  写明押注的行业维度，不得伪装成行业内 alpha；
- 是否行业中性化取决于你是否故意押注行业景气：押 → 不中性化并自认暴露；
  不押 → 用 `CS_NEUTRALIZE(信号, $industry_sw_l1)` 剥掉行业共同部分。

**3. comment 示例（行业内 alpha 型）。**

```
行业内 alpha：CS_ZSCORE(CS_NEUTRALIZE(TS_SUM($ret, 5), $industry_sw_l1))；不押注
行业景气（不对行业动量暴露），行业暴露 R² 应接近 0；评估 sector 诊断应显示
within_rank_ic 与全市场相当。若诊断显示行业暴露偏高，需回到组内变换重做。
```
"""


def render(ctx) -> str:  # noqa: ANN001
    return RAW


def enabled(ctx: PromptContext) -> bool:
    """仅 rotation_policy.enabled=True 时渲染；默认 False 保持旧 prompt 不变。"""
    rp = (getattr(ctx, "research_spec", None) or {}).get("rotation_policy") or {}
    return bool(rp.get("enabled"))
