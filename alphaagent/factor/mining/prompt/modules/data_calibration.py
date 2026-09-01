# -*- coding: utf-8 -*-
"""模块 04 · data_calibration：数据与评估口径 + MLS-FMB 门槛 + label 说明。"""

from alphaagent.factor.mining.mls_thresholds import mls_fmb_thresholds_markdown
from alphaagent.factor.types import DEFAULT_LABEL_COL

_LABEL_DESCRIPTIONS: dict[str, str] = {
    "label_1d_open_to_open": "T+1 开盘 → T+2 开盘（短周期 alpha，默认）",
    "label_1d_close_to_close": "T+1 收盘 → T+2 收盘（1 日持有，**适合价量/短周期因子**）",
    "label_10d_close_to_close": "T+1 收盘 → T+11 收盘（10 日持有，**适合基本面/慢因子**）",
    "label_20d_close_to_close": "T+1 收盘 → T+21 收盘（20 日持有，适合基本面/慢因子）",
}


def _label_section_markdown(label_col: str, *, include_fundamentals: bool = True) -> str:
    desc = _LABEL_DESCRIPTIONS.get(label_col, "panel 内预计算的前瞻收益列")
    lines = [
        f"**本次会话 label 列：`{label_col}`** — {desc}。",
        "所有 `summary.ic` / `rank_ic` / `decile_mean_label` / `mls_fmb` 均相对该列计算。",
        "",
        "panel 内常用 label（启动时可 `--label-col` 切换）：",
        "",
        "| 列名 | 含义 |",
        "|------|------|",
    ]
    for name, meaning in _LABEL_DESCRIPTIONS.items():
        mark = " **← 本次**" if name == label_col else ""
        lines.append(f"| `{name}` | {meaning}{mark} |")
    if label_col not in _LABEL_DESCRIPTIONS:
        lines.append(f"| `{label_col}` | {desc} **← 本次** |")
    lines.extend(
        [
            "",
            "**label 选用建议**（`eval_factor` / 挖掘 CLI 的 `--label-col`）：",
            "",
            "| 因子类型 | 推荐 label |",
            "|----------|------------|",
            *(
                ["| 基本面（主要用 `$funda_*`） | `label_10d_close_to_close` |"]
                if include_fundamentals
                else []
            ),
            "| 价量（OHLC / `$ret` / `$volume` / 筹码等） | `label_1d_close_to_close` |",
            "",
            "本次会话已配置为上表「本次」行；勿在 tool 参数中切换 label。",
        ]
    )
    if label_col.startswith("label_") and "d_close_to_close" in label_col and label_col not in (
        "label_1d_close_to_close",
    ):
        try:
            hold = int(label_col.split("_")[1].replace("d", ""))
            if hold > 1:
                lines.extend(
                    [
                        "",
                        f"**长持有 label 提示**：持有约 **{hold} 个交易日**，因子宜偏基本面/低频结构；"
                        "月度 IC 稳健性与 `cs_pearson_autocorr` 仍适用，但 IC 绝对值通常低于短周期 label。",
                    ]
                )
        except ValueError:
            pass
    return "\n".join(lines)


NAME = "data_calibration"
TITLE = "数据与评估口径 + MLS 门槛 + label"
ORDER = 40
REQUIRED = True
SEP_BEFORE = "\n\n---\n\n"


def render(ctx) -> str:  # noqa: ANN001
    from alphaagent.factor.mining.context import asset_type_label

    mls_block = mls_fmb_thresholds_markdown(label_col=ctx.label_col)
    label_block = _label_section_markdown(ctx.label_col, include_fundamentals=ctx.include_fundamentals)
    return f"""### 数据与评估口径

本仓库为**{asset_type_label(ctx.asset_type)}日频 panel**：索引 `(datetime, instrument)`，主频 **1d**。

- **时序算子**（`TS_*`、`DELTA`、`SLOPE` 等）：在**每个 instrument 各自时间序列**上计算。
- **截面算子**（`RANK`、`CS_ZSCORE`、`CS_DEMEAN`、`CS_WINSORIZE`、`CS_BUCKET`、`CS_NEUTRALIZE`）：在**每个 datetime 截面上**跨 instrument 计算。

评估指标均为**截面**口径：

| 指标 | 含义 |
|------|------|
| `summary.ic` | 逐日横截面 Pearson IC 的均值 |
| `summary.icir` | IC / std(逐日 IC)，即 IC 信息比率 |
| `summary.rank_ic` | 逐日横截面 Spearman Rank IC 的均值 |
| `summary.cs_pearson_autocorr` | 逐日横截面 lag-1 Pearson 自相关均值：`corr_CS(f_t, f_{{t-1}})`，用于衡量因子排名日度延续性；当前为诊断指标，不是两阶段硬门槛 |
| `summary.mls_fmb` | 逐日十分组 MLS-FMB：`mean_rho`（单调性）、`mean_ls`/`ir_ls_annual`（多空 IR）、`mls`（综合）、`nw_t_rho`/`nw_t_ls`（NW t） |

{mls_block}

{label_block}"""
