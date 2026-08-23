"""MLS-FMB 保留级门槛（由已入库因子 train/val 表现 25% 分位校准）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from alphaagent.core.paths import MLS_FMB_PERCENTILES_PATH


def _round_threshold(value: float) -> float:
    x = float(value)
    if x >= 10:
        return round(x, 0)
    if x >= 1:
        return round(x, 1)
    return round(x, 2)


def load_mls_fmb_percentiles(path: Path | None = None) -> dict[str, Any]:
    p = Path(path or MLS_FMB_PERCENTILES_PATH)
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def mls_fmb_prompt_thresholds(path: Path | None = None) -> dict[str, dict[str, float]]:
    """返回 train/val 各指标的 |·| 门槛（p25，两位有效舍入）。"""
    raw = load_mls_fmb_percentiles(path)
    abs_p25 = raw.get("abs_p25") or {}
    out: dict[str, dict[str, float]] = {}
    for split in ("train", "val"):
        src = abs_p25.get(split) or {}
        out[split] = {
            key: _round_threshold(float(src[key]))
            for key in ("mean_rho", "nw_t_rho", "nw_t_ls", "mls")
            if key in src and src[key] is not None
        }
    return out


def mls_fmb_thresholds_markdown(path: Path | None = None, *, label_col: str | None = None) -> str:
    """生成写入挖掘 prompt 的 MLS-FMB 门槛说明。"""
    th = mls_fmb_prompt_thresholds(path)
    if not th.get("train") or not th.get("val"):
        return (
            "**MLS-FMB**（`summary.mls_fmb`）：逐日十分组单调性 × 多空 IR；"
            "门槛文件缺失时仅作参考，详见 `docs/factor_metrics.md`。"
        )

    tr = th["train"]
    va = th["val"]
    meta = load_mls_fmb_percentiles(path)
    n = meta.get("n_factors", "?")
    cal_label = str(meta.get("label_col") or "label_1d_open_to_open")
    rel = MLS_FMB_PERCENTILES_PATH.as_posix()
    session_label = label_col or cal_label
    label_note = ""
    if session_label != cal_label:
        label_note = (
            f"\n\n> **注意**：本次会话 label 为 `{session_label}`，下表门槛按 `{cal_label}` 校准，"
            "仅作**相对参考**；长持有 label 下 MLS 数值尺度可能不同，请结合 IC/ICIR 与十分组判断。"
        )

    return f"""**MLS-FMB**（`summary.mls_fmb`，详见 `docs/factor_metrics.md`）：逐日截面十分组单调性 ρ_t 与 Q10−Q1 多空 LS_t 的 FMB+NW 推断；**保留级候选**在 train/val 上还须（指标与 `ic` **同号**，负 IC 看绝对值）：

| 指标 | train（|·| ≥） | val（|·| ≥） |
|------|----------------|---------------|
| `mls_fmb.mean_rho` | {tr.get("mean_rho", "—")} | {va.get("mean_rho", "—")} |
| `mls_fmb.nw_t_rho` | {tr.get("nw_t_rho", "—")} | {va.get("nw_t_rho", "—")} |
| `mls_fmb.nw_t_ls` | {tr.get("nw_t_ls", "—")} | {va.get("nw_t_ls", "—")} |
| `mls_fmb.mls`（参考） | {tr.get("mls", "—")} | {va.get("mls", "—")} |

门槛来源：当前 factorzoo **{n}** 个已入库因子在 train（2019-01-01~2021-12-31）/ val（2022-01-01~2024-12-31）上的 **25% 分位**（`|·|`，`{cal_label}`；`{rel}`）。{label_note}"""
