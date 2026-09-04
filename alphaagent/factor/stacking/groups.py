"""面感知分组与组合（facet-aware stacking）：候选因子按数据源组分层合成。

泛化契约（N 组，代码不按组名写死任何行为）：
- 组键派生：因子 facets → ``facet_groups()``（expressions.FACET_GROUPS 现成映射），
  跨组融合因子归主组（按 FACET_DEFS 稳定序取第一个组）；无面归属 → UNFACETED_GROUP。
- 组 horizon：成员已验证标签期（label_col 解析天数）众数；FacetGroupPolicy 可按组覆盖。
- 组分数：组内成员（已截面 transform）nanmean，逐行有值即算；
  行覆盖率 < min_coverage 的组当期缺席（组集合动态）。
- 组间权重：equal / icir（时间衰减）/ ridge_nn（非负 Ridge），全部只在 mining 窗口
  学习；组分数先逐日 zscore 再加权，某行缺失部分组时按可用组权重重归一。
- 退化性质：N=1 等价单组组合，N=2 即双组快慢方案；空组/单成员组均合法。
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from .dataset import daily_spearman_ic

UNFACETED_GROUP = "未分面"

_BLEND_METHODS = ("equal", "icir", "ridge_nn")


@dataclass(frozen=True)
class FacetGroupPolicy:
    """面组策略：默认全部从数据派生，这里只存覆盖项（新组零配置生效）。"""

    label_days: dict[str, int] = field(default_factory=dict)  # 组 → 标签期覆盖
    blend_label_days: int | None = None                       # 组间合成标签期覆盖
    min_coverage: float = 0.30                                # 组分数最小行覆盖率
    icir_halflife_days: int = 120                             # icir 权重时间衰减半衰期
    ridge_alpha: float = 10.0                                 # ridge_nn 正则强度


def parse_label_days(label_col: str | None) -> int | None:
    """label_col → 标签期天数：label_10d_close_to_close → 10；无法解析返回 None。"""
    if not label_col:
        return None
    m = re.search(r"label_(\d+)d", str(label_col))
    return int(m.group(1)) if m else None


def derive_group(facets: Sequence[str] | set[str] | None) -> str:
    """因子主组：facet_groups() 按 FACET_GROUPS 插入序取第一个；无面 → 未分面桶。

    延迟导入 expressions（memory/__init__ 会拉 ResearchMemoryStore，
    避免给 stacking 的模块加载链路增重）。
    """
    from alphaagent.factor.mining.memory.expressions import FACET_GROUPS, facet_groups

    groups = facet_groups(set(facets or ()))
    if not groups:
        return UNFACETED_GROUP
    for group in FACET_GROUPS:  # dict 插入序 = 稳定序
        if group in groups:
            return group
    return sorted(groups)[0]  # 防御：FACET_GROUPS 未来扩展漏同步时仍确定性


def assign_groups(members: Sequence) -> dict[str, list]:
    """按主组分组成员（元素需带 facets/expr 属性，如 FactorEntry）。

    facets 为空时按表达式识别面（expr_facets 兜底），仍无面归属进 UNFACETED_GROUP。
    """
    from alphaagent.factor.mining.memory.expressions import expr_facets

    out: dict[str, list] = {}
    for entry in members:
        facets = tuple(getattr(entry, "facets", ()) or ())
        if not facets:
            facets = expr_facets(getattr(entry, "expr", ""))
        out.setdefault(derive_group(facets), []).append(entry)
    return out


def group_horizon(
    member_label_days: Sequence[int | None], group: str, policy: FacetGroupPolicy
) -> int | None:
    """组标签期：policy 按组覆盖 > 成员众数（平局取最小众数）；无从派生 → None。"""
    override = policy.label_days.get(group)
    if override:
        return int(override)
    vals = [int(v) for v in member_label_days if v]
    if not vals:
        return None
    return int(pd.Series(vals).mode().iloc[0])


def derive_blend_horizon(
    group_horizons: Mapping[str, int | None], policy: FacetGroupPolicy | None = None
) -> int | None:
    """组间合成标签期：policy 覆盖 > 各组 horizon 中位数（偶数取更低中位）→ 全 None 则 None。"""
    policy = policy or FacetGroupPolicy()
    if policy.blend_label_days:
        return int(policy.blend_label_days)
    vals = sorted(int(v) for v in group_horizons.values() if v)
    if not vals:
        return None
    return vals[(len(vals) - 1) // 2]


def group_scores(
    assigned: Mapping[str, Sequence[np.ndarray]],
    *,
    min_coverage: float = 0.30,
) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    """组分数 = 组内成员（已截面 transform、行序对齐）nanmean。

    返回 ({组: float32 分数}, {组: 元信息})。覆盖率 < min_coverage 的组不进
    分数表（当期缺席），元信息标注 absent 原因。
    """
    scores: dict[str, np.ndarray] = {}
    meta: dict[str, dict] = {}
    for group in sorted(assigned):
        arrs = [np.asarray(a, dtype=np.float32) for a in assigned[group]]
        info: dict = {"members": len(arrs)}
        if not arrs:
            info["absent"] = "no_members"
            meta[group] = info
            continue
        stack = np.vstack(arrs)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # 全 NaN 行
            score = np.nanmean(stack, axis=0).astype(np.float32)
        score[~np.isfinite(score)] = np.nan
        coverage = float(np.isfinite(score).mean())
        info["coverage"] = round(coverage, 4)
        if coverage < min_coverage:
            info["absent"] = f"coverage<{min_coverage:.2f}"
        else:
            scores[group] = score
        meta[group] = info
    return scores, meta


def daily_zscore(values: np.ndarray, dts: Sequence) -> np.ndarray:
    """逐日截面 rank pct → zscore（组分数合成前的统一尺度；NaN 保持 NaN）。"""
    df = pd.DataFrame({"v": np.asarray(values, dtype=np.float64), "d": np.asarray(dts)})
    df["r"] = df.groupby("d", sort=False)["v"].rank(pct=True)
    g = df.groupby("d", sort=False)["r"]
    mean = g.transform("mean")
    std = g.transform("std")
    z = (df["r"] - mean) / std.where(std > 1e-12)
    out = z.to_numpy(dtype=np.float32)
    out[~np.isfinite(out)] = np.nan
    return out


def _equal_weights(groups: list[str]) -> dict[str, float]:
    return {g: 1.0 / len(groups) for g in groups}


def blend_weights(
    group_scores_map: Mapping[str, np.ndarray],
    label: np.ndarray,
    dts: Sequence,
    *,
    mining_end: pd.Timestamp,
    method: str = "ridge_nn",
    policy: FacetGroupPolicy | None = None,
) -> tuple[dict[str, float], dict]:
    """组间权重，只在 mining_end 之前的行上学习。返回 ({组: 权重}, 诊断)。

    - equal: 1/N。
    - icir: 组分数（逐日 zscore）vs 标签的逐日 Spearman IC，按距 mining_end
      自然日半衰期加权出 ICIR，非负归一；全非正回退 equal。
    - ridge_nn: mining 窗口内组分数（逐日 zscore）对标签做非负 Ridge，系数归一；
      系数全零/样本不足回退 equal。
    """
    policy = policy or FacetGroupPolicy()
    groups = sorted(group_scores_map)
    if not groups:
        return {}, {"method": method, "fallback": "no_groups"}
    diag: dict = {"method": method, "groups": groups}
    m_end = pd.Timestamp(mining_end)
    dts_s = pd.Series(pd.to_datetime(pd.Series(dts)))
    mask = (dts_s <= m_end).to_numpy() & np.isfinite(label)
    if mask.sum() < 100:
        diag["fallback"] = f"mining_rows<{100}"
        return _equal_weights(groups), diag
    idx = np.flatnonzero(mask)

    if method == "equal":
        return _equal_weights(groups), diag

    if method == "icir":
        z = {g: daily_zscore(group_scores_map[g], dts) for g in groups}
        halflife = max(1, int(policy.icir_halflife_days))
        icir: dict[str, float] = {}
        for g in groups:
            ic = daily_spearman_ic(z[g][idx], label[idx], dts_s.iloc[idx])
            if not len(ic):
                icir[g] = 0.0
                continue
            ages = (m_end - pd.DatetimeIndex(ic.index)).days.to_numpy(dtype=np.float64)
            w_days = 0.5 ** (np.clip(ages, 0, None) / halflife)
            w_days = w_days / w_days.sum()
            vals = ic.to_numpy(dtype=np.float64)
            mean = float((vals * w_days).sum())
            std = float(np.sqrt((w_days * (vals - mean) ** 2).sum()))
            icir[g] = mean / std if std > 1e-12 else 0.0
        diag["icir"] = {g: round(v, 4) for g, v in icir.items()}
        pos = {g: max(v, 0.0) for g, v in icir.items()}
        total = sum(pos.values())
        if total <= 1e-12:
            diag["fallback"] = "all_icir_nonpositive"
            return _equal_weights(groups), diag
        return {g: pos[g] / total for g in groups}, diag

    if method == "ridge_nn":
        from sklearn.linear_model import Ridge

        z = {g: daily_zscore(group_scores_map[g], dts) for g in groups}
        S = np.column_stack([z[g] for g in groups])
        ok = mask & np.isfinite(S).all(axis=1)
        if ok.sum() < 100:
            diag["fallback"] = "finite_rows<100"
            return _equal_weights(groups), diag
        model = Ridge(alpha=policy.ridge_alpha, positive=True)
        model.fit(S[ok], label[ok])
        coef = np.clip(np.asarray(model.coef_, dtype=np.float64), 0.0, None)
        diag["coef"] = {g: round(float(c), 5) for g, c in zip(groups, coef)}
        total = float(coef.sum())
        if total <= 1e-12:
            diag["fallback"] = "zero_coef"
            return _equal_weights(groups), diag
        return {g: float(c) / total for g, c in zip(groups, coef)}, diag

    raise ValueError(f"unknown blend method: {method}（可选 {_BLEND_METHODS}）")


def apply_weights(
    weights: Mapping[str, float],
    group_scores_map: Mapping[str, np.ndarray],
    dts: Sequence,
) -> np.ndarray:
    """组分数（逐日 zscore）按权重的 NaN 感知合成。

    某行缺失部分组时按可用组权重重归一（如缺基本面组的股票仍有行情组分数）；
    全部缺失 → NaN。
    """
    if not weights:
        raise ValueError("empty weights")
    z = {
        g: daily_zscore(np.asarray(group_scores_map[g]), dts)
        for g in weights
        if g in group_scores_map
    }
    if not z:
        raise ValueError("no group scores matched weights")
    num: np.ndarray | None = None
    den: np.ndarray | None = None
    for g, w in weights.items():
        arr = z.get(g)
        if arr is None or w <= 0:
            continue
        finite = np.isfinite(arr)
        contrib = np.where(finite, arr * float(w), 0.0)
        wv = np.where(finite, float(w), 0.0)
        num = contrib if num is None else num + contrib
        den = wv if den is None else den + wv
    if num is None:
        raise ValueError("no positive-weight group scores")
    safe_den = np.where(den > 1e-12, den, 1.0)
    out = np.where(den > 1e-12, num / safe_den, np.nan).astype(np.float32)
    return out
