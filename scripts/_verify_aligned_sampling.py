"""验证调仓节拍对齐采样（build_stacking_dataset sample_step_days / --sample-step-days）。

零磁盘副作用：合成面板 + 合成因子，不加载 CNE panel、不写 factor/panel 缓存
（宿主将串行跑三组对比，panel 缓存共享会冲突，本脚本绝不触碰缓存）。

验证点：
1) 行数比例：对齐采样行数 ≈ 日频行数 ÷ step（300 交易日 × 40 股 = 12000 → ÷5 = 2400）；
2) 截面完整性：同一截面日全部股票同时进样本，采样日 == 全量日频 [::5]；
3) 慢频因子（基本面 60 交易日才更新一次 + 发布期 NaN 缺口）在相邻截面日保持
   同一值（不再每日重复），且截面日落在 NaN 缺口时取'截面日前最近值'（ffill）；
4) 快频因子（日频动量秩）在相邻截面日取值不同（未把快因子错误降频）；
5) label 切片正确：截面日 label == 全量前向收益在当天取值；尾部无未来数据仍 NaN；
6) 默认关闭（sample_step_days=0）保持旧行为：行数 == 日频全量；
7) _ffill_along_time / _sampled_day_mask 单元语义；
8) 候选库 label 分布（只读 artifacts，印证 109 因子混合频率动机）。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from alphaagent.factor.stacking import (  # noqa: E402
    FactorEntry,
    build_stacking_dataset,
    forward_return_label,
    parse_label_days,
)
from alphaagent.factor.stacking.dataset import (  # noqa: E402
    _ffill_along_time,
    _sampled_day_mask,
)

N_DAYS, N_INST, STEP = 300, 40, 5
BLOCK = 60  # 慢频基本面 60 个交易日才更新一次


def make_panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    days = pd.bdate_range("2024-01-01", periods=N_DAYS)
    insts = [f"{i:06d}.SZ" for i in range(1, N_INST + 1)]
    idx = pd.MultiIndex.from_product([days, insts], names=["datetime", "instrument"])
    n = len(idx)
    shock = rng.normal(0, 0.002, (N_DAYS, N_INST))
    ret = np.empty_like(shock)
    ret[0] = shock[0]
    for t in range(1, N_DAYS):
        ret[t] = 0.3 * ret[t - 1] + shock[t]
    # C-order（日主序）：与 MultiIndex (datetime, instrument) 行序一致。
    # 注意不可用 order="F"（股票主序）——数值会与行错位，每股"时序"混入其他
    # 股票的数值，导致合成截面上 TS_RANK 等因子在部分日塌缩成常数。
    close = 10.0 * np.exp(np.cumsum(ret, axis=0)).ravel()
    df = pd.DataFrame(index=idx)
    df["adj_close"] = close
    df["adj_open"] = df["adj_close"].values * (1 + rng.normal(0, 5e-4, n))
    df["adj_high"] = np.maximum(df["adj_close"], df["adj_open"]) * 1.002
    df["adj_low"] = np.minimum(df["adj_close"], df["adj_open"]) * 0.998
    df["volume"] = rng.uniform(1e5, 1e7, n)
    df["float_cap"] = rng.uniform(1e8, 1e10, n)
    df["industry_sw_l1"] = (np.arange(n) // N_INST) % 4
    # 慢频基本面列：每 BLOCK 个交易日更新一次；每次更新后头 2 个交易日缺失
    # （模拟报告期数据"迟发布"），其余时间恒定 —— 缺失缺口由 ffill 顶替
    day_idx = np.arange(n) // N_INST
    inst_idx = np.arange(n) % N_INST
    block_b = day_idx // BLOCK
    base = rng.uniform(1, 10, size=(N_DAYS // BLOCK + 2, N_INST))
    funda = base[block_b, inst_idx]
    rel_day = day_idx % BLOCK
    funda = np.where(rel_day < 2, np.nan, funda)
    df["funda_total_assets"] = funda
    return df


def entries() -> list[FactorEntry]:
    return [
        FactorEntry(
            factor_id="f_fast", name="f_fast",
            # 注：TS_RANK 输出唯一值数 ≤ 窗口长度（无并列时恰为窗口长），w=10 只有
            # 10 个离散秩值，会触发 materialize_entries 的 low_discreteness 防御
            # （唯一值 <20 判定为"常数簇过密分箱塌缩"）被剔除；w=60 保证离散度达标，
            # 与生产因子（250/500 长窗 TS_RANK）口径一致。
            expr="TS_RANK($adj_close, 60)",
            library="candidate_technical", label_col="label_1d_open_to_open",
        ),
        FactorEntry(
            factor_id="f_slow", name="f_slow",
            expr="RANK($funda_total_assets)",
            library="candidate_fundamental", label_col="label_20d_close_to_close",
        ),
    ]


def main() -> None:
    panel = make_panel()
    full_rows = len(panel)
    full_days = pd.DatetimeIndex(panel.index.get_level_values("datetime")).unique()
    print(f"[panel] {full_rows:,} 行 = {N_DAYS} 交易日 × {N_INST} 股票")

    mining_end = full_days[N_DAYS - 20]  # = day 280
    # mining 窗口 = [mining_end - 12 个月, mining_end]；从 day ~20 开始，刻意避开
    # TS_RANK warm-up 头部（w=60 局部窗口退化日，截面全同值）。与生产口径一致：
    # 生产 mining 窗口从面板时间 12 个月后起算，天然排除因子 warm-up 区。

    # ── 解析 label_col（groups.parse_label_days 集成） ──
    assert parse_label_days("label_20d_close_to_close") == 20
    assert parse_label_days("label_1d_open_to_open") == 1
    assert parse_label_days(None) is None
    print("[1] parse_label_days：label_20d→20 / label_1d→1 / None→None ✔")

    # ── 默认关闭（旧行为） ──
    ds0 = build_stacking_dataset(
        panel, entries(), label_days=STEP, mining_end=mining_end,
        size_neutral=False, max_corr=0.99, sample_step_days=0,
    )
    assert ds0.panel is panel or ds0.panel.equals(panel)
    assert len(ds0.panel) == full_rows
    assert len(ds0.label) == full_rows
    assert ds0.sample_step_days == 0
    assert set(ds0.feature_names) == {"f_fast", "f_slow"}
    print(f"[2] 默认关闭：行数 == 日频全量 {len(ds0.panel):,} ✔（特征 {ds0.feature_names}）")

    # ── 对齐采样启用 ──
    ds5 = build_stacking_dataset(
        panel, entries(), label_days=STEP, mining_end=mining_end,
        size_neutral=False, max_corr=0.99, sample_step_days=STEP,
    )
    samp_days = pd.DatetimeIndex(ds5.panel.index.get_level_values("datetime")).unique()
    n_samp_day = len(samp_days)
    print(f"[3] 对齐采样：日频 {full_rows:,} 行 → 截面日 {len(ds5.panel):,} 行（{n_samp_day} 个截面日）")
    assert len(ds5.panel) == full_rows // STEP == N_DAYS * N_INST // STEP
    assert n_samp_day == N_DAYS // STEP == 60
    assert len(ds5.feature_matrix) == len(ds5.label) == len(ds5.panel)
    assert ds5.sample_step_days == STEP
    # 采样日 == 全量日频 [::5]（同一截面日全部股票同时进样本）
    assert (samp_days.to_numpy() == full_days[::STEP].to_numpy()).all()
    print(f"[3] 行数比例 = {full_rows}/{len(ds5.panel)} = {full_rows / len(ds5.panel):.0f}（≈ ÷{STEP}）✔ "
          f"采样日 == 全量[::{STEP}] ✔ 截面完整性 ✔")

    # ── 慢频因子：相邻截面日保持同一值；缺口日取截面日前最近值 ──
    slow_idx = ds5.feature_names.index("f_slow")
    fast_idx = ds5.feature_names.index("f_fast")
    slow = ds5.feature_matrix[:, slow_idx].reshape(n_samp_day, N_INST)  # 每截面日 × 股票
    fast = ds5.feature_matrix[:, fast_idx].reshape(n_samp_day, N_INST)
    # 每个 60 交易日块 = 12 个截面日；块内相邻截面日值必须相同（不再每日重复）
    block_changes = 0
    for b in range(n_samp_day - 1):
        col_diff = np.isfinite(slow[b + 1]) & np.isfinite(slow[b])
        if not np.allclose(slow[b + 1][col_diff], slow[b][col_diff]):
            block_changes += 1
    # 预期的变化点只发生在真 · 基本面更新处：day 60,120,180,240（=截面日 12,24,36,48）
    # 块边界后第一个截面日会取新块值，故"相邻截面日值不同"应恰好 4 处
    print(f"[4] 慢频因子：相邻截面日取值变化次数 = {block_changes}（预期 4 = 60/5 个块的更新边界）")
    assert block_changes == N_DAYS // BLOCK - 1 == 4, block_changes
    # 缺口语义：day 60（=截面日索引 12）落在发布期 NaN 缺口内，ffill 后 == 上一截面日值
    assert np.allclose(slow[12], slow[11]), "截面日在 NaN 缺口应取'截面日前最近值'（ffill）"
    # 快频因子：相邻截面日取值必须不同（快因子未被错误降频）
    fast_changes = sum(
        1 for b in range(n_samp_day - 1)
        if np.isfinite(fast[b + 1]).any() and not np.allclose(fast[b + 1], fast[b], equal_nan=True)
    )
    assert fast_changes > n_samp_day * 0.5
    print(f"[4] 快频因子相邻截面日变化 {fast_changes}/{n_samp_day - 1}（>50% 变化 ✔ 未误降频）")

    # ── label：全量日频前向收益在截面日切片 ──
    full_label = forward_return_label(panel, STEP)
    mask = _sampled_day_mask(panel, STEP)
    assert np.array_equal(ds5.label, full_label[mask], equal_nan=True)
    # 尾部无未来数据：最后一个截面日（day 295）位于最后 6 个交易日 → label 全 NaN
    tail_rows = ds5.label.reshape(n_samp_day, N_INST)
    assert np.isnan(tail_rows[-1]).all()
    assert not np.isnan(tail_rows[-7]).all()
    print("[5] label：截面日 label == 全量前向收益切片 ✔；尾部无未来数据保持 NaN ✔")

    # ── _ffill_along_time 单元语义 ──
    fday = pd.bdate_range("2024-01-01", periods=6)
    fidx = pd.MultiIndex.from_product([fday, ["a", "b"]], names=["datetime", "instrument"])
    raw = np.array([
        1.0, 1.0,      # day0: a=1, b=1
        np.nan, np.nan,  # day1: 双双缺失
        5.0, 5.0,      # day2
        10.0, 10.0,    # day3
        np.nan, 10.0,  # day4: a 缺失, b=10
        10.0, 10.0,    # day5
    ])
    ff = _ffill_along_time(raw, pd.DataFrame(index=fidx))
    exp = np.array([1, 1, 1, 1, 5, 5, 10, 10, 10, 10, 10, 10], dtype=np.float32)
    assert np.array_equal(ff, exp, equal_nan=True), (ff, exp)
    # 前导缺失不回填（无前视）
    raw2 = np.array([np.nan, 1.0, np.nan, 2.0, 3.0, 3.0, np.nan, 4.0, 5.0, 5.0, np.nan, 6.0])
    ff2 = _ffill_along_time(raw2, pd.DataFrame(index=fidx))
    assert np.isnan(ff2[0]) and ff2[1] == 1.0 and np.isnan(ff2[2]) and ff2[3] == 2.0
    print("[6] _ffill_along_time：按股票前向填充 ✔、前导 NaN 不回填 ✔")
    sm = _sampled_day_mask(pd.DataFrame(index=fidx), 2)
    assert list(np.flatnonzero(sm)) == [0, 1, 4, 5, 8, 9]
    print("[7] _sampled_day_mask：step=2 → 取 day0/2/4 整截面 ✔")

    # ── 候选库 label 分布（只读；印证 109 因子混合频率动机） ──
    reg_path = ROOT / "artifacts" / "alphaagent" / "factorzoo" / "candidate_main" / "mining_candidate_registry.json"
    if reg_path.is_file():
        import collections
        import json

        reg = json.loads(reg_path.read_text(encoding="utf-8"))
        items = reg.values() if isinstance(reg, dict) else reg
        c = collections.Counter(
            (it.get("metrics") or {}).get("label_col") for it in items if isinstance(it, dict)
        )
        print(f"[8] 候选库 {len(items)} 个因子 label_col 分布：{dict(c)}")
        assert sum(c.values()) == 109
    else:
        print("[8] 候选库 registry 缺失，跳过（不影响本脚本其他断言）")

    # ── import 冒烟（train_ml_composite 模块级加载，不执行 main） ──
    import runpy

    runpy.run_path(str(ROOT / "scripts" / "train_ml_composite.py"), run_name="verify_smoke")
    print("[9] train_ml_composite.py 模块级 import 冒烟 ✔（main 未执行）")

    print("\n=== 全部断言通过 ===")


if __name__ == "__main__":
    main()