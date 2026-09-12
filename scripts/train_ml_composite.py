"""ML 组合因子训练：AlphaAgent 因子池 → walk-forward Ridge/LightGBM → 组合分数。

时间隔离契约（防幸存者偏差传导）：
- ``mining_end``（默认 = 所选因子库里最晚的 created_at）之前的窗口只用于
  衰减对照表，绝不进入训练；
- walk-forward 每折 train 与 OOS 之间留 purge gap；
- engine_gate 在 OOS 段（首折 OOS 起）裁决，非训练段。

产出：
- artifacts/alphaagent/stacking/<run_id>/model.joblib（特征清单+表达式+参数）
- artifacts/alphaagent/stacking/<run_id>/report.json（逐折指标/衰减表/gate 结论）
- data/stock/pred_demo.parquet（date,code,score 长表 → 回测/前端 pred 因子通道）

用法：
  python scripts/train_ml_composite.py --model both --train-months 18 --step-months 6
  python scripts/train_ml_composite.py --scheme hrp --no-gate --no-write-pred  # 简单加权：不拟合模型
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Windows 中文控制台默认 GBK 编码，numpy 负号格式化成 \u2212（MINUS SIGN）会
# 触发 UnicodeEncodeError 中断整条管线（2026-09-12 Phase 0 实跑命中）。显式
# 用 UTF-8 + 兜底替换，保证 report 打印在任何 locale 都不崩。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, ValueError):  # 非交互 stdout（重定向对象无 reconfigure）时忽略
    pass

from alphaagent.data.adapters.cnequity import load_panel_from_cne  # noqa: E402
from alphaagent.factor.cache import FactorValueCache  # noqa: E402
from alphaagent.factor.stacking import (  # noqa: E402
    build_stacking_dataset,
    collect_factor_entries,
    daily_spearman_ic,
    decay_table,
    fit_predict_walkforward,
    walk_forward_splits,
)
from alphaagent.factor.mining.research_spec import default_research_spec  # noqa: E402
from alphaagent.factor.stacking.dataset import _to_utc_naive  # noqa: E402
from core import trading_config  # noqa: E402

SCHEME_LABELS = {
    "ml": "ML 学习加权（Ridge+LGBM）",
    "equal": "等权 1/N",
    "icir": "ICIR 加权",
    "hrp": "HRP 加权",
}


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--modes", nargs="+", default=["technical", "fundamental"],
                    choices=["technical", "fundamental"], help="纳入哪些因子库模式")
    ap.add_argument("--no-candidate", action="store_true", help="只用正式库因子")
    ap.add_argument("--label-days", type=int, default=5, help="前向收益持有天数（对齐调仓频率）")
    ap.add_argument("--model", default="both", choices=["ridge", "lgbm", "both"],
                    help="仅 --scheme ml 生效：拟合的 ML 模型")
    ap.add_argument("--scheme", default="ml", choices=["ml", "equal", "icir", "hrp"],
                    help="组合方法：ml=学习加权（Ridge/LGBM，默认）；equal=等权 1/N；"
                         "icir=ICIR 加权；hrp=HRP 加权。后三者不拟合任何模型，"
                         "符号/权重只由各折 train 段按规则滚动估计，其余链路"
                         "（OOS 报告/smooth/decay/gate/pred）与 ML 模式一致")
    ap.add_argument("--mining-end", default="auto", help="时间隔离边界（YYYY-MM-DD 或 auto=因子库最晚 created_at）")
    ap.add_argument("--end", default=None, help="数据截止日（默认取数据源最新交易日）")
    ap.add_argument("--decay-months", type=int, default=12, help="衰减对照表的 mining 窗口长度")
    ap.add_argument("--train-months", type=int, default=18)
    ap.add_argument("--step-months", type=int, default=6, help="OOS 折长")
    ap.add_argument("--purge-days", type=int, default=5, help="train/OOS 之间 purge gap 交易日数")
    ap.add_argument("--warmup-days", type=int, default=250, help="panel 起点提前量（因子窗口预热）")
    ap.add_argument("--max-corr", type=float, default=0.6, help="跨库因子冗余剔除阈值（默认 0.6，与候选池去重口径一致；相关 0.8 的 20 个因子 ≈ 2-3 个独立信号，放进来只会稀释权重、放大过拟合面）")
    ap.add_argument("--size-neutral/--no-size-neutral", dest="size_neutral", default=True)
    ap.add_argument("--no-gate", action="store_true", help="跳过 engine_gate 回测裁决")
    ap.add_argument(
        "--gate-selection-pct", type=float, default=trading_config.ML_GATE_SELECTION_PCT,
        help="engine_gate 组合选股宽度（动态百分比口径）。默认取 trading_config.ML_GATE_SELECTION_PCT"
             f"（{trading_config.ML_GATE_SELECTION_PCT}，≈15 只）——Phase 2 实证的 gateway 全绿宽度，"
             "而非全局回测 SELECTION_PCT（0.004，excess_sharpe 差 0.02）或单因子门禁 "
             f"GATE_SELECTION_PCT（{trading_config.GATE_SELECTION_PCT}，≈5 只）。"
             "Phase 1 可交易性改造：基线死穴（3.4 只持仓/62%%换手/60 次现金不足拒单）"
             "正是错误复用单因子极窄选股造成。阈值本身不受影响。")
    ap.add_argument(
        "--gate-top-n", type=int, default=None,
        help="engine_gate 改用固定 Top-N 选股（selection_mode=top_n）；缺省保持动态百分比"
             "（top_pct，用 --gate-selection-pct）。传此参数后 --gate-selection-pct 失效。")
    ap.add_argument(
        "--gate-capital", type=float, default=None,
        help="engine_gate 组合门禁资金。缺省按所选宽度等比放大 GATE_CAPITAL（10 万）→ "
             "每只预算密度与单因子门禁（GATE_CAPITAL/GATE_TOP_N=2 万/只）保持一致："
             "top_pct 时 capital = GATE_CAPITAL × selection_pct/GATE_SELECTION_PCT"
             f"（{trading_config.ML_GATE_SELECTION_PCT} 默认宽度 → 30 万）；"
             "top_n 时 = GATE_CAPITAL × top_n/GATE_TOP_N。显式传值则直接用。"
             "Phase 2a：修复组合宽度 × 10 万资金 = 5000 元/只系统性买不起一手"
             "导致的现金不足拒单（0.003 联动 30 万后投入比 97.5%%、gate 全绿）。")
    ap.add_argument("--no-write-pred", action="store_true", help="不写 pred 通道文件")
    ap.add_argument("--subset-curve", action="store_true",
                    help="贡献排序累积子集曲线：按置换贡献降序取 Top-k 逐级重训（成本 ≈ 2n 次拟合），"
                         "输出'子集规模 vs OOS IC'曲线，定位边际收益归零的最优规模")
    ap.add_argument("--score-smooth", type=int, default=0,
                    help="组合分数尾随线性 WMA 平滑窗（按股票、按交易日）。0=自动取 label_days。"
                         "组合分数此前是全链路唯一未平滑的分数——挖掘侧单因子靠 WMA 把日换手压半才过 gate"
                         "（ovdiv：WMA20 后 1.07→0.50）；1=关闭平滑")
    ap.add_argument("--include-factors", nargs="*", default=None,
                    help="因子白名单（按 factor_name 精确匹配）：只训练名单内的因子；缺省=全部")
    ap.add_argument("--recommend-k", type=int, default=0,
                    help=">0 时进入 mRMR 推荐模式：构建数据集后按 '强+互补' 输出 Top-k 推荐清单"
                         "（不训练模型），写 recommend.json 并退出。因子值已进磁盘缓存，随后的完整训练可复用")
    ap.add_argument("--multi-path-shifts", default="",
                    help="多路径对照：逗号分隔的折边界平移月数（如 0,2,4 → 额外 2、4 两条路径）。"
                         "每条平移独立重训一遍同模型，输出 OOS IC/ICIR/Sharpe/回撤 的路径分布——"
                         "单条切法上的表现可能只是那条边界的运气。成本 ≈ 额外路径数 × 一次完整拟合")
    ap.add_argument("--out-dir", default=None, help="输出目录（默认 artifacts/alphaagent/stacking/<时间戳>）；后端托管时传确定性路径")
    ap.add_argument("--isolation", default="strict", choices=["strict", "holdout"],
                    help="strict：walk-forward 仅用挖掘期后干净段（fold 少但指标可信）；"
                         "holdout：全历史训练、挖掘期后干净段整体作留出测试（推荐）")
    ap.add_argument("--pred-out", default=None, help="pred 分数输出路径（默认 data/stock/pred_demo.parquet）")
    return ap.parse_args()


def main() -> None:
    args = _parse_args()
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else ROOT / "artifacts" / "alphaagent" / "stacking" / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # ① 因子枚举
    entries = collect_factor_entries(
        modes=tuple(args.modes), include_candidate=not args.no_candidate, include_production=True
    )
    if args.include_factors:
        wanted = {str(n).strip() for n in args.include_factors if str(n).strip()}
        entries = [e for e in entries if e.name in wanted]
        print(f"因子白名单：请求 {len(wanted)} 个，命中 {len(entries)} 个")
    print(f"因子枚举：{len(entries)} 个（去重后）")
    if len(entries) < 2:
        print("因子数不足（<2），无法组合。请先挖掘入库更多因子（或核对白名单拼写）。")
        sys.exit(1)

    # ② 时间隔离边界：挖掘循环实际评估的右端（registry 的 eval_end），
    #    回退到入库时间。注意 eval_end << ingested_at 是常态：因子 2026-08
    #    入库，但其评估/反馈窗口只到 val_end（如 2025-12-31），其后数据未被
    #    挖掘循环消费，仍是干净的组合训练数据。
    if args.mining_end == "auto":
        eval_ends = [_to_utc_naive(e.eval_end) for e in entries]
        eval_ends = [c for c in eval_ends if c is not None]
        if eval_ends:
            mining_end = max(eval_ends)
            print("mining_end 依据：registry eval_end（挖掘循环真实评估边界）")
        else:
            created = [_to_utc_naive(e.created_at) for e in entries]
            created = [c for c in created if c is not None]
            if not created:
                print("因子库无入库时间/评估边界，无法 auto 推断 mining-end，请显式传 --mining-end")
                sys.exit(1)
            mining_end = max(created)
            print("警告：registry 无 eval_end，退回入库时间推断（偏保守）")
    else:
        mining_end = pd.Timestamp(args.mining_end)
    if mining_end.tzinfo is not None:
        mining_end = mining_end.tz_localize(None)

    # ── 数据可用右端兜底（统一配置中心）──────────────────────────────
    # mining_end 与 end 均不得超过数据源实际最新交易日，否则组合训练会静默
    # 覆盖不存在的数据（历史教训：硬编码日期超过数据截至日）。动态解析
    # resolve_test_end() = 数据源最新交易日，作为右端硬上限。
    from alphaagent.factor.window_config import resolve_test_end

    data_latest = pd.Timestamp(resolve_test_end())
    end = pd.Timestamp(args.end) if args.end else data_latest
    if end > data_latest:
        print(f"[warn] --end={end.date()} 超过数据源最新交易日 {data_latest.date()}，收敛到数据右端")
        end = data_latest
    if mining_end > end:
        print(f"[warn] mining_end={mining_end.date()} 超过数据右端 {end.date()}，收敛到数据右端")
        mining_end = end
    panel_start = mining_end - pd.DateOffset(months=args.decay_months) - pd.DateOffset(days=args.warmup_days)
    print(f"时间隔离边界 mining_end={mining_end.date()}；panel 区间 [{panel_start.date()} ~ {end.date()}]")

    # ③ panel 加载（磁盘缓存命中则秒级）
    print("加载 CNE panel …")
    panel = load_panel_from_cne(start=panel_start, end=end, include_fundamentals=True)
    print(f"panel: {panel.shape[0]} 行 × {panel.shape[1]} 列")

    # ④ 数据集构建（物化 + 预处理 + 冗余过滤）
    cache = FactorValueCache()
    dataset = build_stacking_dataset(
        panel,
        entries,
        label_days=args.label_days,
        mining_end=mining_end,
        size_neutral=args.size_neutral,
        max_corr=args.max_corr,
        cache=cache,
        decay_months=args.decay_months,
        progress=lambda msg: print(" ", msg, flush=True),
    )
    print(f"有效特征 {len(dataset.feature_names)} 个；剔除 {len(dataset.dropped)} 个")
    for d in dataset.dropped:
        print(f"  - drop {d['name']} ({d['library']}): {d['reason']}")

    # ④b mRMR 推荐模式（B 族）：不训练，只输出"强+互补"推荐清单供前端一键勾选
    if args.recommend_k > 0:
        from alphaagent.factor.stacking.model import mrmr_rank_features

        if len(dataset.feature_names) < 2:
            print("有效因子不足（<2），无法推荐组合。")
            sys.exit(1)
        dts = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
        date_series = pd.Series(dts)
        rec_start = mining_end - pd.DateOffset(months=args.decay_months)
        print(f"mRMR 推荐（mining 窗口 [{rec_start.date()} ~ {mining_end.date()}]，Top-{args.recommend_k}）…")
        ranking = mrmr_rank_features(
            dataset.feature_matrix, dataset.label, date_series, dataset.feature_names,
            window_start=rec_start, window_end=mining_end, k=args.recommend_k, beta=0.7,
        )
        meta_by_name = {e.name: e for e in dataset.entries}
        rows = []
        for r in ranking:
            entry = meta_by_name.get(r["name"])
            rows.append({**r, "facets": list(entry.facets) if entry else [],
                         "library": entry.library if entry else ""})
        for r in rows:
            print(f"  #{r['rank']} {r['name']}: |IC|={r['ic_mean']} 与已选冗余={r['redundancy']} "
                  f"score={r['score']}")
        rec = {
            "k": args.recommend_k,
            "beta": 0.7,
            "window": f"[{rec_start.date()} ~ {mining_end.date()}]",
            "n_features": len(dataset.feature_names),
            "ranking": rows,
            "note": "mRMR 贪心：质量 = mining 窗口日均 |IC|；冗余 = 与已选因子最大 |corr|；"
                    "得分 = 质量 × (1 − 0.7 × 冗余)。照单勾选只是起点，是否入选仍由训练后的 "
                    "OOS 与置换贡献说话。",
        }
        (out_dir / "recommend.json").write_text(
            json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8"
        )
        print(f"推荐已写入 {out_dir / 'recommend.json'}（因子值已进磁盘缓存，直接点训练会命中缓存）")
        sys.exit(0)

    # ⑤ walk-forward 训练
    #   strict：train 从 mining_end 起步（只用挖掘期后干净段，fold 少）；
    #   holdout：train 覆盖全历史至 mining_end（拟合挖掘期数据不用于报告），
    #            mining_end 之后的干净段整体作为留出测试集（指标诚实，推荐）。
    dts = pd.DatetimeIndex(panel.index.get_level_values("datetime"))
    date_series = pd.Series(dts)
    holdout_mode = args.isolation == "holdout"
    wf_train_start = panel_start if holdout_mode else mining_end
    base_months = max(3, int((mining_end - panel_start).days / 30.44))
    if holdout_mode:
        # 渐进加窗：第一折的 OOS 必须完全落在 mining_end 之后的干净段
        #（月度算术 + purge 会让边界回退，最多多给 3 个月窗口）
        folds = []
        first_oos = None
        for bump in (0, 1, 2, 3):
            folds = walk_forward_splits(
                dts,
                train_start=wf_train_start,
                train_months=base_months + bump,
                step_months=args.step_months,
                purge_days=max(args.purge_days, args.label_days),
            )
            first_oos = pd.Timestamp(folds[0].oos_dates.min()) if folds else None
            if first_oos is not None and first_oos >= mining_end:
                break
        if folds and first_oos is not None and first_oos >= mining_end:
            # 只保留 OOS 完全在干净段的折（训练含挖掘期数据用于拟合，不影响
            # 各折自身 OOS 的诚实性——purge 保证测试段不在训练集内）
            folds = [f for f in folds if pd.Timestamp(f.oos_dates.min()) >= mining_end]
            print(
                f"holdout 模式：训练 [{panel_start.date()} ~ {pd.Timestamp(folds[0].train_dates.max()).date()}]，"
                f"留出测试 [{pd.Timestamp(folds[0].oos_dates.min()).date()} ~ {pd.Timestamp(folds[-1].oos_dates.max()).date()}]，"
                f"共 {len(folds)} 折（测试段从未进入任何一折的训练集）"
            )
        else:
            folds = []
    else:
        folds = walk_forward_splits(
            dts,
            train_start=mining_end,
            train_months=args.train_months,
            step_months=args.step_months,
            purge_days=max(args.purge_days, args.label_days),
        )
    print(f"walk-forward 折数：{len(folds)}")
    if not folds:
        latest = max((_to_utc_naive(e.created_at) for e in entries if e.created_at), default=None)
        print(
            f"mining_end({mining_end.date()}) 之后样本不足以构成一折"
            f"（需 ≥ {args.train_months} 个月）。"
        )
        if latest is not None and latest >= end - pd.DateOffset(months=args.train_months + args.step_months):
            print(
                "原因：全部因子都在近期入库（最晚 {:%Y-%m-%d}），挖掘窗口之后没有"
                "干净的未来数据可用于组合训练。两条出路：\n"
                "  1. 等待数据积累后重跑（推荐，时间隔离才有意义）；\n"
                "  2. 显式传更早的 --mining-end（如 {} -前推训练期），接受"
                "组合训练与挖掘期重叠的 in-sample 风险——报告会标注该风险，"
                "结论仅作框架演示，不能作为入库依据。".format(latest, end.date())
            )
        sys.exit(1)

    is_simple_scheme = args.scheme != "ml"
    kinds = [] if is_simple_scheme else (["ridge", "lgbm"] if args.model == "both" else [args.model])
    explicit_mining_end = args.mining_end != "auto"
    # 隔离有效性的真实判据：mining_end 是否 ≥ 挖掘循环实际评估右端（eval_end）
    latest_eval_end = max(
        (c for c in (_to_utc_naive(e.eval_end) for e in entries) if c is not None),
        default=None,
    )
    time_isolation = (
        "holdout（训练含挖掘期数据用于拟合；报告指标全部来自挖掘后干净留出段）"
        if holdout_mode
        else ("ok" if (not explicit_mining_end or (latest_eval_end is not None and mining_end >= latest_eval_end))
        else "violated_explicit_override（组合训练期与挖掘期重叠，OOS 结论不可作为入库依据）")
    )
    model_outputs: dict[str, np.ndarray] = {}
    fold_reports: dict[str, list] = {}
    feature_weights: dict[str, list] = {}
    feature_contribution: dict[str, list] = {}
    for kind in kinds:
        print(f"训练 {kind} …")
        pred, report, feat_w, feat_c = fit_predict_walkforward(
            dataset.feature_matrix, dataset.label, date_series, folds, kind=kind,
            feature_names=dataset.feature_names, label_horizon=args.label_days,
        )
        model_outputs[kind] = pred
        fold_reports[kind] = report
        if feat_w:
            feature_weights[kind] = feat_w
            top = "、".join(f"{x['name']}({x['weight']:.1%})" for x in feat_w[:5])
            print(f"  权重 Top5: {top}")
        if feat_c:
            feature_contribution[kind] = feat_c
            top_c = "、".join(f"{x['name']}({x['ic_drop']:+.4f})" for x in feat_c[:5])
            print(f"  置换贡献 Top5: {top_c}")
            dd_drags = [x for x in feat_c if x.get("dd_impact") is not None]
            if dd_drags:
                worst = min(dd_drags, key=lambda x: x["dd_impact"])
                if worst["dd_impact"] < 0:
                    print(f"  回撤拖累最重: {worst['name']}（打乱后回撤收窄 "
                          f"{abs(worst['dd_impact']) * 100:.1f}%，剔除候选）")
        for r in report:
            ic = r.get("ic_mean")
            print(f"  OOS {r['oos_start']}~{r['oos_end']}: n_train={r['n_train']} "
                  f"IC={ic if ic is None else round(ic, 4)} "
                  f"{'SKIP' if r.get('skipped') else ''}")

    # 组合分数：多模型 OOS 预测取平均（折内无 in-sample 污染）
    if not is_simple_scheme:
        usable = [model_outputs[k] for k in kinds if np.isfinite(model_outputs[k]).any()]
        if not usable:
            print("所有模型折均被跳过（样本不足），无组合分数可产出。")
            sys.exit(1)
        stacked = np.nanmean(np.column_stack(usable), axis=1).astype(np.float32)
    else:
        # 简单加权组合：不拟合模型。三种方案的符号/权重由各折 train 段按规则
        # 滚动估计（fit_scheme_compare_scores），此处直接取用户选定的方案分数。
        from alphaagent.factor.stacking.model import fit_scheme_compare_scores

        print(f"简单加权组合（{SCHEME_LABELS[args.scheme]}）："
              f"权重由各折 train 段滚动估计，无模型拟合 …")
        _scheme_all = fit_scheme_compare_scores(
            dataset.feature_matrix, dataset.label, date_series, folds
        )
        stacked = _scheme_all[args.scheme].copy()
        fold_reports = {}
        feature_weights = {}
        feature_contribution = {}

    # ⑤b 组合分数平滑：尾随 WMA（gate/报告/pred 全部消费平滑后分数）
    smooth_n = args.score_smooth if args.score_smooth > 0 else args.label_days
    if smooth_n > 1:
        stacked = _wma_smooth_scores(stacked, panel, smooth_n)
        print(f"组合分数平滑：尾随线性 WMA-{smooth_n}（0=自动取持有天数）")
    else:
        print("组合分数平滑：关闭（--score-smooth 1）")
    score_smooth_used = smooth_n

    # ⑤c 不挑全上对照（D 族，仅 ML 模式）：等权/ICIR/HRP 简单投票 vs 学习加权
    #     权重只由各折 train 段估计（与模型拟合同等信息量）；方案分数同样平滑
    #     后在同一 OOS 行集上与 stacked 对照。
    scheme_compare = None
    if not is_simple_scheme:
        from alphaagent.factor.stacking.model import fit_scheme_compare_scores, scheme_compare_report

        print("不挑全上对照（等权 / ICIR / HRP 简单投票）…")
        scheme_scores = fit_scheme_compare_scores(dataset.feature_matrix, dataset.label, date_series, folds)
        if smooth_n > 1:
            for _sn in scheme_scores:
                scheme_scores[_sn] = _wma_smooth_scores(scheme_scores[_sn], panel, smooth_n)
        scheme_compare = scheme_compare_report(
            scheme_scores, stacked, dataset.label, date_series,
            first_oos=first_oos, label_horizon=args.label_days,
        )
    for _sc in (scheme_compare or {}).get("schemes", []):
        if _sc["scheme"] == "stacked":
            print(f"  [当前组合] OOS IC={_fmt(_sc.get('ic_mean'))} ICIR={_fmt(_sc.get('ic_ir'))} "
                  f"Sharpe={_sc.get('oos_sharpe')} 回撤={_sc.get('oos_max_drawdown')}")
        else:
            gap = _sc.get("ic_gap")
            print(f"  [{_sc['label']}] OOS IC={_fmt(_sc.get('ic_mean'))} ICIR={_fmt(_sc.get('ic_ir'))} "
                  f"Sharpe={_sc.get('oos_sharpe')} 回撤={_sc.get('oos_max_drawdown')} "
                  f"IC−组合={_fmt(gap) if gap is not None else 'None'}")

    # ⑥ 衰减对照表（幸存者偏差量化）
    print("衰减对照表（mining 窗口 IC vs OOS IC）…")
    materialized_for_decay = []
    from alphaagent.factor.ingest import materialize_factor

    for entry in dataset.entries:
        try:
            values = np.asarray(materialize_factor(entry.expr, panel, cache=cache).values, dtype=np.float32)
            materialized_for_decay.append((entry, values))
        except Exception:
            continue
    decay = decay_table(
        materialized_for_decay, panel, dataset.label,
        mining_end=mining_end, decay_months=args.decay_months,
    )
    for row in decay:
        print(f"  {row['name']:32s} mining={_fmt(row['ic_mining'])} oos={_fmt(row['ic_oos'])} "
              f"保留比={_fmt(row['decay_ratio'])}")

    # ⑦ engine_gate 裁决（OOS 段，周调仓口径）
    gate_result = None
    first_oos = folds[0].oos_dates.min()
    if not args.no_gate:
        from alphaagent.factor.mining.engine_gate import run_engine_gate

        policy = default_research_spec(args.modes[0])["delivery_policy"]["production"]["engine_gate"]
        # Phase 1 可交易性改造：组合引擎门禁的选股宽度改用组合口径（top 0.4%≈20 只），
        # 而非单因子门禁的 GATE_SELECTION_PCT(0.001≈5 只)。基线死穴——3.4 只持仓、
        # 62% 换手、60 次现金不足拒单、回撤 −41%——正是窄选股把合成分数逼进不可交易
        # 尾部的连锁反应。阈值（超额年化/夏普/回撤等）一律不动。
        if args.gate_top_n is not None:
            policy = {**policy, "selection_mode": "top_n", "top_n": args.gate_top_n}
        else:
            policy = {**policy, "selection_mode": "top_pct", "selection_pct": args.gate_selection_pct}
        # Phase 2a 资金联动：0.004 组合宽度 × 10 万 = 5000 元/只，买不起中高价股一手
        # → 455 次"现金不足/预算过小"。保持与单因子门禁相同的"2 万/只"预算密度，
        # 按宽度等比放大门禁资金（非作弊——20 只组合实盘本就需更大账户）。
        if args.gate_capital is not None:
            capital = float(args.gate_capital)
        elif policy["selection_mode"] == "top_n":
            capital = trading_config.GATE_CAPITAL * (float(policy["top_n"]) / trading_config.GATE_TOP_N)
        else:
            capital = trading_config.GATE_CAPITAL * (float(policy["selection_pct"]) / trading_config.GATE_SELECTION_PCT)
        policy = {**policy, "capital": capital}
        sel_desc = (
            f"Top-N={policy['top_n']}" if policy["selection_mode"] == "top_n"
            else f"top {policy['selection_pct'] * 100:.2f}%"
        )
        print(f"engine_gate（{first_oos.date()} ~ {end.date()}，freq={policy.get('freq')}，"
              f"{sel_desc}，capital={capital:,.0f}）…")
        gate_result = run_engine_gate(
            panel, stacked, val_start=str(first_oos.date()), val_end=str(end.date()), policy=policy
        )
        gm = gate_result.get("metrics") or {}
        print(f"gate passed={gate_result.get('passed')} fail_reasons={gate_result.get('fail_reasons')}")
        print(
            f"  excess_annual={gm.get('excess_annual')} excess_sharpe={gm.get('excess_sharpe')} "
            f"daily_overlap={gm.get('daily_overlap')} turnover={gate_result.get('diagnostics', {}).get('avg_daily_turnover')}"
        )

    # ⑦b 累积子集曲线（可选）：按置换贡献降序 Top-k 逐级重训，
    #     回答"加到第几个因子后边际收益归零、最优小组合是多少个"
    subset_curve = None
    if args.subset_curve and feature_contribution:
        from alphaagent.factor.stacking.model import cumulative_subset_curve

        ranked_names = [x["name"] for x in feature_contribution.get(kinds[0]) or []]
        if ranked_names:
            print(f"累积子集曲线（{len(ranked_names)} 个特征按置换贡献排序，2×{len(ranked_names)} 次拟合）…")
            subset_curve = cumulative_subset_curve(
                dataset.feature_matrix, dataset.label, date_series, folds,
                ranked_names=ranked_names, feature_names=dataset.feature_names,
                kinds=kinds, first_oos=first_oos, label_horizon=args.label_days,
                progress=lambda msg: print(" ", msg, flush=True),
            )
            valid = [r for r in subset_curve if r.get("blended") is not None]
            if valid:
                best = max(valid, key=lambda r: r["blended"])
                full = valid[-1]["blended"]
                print(f"最优规模 k={best['k']}：blended OOS IC={best['blended']}（全量 {full}，"
                      f"精简 {len(best['names'])} 个因子：{'、'.join(best['names'])}）")
            risk_valid = [r for r in subset_curve if r.get("oos_sharpe") is not None]
            if risk_valid:
                best_r = max(risk_valid, key=lambda r: r["oos_sharpe"])
                print(f"Sharpe 峰值 k={best_r['k']}：OOS Sharpe={best_r['oos_sharpe']} "
                      f"回撤={best_r['oos_max_drawdown']}（IC 峰值 k={best['k']}，"
                      f"两口径不一致时优先看风险口径——IC 高不等于可交易）")

    # ⑦c 多路径对照（可选）：折边界平移后整条路径重训，报告 OOS 指标分布
    multi_path = None
    extra_shifts = [int(s.strip()) for s in args.multi_path_shifts.split(",") if s.strip()]
    extra_shifts = [s for s in extra_shifts if s]
    if extra_shifts and not is_simple_scheme:
        from alphaagent.factor.stacking.model import _portfolio_risk_metrics

        def _path_metrics(score: np.ndarray, p0):
            m = (pd.Series(dts) >= pd.Timestamp(p0)) & np.isfinite(score) & np.isfinite(dataset.label)
            m = m.to_numpy()
            if m.sum() < 20:
                return None
            ic = daily_spearman_ic(score[m], dataset.label[m], pd.Series(dts[m]))
            risk = _portfolio_risk_metrics(
                score[m], dataset.label[m], pd.Series(dts[m]), horizon_days=args.label_days
            )
            if not len(ic):
                return None
            sd = float(ic.std(ddof=1))
            return {
                "ic_mean": round(float(ic.mean()), 6),
                "ic_ir": round(float(ic.mean() / sd), 4) if len(ic) > 2 and sd > 1e-12 else None,
                "oos_sharpe": risk.get("oos_sharpe"),
                "oos_max_drawdown": risk.get("oos_max_drawdown"),
                "n_days": int(len(ic)),
            }

        print(f"多路径对照（额外平移 {extra_shifts} 个月，{len(extra_shifts)} 条路径，各重训 {kinds}）…")
        path_rows: list[dict] = []
        for shift in extra_shifts:
            if holdout_mode:
                pfolds: list = []
                for bump in (0, 1, 2, 3):
                    pfolds = walk_forward_splits(
                        dts, train_start=wf_train_start, train_months=base_months + bump,
                        step_months=args.step_months, purge_days=max(args.purge_days, args.label_days),
                        shift_months=shift,
                    )
                    p0 = pd.Timestamp(pfolds[0].oos_dates.min()) if pfolds else None
                    if p0 is not None and p0 >= mining_end:
                        break
                if pfolds and pd.Timestamp(pfolds[0].oos_dates.min()) >= mining_end:
                    pfolds = [f for f in pfolds if pd.Timestamp(f.oos_dates.min()) >= mining_end]
            else:
                pfolds = walk_forward_splits(
                    dts, train_start=mining_end, train_months=args.train_months,
                    step_months=args.step_months, purge_days=max(args.purge_days, args.label_days),
                    shift_months=shift,
                )
            if not pfolds:
                print(f"  平移 {shift} 个月：无可用折（干净段不足），跳过")
                continue
            p_outs: dict[str, np.ndarray] = {}
            for kind in kinds:
                p, _rep, _w, _c = fit_predict_walkforward(
                    dataset.feature_matrix, dataset.label, date_series, pfolds, kind=kind,
                    feature_names=dataset.feature_names, label_horizon=args.label_days,
                )
                p_outs[kind] = p
            usable = [p_outs[k] for k in kinds if np.isfinite(p_outs[k]).any()]
            if not usable:
                continue
            import warnings as _warnings

            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", category=RuntimeWarning)
                ps = np.nanmean(np.column_stack(usable), axis=1).astype(np.float32)
            if smooth_n > 1:
                ps = _wma_smooth_scores(ps, panel, smooth_n)
            p0 = pfolds[0].oos_dates.min()
            mm_p = _path_metrics(ps, p0)
            mm_base = _path_metrics(stacked, p0)  # 同一窗口上的主路径口径，保证可比
            if mm_p is None:
                continue
            gap = (mm_p["ic_mean"] - mm_base["ic_mean"]) if mm_base else None
            row = {
                "shift_months": shift,
                "folds": len(pfolds),
                "oos_start": str(p0.date()),
                "oos_end": str(pfolds[-1].oos_dates.max().date()),
                **mm_p,
                "ic_gap_vs_main": round(gap, 6) if gap is not None else None,
            }
            path_rows.append(row)
            print(f"  平移 {shift} 个月 [{row['oos_start']} ~ {row['oos_end']}] {len(pfolds)} 折："
                  f"IC={_fmt(mm_p['ic_mean'])} ICIR={_fmt(mm_p['ic_ir'])} "
                  f"Sharpe={mm_p['oos_sharpe']} 回撤={mm_p['oos_max_drawdown']} "
                  f"IC−主路径={_fmt(gap) if gap is not None else 'None'}")
        if path_rows:
            ics = [r["ic_mean"] for r in path_rows if r.get("ic_mean") is not None]
            shp = [r["oos_sharpe"] for r in path_rows if r.get("oos_sharpe") is not None]
            multi_path = {
                "shifts": extra_shifts,
                "paths": path_rows,
                "aggregate": {
                    "ic_min": round(min(ics), 6) if ics else None,
                    "ic_median": round(float(np.median(ics)), 6) if ics else None,
                    "ic_max": round(max(ics), 6) if ics else None,
                    "sharpe_min": round(min(shp), 4) if shp else None,
                    "n_negative_ic": sum(1 for v in ics if v < 0),
                    "n_paths": len(path_rows),
                },
                "note": "同一干净历史换一种折边界切法重训的 OOS 结果。单条路径好不算稳："
                        "若有多条平移路径 IC≤0 或显著低于主路径，说明组合对数据切法敏感，"
                        "结论的可交易性存疑。",
            }
            agg = multi_path["aggregate"]
            print(f"  路径分布：IC [{agg['ic_min']} ~ {agg['ic_max']}] 中位 {agg['ic_median']}，"
                  f"负 IC 路径 {agg['n_negative_ic']}/{agg['n_paths']}")

    # ⑧ 落盘：report + model + pred 通道
    import joblib

    report = {
        "run_id": run_id,
        "mining_end": str(mining_end.date()),
        "time_isolation": time_isolation,
        "panel_start": str(panel_start.date()),
        "panel_end": str(end.date()),
        "label_days": args.label_days,
        "score_smooth": int(score_smooth_used),
        "include_factors": list(args.include_factors) if args.include_factors else None,
        "folds": len(folds),
        "scheme": args.scheme,
        "scheme_label": SCHEME_LABELS[args.scheme],
        "feature_names": dataset.feature_names,
        "dropped": dataset.dropped,
        "fold_metrics": fold_reports,
        "feature_weights": feature_weights,
        "feature_contribution": feature_contribution,
        "subset_curve": subset_curve,
        "decay_table": decay,
        "gate": gate_result,
        "oos_ic_blended": _blended_oos_ic(stacked, dataset.label, dts, first_oos),
        "scheme_compare": scheme_compare,
        "multi_path": multi_path,
    }
    (out_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1, default=str), encoding="utf-8"
    )
    joblib.dump(
        {
            "kind": kinds,
            "entries": [e.__dict__ for e in dataset.entries],
            "feature_names": dataset.feature_names,
            "label_days": args.label_days,
            "size_neutral": args.size_neutral,
            "mining_end": str(mining_end.date()),
            "folds": [(str(f.train_dates.min().date()), str(f.train_dates.max().date()),
                       str(f.oos_dates.min().date()), str(f.oos_dates.max().date())) for f in folds],
        },
        out_dir / "model.joblib",
    )
    print(f"报告与模型已写入 {out_dir}")

    if not args.no_write_pred:
        pred_path = Path(args.pred_out) if args.pred_out else (ROOT / "data" / "stock" / "pred_demo.parquet")
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        write_pred_parquet(stacked, panel, pred_path)
        print(f"组合分数已写入 pred 通道：{pred_path}")
        # 自包含副本：out_dir/scores.parquet（同一份 date/code/score 长表）。
        # 供组合因子库（composite_factors）固化——save_composite_factor 依赖
        # out_dir 内报告 + 分数 parquet 完整，复现命令不依赖会被覆盖的全局 pred_demo。
        scores_path = out_dir / "scores.parquet"
        write_pred_parquet(stacked, panel, scores_path)
        print(f"组合分数已写入 out_dir 自包含副本：{scores_path}")


def _wma_smooth_scores(values: np.ndarray, panel: pd.DataFrame, window: int) -> np.ndarray:
    """组合分数尾随线性 WMA（按股票、按交易日），治换手/持仓重叠。

    挖掘侧单因子靠表达式内 WMA 平滑过换手门（ovdiv：WMA20 日换手 1.07→0.50），
    组合分数此前是全链路唯一未平滑的分数。卷积实现（快）：最近一日权重最大；
    窗口内非有限值不参与加权（den 归一）；原始 NaN 行保持 NaN。
    """
    ser = pd.Series(np.asarray(values, dtype=np.float64), index=panel.index)
    wide = ser.unstack("instrument")
    # 截尾核（降序）：当前日权重最大、往前线性衰减；convolve(a, w)[t] = Σ_j a[t-j]·w[j]
    wd = np.arange(window, 0, -1, dtype=np.float64)
    wd /= wd.sum()
    out: dict[object, np.ndarray] = {}
    for col in wide.columns:
        v = wide[col].to_numpy(dtype=np.float64)
        m = np.isfinite(v)
        if not m.any():
            out[col] = v
            continue
        v0 = np.where(m, v, 0.0)
        num = np.convolve(v0, wd, mode="full")[: len(v)]
        den = np.convolve(m.astype(np.float64), wd, mode="full")[: len(v)]
        sm = np.where(den > 1e-12, num / np.maximum(den, 1e-12), np.nan)
        sm[~m] = np.nan
        out[col] = sm
    sm_wide = pd.DataFrame(out, index=wide.index, columns=wide.columns)
    long = (
        sm_wide.rename_axis("datetime")
        .reset_index()
        .melt(id_vars="datetime", var_name="instrument", value_name="score")
        .set_index(["datetime", "instrument"])["score"]
    )
    return long.reindex(panel.index).to_numpy(dtype=np.float32)


def write_pred_parquet(values: np.ndarray, panel: pd.DataFrame, path: Path) -> None:
    """组合分数 → date,code,score 长表（core.data.load_pred_scores 的读取格式）。

    code 统一为 6 位数字：CNE panel 的 instrument 带交易所后缀（000001.SZ），
    回测引擎面板用 6 位裸码——此前只 zfill 不去后缀，引擎侧 merge 不上会静默全 0 持仓。
    """
    def _norm_code(c: object) -> str:
        m = re.search(r"(\d{6})", str(c))
        return (m.group(1) if m else str(c)).zfill(6)

    valid = np.isfinite(values)
    ser = pd.Series(values[valid].astype(np.float64), index=panel.index[valid])
    wide = ser.unstack("instrument")
    wide.columns = [_norm_code(c) for c in wide.columns]
    long = (
        wide.rename_axis("date")
        .reset_index()
        .melt(id_vars="date", var_name="code", value_name="score")
        .dropna(subset=["score"])
    )
    long.to_parquet(path, index=False)


def _blended_oos_ic(stacked: np.ndarray, label: np.ndarray, dts, first_oos) -> dict | None:
    mask = (pd.Series(dts) >= pd.Timestamp(first_oos)) & np.isfinite(stacked) & np.isfinite(label)
    if mask.sum() < 100:
        return None
    ic = daily_spearman_ic(stacked[mask.to_numpy()], label[mask.to_numpy()], dts[mask.to_numpy()])
    if not len(ic):
        return None
    return {"ic_mean": float(ic.mean()), "ic_ir": float(ic.mean() / ic.std()) if ic.std() > 1e-12 else None,
            "n_days": int(len(ic))}


def _fmt(v) -> str:
    return "None" if v is None else f"{v:.4f}"


if __name__ == "__main__":
    main()
