#!/usr/bin/env python3
"""存量候选池按当前门槛重筛（soft-drop 或 --hard 硬删）。

背景（2026-09-11，观察池口径）：进池线 = train |IC| ≥ 0.020 / |ICIR| > 0.28、
val |IC| ≥ 0.012、盲测 |test IC| ≥ 0.010（外加保留比/方向/coverage/自相关/换手/
相关性）；晋升线（精筛 0.025/0.30 + 引擎裁决）独立于此不动。本脚本把**当前**
门槛回溯应用于存量候选，使池子与新准入策略一致：

- 门槛逐条目按 ``research_mode`` 选档，判定直接复用 ``DeliveryChecker``
  （提交路径同一套代码 = 零口径漂移）：盲测终审 / stage_one 统计 / val 保留比与
  绝对下限 / 相关性；
- 缺省动作 = soft-drop：不通过者打 ``dropped_from_ml`` 标记（ML 组合训练集默认
  剔除，见 ``factor/stacking/dataset.py``），条目/DSL/研究记忆全部保留，可逆；
- ``--hard`` = 硬删除：不通过条目 + DSL 从 registry/库中移出，完整归档到
  ``*.removed-prescreen-<ts>.json`` 与 ``expressions.bak-prescreen-<ts>/``（可整体
  回捞）；研究记忆**不清理**（记忆存的是评估证据，与库成员资格解耦）；
- ``promotion_status == "promoted"`` 一律不动；
- 缺 test 指标的旧条目按盲测门判为不通过（盲测段是唯一诚实样本外，未经验证的
  因子不进池）——这类条目 reason 记 ``blind_test_ic_missing``，可单独复核；
- 已有 ``dropped_from_ml`` 的条目不复活（衰减审计等其它来源的标记保留）；soft-drop
  模式下列出"已标记但按当前门槛应存活"供人工复核，--hard 模式仅复活 prescreen
  自家标记（``dropped_reason`` 以 prescreen_gate 开头）。

用法：
  .venv\\Scripts\\python.exe scripts\\rescreen_candidate_pool.py --data-root D:\\Quant\\quant_ui
  .venv\\Scripts\\python.exe scripts\\rescreen_candidate_pool.py --data-root D:\\Quant\\quant_ui --apply --hard
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

_DROP_REASON_PREFIX = "prescreen_gate"


def _mode_of(entry: dict) -> str:
    mode = str(entry.get("research_mode") or "").strip()
    if mode:
        return mode
    try:
        from alphaagent.factor.mining.infra.registry_io import derive_freq_from_label_col

        cfg = entry.get("ingest_config") if isinstance(entry.get("ingest_config"), dict) else {}
        label = str(entry.get("eval_label") or cfg.get("label_col") or "")
        derived, _freq = derive_freq_from_label_col(label)
        if derived:
            return str(derived)
    except Exception:  # noqa: BLE001
        pass
    return "technical"


def _evidence(entry: dict) -> tuple[dict, dict, dict, dict]:
    """把 registry 存档指标还原成 checker 需要的证据字典（train/val/test/similarity）。"""
    m = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    train: dict = {
        "ic": m.get("train_ic"),
        "icir": m.get("train_icir"),
        "coverage": m.get("coverage", m.get("factor_coverage")),
        "cs_pearson_autocorr": m.get("cs_pearson_autocorr"),
    }
    qp = m.get("quantile_portfolio")
    if isinstance(qp, dict):
        train["quantile_portfolio"] = qp
    val = {"ic": m.get("val_ic"), "n_days": m.get("n_days")}
    test = {"ic": m.get("test_ic")}
    sim = entry.get("similarity") if isinstance(entry.get("similarity"), dict) else {}
    return train, val, test, sim


def _short_reason(reason: str) -> str:
    """换手门 reason 是带中文说明的长串，这里归一化成短标签。"""
    if reason.startswith("avg_daily_side_turnover"):
        return "turnover"
    return reason


def _missing_fields(train: dict, val: dict, test: dict, sim: dict) -> list[str]:
    missing = []
    for key in ("ic", "icir", "coverage", "cs_pearson_autocorr"):
        if train.get(key) is None:
            missing.append(key)
    if train.get("quantile_portfolio", {}).get("avg_daily_side_turnover") is None:
        missing.append("turnover")
    if val.get("ic") is None:
        missing.append("val_ic")
    if test.get("ic") is None:
        missing.append("test_ic")
    if sim.get("max_abs_corr") is None:
        missing.append("max_abs_corr")
    return missing


def rescreen(registry_path: Path, *, apply: bool, hard: bool = False) -> dict:
    from alphaagent.factor.mining.delivery.delivery_checker import DeliveryChecker
    from alphaagent.factor.mining.delivery.delivery_criteria import DeliveryCriteria
    from alphaagent.factor.mining.research_spec import default_research_spec

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if not isinstance(registry, dict):
        raise SystemExit(f"registry 结构异常: {registry_path}")

    checkers: dict[str, DeliveryChecker] = {}
    for mode in ("technical", "fundamental"):
        checkers[mode] = DeliveryChecker(
            DeliveryCriteria.from_spec(default_research_spec(mode))
        )

    to_drop: dict[str, list[str]] = {}
    already_flagged_but_alive: list[str] = []
    skipped_promoted: list[str] = []

    for fid, entry in registry.items():
        if not isinstance(entry, dict):
            continue
        if str(entry.get("promotion_status") or "") == "promoted":
            skipped_promoted.append(fid)
            continue
        mode = _mode_of(entry)
        checker = checkers.get(mode) or checkers["technical"]
        train, val, test, sim = _evidence(entry)
        reasons: list[str] = []
        for tag, res in (
            ("blind", checker.blind_test(train, test)),
            ("stage_one", checker.stage_one_stats(train)),
            ("val", checker.stage_one_val_retention(train, val)),
            ("corr", checker.stage_one_correlation(sim)),
        ):
            reasons.extend(f"{tag}:{_short_reason(r)}" for r in res.fail_reasons)

        if reasons:
            to_drop[fid] = reasons
        elif entry.get("dropped_from_ml"):
            already_flagged_but_alive.append(fid)

    print(f"registry: {registry_path}")
    print(f"候选 {len(registry)} 条 | promoted 跳过 {len(skipped_promoted)} 条 | "
          f"当前门槛不通过 {len(to_drop)} 条 | 存活 "
          f"{len(registry) - len(skipped_promoted) - len(to_drop)} 条")
    if to_drop:
        print("\n不通过明细：")
        for fid, reasons in sorted(to_drop.items()):
            entry = registry.get(fid) or {}
            m = entry.get("metrics") or {}
            miss = _missing_fields(*_evidence(entry))
            tip = f"  [缺字段: {','.join(miss)}]" if miss else ""
            print(f"  - {fid:44s} mode={_mode_of(entry):11s} "
                  f"train_ic={m.get('train_ic')} icir={m.get('train_icir')} "
                  f"val={m.get('val_ic')} test={m.get('test_ic')} "
                  f"-> {','.join(reasons)}{tip}")
    if already_flagged_but_alive:
        action = "复活（仅限 prescreen_gate 自家标记）" if hard else "需人工复核，脚本不复活"
        print(f"\n已标记 dropped_from_ml 但按当前门槛应存活（{action}）：")
        for fid in already_flagged_but_alive:
            print(f"  - {fid}")

    if not apply or not (to_drop or (hard and already_flagged_but_alive)):
        return {"dropped": len(to_drop), "checked": len(registry), "applied": False}

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = registry_path.with_name(f"{registry_path.stem}.bak-prescreen-{stamp}.json")
    backup.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n备份: {backup}")

    if hard:
        # ── 硬删：条目 + DSL 移出，归档保留（可整体回捞）；研究记忆不清理
        # （记忆存的是评估证据/学习材料，与库成员资格解耦，删因子不应抹历史）。──
        lib_dir = registry_path.parent
        archive_dir = lib_dir / f"expressions.bak-prescreen-{stamp}"
        archive_dir.mkdir(exist_ok=True)
        removed: dict[str, dict] = {}
        for fid in to_drop:
            entry = registry.pop(fid, None)
            if entry is None:
                continue
            removed[fid] = entry
            rel = str(entry.get("expression_file") or "")
            dsl_path = Path(rel) if Path(rel).is_absolute() else ROOT / rel
            if not dsl_path.exists():
                dsl_path = lib_dir / "expressions" / f"{fid}.dsl"
            if dsl_path.exists():
                (archive_dir / dsl_path.name).write_text(
                    dsl_path.read_text(encoding="utf-8"), encoding="utf-8"
                )
                dsl_path.unlink()
        # prescreen 自家标记的"假阳性剔除"复活（衰减审计等其它来源/裸标记不动）
        for fid in already_flagged_but_alive:
            entry = registry.get(fid) or {}
            reason = str(entry.get("dropped_reason") or "")
            if reason.startswith(_DROP_REASON_PREFIX):
                entry.pop("dropped_from_ml", None)
                entry.pop("dropped_reason", None)
                entry.pop("dropped_at", None)
        archive = registry_path.with_name(
            f"{registry_path.stem}.removed-prescreen-{stamp}.json"
        )
        archive.write_text(
            json.dumps(removed, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(f"归档: {archive}（{len(removed)} 条，含表达式副本 {archive_dir}）；"
              f"研究记忆未清理（保留学习证据）")
    else:
        now = datetime.now(timezone.utc).isoformat()
        for fid, reasons in to_drop.items():
            entry = registry[fid]
            entry["dropped_from_ml"] = True
            entry["dropped_reason"] = f"{_DROP_REASON_PREFIX}:{','.join(reasons)}"
            entry["dropped_at"] = now

    tmp = registry_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, registry_path)
    action = "硬删除" if hard else "标记 dropped_from_ml"
    print(f"已写入: {registry_path}（{len(to_drop)} 条{action}）")
    return {"dropped": len(to_drop), "checked": len(registry), "applied": True}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=str(ROOT),
                    help="数据根目录（含 artifacts/；默认脚本所在仓库）")
    ap.add_argument("--library", default="candidate_main", help="因子库目录名（默认 candidate_main）")
    ap.add_argument("--apply", action="store_true", help="执行（缺省 dry-run）")
    ap.add_argument("--hard", action="store_true",
                    help="硬删除：条目+DSL 移出并归档（缺省 soft-drop 打标）")
    args = ap.parse_args()

    registry_path = Path(args.data_root) / "artifacts" / "alphaagent" / "factorzoo" / args.library / "mining_candidate_registry.json"
    if not registry_path.exists():
        raise SystemExit(f"registry 不存在: {registry_path}")
    out = rescreen(registry_path, apply=args.apply, hard=args.hard)
    if not out["applied"]:
        print("\ndry-run：确认无误后加 --apply 执行。")


if __name__ == "__main__":
    main()
