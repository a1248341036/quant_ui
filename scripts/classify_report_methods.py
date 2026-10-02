#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P2：给已抽取的研报课题打**方法类型**标签（LLM 判定，不做关键词匹配）。

用途：`ml_model` / `graph_deep`（机器学习、图/深度网络类）暂时不出题——这类研报的"结构"
在 DSL 里无法表达，硬出题只会得到"用任意表达式近似"的伪复现。

用法：
  python scripts/classify_report_methods.py --src question_v2_prod2.jsonl [--limit 0] [--workers 6]
  # 结果写回 --src，并把同一映射同步到在用题库 research_questions.jsonl（按 question_id 对齐）

分类（method_type）：
  rule_formula     规则型公式因子（比值/增长率/波动率/价量统计，可直接写 DSL）
  factor_test      单因子/多因子测试与优选（含少量组合）
  portfolio_combo  组合/合成打分（等权、IC/ICIR 加权、多步筛选）
  event_driven     事件驱动（公告/评级/资金流/股东行为等事件）
  timing_rotation  择时/轮动（因子择时、行业/风格轮动）
  ml_model         机器学习模型（GBDT/XGBoost/随机森林/神经网络做因子合成或预测）
  graph_deep       图模型/深度结构（GNN/GAT/Transformer/RNN/LSTM/自编码器/embedding）
  other            其它/无法归类
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

_HERE = Path(__file__).resolve()
_REPO = _HERE.parent.parent
sys.path.insert(0, str(_REPO))

K = _REPO / "data" / "research_reports" / "knowledge"

LABELS = ("rule_formula", "factor_test", "portfolio_combo", "event_driven",
          "timing_rotation", "ml_model", "graph_deep", "other")

SYS = "你是券商金工研报的分类员，只输出 JSON。"

TMPL = """判断下面这篇研报**提出/测试的核心东西属于哪一类方法**（只选一个标签）：

- `rule_formula`：规则型公式因子（比值/增长率/波动率/价量统计等，可直接写成表达式）
- `factor_test`：单因子/多因子测试与优选（主要是检验既有因子的有效性）
- `portfolio_combo`：组合/合成打分（等权、IC/ICIR 加权、多步筛选等明确的合成方式）
- `event_driven`：事件驱动（公告、评级调整、资金流、股东行为等事件）
- `timing_rotation`：择时/轮动（因子择时、行业/风格轮动）
- `ml_model`：**机器学习模型**（GBDT/XGBoost/随机森林/神经网络等用于因子合成或收益预测）
- `graph_deep`：**图模型/深度结构**（GNN/GAT/Transformer/RNN/LSTM/自编码器/embedding 等）
- `other`：其它/无法归类

判据：**看报告主张的核心机制是否依赖"训练出来的模型/网络结构"**——依赖则归 `ml_model`
或 `graph_deep`（图/深度优先 `graph_deep`）；只是用模型做筛选/回测工具则不算。

**加严规则（2026-10-03，实测漏判后补）**：只要满足下面任一条，就**必须**归 `ml_model`
（若涉及图/深度网络则 `graph_deep`），**不得**归 `factor_test`/`rule_formula`：
1. 标题/课题里出现"机器学习/深度学习/神经网络/图神经网络/GNN/LSTM/Transformer/XGBoost/
   随机森林/GBDT/自编码器/embedding/模型训练/训练模型"等字样，且报告用它来**构造或合成因子**；
2. 报告的因子来自"用模型把高频数据低频化""端到端训练出信号"这类**机制本身依赖训练**的做法；
3. 复现目标里出现"训练/模型/网络/拟合/预测模型"且是核心步骤。
> 反例（不算 ML）：仅用线性回归做中性化/筛选、仅用 IC 加权合成、仅做分组回测。

只输出 JSON：{{"method_type": "...", "reason": "一句话依据"}}

研报信息：
- 标题/课题：{topic}
- 报告类型：{rtype}
- 复现目标（上游抽取）：{target}
- 机制假设：{hyp}
"""


def classify(rec: dict) -> dict:
    from alphaagent.core.llm_provider import chat_json

    prompt = TMPL.format(
        topic=str(rec.get("topic") or "")[:200],
        rtype=str(rec.get("report_type") or ""),
        target=str(rec.get("reproduction_target") or "")[:600],
        hyp=str(rec.get("hypothesis") or rec.get("mechanism") or "")[:400],
    )
    try:
        out = chat_json(SYS, prompt, max_tokens=300, temperature=0.0, timeout=90)
    except Exception as exc:  # noqa: BLE001
        return {"method_type": None, "reason": f"error: {str(exc)[:80]}"}
    label = str((out or {}).get("method_type") or "").strip()
    if label not in LABELS:
        label = "other"
    return {"method_type": label, "reason": str((out or {}).get("reason") or "")[:160]}


def audit_ml_misses(rows: list[dict]) -> int:
    """审计护栏（B）：列出「标题/主题命中 ML 词但未归 ml_model/graph_deep」的题。

    仅用于**发现漏判候选**供复核，不作为分类依据（分类仍由 LLM 完成）。
    背景：2026-10-03 实测 RQ_875028 主题写"机器学习"，却被归为 factor_test，
    没被 `report_policy.exclude_method_types` 排除 → 该题在复现阶段连续 42 次被结构/字段锚
    拦截（`off_reference`）、0 PASS、整轮预算耗在 DSL 表达不了的报告上。
    """
    import re
    pat = re.compile(r"机器学习|深度学习|神经网络|图神经网络|GNN|GAT|XGBoost|随机森林|GBDT|"
                     r"LSTM|Transformer|自编码|embedding|模型训练|训练模型", re.I)
    hits = []
    for r in rows:
        txt = " ".join(str(r.get(k) or "") for k in ("topic", "name", "report_type"))
        if pat.search(txt):
            hits.append(r)
    misses = [r for r in hits if r.get("method_type") not in ("ml_model", "graph_deep")]
    print(f"[审计] {len(rows)} 题中标题/主题命中 ML 词的 {len(hits)} 题；"
          f"其中未归 ml_model/graph_deep 的 {len(misses)} 题（漏判风险）")
    for r in misses:
        print(f"  {r.get('question_id') or r.get('source')}  method={r.get('method_type')}  "
              f"topic={str(r.get('topic'))[:60]}")
    return 0 if not misses else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="question_v2_prod2.jsonl")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--sync-only", action="store_true",
                    help="跳过 LLM 分类，仅把 --src 里已有的 method_type 同步到在用题库（按 source 对齐）")
    ap.add_argument("--audit", action="store_true",
                    help="审计：列出「标题/主题命中 ML 词但未归 ml_model/graph_deep」的题（漏判风险），"
                         "不做判定也不写文件")
    args = ap.parse_args()

    os.environ.setdefault("ALPHA_LLM_PROVIDER", "codex")
    from alphaagent.core.llm_provider import load_codex_provider
    load_codex_provider()

    src = K / args.src
    rows = [json.loads(l) for l in src.read_text(encoding="utf-8", errors="replace").splitlines()
            if l.strip()]
    if args.limit:
        rows = rows[:args.limit]

    if args.audit:
        return audit_ml_misses(rows)

    if args.sync_only:
        results = rows
        if not any(r.get("method_type") for r in results):
            print("!! --src 里没有 method_type，先跑一次分类")
            return 1
    else:
        print(f"待分类 {len(rows)} 篇（model={os.getenv('MODEL')}）", flush=True)
        t0 = time.time()
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(classify, r): r for r in rows}
            for i, fut in enumerate(concurrent.futures.as_completed(futs), 1):
                rec = futs[fut]
                out = fut.result()
                results.append({**rec, **out})
                if i % 10 == 0 or i == len(rows):
                    print(f"  [{i}/{len(rows)}] {time.time()-t0:.0f}s "
                          f"{dict(Counter(x.get('method_type') for x in results))}", flush=True)

    counts = Counter(x.get("method_type") for x in results)
    print("\n=== 方法类型分布 ===")
    for k, v in counts.most_common():
        print(f"  {str(k):<16}{v:>4} 篇  ({v/len(results)*100:.0f}%)")
    excl = {"ml_model", "graph_deep"}
    n_excl = sum(v for k, v in counts.items() if k in excl)
    print(f"\n  机器学习/图深度（暂不出题）={n_excl} 篇；其余可出题 = {len(results)-n_excl} 篇")

    if args.dry_run:
        return 0

    stamp = time.strftime("%Y%m%d-%H%M%S")
    shutil.copy2(src, src.with_suffix(f".jsonl.bak-method-{stamp}"))
    src.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in results) + "\n",
                   encoding="utf-8")
    print(f"  ✓ 写回 {src.name}（备份 .bak-method-{stamp}）")

    # 同步映射到在用题库（**按 source 对齐**：v2 抽取产物没有 question_id，只有 source）
    bank = K / "research_questions.jsonl"
    if bank.is_file():
        mapping = {str(r.get("source")): r.get("method_type") for r in results}
        brows = [json.loads(l) for l in bank.read_text(encoding="utf-8", errors="replace").splitlines()
                 if l.strip()]
        hit = 0
        for r in brows:
            mt = mapping.get(str(r.get("source")))
            if mt:
                r["method_type"] = mt
                hit += 1
        shutil.copy2(bank, bank.with_suffix(f".jsonl.bak-method-{stamp}"))
        bank.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in brows) + "\n",
                        encoding="utf-8")
        print(f"  ✓ 同步到在用题库 {bank.name}：命中 {hit}/{len(brows)} 题（备份 .bak-method-{stamp}）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
