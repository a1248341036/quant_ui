# -*- coding: utf-8 -*-
"""LLM vs 词典 情绪标签一致性审计。"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from . import sentiment as st


def audit_sample(articles, sample_size=300, seed=42, llm_cfg=None, workers=4):
    import random

    random.seed(seed)
    sample = random.sample(articles, min(sample_size, len(articles)))

    def one(a):
        lex_score, lex_net, _, _ = st.score_text(a["title"], a["content"])
        lex = {"label": st.label_of(lex_net), "score": round(lex_score, 4)}
        llm = st.score_with_llm(a["title"], a["content"], llm_cfg) if llm_cfg else {}
        return {
            "title": a["title"],
            "content": a["content"][:200],
            "lex_label": lex["label"],
            "lex_score": lex["score"],
            "llm_label": llm.get("label", "neutral"),
            "llm_score": llm.get("score", 0.0),
        }

    with ThreadPoolExecutor(max_workers=workers) as ex:
        rows = list(ex.map(one, sample))
    return pd.DataFrame(rows)


def summarize(df):
    n = len(df)
    agree = float((df["lex_label"] == df["llm_label"]).mean())
    out = [f"样本数：{n}", f"总体一致率：{agree:.1%}"]
    out.append("")
    out.append("| 词典标签 | n | 与LLM一致率 | LLM标签分布 |")
    out.append("|---|---:|---:|---|")
    for lab in ("positive", "neutral", "negative"):
        sub = df[df["lex_label"] == lab]
        if not len(sub):
            continue
        agree_sub = float((sub["lex_label"] == sub["llm_label"]).mean())
        dist = sub["llm_label"].value_counts().to_dict()
        dist_s = " / ".join(f"{k}:{v}" for k, v in sorted(dist.items()))
        out.append(f"| {lab} | {len(sub)} | {agree_sub:.1%} | {dist_s} |")
    return "\n".join(out)


def write_report(df, path, extra=""):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# LLM vs 词典 情绪标签审计", ""]
    lines.append(summarize(df))
    if extra:
        lines.append("")
        lines.append(extra)
    lines.append("")
    lines.append("## 分歧样本")
    lines.append("")
    bad = df[df["lex_label"] != df["llm_label"]].head(30)
    if len(bad):
        for _, r in bad.iterrows():
            lines.append(
                f"- 词典[{r['lex_label']}({r['lex_score']:+.2f})] LLM[{r['llm_label']}({r['llm_score']:+.2f})] | {r['title'][:80]} | {r['content'][:80]}"
            )
    path.write_text("\n".join(lines), encoding="utf-8")
