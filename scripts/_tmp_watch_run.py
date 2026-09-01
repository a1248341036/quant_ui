# -*- coding: utf-8 -*-
"""临时: run 3754c22c40c8 监控 v2（外部化结果 gz 解析）。用完即删。"""
import gzip
import json
import math
import statistics
from pathlib import Path

RUN = Path(r"D:\Quant\quant_ui\logs\factor_mining\ui\3754c22c40c8")


def load_external(ref: str):
    p = RUN / ref
    try:
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, list) else [data]
    except Exception:
        return []


def load_events():
    out = []
    for f in RUN.glob("run_*.jsonl"):
        for line in f.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def main():
    events = load_events()
    ics = []
    submits = []
    for e in events:
        if e.get("event") != "tool_results":
            continue
        for r in e.get("results") or []:
            items = [r]
            if r.get("result_ref") and r.get("result") is None:
                items = load_external(r["result_ref"]) or [r]
            for it in items:
                name = it.get("name")
                try:
                    args = json.loads(it.get("arguments_raw") or "{}")
                except Exception:
                    args = {}
                res = it.get("result") or {}
                if name == "evaluate_factor":
                    m_all = res.get("metrics") or {}
                    core = m_all.get("cross_sectional_core") or {}
                    ic = core.get("ic")
                    icir = core.get("icir")
                    cov = core.get("factor_coverage")
                    ics.append({
                        "name": args.get("factor_name"),
                        "ic": ic, "icir": icir, "cov": cov,
                        "passed": res.get("passed"),
                        "err": (res.get("error") or "")[:70] if res.get("error") else None,
                        "parent": args.get("parent_factor"),
                        "edit": args.get("edit_note"),
                    })
                elif name == "submit_factor":
                    submits.append({
                        "name": args.get("factor_name"),
                        "stored": bool(res.get("stored")),
                        "candidate": bool(res.get("candidate_stored")),
                        "err": (res.get("error") or "")[:400],
                    })
    print("=== evaluate_factor ===")
    ok = [x for x in ics if isinstance(x.get("ic"), (int, float)) and not math.isnan(x["ic"])]
    for x in ics:
        ic = x.get("ic")
        if isinstance(ic, (int, float)) and not math.isnan(ic):
            mark = "*" if abs(ic) >= 0.015 else " "
            print(f" {mark} {x['name']}: IC={ic:+.4f} ICIR={x['icir']} cov={x['cov']} "
                  f"parent={x['parent']} edit={x['edit']}")
        elif x.get("err"):
            print(f" ! {x['name']}: {x['err']}")
        else:
            print(f" - {x['name']}: IC=nan (全 NaN / 数据缺失)")
    if ok:
        avg = statistics.mean(abs(x["ic"]) for x in ok)
        best = max(ok, key=lambda x: abs(x["ic"]))
        ge = sum(1 for x in ok if abs(x["ic"]) >= 0.015)
        print(f"-- 有效 {len(ok)} 条: avg|IC|={avg:.4f} best={best['name']}({best['ic']:+.4f}) >=0.015: {ge}")
    print("=== submit ===")
    for s in submits:
        print(" ", json.dumps(s, ensure_ascii=False))
    # submit 失败详情（外部化 gz 里的完整 error）
    for e in events:
        if e.get("event") != "tool_results":
            continue
        for r in e.get("results") or []:
            if r.get("name") != "submit_factor" or not r.get("result_ref"):
                continue
            for it in (load_external(r["result_ref"]) or []):
                res = it.get("result") or {}
                err = str(res.get("error") or "")
                if "expressions" in err or "blind_test" in err:
                    try:
                        nm = json.loads(it.get("arguments_raw") or "{}").get("factor_name")
                    except Exception:
                        nm = "?"
                    print("  ERR-DETAIL", nm, "->", err[:400])
    recs = sum(1 for e in events if "本轮记忆推荐" in str(e.get("content") or ""))
    dist = [e for e in events if e.get("event") == "experience_distilled"]
    print(f"任务消息含「本轮记忆推荐」: {recs} 次 | experience_distilled: {len(dist)} 次")
    for d in dist:
        print("  ", json.dumps({k: v for k, v in d.items() if k not in ("event", "ts")}, ensure_ascii=False)[:220])


if __name__ == "__main__":
    import io, sys
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    main()
