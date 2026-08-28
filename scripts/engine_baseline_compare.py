"""引擎基准对比:校验拆分前后回测输出逐字段一致。

用法:
    python scripts/engine_baseline_compare.py --before results/engine_baseline/before \
                                              --after  results/engine_baseline/after

语义:
- 每个场景文件先比较 tree_hash(全量规范化 JSON 的 SHA-256);
- 不一致时递归定位首个差异字段并打印前后值;
- 任何场景不一致则退出码 1。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _first_diff(a, b, path="$", depth: int = 0):
    """递归返回 (路径, 前值, 后值) 或 None。深度保护防病态递归。"""
    if depth > 200:
        return path, a, b
    same_type = isinstance(a, type(b))
    if not same_type and not (isinstance(a, bool) and isinstance(b, bool)):
        return path, a, b
    if isinstance(a, dict):
        if set(a.keys()) != set(b.keys()):
            only_a = sorted(set(a.keys()) - set(b.keys()))
            only_b = sorted(set(b.keys()) - set(a.keys()))
            return (f"{path}.<keys>", f"only_before={only_a} only_after={only_b}",
                    None if only_a else f"only_before={only_a} only_after={only_b}")
        for k in sorted(a.keys()):
            d = _first_diff(a[k], b[k], f"{path}.{k}", depth + 1)
            if d is not None:
                return d
        return None
    if isinstance(a, list):
        if len(a) != len(b):
            return f"{path}.<len>", len(a), len(b)
        for i, (x, y) in enumerate(zip(a, b)):
            d = _first_diff(x, y, f"{path}[{i}]", depth + 1)
            if d is not None:
                return d
        return None
    if isinstance(a, float):
        if a != b:
            return path, a, b
        return None
    if a != b:
        return path, a, b
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--before", required=True)
    ap.add_argument("--after", required=True)
    args = ap.parse_args()

    before_dir = Path(args.before)
    after_dir = Path(args.after)
    before_files = sorted(before_dir.glob("*.json"))
    after_files = sorted(after_dir.glob("*.json"))
    if not before_files:
        print(f"before 目录无快照文件: {before_dir}", file=sys.stderr)
        sys.exit(2)
    missing_after = [p.name for p in before_files
                     if not (after_dir / p.name).exists()]
    if missing_after:
        print(f"after 目录缺少场景: {missing_after}", file=sys.stderr)
        sys.exit(2)

    failures = 0
    for bf in before_files:
        name = bf.stem
        af = after_dir / bf.name
        before = json.loads(bf.read_text(encoding="utf-8"))
        after = json.loads(af.read_text(encoding="utf-8"))
        bh, ah = before["tree_hash"], after["tree_hash"]
        if bh != ah:
            failures += 1
            print(f"[FAIL] {name}: tree_hash 不一致 "
                  f"{bh} vs {ah}", file=sys.stderr)
            d = _first_diff(before["result"], after["result"])
            if d is not None:
                path, va, vb = d
                print(f"  diff at {path}", file=sys.stderr)
                print(f"    before: {str(va)[:400]}", file=sys.stderr)
                print(f"    after : {str(vb)[:400]}", file=sys.stderr)
        else:
            print(f"[OK  ] {name}: tree_hash={bh}")

    if failures:
        print(f"\n共 {failures}/{len(before_files)} 个场景不一致", file=sys.stderr)
        sys.exit(1)
    print(f"\n全部 {len(before_files)} 个场景一致(tree_hash 逐位相同),拆分前后回测输出无漂移。")


if __name__ == "__main__":
    main()