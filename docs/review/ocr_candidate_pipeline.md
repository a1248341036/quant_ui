Review complete: 4 finding(s) across 2 selected item(s).

─── scripts/blind_test_factors.py:273-275 ───
[bug · medium] _append_blind_ledger 调用没有任何异常保护，且位于 report.json 已成功写出之后：一旦写入 ledger.jsonl 发生
OSError（磁盘满、权限、路径异常等），会直接以 traceback 中止整个脚本、返回非零退出码。后果有二：1) post_mining_pipeline 把 blind_test
视为失败而停链并告警——但盲测评估实际已完成、报告已产出，操作者很可能据此重跑，而重跑恰恰是台账要防止的重复消费盲测段；2) 崩溃发生在追加之前，本次盲测消费未被记账，下次
_warn_duplicate_consumption 无法警告这次重叠，审计留下缺口。与 _load_blind_ledger 对 OSError 的容错不对称。建议对追加写入加 OSError
捕获（失败时打印 stderr 警告而非中断主流程），或至少放在 try/except 中保证评估成功的结果不被记账失败拖垮。

+     try:
-     _append_blind_ledger({
+         _append_blind_ledger({
-         "ts": out_dir.name,
+             "ts": out_dir.name,
-         "generated_at": report["generated_at"],
+             "generated_at": report["generated_at"],
+             "test_window": report["test_window"],
+             "label_col": args.label_col,
+             "libs": args.libs,
+             "n_factors": len(rows),
+             "factor_ids": [r["factor_id"] for r in rows],
+             "clean": bool(args.clean),
+             "dry_run": bool(args.dry_run),
+         })
+     except OSError as exc:
+         print(f"[warn] 盲测消费台账追加失败（本次消费未记账）: {exc}")


─── scripts/post_mining_pipeline.py:60-60 ───
[bug · high] dedup_candidate_factors.py 实际只定义了 `--threshold`/`--dry-run`/`--registry`，并不存在 `--apply`
参数。这里 `--apply-dedup` 真删路径会拼出 `--apply`，子进程 argparse 会以“unrecognized arguments”报错退出（rc=2），导致流水线在
dedup 阶段中断、真删永远不会发生，`--apply-dedup` 功能实际不可用。注意该脚本的缺省行为本身就是删除，`--dry-run` 才是“只看不删”；正确写法应是仅在未开启
`--apply-dedup` 时追加 `--dry-run`。

- cmd.append("--apply" if args.apply_dedup else "--dry-run")
+ if not args.apply_dedup:
+             cmd.append("--dry-run")
+         return cmd


─── scripts/post_mining_pipeline.py:85-86 ───
[bug · medium] subprocess.run 既未捕获异常也未设 timeout：子脚本缺失/无执行权限时会抛 FileNotFoundError/PermissionError 穿透
run_chain，既不触发“失败即停 + ops_notify 告警”约定，调用者也看不到中断原因；若子阶段（如 promote/blind_test）挂起，流水线会无限阻塞。建议用
try/except 将文件级错误转换为非零退出码，并加 timeout（如 subprocess.run(cmd, timeout=...）并在超时时返回非零码）。

- def _run_stage(cmd: list[str]) -> int:
-     return subprocess.run(cmd).returncode
+ def _run_stage(cmd: list[str], timeout: int | None = 3600) -> int:
+     try:
+         return subprocess.run(cmd, timeout=timeout).returncode
+     except (OSError, subprocess.TimeoutExpired) as exc:
+         print(f"阶段执行失败: {exc}", file=sys.stderr)
+         return 1


─── scripts/post_mining_pipeline.py:131-131 ───
[maintainability · low] 确认门条件 `'promote_dry' in stages` 在 `--stage promote_apply` 续跑时为 False，会跳过
confirm_promote 直接执行晋升写操作；即使处于非交互环境且未传 --yes 也会放行，与模块头“非交互环境必须 --yes 才能执行晋升
apply”的安全缺省说明矛盾。虽然断点续跑语义可理解为“已确认”，但建议在非交互且未 --yes 时仍给出显式确认/警告，避免计划任务误触写操作。



LLM retry report summary: 2 of 30 requests affected -- 2 requests recovered after retry

Review planning (2 requests):
- scripts/blind_test_factors.py: provider error (HTTP 500) -> succeeded
- scripts/post_mining_pipeline.py: provider error (HTTP 500) -> provider error (HTTP 500) -> provider error (HTTP 500) -> succeeded

Per-attempt detail: --format json (retry_report).
