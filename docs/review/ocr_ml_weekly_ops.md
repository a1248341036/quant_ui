Review complete: 5 finding(s) across 1 selected item(s).

─── scripts/ml_weekly_ops.py:199-199 ───
[bug · medium] 错误通知只覆盖训练阶段 rc!=0 分支：名单/归因/写台账阶段一旦抛异常（如 scores.parquet 空表时 df["date"].max() 抛
ValueError、缺列抛 KeyError、panel 加载失败等），会直接裸 traceback 退出，没有任何 error 通知，周五晚计划任务失败将无人感知。建议把 build_list →
write_list → exposure_snapshot → append_exposure_ledger 整段包一层 try/except，并复用下面训练失败的同款 ops_notify
error 通知。

+     try:
-     top, latest = build_list(scores, args.top_n)
+         top, latest = build_list(scores, args.top_n)
+         md = write_list(top, latest, run_id)
+         ...
+     except Exception as exc:
+         print(f"[abort] 名单/归因阶段失败: {exc}")
+         if args.notify:
+             try:
+                 import ops_notify
+                 ops_notify.send("ML 周更失败", f"名单/归因阶段异常: {exc}", level="error")
+             except Exception:
+                 pass
+         return 1


─── scripts/ml_weekly_ops.py:164-164 ───
[bug · medium] run_train 的 subprocess.run 既未设置 timeout 也未捕获启动异常：train_ml_composite.py
若因数据源卡住而挂起，周五晚任务会无限期阻塞，后续名单/台账/通知全部停摆且无任何告警；若脚本文件不存在等启动异常（FileNotFoundError/OSError），也不会进入调用方 rc!=0
的错误通知分支。建议加 timeout 并把 TimeoutExpired 等异常纳入错误通知路径。

-     return subprocess.run(cmd).returncode
+     try:
+         return subprocess.run(cmd, timeout=args.train_timeout).returncode
+     except subprocess.TimeoutExpired as exc:
+         raise RuntimeError(f"训练超时: {exc}") from exc


─── scripts/ml_weekly_ops.py:123-123 ───
[bug · low] 暴露值存在全 NaN 风险：score_mask 硬编码“当日有效 score ≥ 50”，若股票池不足 50 或数据缺失导致所有日期被掩掉，profile["coef"]
全为 NaN，coef.mean() 得 nan；json.dumps 默认 allow_nan=True 会把 nan 序列化成非法 JSON 的 NaN 写入台账，且 compute_drift
中 abs(NaN) >= threshold 恒为 False，漂移 warn 告警会被静默吞掉。建议写入台账前过滤非有限值，并在出现 NaN 时显式告警。

-         "style_exposure": {n: round(float(coef[n]), 4) for n in names[:n_style]},
+         "style_exposure": {n: (round(float(coef[n]), 4) if np.isfinite(coef[n]) else None)
+                             for n in names[:n_style]},


─── scripts/ml_weekly_ops.py:0-0 ───
[bug · medium] ops_notify 模块在当前仓库中不存在（全库检索无定义），而 import 被包在 except Exception
里只打印一行“通知失败（忽略）”，因此“完成/失败/漂移告警”全部会静默失效，背离了本脚本的告警职责。建议在本 PR 补齐 ops_notify，或至少把缺失依赖当作显式错误（非静默吞掉）处理。

              import ops_notify
  
              ops_notify.send("ML 周更完成", body, level="warn" if drifted else "info")


─── scripts/ml_weekly_ops.py:63-67 ───
[maintainability · low] latest_stacking_run 只按目录名字典序取“最新”且仅校验 scores.parquet 存在性：若最新 run 因
--no-write-pred 或训练中断只创建了目录而未写 scores.parquet，会无提示回退到更旧
run，名单、暴露画像、台账和告警将全部基于过期数据生成且无任何预警。建议在发生回退（首个候选非预期 run）时打 warn 或直接失败退出。

      for d in sorted((x for x in stacking_dir.iterdir() if x.is_dir()), reverse=True):
          scores = d / "scores.parquet"
          if scores.is_file():
+             print(f"[warn] stacking run 回退到较旧目录: {d.name}")
              return d.name, scores
      return None


LLM retry report summary: 1 of 28 requests affected -- 1 request recovered after retry

Review planning (1 request):
- scripts/ml_weekly_ops.py: provider error (HTTP 500) -> provider error (HTTP 500) -> succeeded

Per-attempt detail: --format json (retry_report).
