Review complete: 11 finding(s) across 3 selected item(s).

─── scripts/ops_notify.py:96-97 ───
[bug · medium] `_send_onebot` 只按 HTTP 状态码判定成功（`return f"http {resp.status}"`），从未读取响应体。OneBot 11 的
HTTP 接口在业务失败时（target_id 不存在、消息被风控、触发频率限制等）仍返回 HTTP 200，仅在 JSON 响应体中带非零 `retcode` / `status:
failed`。当前实现会把这类失败误报为 "ok"，而调用方 finalize_run.py 并不消费返回的
status——对本模块"可靠告警"的定位是实质性缺陷，运维会误以为通知已送达。建议解析响应体，`retcode == 0` 才算成功，否则返回错误信息。

      with urllib.request.urlopen(req, timeout=15) as resp:
-         return f"http {resp.status}"
+         try:
+             data = json.loads(resp.read().decode("utf-8"))
+         except (json.JSONDecodeError, OSError) as exc:
+             return f"error: bad resp body: {exc}"
+         if int(data.get("retcode", -1)) != 0 or data.get("status") == "failed":
+             return f"error: retcode={data.get('retcode')} status={data.get('status')} {str(data.get('wording') or '')[:120]}".strip()
+         return "ok"


─── scripts/ops_notify.py:130-130 ───
[bug · medium] qqbot 通道通过裸 `import healthcheck` 复用 scripts/healthcheck.py，只有当 scripts/ 目录在 sys.path
中时才能成功导入。经 `python scripts/ops_notify.py` 或 finalize_run.py（其显式把 SCRIPTS_DIR 插入
sys.path）调用没问题，但若被其他目录脚本以 `from scripts.ops_notify import send` 或类似形式导入（比如本 diff 之外的 monitor 钩子），会抛
ModuleNotFoundError 并被 `except Exception` 吞掉，`"qqbot": true` 时通道静默不可用。另外 healthcheck.send_qq_alert
内部失败时只 print 不抛出，status 里的 "ok" 也不能证明推送成功。建议在导入前把本文件所在目录补进 sys.path（与 finalize_run.py
的做法一致），保证任何入口下都可用。

+             _here = Path(__file__).resolve().parent
+             if str(_here) not in sys.path:
+                 sys.path.insert(0, str(_here))
              import healthcheck  # 复用 QQ 官方机器人推送（凭据 ~/qqbot/.env，零额外配置）


─── scripts/ops_notify.py:78-78 ───
[bug · low] `_send_qqmail` 中 `", ".join(cfg.get("to") or [user])` 与 `smtp.sendmail(user,
cfg.get("to") or [user], ...)` 都假设 `to` 是 list/tuple。配置示例写的是 `["you@qq.com"]`，但若用户在 notify.json 中把
`to` 误配成单个字符串，join 会按字符拆分出 "y, o, u, @, q, q, …" 这种非法收件人，导致发送必然失败，且报错信息对配置类型问题毫无提示。建议先归一化类型：str
转成单元素 list 再使用。

-     msg["To"] = ", ".join(cfg.get("to") or [user])
+     to = cfg.get("to") or [user]
+     if isinstance(to, str):
+         to = [to]
+     msg["To"] = ", ".join(to)


─── scripts/ops_notify.py:63-63 ───
[bug · medium] `_load_config` 的 `json.loads(...) or {}` 只兜住了假值（空列表/None），若配置文件是语法合法但类型错误的 JSON（如数组
`[1,2]` 或字符串），会原样返回非 dict；随后 send() 里 `cfg.get("qqmail")` 抛 AttributeError，且发生在所有 try
之外，直接击穿模块头注释承诺的"通知永远不能弄挂调用方"这一核心约束。配置文件由人工维护，无法保证类型，建议显式校验返回类型。

-         return json.loads(p.read_text(encoding="utf-8")) or {}
+     try:
+         data = json.loads(p.read_text(encoding="utf-8"))
+         return data if isinstance(data, dict) else {}
+     except Exception:
+         return {}


─── scripts/ops_notify.py:39-40 ───
[maintainability · low] `urllib.error` 被导入但全文件从未引用（urllib 错误全部由 send() 中的 `except Exception`
兜住），属于死代码。建议删除该导入，只保留 `urllib.request`，避免误导读入意图。

- import urllib.error
  import urllib.request


─── scripts/night_summary_report.py:46-49 ───
[other · low] 台账行缺少类型校验：json.loads 成功但返回非 dict 的行（如行内容为 null、数组、字符串，可能来自手改或损坏的续写）会被原样放入列表，后续
append_ledger 的 r.get("since") 与 load_prev_baseline 的 key 索引会直接抛
AttributeError/KeyError，整份夜报中断且无有效错误信息。既然此处已有容错 JSONDecodeError 的意图，建议同时过滤非 dict 行：`if not
isinstance(ln, dict): continue`（需在 json.loads 后先取变量再判断）。

          try:
-             out.append(json.loads(ln))
+             parsed = json.loads(ln)
          except json.JSONDecodeError:
              continue
+         if isinstance(parsed, dict):
+             out.append(parsed)


─── scripts/night_summary_report.py:205-208 ───
[other · low] 对上一夜基线的访问方式不一致：base_pass 用 prev.get("tot_judge") 防御式取值，而 base_rate / base_eval /
base_src 却直接索引
prev["dual_gate_rate"]、prev["tot_eval"]、prev["since"]。台账是跨版本持久化文件，若旧版本或手工编辑的行缺少任一字段，会直接 KeyError
中断整份夜报。建议统一改为 .get 并给出默认值兜底。



─── scripts/night_summary_report.py:55-56 ───
[bug · low] append_ledger 把被替换的重跑行追加到文件末尾，破坏了台账的时序顺序：若运营对某个较早的夜晚（如 10-02）做幂等重跑，台账会从 [10-01, 10-02,
10-03] 变成 [10-01, 10-03, 10-02]，此后 load_prev_baseline 对下一个新夜晚取 reversed 首条不同 since
的行，会拿到被重插到末尾的旧夜晚（10-02）而不是真正的上一夜（10-03），「与上一夜对照」基线被静默指错。建议在写入前按 since（或 generated_at）重新排序，或
load_prev_baseline 选 since 时间上最近的一条，而不是文件序上的最近一条。



─── scripts/finalize_run.py:102-103 ───
[bug · medium] `--runs` 分支没有校验目录是否存在 / 是否含 steps.log，而第 116/118 行的 `(d / "steps.log").stat()`
是无保护的。离线模式（`--backend ""`）或后端不可达时，只要 `--runs` 里的 run_id 打错、目录被清理、或该目录缺 steps.log，就会直接抛
FileNotFoundError 中断整批收尾，与模块注释宣称的「单 run 失败不影响其他 run」相悖。建议在 only 分支对每个候选先做 `d.is_dir() and (d /
"steps.log").is_file()` 检查，不满足就跳过并打印告警。

      if only:
-         cands = [ui / r.strip() for r in only.split(",") if r.strip()]
+         cands = [ui / r.strip() for r in only.split(",") if r.strip()
+                  if (ui / r.strip()).is_dir() and (ui / r.strip() / "steps.log").is_file()]


─── scripts/finalize_run.py:254-256 ───
[other · low] 幂等性依赖「读台账 → 判断 → 收尾 → 追加」这个时序，但模块头注释明确描述 monitor 在 run 结束时与早上手动兜底这两种调用会同时出现。这是典型的
check-then-act 竞态：两个进程可能同时读到相同 run 未收尾，都执行 finalize_one 并向 ledger 追加重复行，导致日报同一 run
出现两行、合计被重复计数（patrol/scorecard 也会被重复执行）。建议对 ledger 追加采用文件锁/原子创建（如以 run_id 为名的锁文件或 flock），或至少让
write_summary_md 对同 run_id 去重后再汇总。



─── scripts/finalize_run.py:118-119 ───
[bug · medium] CLI 帮助明确写着 `--runs`「给定时忽略 API/空闲判定」，但实现只在 `only` 分支替换了候选目录来源，循环体里的状态/空闲判定对 `--runs`
指定的 run 照常生效：离线模式（`--backend ""`）或后端临时不可达时，一个刚结束、steps.log 静默不足 idle_minutes 的 run 会被这条 `elif`
静默跳过，最终输出「没有待收尾的 run。」——与 monitor 钩子「run 刚结束就调 --runs 收尾」的用法相悖。建议在 `only` 给定的分支直接跳过
API/空闲判定（与帮助文本一致），或修正 CLI 帮助文案。



LLM retry report summary: 2 of 41 requests affected -- 1 request failed, 1 request recovered after retry

Review planning (2 requests):
- scripts/finalize_run.py: provider error (HTTP 500) -> provider error (HTTP 500) -> provider error (HTTP 500) -> provider error (HTTP 500) -> provider error (HTTP 500) -> provider error (HTTP 500) -> failed
- scripts/night_summary_report.py: provider error (HTTP 500) -> provider error (HTTP 500) -> provider error (HTTP 500) -> succeeded

Per-attempt detail: --format json (retry_report).
