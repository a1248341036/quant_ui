Review complete: 12 finding(s) across 3 selected item(s).

─── scripts/build_frontend.ps1:19-20 ───
[bug · medium] `git worktree add` 的失败未被检查（前面 remove/prune 是幂等清理，可忽略，但 add 是必需步骤）：当 $Ref
不存在、路径被占用等情况下，git 退出码非 0，脚本仍会打印 "[1/3] worktree 就绪" 并继续，随后在 Push-Location 处才以晦涩的“路径不存在”报错。建议对 add 检查
$LASTEXITCODE 并直接 throw，让失败原因清晰可见。

- & git -C $RepoRoot worktree add --detach $wt $Ref | Out-Null
+ & git -C $RepoRoot worktree add --detach $wt $Ref 2>$null
+ if ($LASTEXITCODE -ne 0) { throw "git worktree add 失败: $Ref (exit $LASTEXITCODE)" }
  Write-Host "[1/3] worktree 就绪: $wt @ $Ref"


─── scripts/build_frontend.ps1:41-42 ───
[bug · medium] 拷贝回写顺序有边界隐患：若 npm run build 因配置问题“成功退出但未产出 dist”（例如构建输出目录被改），这里会先把主工作区现有 dist 改名成
.bak，随后 Copy-Item 因源目录不存在而失败——结果主工作区 dist 直接缺失，且 Build 明明失败了却动了 dist，违背头部注释“失败不动 dist”的承诺。建议先校验
worktree 内 dist 存在，再改名与拷贝。

+     $distSrc = Join-Path $wt "static\dist"
+     if (-not (Test-Path $distSrc)) { throw "构建未产出 dist，已保留主工作区原有产物" }
      if (Test-Path $dist) { Rename-Item $dist "$dist.bak" }
-     Copy-Item (Join-Path $wt "static\dist") $dist -Recurse
+     Copy-Item $distSrc $dist -Recurse


─── scripts/build_frontend.ps1:15-15 ───
[other · medium] 该脚本的核心场景是“并行 agent 各建各的分支”，但 worktree 路径固定为 `.worktrees\frontend-build`：若两个 agent
同时运行，后启动的一方在开头 `worktree remove --force` 会直接把先启动一方正在构建的 worktree 强拆，导致构建目录消失/互相踩踏。建议按 Agent/任务标识（如
`$env:AGENT_RUN_ID`）唯一化 worktree 名，或加互斥锁串行化构建。



─── scripts/build_frontend.ps1:13-13 ───
[bug · low] `$ErrorActionPreference = "Stop"` 下，若用 PowerShell 7.2+（pwsh）执行，原生命令 stderr
会被当作终止错误（`$PSNativeCommandUseErrorActionPreference` 在 7.2 默认开启）。npm 的 warn/进度信息常写 stderr，会导致构建实际成功却因
stderr 输出被中断，走不到 $LASTEXITCODE 检查。建议显式关闭该行为，确保跨 powershell/pwsh 版本行为一致。

  $ErrorActionPreference = "Stop"
+ $PSNativeCommandUseErrorActionPreference = $false


─── scripts/build_frontend.ps1:10-10 ───
[maintainability · low] 默认值 `D:\Quant\quant_ui` 是机器相关硬编码路径，换环境/换机器运行时容易误指向不存在的仓库。可推理脚本自身位置：脚本位于仓库
scripts/ 下，默认值用 `Split-Path -Parent $PSScriptRoot` 推导即可，同时保持可用参数覆盖。

-     [string]$RepoRoot = "D:\Quant\quant_ui",
+     [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),


─── scripts/build_frontend.ps1:26-27 ───
[bug · medium] `-SkipInstall` 开关实际无法生效：每次运行开头都会先 `worktree remove --force` 再重新 `worktree
add`，worktree 目录总是全新创建的，`node_modules` 永远不会预先存在。因此：1) 一旦传入 `-SkipInstall`，该条件直接为假，npm ci 被跳过，`npm
run build` 会在缺依赖的全新目录下运行，几乎必然以晦涩的模块缺失报错失败；2) 不传开关时 npm ci 又会每次全量重装，`Test-Path node_modules`
的缓存判断从未命中。建议：传入 `-SkipInstall` 时先检查 node_modules 是否存在，不存在则 throw 明确错误（如提示“请先不带 -SkipInstall
完整执行一次安装”）；若确需复用依赖，可考虑将主工作区 node_modules 复制/符号链接进 worktree，否则建议移除该开关避免误导。

-         if (-not $SkipInstall -and -not (Test-Path (Join-Path $staticDir "node_modules"))) {
+         if (Test-Path (Join-Path $staticDir "node_modules")) {
+             Write-Host "[2/3] 发现已有 node_modules，跳过 npm ci..."
+         } elseif ($SkipInstall) {
+             throw "已指定 -SkipInstall，但 worktree 是全新目录且无 node_modules，请先不带该开关完整安装一次"
+         } else {
              Write-Host "[2/3] npm ci（worktree 内首次安装，可能较慢）..."
+             & npm ci
+             if ($LASTEXITCODE -ne 0) { throw "npm ci 失败 (exit $LASTEXITCODE)" }
+         }


─── scripts/factorzoo_weekly.py:66-68 ───
[bug · medium] 后端 `/api/alphaagent/runs` 的 snapshot（backend/alphaagent_service.py:261）只含
`run_id/status/created_at`，并没有 `ended_at/updated_at/finished_at` 字段。因此这里对每个 run 都会无条件回退到
`created_at`，实际选归档阈值按的是「创建时间」而非帮助文本声称的「run 结束时间」——一个 20 天前创建、昨天才跑完的终态 run 会被当旧 run 立即归档。建议在后端
snapshot 中补充真实结束时间字段（可用事件轨迹末尾时间），或至少在此处去掉并不存在的字段并明示按 created_at 的口径。



─── scripts/factorzoo_weekly.py:101-103 ───
[bug · medium] 当 `_old/` 下已存在同名日志时，`shutil.move` 会直接覆盖旧归档：POSIX 上 os.rename 覆盖目标；Windows 上 rename
失败后会走 copy+unlink 回退同样覆盖。若日志使用固定文件名（如 app.log）且应用间歇停写，多周运行后历史归档会被新文件覆盖，与模块
docstring「移动而非删除，可随时找回」的承诺相悖。建议目标已存在时改名保留（如追加时间戳后缀）。

          if f.stat().st_mtime < cutoff:
-             shutil.move(str(f), str(dst / f.name))
+             dst_file = dst / f.name
+             if dst_file.exists():
+                 dst_file = dst / f"{f.stem}_{time.strftime('%Y%m%d%H%M%S')}{f.suffix}"
+             shutil.move(str(f), str(dst_file))
              moved += 1


─── scripts/factorzoo_weekly.py:155-157 ───
[bug · medium] 当前代码库中不存在 `ops_notify` 模块（scripts/ 及全仓库均未找到），`import ops_notify` 必然抛
ModuleNotFoundError，被外层 except 吞掉后每次都打印「通知失败（忽略）」，通知功能实际从未生效。若非本次 PR 未包含该模块文件，建议补充；否则应将缺失模块列为显式错误（非
ignore 级别）以免周维护报警通道静默失效。



─── scripts/factorzoo_weekly.py:109-111 ───
[bug · low] `run_dedup_dry` 仅处理了子进程返回码非零的情况；若 `dedup_candidate_factors.py`
被删除或解释器启动失败，`subprocess.run` 会抛出 FileNotFoundError/OSError，而 main 第 1 步未捕获，整个脚本会在归档/日志轮转/通知执行前崩溃，违背
docstring「dedup 失败不阻断后两项」的设计意图（当前文件虽存在，但该依赖是脆弱的硬引用）。建议用 try/except 包住并计入 results。

      cmd = [sys.executable, str(SCRIPTS / "dedup_candidate_factors.py"), "--dry-run"]
      print(f"$ {' '.join(cmd[1:])}", flush=True)
+     try:
-     return subprocess.run(cmd).returncode
+         return subprocess.run(cmd).returncode
+     except OSError as exc:
+         print(f"dedup 干跑启动失败：{exc}", flush=True)
+         return 127


─── scripts/factorzoo_weekly.py:31-31 ───
[maintainability · low]
模块级默认日志目录硬编码了个人机器的绝对路径（`C:\Users\zhoubw\...`），换用户/换机器运行时默认值即失效，且把个人路径写进仓库代码也不利于向他人分发。建议改为从环境变量（如
`FACTORZOO_LOG_DIR`）读取，缺省时回退到相对路径或空值（等价于不归档）。



─── scripts/factorzoo_weekly.py:162-162 ───
[other · low] main() 的退出码与通知级别完全由 dedup 干跑的 rc 决定（`return 0 if rc == 0 else rc`、`level="error" if rc
!= 0 else "info"`）。归档步骤即使存在 per-run 失败（`failed` 列表非空，例如个别 run 返回 5xx），也只是写进 results 文本，退出码仍为
0、通知级别仍为 info。对挂周计划任务的无人值守维护而言，「归档部分失败」才是真正需要调度器/告警感知的信号，当前会被误判为整体成功。建议：当 `failed`
非空（排除后端整体不可达的“跳过”）时让退出码非零并把通知级别升为 error，或至少将归档失败计入返回值语义。



LLM retry report summary: 1 of 40 requests affected -- 1 request recovered after retry

Review planning (1 request):
- scripts/factorzoo_weekly.py: provider error (HTTP 500) -> succeeded

Per-attempt detail: --format json (retry_report).
