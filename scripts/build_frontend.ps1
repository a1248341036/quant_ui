# 前端 worktree 构建一条龙（AGENTS.md 构建纪律的脚本化）
# 解决的问题：共享工作区检出分支随并行 agent 漂移，直接在共享工作区 npm run build
# 的产物 = "碰巧在哪个 agent 手里"的分支。本脚本强制在包含目标提交的独立 worktree
# 里构建，构建成功才把 static/dist 拷回主工作区（失败不动 dist）。
# 并行安全：worktree 目录带时间戳唯一化，多个 agent 同时构建互不踩踏；
# 崩溃残留的注册 worktree 可用 `git worktree list` 找到后手动 remove。
#
# 用法：
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_frontend.ps1 [-Ref main] [-SkipInstall]
param(
    [string]$Ref = "main",
    [string]$RepoRoot = (Split-Path -Parent $PSScriptRoot),   # 缺省=脚本所在仓库根
    [switch]$SkipInstall                                      # 仅当 worktree 内已有 node_modules 时有效
)
$ErrorActionPreference = "Stop"
$PSNativeCommandUseErrorActionPreference = $false   # pwsh 7.2+：npm 的 stderr 进度不当终止错误

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$wt = Join-Path $RepoRoot ".worktrees\frontend-build-$stamp"
& git -C $RepoRoot worktree add --detach $wt $Ref 2>$null
if ($LASTEXITCODE -ne 0) { throw "git worktree add 失败: $Ref (exit $LASTEXITCODE)" }
Write-Host "[1/3] worktree 就绪: $wt @ $Ref"

try {
    $staticDir = Join-Path $wt "static"
    Push-Location $staticDir
    try {
        if (Test-Path (Join-Path $staticDir "node_modules")) {
            Write-Host "[2/3] worktree 内已有 node_modules，跳过 npm ci..."
        } elseif ($SkipInstall) {
            throw "已指定 -SkipInstall，但 worktree 是全新目录且无 node_modules；请先不带该开关完整执行一次"
        } else {
            Write-Host "[2/3] npm ci（worktree 内首次安装，可能较慢）..."
            & npm ci
            if ($LASTEXITCODE -ne 0) { throw "npm ci 失败 (exit $LASTEXITCODE)" }
        }
        Write-Host "[2/3] npm run build（在 worktree 内，产物 = $Ref 的源码）..."
        & npm run build
        if ($LASTEXITCODE -ne 0) { throw "npm run build 失败 (exit $LASTEXITCODE)" }
    } finally {
        Pop-Location
    }

    # 构建成功且确认产出 dist 才动主工作区；旧 dist 留 .bak（仅保留最近一次）
    $distSrc = Join-Path $wt "static\dist"
    if (-not (Test-Path $distSrc)) { throw "构建未产出 dist，已保留主工作区原有产物" }
    $dist = Join-Path $RepoRoot "static\dist"
    if (Test-Path "$dist.bak") { Remove-Item "$dist.bak" -Recurse -Force }
    if (Test-Path $dist) { Rename-Item $dist "$dist.bak" }
    Copy-Item $distSrc $dist -Recurse
    Write-Host "[3/3] 已拷回 $dist（旧产物在 dist.bak，确认页面正常后可删）"
} finally {
    & git -C $RepoRoot worktree remove --force $wt 2>$null | Out-Null
    & git -C $RepoRoot worktree prune | Out-Null
}
