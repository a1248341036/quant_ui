# 前端 worktree 构建一条龙（AGENTS.md 构建纪律的脚本化）
# 解决的问题：共享工作区检出分支随并行 agent 漂移，直接在共享工作区 npm run build
# 的产物 = "碰巧在哪个 agent 手里"的分支。本脚本强制在包含目标提交的独立 worktree
# 里构建，构建成功才把 static/dist 拷回主工作区（失败不动 dist）。
#
# 用法：
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts\build_frontend.ps1 [-Ref main] [-SkipInstall]
param(
    [string]$Ref = "main",                                  # 要构建的提交/分支
    [string]$RepoRoot = "D:\Quant\quant_ui",
    [switch]$SkipInstall                                    # worktree 里已有 node_modules 时跳过安装
)
$ErrorActionPreference = "Stop"

$wt = Join-Path $RepoRoot ".worktrees\frontend-build"
# 上次残留的注册 worktree 先清掉（幂等）
& git -C $RepoRoot worktree remove --force $wt 2>$null | Out-Null
& git -C $RepoRoot worktree prune | Out-Null
& git -C $RepoRoot worktree add --detach $wt $Ref | Out-Null
Write-Host "[1/3] worktree 就绪: $wt @ $Ref"

try {
    $staticDir = Join-Path $wt "static"
    Push-Location $staticDir
    try {
        if (-not $SkipInstall -and -not (Test-Path (Join-Path $staticDir "node_modules"))) {
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

    # 构建成功才拷回；旧 dist 先留一份 .bak（仅保留最近一次，下次覆盖）
    $dist = Join-Path $RepoRoot "static\dist"
    if (Test-Path "$dist.bak") { Remove-Item "$dist.bak" -Recurse -Force }
    if (Test-Path $dist) { Rename-Item $dist "$dist.bak" }
    Copy-Item (Join-Path $wt "static\dist") $dist -Recurse
    Write-Host "[3/3] 已拷回 $dist（旧产物在 dist.bak，确认页面正常后可删）"
} finally {
    & git -C $RepoRoot worktree remove --force $wt 2>$null | Out-Null
    & git -C $RepoRoot worktree prune | Out-Null
}
