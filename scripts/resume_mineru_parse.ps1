# MinerU 全量解析：一键续跑（重启电脑后用这个）
#
#   pwsh -File D:\Quant\quant_ui\scripts\resume_mineru_parse.ps1
#
# 做三件事（幂等，可反复执行）：
#   1) 已有驱动在跑 → 直接提示退出，不重复启动
#   2) MinerU 服务没起 → 拉起并等就绪（最多 90s）
#   3) 以「2020 年起、最新优先、2 并发、脱离会话、无黑框」启动续跑（已完成篇自动跳过）
param(
    [string]$Root = "D:\Quant\quant_ui",
    [int]$MinYear = 2020,
    [int]$Workers = 2,
    [string]$Mineru = ""    # 留空时取 $env:MINERU_EXE，再退回本机默认安装路径
)

$mineru = if ($Mineru) { $Mineru } elseif ($env:MINERU_EXE) { $env:MINERU_EXE } else { "D:\Quant\MinerU\.venv\Scripts\mineru.exe" }
$logDir = Join-Path $Root "logs"
$out = Join-Path $logDir "batch_mineru_recent.out"
# 日志轮转：Start-Process 的重定向是**覆盖**写，直接续跑会把上一轮日志清空
# （2026-09-30 review 修）→ 先把上一轮 recent 归档为带时间戳的名字，再开新 recent。
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
foreach ($f in @($out, (Join-Path $logDir "batch_mineru_recent.err"))) {
    if (Test-Path $f) {
        $stamp = (Get-Item $f).LastWriteTime.ToString("yyyyMMdd_HHmmss")
        Move-Item -LiteralPath $f -Destination ($f -replace "_recent\.", "_$stamp.") -Force
    }
}
$env:MINERU_MODEL_SOURCE = "modelscope"
$env:MINERU_EXE = $mineru    # 传给子进程；batch_mineru_parse.py 读该变量

# 1) 幂等：已有驱动就不重复起
$running = Get-CimInstance Win32_Process -Filter "Name='pythonw.exe' or Name='python.exe'" |
    Where-Object { $_.CommandLine -match "batch_mineru_parse" }
if ($running) {
    Write-Host "批次已在运行 (PID $($running[0].ProcessId))，不重复启动。" -ForegroundColor Yellow
    Write-Host "查进度: pwsh -File $Root\scripts\mineru_progress.ps1"
    exit 0
}

# 2) 服务
$ok = $false
& $mineru server status *> $null
if ($LASTEXITCODE -eq 0) { $ok = $true }
if (-not $ok) {
    Write-Host "启动 MinerU 服务…" -ForegroundColor Cyan
    Start-Process -FilePath $mineru -ArgumentList "server", "start" -WindowStyle Hidden | Out-Null
    for ($i = 0; $i -lt 18; $i++) {
        Start-Sleep -Seconds 5
        & $mineru server status *> $null
        if ($LASTEXITCODE -eq 0) { $ok = $true; break }
    }
}
if (-not $ok) { Write-Host "服务启动失败，请手动执行: $mineru server start" -ForegroundColor Red; exit 1 }
Write-Host "服务就绪。" -ForegroundColor Green

# 3) 续跑（脱离会话，子进程 CREATE_NO_WINDOW 不弹黑框）
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$b = Start-Process -FilePath (Join-Path $Root ".venv\Scripts\pythonw.exe") `
    -ArgumentList "scripts\batch_mineru_parse.py", "--workers", "$Workers", "--min-year", "$MinYear", "--order", "recent" `
    -WorkingDirectory $Root -RedirectStandardOutput $out `
    -RedirectStandardError (Join-Path $logDir "batch_mineru_recent.err") -WindowStyle Hidden -PassThru
Write-Host "已启动续跑，驱动 PID=$($b.Id)（2020+, 最新优先, $Workers 并发）" -ForegroundColor Green
Write-Host "查进度: pwsh -File $Root\scripts\mineru_progress.ps1"
