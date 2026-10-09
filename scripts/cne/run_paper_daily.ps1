#Requires -Version 7
<#
.SYNOPSIS
  模拟盘每日定时执行（Windows / PowerShell 版）。
  排在 CNE 数据同步（16:30）之后，16:40 自动执行所有启用账户。

  逻辑：
    1. 探测后端 17891 是否在线（GET /api/paper/accounts）
    2. 后端在线 → POST /api/paper/run 执行全部启用账户
    3. 后端不在线 → 写日志跳过（非交易日也跳过）
    4. 重复执行同一交易日会自动跳过（幂等，paper_core 内置）

  幂等性：同一 exec_date 重复执行自动跳过；订单表带唯一约束。

.EXAMPLE
  .\run_paper_daily.ps1
  .\run_paper_daily.ps1 -ExecDate 2026-09-02
#>
[CmdletBinding()]
param(
    [string]$ExecDate = ""
)

$ErrorActionPreference = "Continue"

# ── 路径常量 ──────────────────────────────────────────────────────────
$RepoRoot   = "D:\Quant\quant_ui"
$LogDir     = Join-Path $RepoRoot "logs\paper"
$Stamp      = Get-Date -Format "yyyyMMdd"
$LogFile    = Join-Path $LogDir "paper-daily-$Stamp.log"
$ApiBase    = "http://127.0.0.1:17891/api"
$Py         = Join-Path $RepoRoot ".venv\Scripts\python.exe"

# ── 环境变量 ──────────────────────────────────────────────────────────
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$null = New-Item -ItemType Directory -Force -Path $LogDir

# ── 工具函数 ──────────────────────────────────────────────────────────
function Write-Log([string]$Msg) {
    $ts = Get-Date -Format "HH:mm:ss"
    $line = "[$ts] $Msg"
    Write-Host $line
    [System.IO.File]::AppendAllText($LogFile, "$line`n", [System.Text.Encoding]::UTF8)
}

# ── 主流程 ────────────────────────────────────────────────────────────
Write-Log "==== paper daily start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') exec_date=$(if ($ExecDate) { $ExecDate } else { 'today' }) ===="

# 1. 探测后端是否在线
try {
    $probe = Invoke-RestMethod -Uri "$ApiBase/paper/accounts" -Method GET -TimeoutSec 5
    $activeAccounts = @($probe | Where-Object { $_.status -eq 'active' })
    Write-Log "backend online, active accounts: $($activeAccounts.Count)"
    if ($activeAccounts.Count -eq 0) {
        Write-Log "no active accounts, skipping"
        Write-Log "==== paper daily DONE (no accounts) ===="
        exit 0
    }
} catch {
    # ── 后端离线：降级走离线重放（进程内 run_paper_trade，不依赖 17891）──
    Write-Log "backend offline (port 17891 not responding), falling back to offline replay"
    $offlineArgs = @("-X", "utf8", (Join-Path $RepoRoot "scripts\paper_offline_run.py"))
    if ($ExecDate) { $offlineArgs += $ExecDate }
    $out = & $Py @offlineArgs 2>&1
    $outText = ($out | Out-String).Trim()
    foreach ($line in ($outText -split "`n")) { if ($line.Trim()) { Write-Log "  [py] $($line.Trim())" } }
    try {
        $j = ($outText -split "`n" | Where-Object { $_.Trim().StartsWith("{") } | Select-Object -Last 1) | ConvertFrom-Json
        if ($j.ok) {
            Write-Log "offline replay OK: run_date=$($j.run_date), accounts=$(@($j.accounts).Count)"
            Write-Log "==== paper daily DONE (offline) ===="
            exit 0
        }
        Write-Log "offline replay FAILED: $($j.error)"
        Write-Log "==== paper daily DONE (offline failed) ===="
        exit 1
    } catch {
        Write-Log "offline replay FAILED: unparseable output"
        Write-Log "==== paper daily DONE (offline failed) ===="
        exit 1
    }
}

# 2. 执行模拟盘
$body = @{}
if ($ExecDate) { $body["exec_date"] = $ExecDate }
$jsonBody = $body | ConvertTo-Json -Compress

try {
    Write-Log "POST /api/paper/run ..."
    $result = Invoke-RestMethod -Uri "$ApiBase/paper/run" -Method POST `
        -ContentType "application/json" -Body $jsonBody -TimeoutSec 300

    $runDate = $result.run_date
    $accs = $result.accounts
    Write-Log "run_date: $runDate, accounts processed: $($accs.Count)"

    foreach ($a in $accs) {
        if ($a.error) {
            Write-Log "  account $($a.id): ERROR $($a.error)"
        } else {
            Write-Log "  account $($a.id): OK"
        }
    }
    Write-Log "==== paper daily DONE ok ===="
    exit 0
} catch {
    $errMsg = $_.Exception.Message
    Write-Log "paper run FAILED: $errMsg"
    Write-Log "==== paper daily DONE (failed) ===="
    exit 1
}
