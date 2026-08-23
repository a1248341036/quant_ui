#Requires -Version 7
<#
.SYNOPSIS
  CNE 数据湖每日流水线（Windows / PowerShell 版）。
  对标 CNEquity/scripts/daily_pipeline.sh，功能一致：
    1. 按 wave 依赖顺序依次执行 cne run daily（前台输出实时进度 + 写日志）
    2. stale 补抓（可选）
    3. health check（audit + status）
    4. meta 备份
    5. staging 清理
  一个 gate wave 失败 → 脚本 exit 1；soft wave 失败只告警。
  非交易日运行为 no-op（cne 内部跳过，exit 0）。

  事件/财务数据（balancesheet/income/cashflow/fina_indicator/report_rc/
  dividend/share_float_external/namechange/forecast/express/stk_surv）
  已迁移为 CNE curated step，由 events/fundamentals wave 内部抓取，
  不再需要外部 sync_tushare_to_parquet.py 同步。

.PARAMETER TradeDate
  指定交易日补跑，格式 YYYY-MM-DD，默认 today。

.PARAMETER NoStaleRetry
  跳过 stale 补抓环节。

.PARAMETER NoBackup
  跳过 meta 备份。

.PARAMETER Quiet
   传给 cne run daily --quiet，只输出 WARNING 及以上。

.PARAMETER SkipClean
   跳过 staging 清理。

.PARAMETER SkipEtfFund
   跳过 ETF/基金/指数刷新（refresh_data.py）。

.EXAMPLE
  .\run_cne_daily.ps1
  .\run_cne_daily.ps1 -TradeDate 2026-08-20
  .\run_cne_daily.ps1 -SkipEtfFund   # 只跑 CNE 流水线
#>
[CmdletBinding()]
param(
    [string]$TradeDate = "",
    [switch]$NoStaleRetry,
    [switch]$NoBackup,
    [switch]$Quiet,
    [switch]$SkipClean,
    [switch]$SkipEtfFund
)

$ErrorActionPreference = "Stop"

# ── 路径常量 ──────────────────────────────────────────────────────────
$RepoRoot   = "D:\Quant\quant_ui"
$CneRoot    = Join-Path $RepoRoot "CNEquity"
$Cne        = Join-Path $RepoRoot ".venv\Scripts\cne.exe"
$Config     = Join-Path $CneRoot "configs\cnequity.quant_dataset.toml"
$LogDir     = Join-Path $CneRoot "data\cnequity\logs"
$BackupDir  = Join-Path $CneRoot "data\cnequity\backups"
$Py         = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$RefreshData= Join-Path $RepoRoot "scripts\refresh_data.py"

# 额外日志（终端已有实时输出，此文件做留底）
$Stamp   = Get-Date -Format "yyyyMMdd"
$LogFile = Join-Path $LogDir "daily-$Stamp.log"

$null = New-Item -ItemType Directory -Force -Path $LogDir
$null = New-Item -ItemType Directory -Force -Path $BackupDir

# ── 环境变量 ──────────────────────────────────────────────────────────
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── 配置 ──────────────────────────────────────────────────────────────
# wave 依赖顺序（core 先跑提供 instruments，后续 wave 依赖它）
$WaveList      = @("core", "fundamentals", "events", "capital", "macro_risk", "research", "finalize")
$GateWaves     = @("core", "finalize")
$SoftFailOk    = $true   # gate OK 时 soft wave 失败只告警
$StaleRetry    = -not $NoStaleRetry
$StaleDelaySec = 1800    # stale 补抓前等待秒数

# ── 工具函数 ──────────────────────────────────────────────────────────
function Write-Log([string]$Msg) {
    $ts = Get-Date -Format "HH:mm:ss"
    $line = "[$ts] $Msg"
    Write-Host $line
    $line | Out-File -FilePath $LogFile -Append -Encoding UTF8
}

function Invoke-Cne([string[]]$Args) {
    if ($Quiet -and ($Args -contains "run" -and $Args -contains "daily")) {
        $Args = @($Args) + "--quiet"
    }
    & $Cne @Args --config $Config
    return $LASTEXITCODE
}

function Invoke-CneWithLog([string[]]$Args) {
    # 前台实时输出 + tee 到日志
    if ($Quiet -and ($Args -contains "run" -and $Args -contains "daily")) {
        $Args = @($Args) + "--quiet"
    }
    $fullArgs = @($Args) + "--config", $Config

    # 前台输出同时写日志：用 Tee 流
    $pipeLine = & $Cne @fullArgs 2>&1 |
        ForEach-Object {
            $line = $_.ToString()
            Write-Host $line
            $line | Out-File -FilePath $LogFile -Append -Encoding UTF8
        }
    return $LASTEXITCODE
}

function Invoke-PyWithLog([string[]]$Args, [string]$JobName) {
    # 前台输出 + tee 到日志
    Write-Log "$JobName start"
    $fullArgs = @($Args)
    & $Py @fullArgs 2>&1 |
        ForEach-Object {
            $line = $_.ToString()
            Write-Host $line
            $line | Out-File -FilePath $LogFile -Append -Encoding UTF8
        }
    $exitCode = $LASTEXITCODE
    if ($exitCode -eq 0) {
        Write-Log "$JobName OK"
    } else {
        Write-Log "$JobName FAILED (exit=$exitCode)"
    }
    return $exitCode
}

function Invoke-EtfFund {
    # 刷新 ETF/基金/指数面板（refresh_data.py：ETF/基金 akshare + 腾讯行情）
    # --skip-stock-panel:        跳过腾讯股票日线抓取（股票行情由 CNE tushare_wide 承担）
    # --no-sync-pg:              不同步 Tushare 日线到 pg_parquet（避免与 CNE daily 重复）
    # --no-rebuild-panel:        不重建股票 panel.parquet（股票 panel 由 CNE daily_bars 提供）
    # 这样 refresh_data 只负责 ETF/基金/指数，股票相关完全交给 CNE 流水线。
    $args = @($RefreshData, "--skip-stock-panel", "--no-sync-pg", "--no-rebuild-panel")
    return Invoke-PyWithLog $args "quant_ui:refresh_data"
}

function Backup-Meta {
    if ($NoBackup) { return }
    $metaDir = Join-Path $CneRoot "data\cnequity\meta"
    if (-not (Test-Path $metaDir)) {
        Write-Log "backup: meta dir not found: $metaDir"
        return
    }
    $ts = Get-Date -Format "yyyyMMdd-HHmmss"
    $archive = Join-Path $BackupDir "meta-$ts.zip"

    # 用 sqlite3 备份 manifest.db（避免撕裂），没有就用 Copy-Item
    $manifest = Join-Path $metaDir "manifest.db"
    $tempDir  = Join-Path $env:TEMP "cne_backup_$ts"
    $null = New-Item -ItemType Directory -Force -Path $tempDir

    if (Test-Path $manifest) {
        $sqliteExe = Get-Command sqlite3 -ErrorAction SilentlyContinue
        if ($sqliteExe) {
            & sqlite3 $manifest ".backup '$(Join-Path $tempDir 'manifest.db')'"
        } else {
            Copy-Item $manifest (Join-Path $tempDir "manifest.db")
        }
    }
    foreach ($sub in @("state", "quality")) {
        $src = Join-Path $metaDir $sub
        if (Test-Path $src) {
            Copy-Item $src $tempDir -Recurse
        }
    }

    # 压缩
    Compress-Archive -Path (Join-Path $tempDir "*") -DestinationPath $archive -Force
    Remove-Item $tempDir -Recurse -Force

    # 清理 14 天前的备份
    $cutoff = (Get-Date).AddDays(-14)
    Get-ChildItem $BackupDir -Filter "meta-*.zip" |
        Where-Object { $_.LastWriteTime -lt $cutoff } |
        Remove-Item -Force

    $size = [math]::Round((Get-Item $archive).Length / 1KB, 1)
    Write-Log "backup: wrote $archive (${size}KB; retention 14d)"
}

# ── 主流程 ────────────────────────────────────────────────────────────
$dateArg = @()
if ($TradeDate) { $dateArg = @("--trade-date", $TradeDate) }

Write-Log "==== CNE daily pipeline start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') trade_date=$($TradeDateOrDefault ?? 'today') ===="

# 先 reconcile：清理上次崩溃残留的 running 状态
Write-Log "--- reconcile stale runs ---"
$null = Invoke-CneWithLog @("clean", "--reconcile-runs", "--dry-run")

# ── 1. 按 wave 顺序执行 ──────────────────────────────────────────────
$failedGates  = @()
$failedSoft   = @()
$summary      = [System.Collections.Generic.List[pscustomobject]]::new()

foreach ($wave in $WaveList) {
    Write-Log "--- wave: $wave ---"

    $exitCode = Invoke-CneWithLog (@("run", "daily", "--group", $wave) + $dateArg)

    $isGate = $GateWaves -contains $wave
    if ($exitCode -eq 0) {
        Write-Log "wave $wave OK"
        $summary.Add([pscustomobject]@{ Wave = $wave; Status = "OK"; Kind = $(if ($isGate) { "gate" } else { "soft" }) })
    } else {
        Write-Log "wave $wave FAILED (exit=$exitCode, see $LogFile)"
        $summary.Add([pscustomobject]@{ Wave = $wave; Status = "FAILED"; Kind = $(if ($isGate) { "gate" } else { "soft" }) })
        if ($isGate) {
            $failedGates += $wave
        } else {
            $failedSoft += $wave
        }
    }
}

# ── 2. stale 补抓 ────────────────────────────────────────────────────
$staleStatus = "skipped"
if ($StaleRetry -and $failedGates.Count -eq 0) {
    Write-Log "--- stale probe ---"
    $staleExit = Invoke-Cne @("status", "--datasets")

    if ($staleExit -eq 0) {
        Write-Log "nothing stale — no retry needed"
        $staleStatus = "not needed"
    } else {
        Write-Log "something is stale; waiting ${StaleDelaySec}s before re-fetching"
        Start-Sleep -Seconds $StaleDelaySec
        Write-Log "--- stale retry ---"
        $retryExit = Invoke-CneWithLog (@("run", "daily", "--stale-only") + $dateArg)
        if ($retryExit -eq 0) {
            Write-Log "stale retry OK"
            $staleStatus = "OK"
        } else {
            Write-Log "stale retry FAILED (see $LogFile)"
            $staleStatus = "FAILED"
            $failedSoft += "stale-retry"
        }
    }
}

# ── 2.5 ETF/基金/指数刷新（可选）───────────────────────────────────
$etfStatus = "skipped"
if (-not $SkipEtfFund) {
    Write-Log "--- ETF/基金/指数刷新 ---"
    $etfExit = Invoke-EtfFund
    if ($etfExit -eq 0) {
        $etfStatus = "OK"
    } else {
        $etfStatus = "FAILED"
        $failedSoft += "etf-fund"
    }
}

# ── 3. health check ──────────────────────────────────────────────────
Write-Log "--- health check ---"
$auditExit = Invoke-CneWithLog @("audit", "--full")
# status --datasets 退出码 1 表示有 STALE，不一定是硬错误
$statusExit = Invoke-CneWithLog @("status", "--datasets")
if ($auditExit -ne 0) {
    Write-Log "health check: audit reported problems (exit=$auditExit)"
}

# ── 4. meta 备份 ─────────────────────────────────────────────────────
Write-Log "--- backup ---"
Backup-Meta

# ── 5. staging 清理 ──────────────────────────────────────────────────
if (-not $SkipClean) {
    Write-Log "--- clean staging ---"
    $cleanExit = Invoke-CneWithLog @("clean")
    if ($cleanExit -ne 0) {
        Write-Log "staging cleanup FAILED (non-fatal)"
    }
}

# ── 汇总 ─────────────────────────────────────────────────────────────
Write-Log "---- wave summary (gate=$($GateWaves -join ',')) ----"
foreach ($s in $summary) {
    Write-Log ("  {0}: {1}  [{2}]" -f $s.Wave, $s.Status, $s.Kind)
}
Write-Log "  stale-retry: $staleStatus"
Write-Log "  etf-fund-refresh: $etfStatus"

if ($failedGates.Count -gt 0) {
    $softStr = if ($failedSoft.Count -gt 0) { $failedSoft -join ", " } else { "none" }
    Write-Log "==== daily pipeline DONE — GATE FAILED: $($failedGates -join ', ') (soft also: $softStr) ===="
    exit 1
}
if ($failedSoft.Count -gt 0) {
    if ($SoftFailOk) {
        Write-Log "==== daily pipeline DONE — gate OK, soft FAILED (warn-only): $($failedSoft -join ', ') ===="
        exit 0
    }
    Write-Log "==== daily pipeline DONE — gate OK, soft FAILED: $($failedSoft -join ', ') ===="
    exit 1
}

Write-Log "==== daily pipeline DONE ok ===="
exit 0
