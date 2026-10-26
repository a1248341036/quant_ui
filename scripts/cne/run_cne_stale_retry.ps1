#Requires -Version 7
<#
.SYNOPSIS
  23:00 晚间 stale 补抓（独立计划任务 QuantUIDataStaleRetry，排在
  21:30 日更流水线之后 1.5 小时）：把流水线收尾时仍滞后/失败的批次再抓
  一遍，推进水位、消除 STALE；全量已对齐时为 no-op。

  2026-10-25 补：整晚没跑（关机/休眠错过 21:30）时，下一个触发点可能是周末，
  非交易日会被直接 skip，`--stale-only` 用"今天"做 trade_date 同样被 skip，
  于是整个交易日静默丢失（2026-10-23 就是这样丢的）。这里先探
  `cne status --datasets`，发现水位落后于上一个交易日时先
  `cne run catchup --trade-date <anchor>` 补齐，再补滞后项。
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"
$RepoRoot = "D:\Quant\quant_ui"
$CneRoot  = Join-Path $RepoRoot "CNEquity"
$Cne      = Join-Path $RepoRoot ".venv\Scripts\cne.exe"
$Py       = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Config = Join-Path $CneRoot "configs\cnequity.quant_dataset.toml"
$LogDir = Join-Path $CneRoot "data\cnequity\logs"
$Stamp = Get-Date -Format "yyyyMMdd"
$LogFile = Join-Path $LogDir "stale-retry-$Stamp.log"
$null = New-Item -ItemType Directory -Force -Path $LogDir

function Write-Log([string]$Msg) {
    $line = "[$(Get-Date -Format 'HH:mm:ss')] $Msg"
    Write-Host $line
    try { [System.IO.File]::AppendAllText($LogFile, "$line`n", [System.Text.Encoding]::UTF8) } catch {}
}

Write-Log "==== stale retry start ===="

# ── 1) 探测：退出码非 0 表示有数据集落后上一个交易日（含"整晚没跑"）───────
$anchor = ""
$probeExit = 0
try {
    $probe = & $Cne status --datasets --config $Config 2>&1
    $probeExit = $LASTEXITCODE
    foreach ($line in $probe) {
        $text = $line.ToString()
        if ($text -match 'last trading day:\s*(\d{4}-\d{2}-\d{2})') {
            $anchor = $matches[1]
            break
        }
    }
} catch {
    Write-Log "probe failed: $($_.Exception.Message)"
}
Write-Log "probe exit=$probeExit anchor=$anchor"

$today = Get-Date -Format "yyyy-MM-dd"
$missed = ($probeExit -ne 0 -and $anchor -and $anchor -lt $today)
if ($missed) {
    Write-Log "missed session $anchor — catchup core before stale-only"
    & $Cne run catchup --trade-date $anchor --config $Config 2>&1 | ForEach-Object { Write-Log $_.ToString() }
    Write-Log "catchup exit=$LASTEXITCODE"
}

# ── 2) stale-only：漏跑时把 trade_date 钉在 anchor（今天可能是周末会被 skip）──
$staleArgs = @("run", "daily", "--stale-only", "--config", $Config)
if ($missed) { $staleArgs += @("--trade-date", $anchor) }
& $Cne @staleArgs 2>&1 | ForEach-Object { Write-Log $_.ToString() }
Write-Log "stale-only exit=$LASTEXITCODE"

# fund_nav 补抓成功后重建基金面板（fund_bars 数据源），否则 fund_bars 会
# 持续显示 STALE 直到次日 21:30 流水线才重建。
Write-Log "rebuild fund panel"
& $Py -c 'import sys; sys.path.insert(0, "D:/Quant/quant_ui"); from core.updater import rebuild_fund_panel; rebuild_fund_panel()' 2>&1 | ForEach-Object { Write-Log $_.ToString() }
Write-Log "rebuild fund panel exit=$LASTEXITCODE"

# 顺带清理 staging（保留最近 2 天外的）
& $Cne clean --config $Config 2>&1 | ForEach-Object { Write-Log $_.ToString() }
Write-Log "clean exit=$LASTEXITCODE"
Write-Log "==== stale retry done ===="
