#Requires -Version 7
<#
.SYNOPSIS
  18:00 晚间 stale 补抓（独立计划任务，避免 16:30 流水线在 17:00 睡眠时
  丢失的尾部重试）。macro_indicators 等日频序列在 16:30 尚未发布当日值，
  18:00 时通常已出，此任务把水位推进、消除 STALE。
#>
[CmdletBinding()]
param()

$ErrorActionPreference = "Continue"
$CneRoot = "D:\Quant\quant_ui\CNEquity"
$Cne = "D:\Quant\quant_ui\CNEquity\.venv\Scripts\cne.exe"
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

# 错过补跑（开机/唤醒后 StartWhenAvailable 触发）时，交易日历判断在 cne 内部，
# 非交易日 stale-only 是 no-op。
Write-Log "==== stale retry start ===="
& $Cne run daily --stale-only --config $Config 2>&1 | ForEach-Object { Write-Log $_.ToString() }
Write-Log "stale-only exit=$LASTEXITCODE"

# 顺带清理 staging（保留最近 2 天外的）
& $Cne clean --config $Config 2>&1 | ForEach-Object { Write-Log $_.ToString() }
Write-Log "clean exit=$LASTEXITCODE"
Write-Log "==== stale retry done ===="
