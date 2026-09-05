#Requires -Version 7
<#
.SYNOPSIS
  Post-wipe curated deep-history rebuild. Two source-separated lanes run in
  PARALLEL; within each lane datasets run serially (same-source limiter is
  the shared bottleneck).
  Lane eastmoney: EastMoney + THS + snapshot datasets.
  Lane tushare:   tushare event/financial + minute-bar curated registration.
#>
[CmdletBinding()]
param(
    [string]$Start = "2016-01-01",
    [switch]$Resume,   # skip steps whose marker file exists
    [ValidateSet("eastmoney", "tushare")]
    [string]$Lane = "eastmoney",
    [string]$Datasets = ""   # comma-separated override; runs only these (args looked up from the lane tables)
)

$ErrorActionPreference = "Continue"
# Decode native cne output (UTF-8) correctly under scheduled-task pwsh.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
# --- Load Tushare credentials from .env (no hardcoded secrets) ---
$EnvFile = Join-Path (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)) ".env"
if (Test-Path $EnvFile) {
    Get-Content $EnvFile | Where-Object { $_ -match '^\s*(TUSHARE_TOKEN|TUSHARE_URL)\s*=' } | ForEach-Object {
        $kv = $_ -split '=', 2
        $name = $kv[0].Trim(); $val = $kv[1].Trim().Trim('"').Trim("'")
        Set-Item -Path "Env:$name" -Value $val
    }
}
$Cne    = "D:\Quant\quant_ui\.venv\Scripts\cne.exe"
$Config = "D:\Quant\quant_ui\CNEquity\configs\cnequity.quant_dataset.toml"
$MarkerDir = "D:\Quant\quant_ui\CNEquity\data\cnequity\backfill_markers"
$null = New-Item -ItemType Directory -Force -Path $MarkerDir
$Log = "D:\Quant\quant_ui\CNEquity\data\cnequity\logs\rebuild-backfill-$Lane-$(Get-Date -Format 'yyyyMMdd').log"

function Write-Log([string]$Msg) {
    $line = "[$(Get-Date -Format 'MM-dd HH:mm:ss')][$Lane] $Msg"
    Write-Host $line
    [System.IO.File]::AppendAllText($Log, "$line`n", [System.Text.Encoding]::UTF8)
}

if ($Lane -eq "eastmoney") {
    $queue = [ordered]@{
        # --- 快照/小表（秒级～分钟级） ---
        "share_unlock_schedule"   = @()
        # 注：market_breadth / commodity_bars / hot_rank / news_headlines /
        # flash_news_wire / sector_fund_flow / sector_members 是纯快照语义，
        # backfill 明确不支持（源端无历史），不从队列重试。
        # --- date-walking（逐日一请求；底层统一走 tushare 代理，需 token env） ---
        "dragon_tiger"            = @("--start", $Start, "--workers", "2")
        "block_trades"            = @("--start", $Start, "--workers", "2")
        "margin_trading"          = @("--start", $Start, "--workers", "2")
        "fund_flow"               = @("--start", $Start, "--workers", "2")
        "northbound_holdings"     = @("--start", $Start, "--workers", "2")
        "valuation_metrics"       = @("--start", $Start, "--workers", "2")
        "trading_status"          = @("--start", $Start, "--workers", "2")
        "analyst_consensus"       = @("--start", $Start, "--workers", "2")
        "announcement_index"      = @("--start", $Start, "--workers", "2")
        "institutional_holdings"  = @("--start", $Start, "--workers", "2")
        # --- THS（独立源，本 lane 尾部） ---
        "sector_bars"             = @("--start", $Start)
    }
} else {
    $queue = [ordered]@{
        # --- tushare 事件/财务（源有全历史） ---
        "dividend"                = @("--start", "2009-01-01")
        "forecast"                = @("--start", "2009-01-01")
        "express"                 = @("--start", "2009-01-01")
        "namechange"              = @("--start", "2009-01-01")
        "share_float_external"    = @("--start", "2009-01-01")
        "stk_surv"                = @("--start", "2009-01-01")
        "share_structure"         = @("--start", "2009-01-01")
        "shareholder_counts"      = @("--start", "2009-01-01")
        "top_holders"             = @("--start", "2009-01-01")
        "financial_statement_items" = @("--start", "2009-01-01")
        "earnings_disclosure_schedule" = @("--start", "2009-01-01")
        "report_rc"               = @("--start", "2016-01-01")
        "delisting_events"        = @()
        # --- 分钟线 curated 登记（external 本体在本地） ---
        "minute_bars"             = @()
        "minute_bars_5m"          = @()
    }
}

if ($Datasets) {
    # Explicit subset: pull each dataset's args from either lane table (unknown
    # datasets get no extra args -- the token env + --start still apply).
    $list = $Datasets.Split(',') | ForEach-Object { $_.Trim() } | Where-Object { $_ }
    $override = [ordered]@{}
    foreach ($d in $list) {
        if ($queue.Contains($d)) { $override[$d] = $queue[$d] }
        else { $override[$d] = @("--start", $Start) }
    }
    $queue = $override
}

$total = $queue.Keys.Count
$i = 0
foreach ($ds in $queue.Keys) {
    $i++
    $marker = Join-Path $MarkerDir "$ds.done"
    if ($Resume -and (Test-Path $marker)) {
        Write-Log "[$i/$total] $ds SKIP (marker exists)"
        continue
    }
    $extra = $queue[$ds]
    Write-Log "[$i/$total] backfill $ds start $($extra -join ' ')"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Cne backfill $ds @extra --config $Config 2>&1 | Out-Null
    $code = $LASTEXITCODE
    $sw.Stop()
    if ($code -eq 0) {
        Set-Content -Path $marker -Value "ok $(Get-Date -Format s)"
        Write-Log "[$i/$total] backfill $ds OK ($([math]::Round($sw.Elapsed.TotalMinutes,1)) min)"
    } else {
        Write-Log "[$i/$total] backfill $ds FAILED exit=$code ($([math]::Round($sw.Elapsed.TotalMinutes,1)) min) — continuing"
    }
}
Write-Log "==== lane $Lane done ===="
