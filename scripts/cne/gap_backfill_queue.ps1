#Requires -Version 7
<#
.SYNOPSIS
  Final post-wipe gap backfill (2026-09-05). Only datasets still flagged by
  `cne verify` after: A-class local materialization + leftover staging recovery.
  Two source-separated lanes run in PARALLEL; within a lane, serial.
    Lane eastmoney: EM datacenter/F10 datasets (incl. shareholders family).
    Lane misc:      tushare (delisting), pboc (macro), local-derive attempts.
  Excluded on purpose:
    - daily_bars / minute_bars / etf_bars / fund_bars / fund_nav / fund_list /
      etf_list / fund_fees  -> external/local-asset bridges, verify false positive
    - analyst_consensus / hot_rank / sector_fund_flow / sector_members -> source
      has no history (permanently lost)
    - adj_factors -> phantom gap; real coverage 4294/4294 matches bar evidence
    - sentiment_scores -> filled by the daily research group (local scoring)
#>
[CmdletBinding()]
param(
    [switch]$Resume,   # skip steps whose marker file exists
    [ValidateSet("eastmoney", "misc", "fundamentals")]
    [string]$Lane = "eastmoney"
)

$ErrorActionPreference = "Continue"
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
$Log = "D:\Quant\quant_ui\CNEquity\data\cnequity\logs\gap-backfill-$Lane-$(Get-Date -Format 'yyyyMMdd').log"

function Write-Log([string]$Msg) {
    $line = "[$(Get-Date -Format 'MM-dd HH:mm:ss')][$Lane] $Msg"
    Write-Host $line
    [System.IO.File]::AppendAllText($Log, "$line`n", [System.Text.Encoding]::UTF8)
}

if ($Lane -eq "eastmoney") {
    $queue = [ordered]@{
        # date-walking (one request per day)
        "announcement_index"           = @("--start", "2016-01-01", "--workers", "2")
        "institutional_holdings"       = @("--start", "2016-01-01", "--workers", "2")
        "northbound_holdings"          = @("--start", "2016-01-01", "--workers", "2")
        "fund_flow"                    = @("--start", "2016-01-01", "--workers", "2")
        # curated ends 2022-06-27; extend from there
        "margin_trading"               = @("--start", "2022-06-28", "--workers", "2")
        "regulatory_events"            = @("--start", "2016-01-01")
    }
} elseif ($Lane -eq "fundamentals") {
    # Re-queued 2026-09-05 16:xx: the first pass "succeeded" with 0 rows because
    # a shadowing skip-step registration ate the real EastMoney steps (fixed on
    # main, bc40a15). Watermark JSONs cleared; these now actually fetch.
    $queue = [ordered]@{
        "financial_statement_items"    = @("--start", "2009-01-01")
        "earnings_disclosure_schedule" = @("--start", "2009-01-01")
        "share_structure"              = @("--start", "2009-01-01")
        "top_holders"                  = @("--start", "2009-01-01")
    }
} else {
    $queue = [ordered]@{
        "delisting_events"             = @()
        "macro_indicators"             = @("--start", "2009-01-01")
        # attempt once; fails fast if snapshot semantics refuse backfill
        "market_breadth"               = @()
    }
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
