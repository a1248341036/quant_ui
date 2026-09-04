#Requires -Version 7
# Serial queue for the four per-stock incremental L3 datasets (tushare
# middleware). balancesheet runs separately already; this waits for it, then
# runs the remaining three. Marks per dataset for -Resume safety.
param(
    [string]$Start = "2009-01-01",
    [string[]]$Datasets = @("income", "cashflow", "fina_indicator")
)
$ErrorActionPreference = "Continue"
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
$Log = "D:\Quant\quant_ui\CNEquity\data\cnequity\logs\rebuild-l3-incremental-$(Get-Date -Format 'yyyyMMdd').log"
function Write-Log([string]$Msg) {
    $line = "[$(Get-Date -Format 'MM-dd HH:mm:ss')][l3] $Msg"
    Write-Host $line
    [System.IO.File]::AppendAllText($Log, "$line`n", [System.Text.Encoding]::UTF8)
}

# Wait for balancesheet to finish (its cne process holds the middleware share)
Write-Log "waiting for balancesheet to finish..."
while (Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -match 'backfill balancesheet' }) {
    Start-Sleep -Seconds 60
}
Write-Log "balancesheet done; starting queue"

foreach ($ds in $Datasets) {
    $marker = Join-Path $MarkerDir "$ds.done"
    if (Test-Path $marker) { Write-Log "$ds SKIP (marker)"; continue }
    Write-Log "backfill $ds start --start $Start"
    $sw = [System.Diagnostics.Stopwatch]::StartNew()
    & $Cne backfill $ds --start $Start --config $Config 2>&1 | Out-Null
    $code = $LASTEXITCODE
    $sw.Stop()
    if ($code -eq 0) {
        Set-Content -Path $marker -Value "ok $(Get-Date -Format s)"
        Write-Log "backfill $ds OK ($([math]::Round($sw.Elapsed.TotalMinutes,1)) min)"
    } else {
        Write-Log "backfill $ds FAILED exit=$code ($([math]::Round($sw.Elapsed.TotalMinutes,1)) min)"
    }
}
Write-Log "==== l3 incremental queue done ===="
