# Compact leftover staging runs from interrupted backfills (idempotent, PK-deduped).
# Safe to re-run: already-compacted runs simply no-op once staging is gone.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = 'utf-8'
$Cne = 'D:\Quant\quant_ui\.venv\Scripts\cne.exe'
$Cfg = 'C:\Quant\quant_ui\CNEquity\configs\cnequity.quant_dataset.toml'
$Cfg = 'D:\Quant\quant_ui\CNEquity\configs\cnequity.quant_dataset.toml'
$Log = 'D:\Quant\quant_ui\CNEquity\data\cnequity\logs\compact-leftovers.log'

$Runs = @(
    'd40126a8-5d0d-46b8-919d-59d5b5192997', # margin_trading 25 files
    'ff13237b-078d-4256-b6f0-993363bc74cd', # margin_trading 10 files
    'c103dbda-bfcb-4673-9071-df59cc8e60d7', # margin_trading 4 files
    '10b73444-a2e7-4697-9010-3480f18664f4', # margin_trading 2 files
    'd9936824-5c21-430c-8373-b440c4c4216b', # balancesheet 75MB
    'a0ba2230-45b8-49b8-8840-98f83d7758ab', # balancesheet 2MB
    '72209f4c-52fd-45c3-8c16-6ad9e2c0e756', # valuation_metrics 31 files
    '909f2f1b-0898-42f8-9278-8d2143f9a118', # valuation_metrics 26 files
    'a0590392-c60c-4e71-aae0-98fe8d6d1b61', # block_trades 44 files
    '68a86cbf-dcc9-42e0-a952-d803dcc92cf4', # block_trades 30 files
    'ef60ecc4-f6a8-4ec0-b332-e4b0f4280b34', # block_trades 6 files
    'df3d4bbe-d43a-4d5c-9312-e303b703445c', # sector_bars 11 files
    '2ae67267-7479-453d-9f6b-89958e574be4', # fund_flow 2 files
    '3595f4c2-9ab6-4d32-a69b-ecfe415c894e', # fund_flow 1 file
    'b3d71a04-4cff-4872-b647-642fce913608', # announcement_index 5 files
    '70ae852d-c5da-4f4c-9b0b-1acd9e517c40', # announcement_index 1 file
    '0cb4af75-227c-43c5-8a27-7abba33ce0ce', # northbound_holdings
    'bdceb27a-a79b-41a7-ac2c-07af0156ebc2', # northbound_holdings
    '782c3600-74c6-4a75-b456-e623db648707', # flash_news_wire
    'dab69ee8-1d9e-4e3b-b735-9101233c26f0', # news_headlines
    'd50e0490-b8fe-4553-b014-9bfb5f5c21ac', # shareholder_counts
    'd8b0f0ea-b440-4b9c-b7af-472ea6f70098', # share_unlock_schedule
    '9b3cd583-464b-4f6d-b43d-fe5b02d1c8dc', # trading_status
    '6d5709c6-65f4-463d-ae6e-f0844c6c1273', # instruments
    'c3d90f54-fc7a-4c6b-ae3b-b88ffedfb783'  # market_breadth
)

function Log([string]$m) {
    $line = "[$(Get-Date -Format 'MM-dd HH:mm:ss')] $m"
    Write-Host $line
    [System.IO.File]::AppendAllText($Log, "$line`n", [System.Text.Encoding]::UTF8)
}

Log "==== compact leftovers start: $($Runs.Count) run(s) ===="
$ok = 0; $fail = 0
foreach ($r in $Runs) {
    $staging = "D:\Quant\quant_ui\CNEquity\data\quant_dataset\_cnequity\staging"
    $hasStaging = Get-ChildItem $staging -Directory | Where-Object {
        Test-Path (Join-Path $_.FullName "run_id=$r")
    }
    if (-not $hasStaging) { Log "skip $r (no staging)"; continue }
    $out = & $Cne compact --run-id $r --config $Cfg 2>&1 | Out-String
    $code = $LASTEXITCODE
    if ($code -eq 0) {
        $ok++
        $rows = if ($out -match '"rows_written":\s*(\d+)') { $Matches[1] } else { '?' }
        Log "OK   $r rows_written=$rows"
    } else {
        $fail++
        Log "FAIL $r exit=$code"
        Log ($out.Trim() | Select-Object -First 1)
    }
}
Log "==== compact leftovers done: ok=$ok fail=$fail ===="
