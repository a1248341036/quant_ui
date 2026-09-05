#Requires -Version 7
# Boot-time resume for the post-wipe lake rebuild. Idempotent:
# verify re-computes gaps, so already-repaired datasets are skipped.
$ErrorActionPreference = "Continue"
# Native command output (cne emits UTF-8) must be decoded as UTF-8: a
# scheduled-task pwsh has no console and falls back to the GBK code page,
# which mojibake'd every Chinese line in the log (2026-09-04).
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$env:TUSHARE_TOKEN = "6cc06f993ea2f04821a8b05a0ac3a75a3512ade625da24a1f0f4718d"
$env:TUSHARE_URL   = "https://t.xiaodefa.top/"
$env:PYTHONIOENCODING = "utf-8"
$Cne    = "D:\Quant\quant_ui\.venv\Scripts\cne.exe"
$Config = "D:\Quant\quant_ui\CNEquity\configs\cnequity.quant_dataset.toml"
$Log    = "D:\Quant\quant_ui\CNEquity\data\cnequity\logs\rebuild-resume-$(Get-Date -Format 'yyyyMMdd').log"

function Write-Log([string]$Msg) {
    $line = "[$(Get-Date -Format 'MM-dd HH:mm:ss')] $Msg"
    Write-Host $line
    [System.IO.File]::AppendAllText($Log, "$line`n", [System.Text.Encoding]::UTF8)
}

Write-Log "==== resume after boot ===="
Start-Sleep -Seconds 60   # let network/dashboard settle

Write-Log "reconcile stale runs"
& $Cne clean --reconcile-runs --config $Config 2>&1 | Out-Null

Write-Log "verify --repair (resumes remaining gaps)"
& $Cne verify --repair --config $Config 2>&1 | ForEach-Object { Write-Log $_.ToString() }

Write-Log "stats rebuild"
& $Cne stats rebuild --config $Config 2>&1 | Select-Object -First 2 | ForEach-Object { Write-Log $_.ToString() }

Write-Log "==== resume done ===="
