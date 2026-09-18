#Requires -Version 7
# Boot-time resume for the post-wipe lake rebuild. Idempotent:
# verify re-computes gaps, so already-repaired datasets are skipped.
$ErrorActionPreference = "Continue"
# Native command output (cne emits UTF-8) must be decoded as UTF-8: a
# scheduled-task pwsh has no console and falls back to the GBK code page,
# which mojibake'd every Chinese line in the log (2026-09-04).
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
