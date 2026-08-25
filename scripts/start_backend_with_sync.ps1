$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# --- Kill existing services on ports 17891 and 8787 before launching ---
function Stop-PortListener([int]$Port) {
    try {
        $conns = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
        foreach ($conn in $conns) {
            if ($conn.OwningProcess -and $conn.OwningProcess -ne 0) {
                Write-Host "[launcher] Killing PID $($conn.OwningProcess) on port $Port" -ForegroundColor Yellow
                Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
            }
        }
    } catch { }
}

Write-Host '[launcher] Stopping existing services' -ForegroundColor DarkGray
Stop-PortListener 17891
Stop-PortListener 8787
Start-Sleep -Seconds 1

# [DEPRECATED 2026-08-22] 远程 sync 已废弃，服务器停用，日更切换至 CNE 数据湖流水线
# & $python (Join-Path $PSScriptRoot "startup_remote_sync.py")

# --- Launch both services as child processes so they share the console ---

# Quant UI backend (port 17891)
$backendArgs = @("-m", "uvicorn", "backend.main:app", "--host", "127.0.0.1", "--port", "17891")
$backend = Start-Process -FilePath $python -ArgumentList $backendArgs -WorkingDirectory $root -NoNewWindow -PassThru

# CNE dashboard (port 8787)
$env:TUSHARE_TOKEN = "REDACTED_TUSHARE_TOKEN"
$env:TUSHARE_URL = "https://t.xiaodefa.top/"
$cneDir = Join-Path $root "CNEquity"
$cneArgs = @("-m", "cnequity.cli.main", "serve", "--config", "configs/cnequity.quant_dataset.toml")
$cne = Start-Process -FilePath $python -ArgumentList $cneArgs -WorkingDirectory $cneDir -NoNewWindow -PassThru

Write-Host "[launcher] Quant UI backend  -> http://127.0.0.1:17891" -ForegroundColor Cyan
Write-Host "[launcher] CNE dashboard    -> http://127.0.0.1:8787"   -ForegroundColor Cyan
Write-Host "[launcher] Press Ctrl+C to stop both services."           -ForegroundColor Yellow

# Wait for either process to exit; when one dies, kill the other and stop.
while (-not $backend.HasExited -and -not $cne.HasExited) {
    Start-Sleep -Seconds 1
}

if ($backend.HasExited) {
    Write-Host ('[launcher] backend exited (code ' + $backend.ExitCode + ')') -ForegroundColor Red
} else {
    $cneExitCode = $cne.ExitCode
    Write-Host ('[launcher] CNE exited (code ' + $cneExitCode + ')') -ForegroundColor Red
}

# Kill the survivor
if (-not $backend.HasExited) { Stop-Process -Id $backend.Id -Force -ErrorAction SilentlyContinue }
if (-not $cne.HasExited)     { Stop-Process -Id $cne.Id -Force -ErrorAction SilentlyContinue }
