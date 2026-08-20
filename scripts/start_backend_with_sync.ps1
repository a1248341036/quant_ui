$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root ".venv\Scripts\python.exe"

$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# The sync helper is fail-open by default: a remote outage does not stop the local API.
& $python (Join-Path $PSScriptRoot "startup_remote_sync.py")

# Preserve the existing local backend startup command.
& $python -m uvicorn backend.main:app --host 0.0.0.0 --port 17891
