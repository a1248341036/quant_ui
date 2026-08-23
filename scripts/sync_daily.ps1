# [DEPRECATED 2026-08-22] 此脚本已废弃。日更已切换至 CNE 数据湖流水线。
# 新脚本：scripts/run_cne_daily.ps1（计划任务 QuantUIDataSync 已指向新脚本）
# 此文件保留仅作参考，不再被任何计划任务或自动化流程调用。
# ---------------------------------------------------------------------------
$ErrorActionPreference = "Stop"
$root = "D:\Quant\quant_ui"
$log = Join-Path $root "data\sync_run.log"
$ts = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
"[$ts] ===== sync start =====" | Out-File -FilePath $log -Append -Encoding UTF8
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
try {
    & (Join-Path $root ".venv\Scripts\python.exe") (Join-Path $root "scripts\sync_server.py") --apply --rebuild-panel-remote 2>&1 | Out-File -FilePath $log -Append -Encoding UTF8
    if ($LASTEXITCODE -ne 0) {
        "[$ts] ===== sync FAIL exit=$LASTEXITCODE =====" | Out-File -FilePath $log -Append -Encoding UTF8
        exit 1
    }
    "[$ts] ===== sync done =====" | Out-File -FilePath $log -Append -Encoding UTF8
    exit 0
} catch {
    "[$ts] ===== sync FAIL: $($_.Exception.Message) =====" | Out-File -FilePath $log -Append -Encoding UTF8
    exit 1
}