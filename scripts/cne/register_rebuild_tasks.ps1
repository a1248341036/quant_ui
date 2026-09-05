#Requires -Version 7
# Register + start one-shot scheduled tasks for the two backfill lanes,
# detached from this session so they survive session/window teardown.
$ErrorActionPreference = "Stop"

$taskA = "CNERebuildLaneEM"
$taskB = "CNERebuildLaneTS"
$script = "D:\Quant\quant_ui\scripts\cne\rebuild_backfill_queue.ps1"
$pwsh = "C:\Program Files\PowerShell\7\pwsh.exe"
if (-not (Test-Path $pwsh)) { $pwsh = "C:\Program Files (x86)\PowerShell\7\pwsh.exe" }

foreach ($t in @($taskA, $taskB)) {
    if (Get-ScheduledTask -TaskName $t -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $t -Confirm:$false
    }
}

$argA = "-NoProfile -ExecutionPolicy Bypass -File `"$script`" -Lane eastmoney -Start 2016-01-01 -Resume"
$argB = "-NoProfile -ExecutionPolicy Bypass -File `"$script`" -Lane tushare -Start 2016-01-01 -Resume"

Register-ScheduledTask -TaskName $taskA `
    -Action (New-ScheduledTaskAction -Execute $pwsh -Argument $argA -WorkingDirectory "D:\Quant\quant_ui") `
    -Trigger (New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)) `
    -Settings (New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 12)) | Out-Null

Register-ScheduledTask -TaskName $taskB `
    -Action (New-ScheduledTaskAction -Execute $pwsh -Argument $argB -WorkingDirectory "D:\Quant\quant_ui") `
    -Trigger (New-ScheduledTaskTrigger -Once -At (Get-Date).AddSeconds(5)) `
    -Settings (New-ScheduledTaskSettingsSet -StartWhenAvailable -DontStopIfGoingOnBatteries -AllowStartIfOnBatteries -ExecutionTimeLimit (New-TimeSpan -Hours 12)) | Out-Null

Start-ScheduledTask -TaskName $taskA
Start-ScheduledTask -TaskName $taskB
Start-Sleep -Seconds 8
Get-ScheduledTask -TaskName $taskA, $taskB | Select-Object TaskName, State
"日志: CNEquity\data\cnequity\logs\rebuild-backfill-{eastmoney,tushare}-20260903.log"
