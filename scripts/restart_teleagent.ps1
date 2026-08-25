$ErrorActionPreference = 'SilentlyContinue'

$app = 'D:\dxRD\software\TeleAgent\TeleAgent.exe'
$runtimeRoot = 'C:\Users\zhoubw\.local\share\TeleAgent'

function Get-TeleAgentProcesses {
    @(Get-CimInstance Win32_Process | Where-Object {
        ($_.Name -eq 'TeleAgent.exe' -and ($_.ExecutablePath -eq $app -or $_.ExecutablePath -like "$runtimeRoot\runtimes\super-agent-code\bin\TeleAgent.exe")) -or
        ($_.Name -eq 'node.exe' -and $_.CommandLine -like "*$runtimeRoot*") -or
        ($_.Name -eq 'cmd.exe' -and $_.CommandLine -like "*$runtimeRoot*")
    })
}

$before = Get-TeleAgentProcesses
$main = @($before | Where-Object { $_.ExecutablePath -eq $app -and $_.CommandLine -notmatch '--type=' }) | Select-Object -First 1
if ($main) {
    & taskkill.exe /PID $main.ProcessId /T /F | Out-Null
}

Start-Sleep -Seconds 2
foreach ($p in @(Get-TeleAgentProcesses)) {
    if ($p.ProcessId -ne $PID) { Stop-Process -Id $p.ProcessId -Force }
}

$deadline = (Get-Date).AddSeconds(20)
while ((Get-Date) -lt $deadline -and (Get-TeleAgentProcesses).Count -gt 0) {
    Start-Sleep -Milliseconds 500
}

$started = Start-Process -FilePath $app -WorkingDirectory (Split-Path $app) -PassThru
$health = $false
$deadline = (Get-Date).AddSeconds(45)
while ((Get-Date) -lt $deadline) {
    try {
        $response = Invoke-WebRequest -UseBasicParsing -Uri 'http://127.0.0.1:4397/health' -TimeoutSec 2
        if ($response.StatusCode -eq 200) { $health = $true; break }
    } catch {}
    Start-Sleep -Seconds 1
}

[ordered]@{
    started_pid = $started.Id
    health_4397 = $health
    health_url = 'http://127.0.0.1:4397/health'
    time = (Get-Date).ToString('o')
} | ConvertTo-Json
