$ErrorActionPreference = 'SilentlyContinue'

$root = 'C:\Users\zhoubw\.local\share\TeleAgent'
$logPath = Join-Path $root 'logs\main-2.3.1.log'
$runtimeLogPath = Join-Path $root 'log\super-agent-runtime.log'
$outPath = Join-Path $PSScriptRoot '..\.temp\teleagent-restart-monitor.log'
$statusPath = Join-Path $PSScriptRoot '..\.temp\teleagent-restart-monitor.json'
$stopPath = Join-Path $PSScriptRoot '..\.temp\teleagent-restart-monitor.stop'

Remove-Item -LiteralPath $stopPath -Force
Set-Content -LiteralPath $outPath -Value '' -Encoding UTF8

function Write-Record([string] $kind, [object] $data) {
    $record = [ordered]@{
        time = (Get-Date).ToString('o')
        kind = $kind
        data = $data
    }
    $line = $record | ConvertTo-Json -Compress -Depth 5
    Add-Content -LiteralPath $outPath -Value $line -Encoding UTF8
}

function Redact([string] $line) {
    $line = $line -replace '(?i)(API Token\s*:\s*)[^\s,}]+', '$1<REDACTED>'
    $line = $line -replace '(?i)(Authorization\s*[=:]\s*(?:Bearer\s+)?)[^\s,}]+', '$1<REDACTED>'
    $line = $line -replace '(?i)(apiToken|SCHEDULER_API_TOKEN|SUPER_AGENT_LOCAL_SESSION_KEY)\s*[=:]\s*[^\s,}]+', '$1=<REDACTED>'
    return $line
}

function Read-NewLines([string] $path, [hashtable] $state) {
    if (-not (Test-Path -LiteralPath $path)) { return }
    $lines = @(Get-Content -LiteralPath $path -Tail 120 -Encoding UTF8)
    $key = [IO.Path]::GetFullPath($path)
    if (-not $state.ContainsKey($key)) { $state[$key] = '' }
    $marker = $state[$key]
    $start = 0
    if ($marker) {
        $found = [Array]::IndexOf($lines, $marker)
        if ($found -ge 0) { $start = $found + 1 }
    }
    foreach ($line in $lines[$start..($lines.Count - 1)]) {
        $clean = Redact ([string] $line)
        if ($clean -match '(?i)(API Token|SuperAgent API|local_auth|superagent-auth|Starting server|server started|SERVER_PORT|SUPER_AGENT|auth|token)') {
            Write-Record 'log' @{ path = $path; line = $clean }
        }
    }
    if ($lines.Count -gt 0) { $state[$key] = [string]$lines[-1] }
}

$state = @{}
$previous = @{}
$started = Get-Date
Write-Record 'monitor_started' @{ pid = $PID; root = $root }

while (-not (Test-Path -LiteralPath $stopPath) -and ((Get-Date) - $started).TotalSeconds -lt 180) {
    $processes = @(Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '(?i)TeleAgent|node.exe' -and (
            $_.Name -match '(?i)TeleAgent' -or $_.CommandLine -match '(?i)TeleAgent'
        )
    } | ForEach-Object {
        [ordered]@{ pid = $_.ProcessId; name = $_.Name; parent = $_.ParentProcessId; command = Redact ([string]$_.CommandLine) }
    })
    $ports = @(Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -in 4397, 19876, 8080 } | ForEach-Object {
        [ordered]@{ address = $_.LocalAddress; port = $_.LocalPort; pid = $_.OwningProcess }
    })
    $snapshot = @{ processes = $processes; ports = $ports } | ConvertTo-Json -Compress -Depth 6
    if ($snapshot -ne $previous.snapshot) {
        Write-Record 'runtime' @{ processes = $processes; ports = $ports }
        $previous.snapshot = $snapshot
    }
    Read-NewLines $logPath $state
    Read-NewLines $runtimeLogPath $state
    Set-Content -LiteralPath $statusPath -Value (@{ pid = $PID; running = $true; output = $outPath; updated = (Get-Date).ToString('o') } | ConvertTo-Json) -Encoding UTF8
    Start-Sleep -Milliseconds 700
}

Write-Record 'monitor_stopped' @{ reason = if (Test-Path -LiteralPath $stopPath) { 'stop_file' } else { 'timeout' } }
Set-Content -LiteralPath $statusPath -Value (@{ pid = $PID; running = $false; output = $outPath; updated = (Get-Date).ToString('o') } | ConvertTo-Json) -Encoding UTF8
