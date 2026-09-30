#Requires -Version 7.0
# MinerU 全量解析进度速查（用途：一眼看到进度，不必翻日志）
# 用法:  pwsh -File D:\Quant\quant_ui\scripts\mineru_progress.ps1
# 注：本脚本用了 PS7 的 `??` 空合并运算符，Windows 自带的 PS 5.1 会直接解析失败，
# 故用 #Requires 提前给出明确报错（2026-09-30 review 修）。
param(
    [string]$Root = "D:\Quant\quant_ui",
    [int]$PerDocSeconds = 11   # 单篇有效耗时（2 并发实测约 10.8s），用于估时
)

$out = Join-Path $Root "data\research_reports\parsed_mineru"
$log = Join-Path $Root "logs\batch_mineru_recent.out"

Write-Host "=== MinerU 解析进度 ===" -ForegroundColor Cyan

# 1) 进程 + GPU
$procs = Get-CimInstance Win32_Process -Filter "Name='mineru.exe' or Name='pythonw.exe'" |
    Where-Object { $_.CommandLine -match "batch_mineru_parse|mineru.exe parse" }
$driver = $procs | Where-Object { $_.CommandLine -match "batch_mineru_parse" }
Write-Host ("驱动: {0}   解析子进程: {1} 个" -f `
    ($(if ($driver) { "运行中 (PID $($driver[0].ProcessId))" } else { "未运行 ⚠" })), `
    (($procs | Where-Object { $_.CommandLine -match "mineru.exe parse" }).Count))
try {
    $g = (nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader 2>$null | Select-Object -First 1)
    if ($g) { Write-Host "GPU: $g" }
} catch { }

# 2) 日志最后一行（含 [done/total] ok/skip/err 与脚本自估 ETA）
if (Test-Path $log) {
    $last = Get-Content $log -Encoding UTF8 -Tail 500 | Select-String -Pattern "^\[\d+/" | Select-Object -Last 1
    if ($last) { Write-Host "日志: $($last.Line)" } else { Write-Host "日志: (尚无进度行)" }
} else { Write-Host "日志: 未找到 $log" -ForegroundColor Yellow }

# 3) 按年份的完成矩阵
$root = Join-Path $Root "data\research_reports"
$tot = @{}; $done = @{}
foreach ($p in Get-ChildItem $root -Recurse -Filter *.pdf | Where-Object { $_.FullName -notmatch "\\parsed_mineru" }) {
    if ($p.Name -match "(20\d{2})") { $y = [int]$Matches[1]; $tot[$y] = 1 + ($tot[$y] ?? 0) }
}
$mdFiles = Get-ChildItem $out -Recurse -Filter *.md -ErrorAction SilentlyContinue | Where-Object { $_.Length -gt 1000 }
foreach ($m in $mdFiles) {
    if ($m.Name -match "(20\d{2})") { $y = [int]$Matches[1]; $done[$y] = 1 + ($done[$y] ?? 0) }
}
Write-Host "`n年份   总量  已完成  剩余"
$tAll = 0; $dAll = 0
foreach ($y in ($tot.Keys | Sort-Object)) {
    $t = $tot[$y]; $d = $done[$y] ?? 0; $tAll += $t; $dAll += $d
    $mark = if ($y -lt 2020) { " (旧, 已停)" } else { "" }
    Write-Host ("{0}  {1,5}  {2,6}  {3,5}{4}" -f $y, $t, $d, ($t - $d), $mark)
}
$rest = $tAll - $dAll
Write-Host ("`n合计: {0}/{1} 完成   剩余 {2} 篇 ≈ {3:N1} 小时" -f $dAll, $tAll, $rest, ($rest * $PerDocSeconds / 3600))
Write-Host "产物: $out"
