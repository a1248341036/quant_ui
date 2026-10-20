<#
.SYNOPSIS
  OCR review 安全包装：自动分流大文件、统一参数、超时止损、结果归档。

.DESCRIPTION
  背景（2026-10-20 实测，详见 docs/ocr_review_pitfalls.md）：
  OCR 的 plan 阶段在「组内最大改动文件 >=50 行」或「组内合计 >=100 行」时触发，
  单次吃 16-30 万 token，而该阈值不可配、--exclude 也不影响 plan 阶段的输入量。
  于是大文件和小文件混在一组时，200k 窗口模型直接 HTTP 500 爆窗、全组报废。

  本脚本的做法：
    1) 先跑 `ocr delegate preview`（1 秒、零 token）拿到"会被评审的文件 + 增删行数"；
    2) 把改动 >=BigFileLines 的文件判为"大文件"，从小文件组里 --exclude 掉；
    3) 先跑小文件组（这是能稳定出结论的那批）；
    4) 大文件再逐个单独跑（每个都被 --exclude 成唯一文件），各自独立超时；
    5) 全部结果落到 artifacts/ocr_reviews/<timestamp>/ 下，逐份打印路径。

.PARAMETER Base
  diff 起点 ref，默认 main。

.PARAMETER Branch
  diff 终点 ref（你的功能分支）。

.PARAMETER Background
  业务背景，喂给 review 提升质量（必填）。

.PARAMETER DryRun
  只做 preview 与分流计划，不真正调用 LLM。

.EXAMPLE
  pwsh -File scripts/ocr_review_safe.ps1 -Branch fix/panel-need-columns `
    -Background "面板按需取列：columns 从 SessionCreateRequest 接到 read_panel_arrow_mmap"
#>
param(
  [string]$Base = "main",
  [Parameter(Mandatory = $true)][string]$Branch,
  [Parameter(Mandatory = $true)][string]$Background,
  [string]$Provider = "easycli",
  [string]$Model = "DeepSeek-V4-Flash-0731",
  [int]$TimeoutMin = 7,
  [int]$BigFileLines = 50,
  [switch]$DryRun
)

$ErrorActionPreference = "Stop"
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}

$repoRoot = (Get-Location).Path
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$outDir = Join-Path $repoRoot "artifacts\ocr_reviews\$stamp"
New-Item -ItemType Directory -Path $outDir -Force | Out-Null

# ── 1. preview：拿到文件清单与增删行数（零 token） ──────────────────
Write-Host "[1/4] preview $Base..$Branch"
$preview = & ocr delegate preview --from $Base --to $Branch 2>&1 | Out-String
$preview | Set-Content -Path (Join-Path $outDir "preview.txt") -Encoding UTF8

$files = @()
foreach ($line in ($preview -split "`r?`n")) {
  # 形如:   - `path/to/x.py` [modified] +76/-15
  # 被排除: ~~- `path` [added] +128/-0 (excluded: default_path)~~
  if ($line -match '^\s*-{0,2}\s*`([^`]+)`.*?\+(\d+)/-(\d+)') {
    $path = $Matches[1]
    $added = [int]$Matches[2]
    $deleted = [int]$Matches[3]
    $excluded = $line -match 'excluded:'
    if (-not $excluded) {
      $files += [pscustomobject]@{ Path = $path; Changed = $added + $deleted }
    }
  }
}

if ($files.Count -eq 0) {
  Write-Host "没有可评审文件（全部被规则排除，或 diff 为空）。"
  exit 0
}

$big = $files | Where-Object { $_.Changed -ge $BigFileLines }
$small = $files | Where-Object { $_.Changed -lt $BigFileLines }

Write-Host ("      可评审 {0} 个：小文件 {1} 个 / 大文件 {2} 个（>= {3} 行触发 plan 阶段）" -f `
  $files.Count, $small.Count, $big.Count, $BigFileLines)
foreach ($f in $big) { Write-Host ("      [大] {0} ({1} 行)" -f $f.Path, $f.Changed) }
foreach ($f in $small) { Write-Host ("      [小] {0} ({1} 行)" -f $f.Path, $f.Changed) }

if ($DryRun) {
  Write-Host "[DryRun] 计划已写入 $outDir\preview.txt"
  exit 0
}

# ── 2. 通用调用参数 ────────────────────────────────────────────────
$commonArgs = @(
  "review", "--audience", "agent",
  "--effort", "low",                    # 2 轮 -> 1 轮（滚雪球主因）
  "--timeout", "$TimeoutMin",
  "--max-tokens", "60000",
  "--max-tokens-budget", "400000",
  "--provider", $Provider, "--model", $Model,
  "--from", $Base, "--to", $Branch,
  "--background", $Background
)
$results = @()

# ── 3. 小文件组（能稳定出结论） ────────────────────────────────────
if ($small.Count -gt 0) {
  $exclude = ($big | ForEach-Object { "**/" + (Split-Path $_.Path -Leaf) }) -join ","
  $outFile = Join-Path $outDir "small_files.md"
  Write-Host ("[2/4] 跑小文件组（{0} 个文件，排除 {1} 个大文件）" -f $small.Count, $big.Count)
  $args2 = $commonArgs.Clone()
  if ($exclude) { $args2 += @("--exclude", $exclude) }
  $args2 += @("--output", $outFile)
  & ocr @args2 2>&1 | Select-Object -Last 4
  $results += [pscustomobject]@{ Scope = "small_files($($small.Count))"; File = $outFile; Exit = $LASTEXITCODE }
}

# ── 4. 大文件逐个单独跑 ───────────────────────────────────────────
$idx = 0
foreach ($f in $big) {
  $idx++
  # 排除除它之外的所有已知文件，确保该组只剩它一个
  $others = ($files | Where-Object { $_.Path -ne $f.Path } |
    ForEach-Object { "**/" + (Split-Path $_.Path -Leaf) }) -join ","
  $safeName = ($f.Path -replace '[\\/]', '_')
  $outFile = Join-Path $outDir ("big_" + $safeName + ".md")
  Write-Host ("[3/4] 跑大文件 {0}/{1}：{2}（{3} 行，可能仍触发 plan 阶段）" -f $idx, $big.Count, $f.Path, $f.Changed)
  $args3 = $commonArgs.Clone()
  if ($others) { $args3 += @("--exclude", $others) }
  $args3 += @("--output", $outFile)
  & ocr @args3 2>&1 | Select-Object -Last 4
  $results += [pscustomobject]@{ Scope = "big:" + $f.Path; File = $outFile; Exit = $LASTEXITCODE }
}

# ── 5. 汇总 ───────────────────────────────────────────────────────
Write-Host "[4/4] 汇总"
$summary = Join-Path $outDir "SUMMARY.txt"
$lines = @()
foreach ($r in $results) {
  $status = if ($r.Exit -eq 0) { "OK  " } else { "FAIL" }
  $exists = Test-Path $r.File
  $size = if ($exists) { (Get-Item $r.File).Length } else { 0 }
  $line = "{0}  {1,-10}  {2}" -f $status, $r.Exit, $r.Scope
  $lines += $line
  Write-Host ("      {0}  -> {1} ({2} bytes)" -f $line, $r.File, $size)
}
$lines += ""
$lines += "提示：Exit!=0 或输出含 'review failed' 时，该组仍可能爆窗；"
$lines += "      把该文件的改动拆成 <$BigFileLines 行的小 commit 后单独 review，或由宿主 agent 直接审。"
$lines | Set-Content -Path $summary -Encoding UTF8
Write-Host ("      归档目录：{0}" -f $outDir)
