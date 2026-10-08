#Requires -Version 7
<#
.SYNOPSIS
  CNE 数据湖每日流水线（Windows / PowerShell 版）。
  功能：
    1. 弹出现代化半透明暗色进度卡片看板（自动隐藏控制台黑框）
    2. 按 wave 依赖顺序依次执行 cne run daily（实时推送进度条与步骤）
    3. ETF/场外基金/指数刷新
    4. stale 补抓（可选）
    5. health check（audit + status）与 meta 备份、staging 清理
    6. 跑完后展示清晰直观的数据集对账清单（包含各数据集行数、耗时、状态）
    7. 任务完成常驻卡片，待用户确认后关闭退出（不再闪退）

.PARAMETER TradeDate
  指定交易日补跑，格式 YYYY-MM-DD，默认 today。

.PARAMETER NoGui
  强制纯命令行/无图形模式（禁用 WPF 进度看板）。

.PARAMETER NoStaleRetry
  跳过 stale 补抓环节。

.PARAMETER NoBackup
  跳过 meta 备份。

.PARAMETER Quiet
  传给 cne run daily --quiet，只输出 WARNING 及以上。

.PARAMETER SkipClean
  跳过 staging 清理。

.PARAMETER SkipEtfFund
  跳过 ETF/基金/指数刷新（refresh_data.py）。
#>
[CmdletBinding()]
param(
    [string]$TradeDate = "",
    [string]$StatusFile = "",
    [switch]$NoGui,
    [switch]$NoStaleRetry,
    [switch]$NoBackup,
    [switch]$Quiet,
    [switch]$SkipClean,
    [switch]$SkipEtfFund
)

$ErrorActionPreference = "Stop"

# ── 代理健康探测与自愈（防止死代理 10061 拖死流水线）─────────────────
try {
    $proxyEnv = $env:HTTP_PROXY ?? $env:HTTPS_PROXY ?? $env:ALL_PROXY
    if ($proxyEnv -and $proxyEnv -match "127\.0\.0\.1:(\d+)") {
        $pPort = [int]$matches[1]
        $tcp = [System.Net.Sockets.TcpClient]::new()
        $ar = $tcp.BeginConnect("127.0.0.1", $pPort, $null, $null)
        $wh = $ar.AsyncWaitHandle.WaitOne(500)
        if (-not $wh -or -not $tcp.Connected) {
            # 代理端口无法建立连接，清除环境变量强制直连，防止批处理全部因 10061 拒绝退出
            $env:HTTP_PROXY = ""
            $env:HTTPS_PROXY = ""
            $env:ALL_PROXY = ""
        }
        $tcp.Dispose()
    }
} catch {}

# ── 路径常量 ──────────────────────────────────────────────────────────
$RepoRoot   = "D:\Quant\quant_ui"
$CneRoot    = Join-Path $RepoRoot "CNEquity"
$Cne        = Join-Path $RepoRoot ".venv\Scripts\cne.exe"
$Config     = Join-Path $CneRoot "configs\cnequity.quant_dataset.toml"
$LogDir     = Join-Path $CneRoot "data\cnequity\logs"
$BackupDir  = Join-Path $CneRoot "data\cnequity\backups"
$Py         = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$RefreshData= Join-Path $RepoRoot "scripts\refresh_data.py"
$GuiModule  = Join-Path $PSScriptRoot "CneDailyGui.psm1"

# 预置步骤清单：CLI worker（-NoGui 子进程）也必须加载。GUI 模块含 WPF 不便在子进程导入，
# 而没有这份清单，波次失败时无法把受阻数据集标记进对账清单——
# 2026-10-08 看板报"存在失败项"却一个名字都不点、清单全空，即此因。
$StepsData = Join-Path $PSScriptRoot "CneDailySteps.ps1"
if ((-not $Global:CnePredefinedSteps) -and (Test-Path $StepsData)) { . $StepsData }

# 额外日志（终端已有实时输出，此文件做留底）
$Stamp   = Get-Date -Format "yyyyMMdd"
$LogFile = Join-Path $LogDir "daily-$Stamp.log"

$null = New-Item -ItemType Directory -Force -Path $LogDir
$null = New-Item -ItemType Directory -Force -Path $BackupDir

# ── 环境变量 ──────────────────────────────────────────────────────────
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

# ── Win32 控制台隐藏支持 ───────────────────────────────────────────────
try {
    Add-Type -Name WinUtil -Namespace Cne -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("Kernel32.dll")]
public static extern IntPtr GetConsoleWindow();

[System.Runtime.InteropServices.DllImport("user32.dll")]
public static extern bool ShowWindow(IntPtr hWnd, int nCmdShow);
"@ -ErrorAction SilentlyContinue
} catch {}

# ── 配置 ──────────────────────────────────────────────────────────────
$WaveList      = @("core", "fundamentals", "capital", "macro_risk", "signals", "research")
$GateWaves     = @("core")
$SoftFailOk    = $true
$StaleRetry    = -not $NoStaleRetry
$StaleDelaySec = 1800

# ── 判断是否启用 GUI 模式 ─────────────────────────────────────────────
$UseGui = (-not $NoGui) -and [System.Environment]::UserInteractive -and (Test-Path $GuiModule)

# ── 核心工作函数（可被前台或后台线程调用）─────────────────────────────
function Execute-CnePipeline {
    param(
        [hashtable]$SyncContext = $null,
        [string]$StatusFile = "",
        [string]$TradeDate = "",
        [bool]$NoStaleRetry = $false,
        [bool]$NoBackup = $false,
        [bool]$Quiet = $false,
        [bool]$SkipClean = $false,
        [bool]$SkipEtfFund = $false
    )

    $StaleRetry = -not $NoStaleRetry
    $delaySec = if ($env:CNE_STALE_DELAY_SEC) { [int]$env:CNE_STALE_DELAY_SEC } else { $StaleDelaySec }

    # 结构化状态管理（供 GUI 跨进程/跨线程原子读取）
    $resultsList = [System.Collections.Generic.List[psobject]]::new()
    $statusObj = [ordered]@{
        phase_name     = "环境准备与状态清理"
        phase_index    = 1
        phase_total    = 8
        progress_pct   = 0.0
        current_step   = "正在初始化数据同步环境..."
        active_dataset = ""
        results        = $resultsList
        is_completed   = $false
        is_success     = $false
        message        = ""
        updated_at     = (Get-Date -Format "HH:mm:ss")
    }

    function Save-StatusState {
        if ($StatusFile) {
            try {
                $statusObj.updated_at = (Get-Date -Format "HH:mm:ss")
                $json = $statusObj | ConvertTo-Json -Depth 5 -Compress
                [System.IO.File]::WriteAllText($StatusFile, $json, [System.Text.Encoding]::UTF8)
            } catch {}
        }
    }

    function Write-PipelineLog([string]$Msg) {
        $ts = Get-Date -Format "HH:mm:ss"
        $line = "[$ts] $Msg"
        Write-Host $line
        try { [System.IO.File]::AppendAllText($LogFile, "$line`n", [System.Text.Encoding]::UTF8) } catch {}
    }

    function Notify-Step([string]$stepText) {
        if ([string]::IsNullOrWhiteSpace($stepText)) { return }
        $trimmed = $stepText.Trim()
        # 彻底过滤 JSON 符号与行首行尾单字符
        if ($trimmed -match '^[\s\{\}\[\]\,\"''\:]+$') { return }
        if ($trimmed -match '^\d+$') { return }

        $statusObj.current_step = $trimmed
        Save-StatusState
        if ($SyncContext) {
            $SyncContext.CurrentStep = $trimmed
        }
    }

    function Notify-ActiveDataset([string]$dataset) {
        if ([string]::IsNullOrWhiteSpace($dataset)) { return }
        $statusObj.active_dataset = $dataset
        Save-StatusState
    }

    function Notify-Phase([string]$phaseName, [int]$idx, [int]$total, [double]$pct) {
        $statusObj.phase_name   = $phaseName
        $statusObj.phase_index  = $idx
        $statusObj.phase_total  = $total
        $statusObj.progress_pct = [math]::Round($pct, 1)
        Save-StatusState
        if ($SyncContext) {
            $SyncContext.PhaseName  = $phaseName
            $SyncContext.PhaseIndex = $idx
            $SyncContext.PhaseTotal = $total
            $SyncContext.ProgressPct= $pct
            $SyncContext.PhaseChanged = $true
        }
    }

    function Record-Result([string]$dataset, [string]$status, [int]$rows, [string]$elapsed, [string]$note) {
        $item = [pscustomobject]@{
            Dataset = $dataset
            Status  = $status
            Rows    = $rows
            Elapsed = $elapsed
            Note    = $note
        }
        $resultsList.Add($item)
        Save-StatusState
        if ($SyncContext -and $SyncContext.NewResults) {
            $SyncContext.NewResults.Enqueue($item)
        }
    }

    function Run-CneCommand([string[]]$CmdArgs) {
        if ($Quiet -and ($CmdArgs -contains "run" -and $CmdArgs -contains "daily")) {
            $CmdArgs = @($CmdArgs) + "--quiet"
        }
        $fullArgs = @($CmdArgs) + "--config", $Config

        $lines = [System.Collections.Generic.List[string]]::new()
        $script:LastCmdErrors = [System.Collections.Generic.List[string]]::new()
        & $Cne @fullArgs 2>&1 | ForEach-Object {
            $line = $_.ToString()
            Write-Host $line
            $lines.Add($line)

            # 解析实时 step 进度与有业务含义的状态
            if ($line -match "Step\s+(\w+)\s+success\s+in\s+([\d\.]+)s\s+\((\d+)\s+rows\)") {
                $ds = $matches[1]
                Record-Result $ds "success" ([int]$matches[3]) "$($matches[2])s" "更新成功"
                $cn = if ($Global:CneDatasetNames[$ds]) { $Global:CneDatasetNames[$ds] } else { $ds }
                Notify-Step "$cn ($ds) 增量同步完成 (+$($matches[3]) 行)"
            } elseif ($line -match "Step\s+(\w+)\s+warning\s+in\s+([\d\.]+)s\s+\((\d+)\s+rows\)") {
                $ds = $matches[1]
                Record-Result $ds "warning" ([int]$matches[3]) "$($matches[2])s" "有警告提示"
                $cn = if ($Global:CneDatasetNames[$ds]) { $Global:CneDatasetNames[$ds] } else { $ds }
                Notify-Step "$cn ($ds) 带有警告提示完成"
            } elseif ($line -match "Step\s+(\w+)\s+failed\s+after\s+([\d\.]+)s") {
                $ds = $matches[1]
                Record-Result $ds "failed" 0 "$($matches[2])s" "执行失败"
                $cn = if ($Global:CneDatasetNames[$ds]) { $Global:CneDatasetNames[$ds] } else { $ds }
                Notify-Step "⚠️ $cn ($ds) 同步失败"
            } elseif ($line -match "(\w+):\s+cadence\s+skip") {
                $ds = $matches[1]
                Record-Result $ds "skip" 0 "<1s" "同周期最新无需重扫"
                $cn = if ($Global:CneDatasetNames[$ds]) { $Global:CneDatasetNames[$ds] } else { $ds }
                Notify-Step "$cn ($ds) 本周期已最新，自动跳过"
            } elseif ($line -match "(\w+):\s+fell\s+back\s+to\s+(\w+)\s+\((\d+)\s+rows\)") {
                $ds = $matches[1]
                Record-Result $ds "success" ([int]$matches[3]) "--" "降级至 $($matches[2]) 成功"
                $cn = if ($Global:CneDatasetNames[$ds]) { $Global:CneDatasetNames[$ds] } else { $ds }
                Notify-Step "$cn ($ds) 东财断连，降级至 $($matches[2]) 完成"
            } elseif ($line -match "(\w+):\s+fetching\s+(.*)") {
                $ds = $matches[1]
                $cn = if ($Global:CneDatasetNames[$ds]) { $Global:CneDatasetNames[$ds] } else { $ds }
                Notify-ActiveDataset $ds
                Notify-Step "正在从数据源拉取: $cn ($ds)..."
            } elseif ($line -match "(\w+):\s+(\d+)\s+new\s+rows\s+merged") {
                $ds = $matches[1]
                $cn = if ($Global:CneDatasetNames[$ds]) { $Global:CneDatasetNames[$ds] } else { $ds }
                Notify-Step "正在合并写入数据湖: $cn ($ds) (+$($matches[2]) 行)"
            } elseif ($line -match "THS sweep:\s+(\d+/\d+)\s+boards") {
                Notify-ActiveDataset "sector_bars"
                Notify-Step "正在扫描同花顺板块行情: $($matches[1])..."
            } elseif ($line -match "st_coverage:\s+skipping\s+receipt") {
                # 忽略内部未就绪凭证跳过日志
            } elseif ($line -match '^(Error|Traceback)\b' -or $line -match '\b(ERROR|CRITICAL)\b') {
                # 捕获失败原因（摄入锁被占用、Traceback 等），供完成摘要与受阻标记点名
                $script:LastCmdErrors.Add($line)
            }
        }
        if ($lines.Count -gt 0) {
            $text = ($lines -join "`n") + "`n"
            try { [System.IO.File]::AppendAllText($LogFile, $text, [System.Text.Encoding]::UTF8) } catch {}
        }
        return $LASTEXITCODE
    }

    function Run-PyCommand([string[]]$CmdArgs, [string]$JobName) {
        $script:LastCmdErrors = [System.Collections.Generic.List[string]]::new()
        Write-PipelineLog "$JobName start"
        Notify-Step "正在执行: $JobName"
        $lines = [System.Collections.Generic.List[string]]::new()
        & $Py @CmdArgs 2>&1 | ForEach-Object {
            $line = $_.ToString()
            Write-Host $line
            $lines.Add($line)
            Notify-Step $line
            if ($line -match '^(Error|Traceback)\b' -or $line -match '\b(ERROR|CRITICAL)\b') {
                $script:LastCmdErrors.Add($line)
            }
        }
        if ($lines.Count -gt 0) {
            $text = ($lines -join "`n") + "`n"
            try { [System.IO.File]::AppendAllText($LogFile, $text, [System.Text.Encoding]::UTF8) } catch {}
        }
        $exitCode = $LASTEXITCODE
        if ($exitCode -eq 0) {
            Write-PipelineLog "$JobName OK"
            Record-Result $JobName "success" 0 "--" "更新成功"
        } else {
            Write-PipelineLog "$JobName FAILED (exit=$exitCode)"
            foreach ($e in $script:LastCmdErrors) { $failErrors.Add($e) }
            $errFirst = ""
            if ($script:LastCmdErrors.Count -gt 0) { $errFirst = ($script:LastCmdErrors[0] -replace '\s+', ' ').Trim() }
            $note = if ($errFirst) { "失败: $errFirst" } else { "执行失败" }
            Record-Result $JobName "failed" 0 "--" $note
        }
        return $exitCode
    }

    # ── 开始执行流程 ──────────────────────────────────────────────────
    $dateArg = @()
    if ($TradeDate) { $dateArg = @("--trade-date", $TradeDate) }
    $dispDate = if ($TradeDate) { $TradeDate } else { "今日 (Today)" }

    Write-PipelineLog "==== CNE daily pipeline start $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') trade_date=$dispDate ===="

    # 阶段 0: 环境准备与清理
    Notify-Phase "环境准备与状态清理" 1 8 5.0
    Write-PipelineLog "--- reconcile stale runs ---"
    $null = Run-CneCommand @("clean", "--reconcile-runs", "--dry-run")

    # 失败原因收集（跨波次/跨命令累积，供完成摘要点名）
    $failErrors = [System.Collections.Generic.List[string]]::new()
    $failedGates  = @()
    $failedSoft   = @()
    $summary      = [System.Collections.Generic.List[pscustomobject]]::new()

    # 阶段 1..6: 依次执行 WaveList
    $waveNamesCn = @{
        "core"         = "Core (核心行情与基础参考)"
        "fundamentals" = "Fundamentals (财务基本面与披露)"
        "capital"      = "Capital (资金流与估值)"
        "macro_risk"   = "Macro & Risk (宏观与风险)"
        "signals"      = "Signals (事件披露与信号)"
        "research"     = "Research (舆情分析与研究)"
    }

    $waveIdx = 0
    foreach ($wave in $WaveList) {
        $waveIdx++
        $cnTitle = if ($waveNamesCn[$wave]) { $waveNamesCn[$wave] } else { $wave }
        $pct = 10.0 + ($waveIdx / $WaveList.Count) * 60.0
        Notify-Phase $cnTitle ($waveIdx + 1) 8 $pct
        Write-PipelineLog "--- wave: $wave ($cnTitle) ---"

        $exitCode = Run-CneCommand (@("run", "daily", "--group", $wave) + $dateArg)

        $isGate = $GateWaves -contains $wave
        if ($exitCode -eq 0) {
            Write-PipelineLog "wave $wave OK"
            $summary.Add([pscustomobject]@{ Wave = $wave; Status = "OK"; Kind = $(if ($isGate) { "gate" } else { "soft" }) })
        } else {
            Write-PipelineLog "wave $wave FAILED (exit=$exitCode, see $LogFile)"
            $summary.Add([pscustomobject]@{ Wave = $wave; Status = "FAILED"; Kind = $(if ($isGate) { "gate" } else { "soft" }) })
            foreach ($e in $script:LastCmdErrors) { $failErrors.Add($e) }
            $errFirst = ""
            if ($script:LastCmdErrors.Count -gt 0) { $errFirst = ($script:LastCmdErrors[0] -replace '\s+', ' ').Trim() }
            $blockedNote = if ($errFirst) { "受阻: $errFirst" } else { "受阻未执行: 前置步骤报错导致波次提前中断" }
            if ($isGate) {
                $failedGates += $wave
            } else {
                $failedSoft += $wave
            }
            # 如果波次执行失败，将该波次下所有尚未产生执行记录的预置数据集显式标记为"未执行/已受阻"
            if ($Global:CnePredefinedSteps) {
                $recordedNames = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
                foreach ($r in $resultsList) { [void]$recordedNames.Add($r.Dataset) }
                foreach ($pre in $Global:CnePredefinedSteps) {
                    if ($pre.Wave -eq $wave -and -not $recordedNames.Contains($pre.Dataset)) {
                        Record-Result $pre.Dataset "failed" 0 "--" $blockedNote
                    }
                }
            }
        }
    }

    # 阶段 7: ETF/基金/指数面板刷新
    $etfStatus = "skipped"
    Notify-Phase "ETF/基金/指数面板刷新" 7 8 75.0
    if (-not $SkipEtfFund) {
        Write-PipelineLog "--- ETF/基金/指数刷新 ---"
        $cmdArgs = @($RefreshData, "--skip-stock-panel", "--no-sync-pg", "--no-rebuild-panel")
        $etfExit = Run-PyCommand $cmdArgs "etf_fund_refresh"
        if ($etfExit -eq 0) {
            $etfStatus = "OK"
        } else {
            $etfStatus = "FAILED"
            $failedSoft += "etf-fund"
        }
    }

    # 阶段 8: stale 补抓与总结审计
    Notify-Phase "滞后补抓与健康检查" 8 8 90.0
    $staleStatus = "skipped"
    if ($StaleRetry -and $failedGates.Count -eq 0) {
        Write-PipelineLog "--- stale probe ---"
        $staleExit = Run-CneCommand @("status", "--datasets")
        if ($staleExit -eq 0) {
            Write-PipelineLog "nothing stale — no retry needed"
            $staleStatus = "not needed"
            Record-Result "stale_retry" "skip" 0 "--" "全量对齐，无需补抓"
        } else {
            # 只有在非交互环境或未指定立即跳过时才等待大延时，避免阻塞界面
            Write-PipelineLog "something is stale; waiting ${delaySec}s before re-fetching"
            Notify-Step "部分数据滞后，等待重试补抓 (${delaySec}s)..."
            if ($delaySec -gt 0) {
                # 细分 sleep 期间更新倒计时提示
                $rem = $delaySec
                while ($rem -gt 0) {
                    Notify-Step "部分数据滞后，等待重试补抓 (${rem}s)..."
                    $sleepStep = [math]::Min(5, $rem)
                    Start-Sleep -Seconds $sleepStep
                    $rem -= $sleepStep
                }
            }
            $retryExit = Run-CneCommand (@("run", "daily", "--stale-only") + $dateArg)
            if ($retryExit -eq 0) {
                Write-PipelineLog "stale retry OK"
                $staleStatus = "OK"
                Record-Result "stale_retry" "success" 0 "--" "补抓完成"
            } else {
                Write-PipelineLog "stale retry FAILED (see $LogFile)"
                $staleStatus = "FAILED"
                $failedSoft += "stale-retry"
                foreach ($e in $script:LastCmdErrors) { $failErrors.Add($e) }
                $errFirst = ""
                if ($script:LastCmdErrors.Count -gt 0) { $errFirst = ($script:LastCmdErrors[0] -replace '\s+', ' ').Trim() }
                $note = if ($errFirst) { "补抓失败: $errFirst" } else { "部分补抓失败" }
                Record-Result "stale_retry" "failed" 0 "--" $note
            }
        }
    }

    # Meta 备份与清理
    Write-PipelineLog "--- backup & clean ---"
    Notify-Step "正在执行元数据备份与清理..."
    if (-not $NoBackup) {
        try {
            $metaDir = Join-Path $CneRoot "data\quant_dataset\_cnequity\meta"
            if (Test-Path $metaDir) {
                $ts = Get-Date -Format "yyyyMMdd-HHmmss"
                $archive = Join-Path $BackupDir "meta-$ts.zip"
                $tempDir  = Join-Path $env:TEMP "cne_backup_$ts"
                $null = New-Item -ItemType Directory -Force -Path $tempDir
                foreach ($sub in @("state", "quality")) {
                    $src = Join-Path $metaDir $sub
                    if (Test-Path $src) { Copy-Item $src $tempDir -Recurse }
                }
                Compress-Archive -Path (Join-Path $tempDir "*") -DestinationPath $archive -Force
                Remove-Item $tempDir -Recurse -Force
                Record-Result "meta_backup" "success" 0 "--" "备份完成: $(Split-Path $archive -Leaf)"
            }
        } catch {}
    }

    if (-not $SkipClean) {
        $null = Run-CneCommand @("clean")
    }

    # ── 汇总评估 ──────────────────────────────────────────────────────────
    Notify-Phase "流水线更新完成" 8 8 100.0
    $doneMessage = ""

    # 失败原因摘要：取第一条错误行压成单行并截断，让完成摘要直接点名原因
    $reasonText = ""
    if ($failErrors.Count -gt 0) {
        $r = (($failErrors | Select-Object -First 1) -replace '\s+', ' ').Trim()
        if ($r.Length -gt 90) { $r = $r.Substring(0, 90) + "…" }
        $reasonText = " — $r"
    }

    if ($failedGates.Count -gt 0) {
        $isAllSuccess = $false
        $doneMessage = "核心门禁失败: $($failedGates -join ', ')$reasonText"
        Write-PipelineLog "==== daily pipeline DONE — GATE FAILED: $($failedGates -join ', ')$reasonText ===="
    } elseif ($failedSoft.Count -gt 0) {
        $isAllSuccess = $SoftFailOk
        # 点名失败项：优先逐个数据集（中文名对照），超过 8 个收敛为计数
        $failedNames = @(
            $resultsList | Where-Object { $_.Status -eq 'failed' } | ForEach-Object {
                $cn = if ($Global:CneDatasetNames[$_.Dataset]) { $Global:CneDatasetNames[$_.Dataset] } else { $_.Dataset }
                "$cn($($_.Dataset))"
            } | Select-Object -Unique
        )
        $names = if ($failedNames.Count -gt 0) {
            $head = ($failedNames | Select-Object -First 8) -join '、'
            if ($failedNames.Count -gt 8) { "$head 等 $($failedNames.Count) 项" } else { $head }
        } else {
            $failedSoft -join ', '
        }
        $doneMessage = "数据同步完成，失败项: $names$reasonText"
        Write-PipelineLog "==== daily pipeline DONE — soft FAILED (warn-only): $($failedSoft -join ', ')$reasonText ===="
    } else {
        $isAllSuccess = $true
        $doneMessage = "全部数据集与波次同步成功！"
        Write-PipelineLog "==== daily pipeline DONE ok ===="
    }

    $statusObj.is_completed = $true
    $statusObj.is_success   = $isAllSuccess
    $statusObj.message      = $doneMessage
    Save-StatusState

    if ($SyncContext) {
        $SyncContext.IsSuccess   = $isAllSuccess
        $SyncContext.Message     = $doneMessage
        $SyncContext.IsCompleted = $true
    }

    return $(if ($failedGates.Count -gt 0) { 1 } else { 0 })
}

# ── 执行分支：GUI 桌面浮窗模式 vs CLI 纯命令行模式 ───────────────────────
if ($UseGui) {
    # 隐藏控制台黑框窗口
    try {
        $consolePtr = [Cne.WinUtil]::GetConsoleWindow()
        if ($consolePtr -and $consolePtr -ne [IntPtr]::Zero) {
            [void][Cne.WinUtil]::ShowWindow($consolePtr, 0) # 0 = SW_HIDE
        }
    } catch {}

    # 加载 WPF GUI 模块
    Add-Type -AssemblyName PresentationFramework, PresentationCore, WindowsBase
    Import-Module $GuiModule -Force

    $effectiveDate = if ($TradeDate) { $TradeDate } else { (Get-Date -Format "yyyy-MM-dd") }
    $app = New-CneDailyGuiApp -TradeDate $effectiveDate -LogPath $LogFile

    # 状态文件路径（CLI 子进程与前台 GUI 通信媒介）
    $activeStatusFile = if ($StatusFile) { $StatusFile } else { Join-Path $LogDir "daily-sync-status.json" }
    try { Remove-Item $activeStatusFile -Force -ErrorAction SilentlyContinue } catch {}

    # 启动后台 CLI 独立工作进程（彻底解决 ThreadJob 跨 Runspace 丢失函数与变量的问题）
    $pwshBin = (Get-Command pwsh.exe -ErrorAction SilentlyContinue).Source
    if (-not $pwshBin) { $pwshBin = "pwsh.exe" }

    $workerJob = Start-ThreadJob -ScriptBlock {
        param($bin, $script, $td, $sf, $nsr, $nb, $q, $sc, $sef)
        $argsList = @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", $script, "-NoGui", "-StatusFile", $sf)
        if ($td)  { $argsList += @("-TradeDate", $td) }
        if ($nsr) { $argsList += "-NoStaleRetry" }
        if ($nb)  { $argsList += "-NoBackup" }
        if ($q)   { $argsList += "-Quiet" }
        if ($sc)  { $argsList += "-SkipClean" }
        if ($sef) { $argsList += "-SkipEtfFund" }

        & $bin @argsList
        return $LASTEXITCODE
    } -ArgumentList $pwshBin, $PSCommandPath, $TradeDate, $activeStatusFile, [bool]$NoStaleRetry, [bool]$NoBackup, [bool]$Quiet, [bool]$SkipClean, [bool]$SkipEtfFund

    # 前台 Dispatcher 轮询定时器
    $summaryShown = $false
    $recordedKeys = [System.Collections.Generic.HashSet[string]]::new()
    $lastPhase = ""
    $lastStep  = ""

    $pollTimer = [System.Windows.Threading.DispatcherTimer]::new()
    $pollTimer.Interval = [TimeSpan]::FromMilliseconds(100)

    $pollTimer.add_Tick({
        # 1. 尝试读取状态文件
        if (Test-Path $activeStatusFile) {
            try {
                $rawJson = [System.IO.File]::ReadAllText($activeStatusFile, [System.Text.Encoding]::UTF8)
                if ($rawJson) {
                    $st = $rawJson | ConvertFrom-Json
                    if ($st) {
                        if ($st.phase_name -and $st.phase_name -ne $lastPhase) {
                            $app.UpdatePhase($st.phase_name, [int]$st.phase_index, [int]$st.phase_total, [double]$st.progress_pct)
                            $lastPhase = $st.phase_name
                        }
                        if ($st.active_dataset) {
                            $app.SetStepRunning($st.active_dataset)
                        }
                        if ($st.current_step -and $st.current_step -ne $lastStep) {
                            $app.UpdateStep($st.current_step)
                            $lastStep = $st.current_step
                        }
                        if ($st.results) {
                            foreach ($r in $st.results) {
                                $rKey = "$($r.Dataset)_$($r.Status)_$($r.Rows)_$($r.Elapsed)"
                                if ($recordedKeys.Add($rKey)) {
                                    $app.RecordStepResult($r.Dataset, $r.Status, [int]$r.Rows, $r.Elapsed, $r.Note)
                                }
                            }
                        }
                        if ($st.is_completed -and -not $summaryShown) {
                            $summaryShown = $true
                            $app.ShowCompletionSummary([bool]$st.is_success, [string]$st.message)
                            $pollTimer.Stop()
                            return
                        }
                    }
                }
            } catch {}
        }

        # 2. 检查子工作进程状态（防呆异常捕获，绝不挂死在启动中）
        if ($workerJob.State -ne 'Running' -and -not $summaryShown) {
            $summaryShown = $true
            $jobExit = $null
            try { $jobExit = Receive-Job $workerJob } catch {}
            $isOk = ($jobExit -eq 0)
            $msg = if ($isOk) { "全部数据集与波次同步成功！" } else { "流水线执行异常终止 (退出码: $jobExit)，请查阅详细日志。" }
            $app.ShowCompletionSummary($isOk, $msg)
            $pollTimer.Stop()
        }
    })

    $pollTimer.Start()

    # 阻塞前台主线程展示 WPF 浮窗，直到用户点击“确定关闭”
    [void]$app.Window.ShowDialog()

    # 清理后台作业
    try {
        Wait-Job $workerJob -Timeout 2 | Out-Null
        Receive-Job $workerJob | Out-Null
        Remove-Job $workerJob -Force
    } catch {}

    exit 0
} else {
    # 纯命令行控制台模式（CLI）
    $activeStatusFile = if ($StatusFile) { $StatusFile } else { Join-Path $LogDir "daily-sync-status.json" }
    $exit = Execute-CnePipeline -StatusFile $activeStatusFile -TradeDate $TradeDate -NoStaleRetry ([bool]$NoStaleRetry) -NoBackup ([bool]$NoBackup) -Quiet ([bool]$Quiet) -SkipClean ([bool]$SkipClean) -SkipEtfFund ([bool]$SkipEtfFund)
    exit $exit
}
