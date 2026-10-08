using namespace System.Windows
using namespace System.Windows.Controls
using namespace System.Windows.Threading
using namespace System.Windows.Media

<#
.SYNOPSIS
  CNE 每日数据同步 - 现代化可视化进度与对账看板
#>

Add-Type -AssemblyName PresentationFramework, PresentationCore, WindowsBase, System.Drawing, System.Windows.Forms

# ── 预置步骤清单：数据外置到 CneDailySteps.ps1（纯数据、无 WPF 依赖）─────
# CLI worker（run_cne_daily.ps1 的 -NoGui 子进程）也要用同一份清单标记受阻数据集，
# 因此抽成独立数据文件，两边 dot-source 同一真源，禁止两处维护。
. (Join-Path $PSScriptRoot "CneDailySteps.ps1")

$Global:CneGuiXaml = @"
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="CNE 数据湖 · 每日同步看板" Width="840" Height="650"
        WindowStartupLocation="CenterScreen" ResizeMode="CanResizeWithGrip"
        Background="#1E1E2E" FontFamily="Segoe UI, Microsoft YaHei UI"
        Topmost="False">
    <Window.Resources>
        <Style TargetType="TextBlock">
            <Setter Property="Foreground" Value="#CDD6F4"/>
        </Style>
        <Style x:Key="HeaderTitle" TargetType="TextBlock">
            <Setter Property="FontSize" Value="16"/>
            <Setter Property="FontWeight" Value="SemiBold"/>
            <Setter Property="Foreground" Value="#89B4FA"/>
        </Style>
        <Style x:Key="SubTitle" TargetType="TextBlock">
            <Setter Property="FontSize" Value="12"/>
            <Setter Property="Foreground" Value="#A6ADC8"/>
        </Style>
        <Style x:Key="ActionButton" TargetType="Button">
            <Setter Property="Background" Value="#313244"/>
            <Setter Property="Foreground" Value="#CDD6F4"/>
            <Setter Property="BorderThickness" Value="1"/>
            <Setter Property="BorderBrush" Value="#45475A"/>
            <Setter Property="Padding" Value="14,6"/>
            <Setter Property="FontSize" Value="12"/>
            <Setter Property="Cursor" Value="Hand"/>
            <Setter Property="Template">
                <Setter.Value>
                    <ControlTemplate TargetType="Button">
                        <Border Background="{TemplateBinding Background}"
                                BorderBrush="{TemplateBinding BorderBrush}"
                                BorderThickness="{TemplateBinding BorderThickness}"
                                CornerRadius="4">
                            <ContentPresenter HorizontalAlignment="Center" VerticalAlignment="Center"/>
                        </Border>
                    </ControlTemplate>
                </Setter.Value>
            </Setter>
        </Style>
        <Style x:Key="PrimaryButton" TargetType="Button" BasedOn="{StaticResource ActionButton}">
            <Setter Property="Background" Value="#89B4FA"/>
            <Setter Property="Foreground" Value="#11111B"/>
            <Setter Property="FontWeight" Value="SemiBold"/>
            <Setter Property="BorderThickness" Value="0"/>
        </Style>
        <Style TargetType="ProgressBar">
            <Setter Property="Background" Value="#313244"/>
            <Setter Property="Foreground" Value="#89B4FA"/>
            <Setter Property="BorderThickness" Value="0"/>
            <Setter Property="Template">
                <Setter.Value>
                    <ControlTemplate TargetType="ProgressBar">
                        <Grid>
                            <Border Background="{TemplateBinding Background}" CornerRadius="4"/>
                            <Border Name="PART_Indicator" Background="{TemplateBinding Foreground}" CornerRadius="4" HorizontalAlignment="Left"/>
                        </Grid>
                    </ControlTemplate>
                </Setter.Value>
            </Setter>
        </Style>
    </Window.Resources>

    <Grid Margin="18">
        <Grid.RowDefinitions>
            <RowDefinition Height="Auto"/> <!-- 顶部标题与总体状态 -->
            <RowDefinition Height="Auto"/> <!-- 进度条与当前步骤 -->
            <RowDefinition Height="*"/>    <!-- 核心数据集对账看板表格 -->
            <RowDefinition Height="Auto"/> <!-- 底部控制栏 -->
        </Grid.RowDefinitions>

        <!-- 顶部导航栏 / 标题 -->
        <Border Grid.Row="0" Background="#252538" CornerRadius="6" Padding="14,10" Margin="0,0,0,12" BorderBrush="#313244" BorderThickness="1">
            <Grid>
                <Grid.ColumnDefinitions>
                    <ColumnDefinition Width="*"/>
                    <ColumnDefinition Width="Auto"/>
                </Grid.ColumnDefinitions>
                <StackPanel Grid.Column="0">
                    <StackPanel Orientation="Horizontal" VerticalAlignment="Center">
                        <TextBlock Text="🌊" FontSize="18" Margin="0,0,8,0"/>
                        <TextBlock Text="CNE 数据湖 · 每日流水线更新看板" Style="{StaticResource HeaderTitle}"/>
                        <Border Name="badgeOverallStatus" Background="#1E382B" CornerRadius="3" Padding="6,2" Margin="12,0,0,0" VerticalAlignment="Center">
                            <TextBlock Name="txtOverallStatus" Text="同步中..." Foreground="#A6E3A1" FontSize="11" FontWeight="SemiBold"/>
                        </Border>
                    </StackPanel>
                    <TextBlock Name="txtSubHeader" Text="准备执行数据同步..." Style="{StaticResource SubTitle}" Margin="26,3,0,0"/>
                </StackPanel>
                <Border Grid.Column="1" Background="#313244" CornerRadius="4" Padding="10,4" VerticalAlignment="Center">
                    <TextBlock Name="txtTimer" Text="用时: 00:00" Foreground="#F9E2AF" FontSize="12" FontFamily="Consolas, Segoe UI"/>
                </Border>
            </Grid>
        </Border>

        <!-- 进度与当前动作指示区 -->
        <Border Grid.Row="1" Background="#252538" CornerRadius="6" Padding="14,12" Margin="0,0,0,12" BorderBrush="#313244" BorderThickness="1">
            <Grid>
                <Grid.RowDefinitions>
                    <RowDefinition Height="Auto"/>
                    <RowDefinition Height="Auto"/>
                    <RowDefinition Height="Auto"/>
                    <RowDefinition Height="Auto"/>
                </Grid.RowDefinitions>

                <!-- 当前波次指示 -->
                <Grid Grid.Row="0" Margin="0,0,0,6">
                    <Grid.ColumnDefinitions>
                        <ColumnDefinition Width="*"/>
                        <ColumnDefinition Width="Auto"/>
                    </Grid.ColumnDefinitions>
                    <TextBlock Name="txtCurrentPhase" Text="正在执行: Core (核心行情与基础参考)" FontSize="13" FontWeight="SemiBold" Foreground="#89B4FA"/>
                    <TextBlock Name="txtPhaseIndicator" Grid.Column="1" Text="阶段 1 / 8" Foreground="#89DCEB" FontSize="12"/>
                </Grid>

                <!-- 当前具体操作说明 (过滤后的干净业务文本) -->
                <TextBlock Name="txtCurrentStep" Grid.Row="1" Text="正在初始化数据环境..." FontSize="12" Foreground="#A6ADC8" Margin="0,0,0,8" TextTrimming="CharacterEllipsis"/>

                <!-- 总进度条与百分比 -->
                <Grid Grid.Row="2" Margin="0,0,0,10">
                    <Grid.ColumnDefinitions>
                        <ColumnDefinition Width="*"/>
                        <ColumnDefinition Width="Auto"/>
                    </Grid.ColumnDefinitions>
                    <ProgressBar Name="pbTotal" Height="8" Minimum="0" Maximum="100" Value="0" VerticalAlignment="Center" Margin="0,0,12,0"/>
                    <TextBlock Name="txtProgressPct" Grid.Column="1" Text="0%" Foreground="#89B4FA" FontWeight="Bold" FontSize="13" FontFamily="Consolas, Segoe UI"/>
                </Grid>

                <!-- 统计计数分布 -->
                <StackPanel Grid.Row="3" Orientation="Horizontal">
                    <TextBlock Text="已成功: " FontSize="11" Foreground="#6C7086"/>
                    <TextBlock Name="cntSuccess" Text="0" FontSize="11" FontWeight="Bold" Foreground="#A6E3A1" Margin="0,0,16,0"/>

                    <TextBlock Text="同步中: " FontSize="11" Foreground="#6C7086"/>
                    <TextBlock Name="cntRunning" Text="0" FontSize="11" FontWeight="Bold" Foreground="#89B4FA" Margin="0,0,16,0"/>

                    <TextBlock Text="周期跳过: " FontSize="11" Foreground="#6C7086"/>
                    <TextBlock Name="cntSkip" Text="0" FontSize="11" FontWeight="Bold" Foreground="#A6ADC8" Margin="0,0,16,0"/>

                    <TextBlock Text="失败项: " FontSize="11" Foreground="#6C7086"/>
                    <TextBlock Name="cntFailed" Text="0" FontSize="11" FontWeight="Bold" Foreground="#F38BA8" Margin="0,0,16,0"/>

                    <TextBlock Text="排队等待: " FontSize="11" Foreground="#6C7086"/>
                    <TextBlock Name="cntWaiting" Text="0" FontSize="11" FontWeight="Bold" Foreground="#6C7086"/>
                </StackPanel>
            </Grid>
        </Border>

        <!-- 中部主体：核心数据集对账表格看板 -->
        <Border Grid.Row="2" Background="#252538" CornerRadius="6" Padding="12" BorderBrush="#313244" BorderThickness="1">
            <Grid>
                <Grid.RowDefinitions>
                    <RowDefinition Height="Auto"/> <!-- 表头 -->
                    <RowDefinition Height="Auto"/> <!-- 分割线 -->
                    <RowDefinition Height="*"/>    <!-- 可滚动表体 -->
                </Grid.RowDefinitions>

                <!-- 表头 -->
                <Grid Grid.Row="0" Margin="4,2,4,6">
                    <Grid.ColumnDefinitions>
                        <ColumnDefinition Width="65"/>  <!-- 状态 -->
                        <ColumnDefinition Width="80"/>  <!-- 板块分类 -->
                        <ColumnDefinition Width="240"/> <!-- 数据集 (中文名称) -->
                        <ColumnDefinition Width="100"/> <!-- 同步行数 -->
                        <ColumnDefinition Width="70"/>  <!-- 耗时 -->
                        <ColumnDefinition Width="*"/>   <!-- 说明 -->
                    </Grid.ColumnDefinitions>
                    <TextBlock Grid.Column="0" Text="状态" FontSize="11" Foreground="#6C7086" FontWeight="SemiBold"/>
                    <TextBlock Grid.Column="1" Text="业务板块" FontSize="11" Foreground="#6C7086" FontWeight="SemiBold"/>
                    <TextBlock Grid.Column="2" Text="数据集及中文对照" FontSize="11" Foreground="#6C7086" FontWeight="SemiBold"/>
                    <TextBlock Grid.Column="3" Text="更新行数" FontSize="11" Foreground="#6C7086" FontWeight="SemiBold"/>
                    <TextBlock Grid.Column="4" Text="耗时" FontSize="11" Foreground="#6C7086" FontWeight="SemiBold"/>
                    <TextBlock Grid.Column="5" Text="执行状态与详情说明" FontSize="11" Foreground="#6C7086" FontWeight="SemiBold"/>
                </Grid>

                <Separator Grid.Row="1" Background="#313244" Margin="0,0,0,4"/>

                <!-- 表体 -->
                <ScrollViewer Grid.Row="2" Name="scrollSummary" VerticalScrollBarVisibility="Auto">
                    <StackPanel Name="panelSummaryList">
                        <!-- 动态填充各数据集行 -->
                    </StackPanel>
                </ScrollViewer>
            </Grid>
        </Border>

        <!-- 底部操作按钮栏 -->
        <Grid Grid.Row="3" Margin="0,12,0,0">
            <Grid.ColumnDefinitions>
                <ColumnDefinition Width="Auto"/>
                <ColumnDefinition Width="*"/>
                <ColumnDefinition Width="Auto"/>
                <ColumnDefinition Width="Auto"/>
            </Grid.ColumnDefinitions>

            <TextBlock Name="txtFooterHint" Grid.Column="0" Text="流水线正在后台并发更新，请勿断网或强制关闭电脑" VerticalAlignment="Center" FontSize="11" Foreground="#6C7086"/>

            <Button Name="btnViewLog" Grid.Column="2" Content="查看详细日志" Style="{StaticResource ActionButton}" Margin="0,0,8,0"/>
            <Button Name="btnClose" Grid.Column="3" Content="确定关闭" Style="{StaticResource PrimaryButton}" IsEnabled="False"/>
        </Grid>
    </Grid>
</Window>
"@

class StepRowWidgets {
    [System.Windows.Controls.Grid]$Container
    [System.Windows.Controls.Border]$StatusBadge
    [System.Windows.Controls.TextBlock]$BadgeText
    [System.Windows.Controls.TextBlock]$RowsText
    [System.Windows.Controls.TextBlock]$TimeText
    [System.Windows.Controls.TextBlock]$NoteText
    [string]$CurrentStatus = "waiting"
}

class CneGuiApp {
    [System.Windows.Window]$Window
    [System.Windows.Controls.TextBlock]$txtSubHeader
    [System.Windows.Controls.TextBlock]$txtTimer
    [System.Windows.Controls.TextBlock]$txtCurrentPhase
    [System.Windows.Controls.TextBlock]$txtCurrentStep
    [System.Windows.Controls.TextBlock]$txtPhaseIndicator
    [System.Windows.Controls.TextBlock]$txtProgressPct
    [System.Windows.Controls.TextBlock]$txtFooterHint
    [System.Windows.Controls.Border]$badgeOverallStatus
    [System.Windows.Controls.TextBlock]$txtOverallStatus
    [System.Windows.Controls.ProgressBar]$pbTotal

    [System.Windows.Controls.TextBlock]$cntSuccess
    [System.Windows.Controls.TextBlock]$cntRunning
    [System.Windows.Controls.TextBlock]$cntSkip
    [System.Windows.Controls.TextBlock]$cntFailed
    [System.Windows.Controls.TextBlock]$cntWaiting

    [System.Windows.Controls.ScrollViewer]$scrollSummary
    [System.Windows.Controls.StackPanel]$panelSummaryList
    [System.Windows.Controls.Button]$btnViewLog
    [System.Windows.Controls.Button]$btnClose

    [System.Diagnostics.Stopwatch]$Stopwatch
    [System.Windows.Threading.DispatcherTimer]$Timer
    [string]$LogFilePath = ""
    [hashtable]$RowMap

    CneGuiApp([string]$tradeDate, [string]$logPath) {
        $this.LogFilePath = $logPath
        $this.RowMap = @{}

        $reader = [System.Xml.XmlReader]::Create([System.IO.StringReader]::new($Global:CneGuiXaml))
        $this.Window = [System.Windows.Markup.XamlReader]::Load($reader)

        $this.txtSubHeader        = $this.Window.FindName("txtSubHeader")
        $this.txtTimer            = $this.Window.FindName("txtTimer")
        $this.txtCurrentPhase     = $this.Window.FindName("txtCurrentPhase")
        $this.txtCurrentStep      = $this.Window.FindName("txtCurrentStep")
        $this.txtPhaseIndicator   = $this.Window.FindName("txtPhaseIndicator")
        $this.txtProgressPct      = $this.Window.FindName("txtProgressPct")
        $this.txtFooterHint       = $this.Window.FindName("txtFooterHint")
        $this.badgeOverallStatus  = $this.Window.FindName("badgeOverallStatus")
        $this.txtOverallStatus    = $this.Window.FindName("txtOverallStatus")
        $this.pbTotal             = $this.Window.FindName("pbTotal")

        $this.cntSuccess          = $this.Window.FindName("cntSuccess")
        $this.cntRunning          = $this.Window.FindName("cntRunning")
        $this.cntSkip             = $this.Window.FindName("cntSkip")
        $this.cntFailed           = $this.Window.FindName("cntFailed")
        $this.cntWaiting          = $this.Window.FindName("cntWaiting")

        $this.scrollSummary       = $this.Window.FindName("scrollSummary")
        $this.panelSummaryList    = $this.Window.FindName("panelSummaryList")
        $this.btnViewLog          = $this.Window.FindName("btnViewLog")
        $this.btnClose            = $this.Window.FindName("btnClose")

        $this.txtSubHeader.Text = "交易日锚点: $tradeDate | 调度环境: 本地数据湖 (CNEquity)"

        # 初始化预制表格行
        $this.BuildPredefinedRows()
        $this.RecountTotals()

        # 计时器
        $this.Stopwatch = [System.Diagnostics.Stopwatch]::StartNew()
        $this.Timer = [System.Windows.Threading.DispatcherTimer]::new()
        $this.Timer.Interval = [TimeSpan]::FromSeconds(1)
        $this.Timer.Tag = $this
        $this.Timer.add_Tick({
            param($sender, $e)
            $self = $sender.Tag
            if ($self -and $self.Stopwatch -and $self.txtTimer) {
                $elapsed = $self.Stopwatch.Elapsed
                $self.txtTimer.Text = "用时: {0:D2}:{1:D2}" -f [int]$elapsed.TotalMinutes, $elapsed.Seconds
            }
        })
        $this.Timer.Start()

        # 按钮事件
        $this.btnClose.Tag = $this
        $this.btnClose.add_Click({
            param($sender, $e)
            $self = $sender.Tag
            if ($self) {
                if ($self.Timer) { $self.Timer.Stop() }
                if ($self.Window) { $self.Window.Close() }
            }
        })

        $this.btnViewLog.Tag = $this
        $this.btnViewLog.add_Click({
            param($sender, $e)
            $self = $sender.Tag
            if ($self -and $self.LogFilePath -and (Test-Path $self.LogFilePath)) {
                Start-Process "notepad.exe" -ArgumentList "`"$($self.LogFilePath)`""
            }
        })

        # 防误关守卫：流水线未结束（btnClose 仍禁用）时拦截标题栏 X / Alt+F4 关闭。
        # 前台脚本关闭时会 Remove-Job -Force 连带杀死后台 CLI worker——
        # 2026-10-08 16:35 的每日同步就是在 fundamentals 波次被这样中断的。
        $this.Window.Tag = $this
        $this.Window.add_Closing({
            param($sender, $e)
            $self = $sender.Tag
            if (-not $self) { return }
            if (-not $self.btnClose.IsEnabled) {
                $e.Cancel = $true
                $self.txtFooterHint.Text = "同步仍在运行，完成后才可关闭；确需中止请用任务计划程序停止 QuantUIDataSync"
                $self.txtFooterHint.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#F38BA8")
            }
        })
    }

    [void]BuildPredefinedRows() {
        foreach ($item in $Global:CnePredefinedSteps) {
            $rowGrid = [System.Windows.Controls.Grid]::new()
            $rowGrid.Margin = [System.Windows.Thickness]::new(4, 3, 4, 3)

            $c1 = [System.Windows.Controls.ColumnDefinition]::new(); $c1.Width = [System.Windows.GridLength]::new(65, [System.Windows.GridUnitType]::Pixel)
            $c2 = [System.Windows.Controls.ColumnDefinition]::new(); $c2.Width = [System.Windows.GridLength]::new(80, [System.Windows.GridUnitType]::Pixel)
            $c3 = [System.Windows.Controls.ColumnDefinition]::new(); $c3.Width = [System.Windows.GridLength]::new(240, [System.Windows.GridUnitType]::Pixel)
            $c4 = [System.Windows.Controls.ColumnDefinition]::new(); $c4.Width = [System.Windows.GridLength]::new(100, [System.Windows.GridUnitType]::Pixel)
            $c5 = [System.Windows.Controls.ColumnDefinition]::new(); $c5.Width = [System.Windows.GridLength]::new(70, [System.Windows.GridUnitType]::Pixel)
            $c6 = [System.Windows.Controls.ColumnDefinition]::new(); $c6.Width = [System.Windows.GridLength]::new(1, [System.Windows.GridUnitType]::Star)
            $rowGrid.ColumnDefinitions.Add($c1); $rowGrid.ColumnDefinitions.Add($c2)
            $rowGrid.ColumnDefinitions.Add($c3); $rowGrid.ColumnDefinitions.Add($c4)
            $rowGrid.ColumnDefinitions.Add($c5); $rowGrid.ColumnDefinitions.Add($c6)

            # 1. 状态徽章
            $statusBadge = [System.Windows.Controls.Border]::new()
            $statusBadge.CornerRadius = [System.Windows.CornerRadius]::new(3)
            $statusBadge.Padding = [System.Windows.Thickness]::new(5, 1, 5, 1)
            $statusBadge.HorizontalAlignment = [System.Windows.HorizontalAlignment]::Left
            $statusBadge.Background = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#313244")

            $badgeText = [System.Windows.Controls.TextBlock]::new()
            $badgeText.FontSize = 10
            $badgeText.FontWeight = [System.Windows.FontWeights]::SemiBold
            $badgeText.Text = "等待中"
            $badgeText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#6C7086")
            $statusBadge.Child = $badgeText
            [System.Windows.Controls.Grid]::SetColumn($statusBadge, 0)

            # 2. 分类
            $catText = [System.Windows.Controls.TextBlock]::new()
            $catText.Text = $item.Category
            $catText.FontSize = 11
            $catText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#89B4FA")
            [System.Windows.Controls.Grid]::SetColumn($catText, 1)

            # 3. 数据集名称 (含中文)
            $nameText = [System.Windows.Controls.TextBlock]::new()
            $nameText.Text = "$($item.Dataset) ($($item.CnName))"
            $nameText.FontSize = 11
            $nameText.FontWeight = [System.Windows.FontWeights]::Medium
            $nameText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#CDD6F4")
            $nameText.TextTrimming = [System.Windows.TextTrimming]::CharacterEllipsis
            [System.Windows.Controls.Grid]::SetColumn($nameText, 2)

            # 4. 行数
            $rowsText = [System.Windows.Controls.TextBlock]::new()
            $rowsText.Text = "--"
            $rowsText.FontSize = 11
            $rowsText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#6C7086")
            [System.Windows.Controls.Grid]::SetColumn($rowsText, 3)

            # 5. 耗时
            $timeText = [System.Windows.Controls.TextBlock]::new()
            $timeText.Text = "--"
            $timeText.FontSize = 11
            $timeText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#6C7086")
            [System.Windows.Controls.Grid]::SetColumn($timeText, 4)

            # 6. 说明
            $noteText = [System.Windows.Controls.TextBlock]::new()
            $noteText.Text = "排队等待中"
            $noteText.FontSize = 11
            $noteText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#6C7086")
            $noteText.TextTrimming = [System.Windows.TextTrimming]::CharacterEllipsis
            [System.Windows.Controls.Grid]::SetColumn($noteText, 5)

            $rowGrid.Children.Add($statusBadge); $rowGrid.Children.Add($catText)
            $rowGrid.Children.Add($nameText); $rowGrid.Children.Add($rowsText)
            $rowGrid.Children.Add($timeText); $rowGrid.Children.Add($noteText)

            $this.panelSummaryList.Children.Add($rowGrid)

            $widgets = [StepRowWidgets]::new()
            $widgets.Container   = $rowGrid
            $widgets.StatusBadge = $statusBadge
            $widgets.BadgeText   = $badgeText
            $widgets.RowsText    = $rowsText
            $widgets.TimeText    = $timeText
            $widgets.NoteText    = $noteText
            $widgets.CurrentStatus = "waiting"

            $this.RowMap[$item.Dataset] = $widgets
        }
    }

    [void]RecountTotals() {
        $nSuccess = 0
        $nRunning = 0
        $nSkip    = 0
        $nFailed  = 0
        $nWaiting = 0

        foreach ($w in $this.RowMap.Values) {
            switch ($w.CurrentStatus) {
                "success" { $nSuccess++ }
                "running" { $nRunning++ }
                "skip"    { $nSkip++ }
                "failed"  { $nFailed++ }
                default   { $nWaiting++ }
            }
        }

        $this.cntSuccess.Text = "$nSuccess"
        $this.cntRunning.Text = "$nRunning"
        $this.cntSkip.Text    = "$nSkip"
        $this.cntFailed.Text  = "$nFailed"
        $this.cntWaiting.Text = "$nWaiting"
    }

    [void]SetStepRunning([string]$dataset) {
        if (-not $this.RowMap.ContainsKey($dataset)) { return }
        $this.Window.Dispatcher.Invoke([Action]{
            $w = $this.RowMap[$dataset]
            if ($w.CurrentStatus -eq "waiting") {
                $w.CurrentStatus = "running"
                $w.StatusBadge.Background = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#1E2D4A")
                $w.BadgeText.Text = "同步中"
                $w.BadgeText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#89B4FA")
                $w.NoteText.Text = "正在从数据源拉取并校验..."
                $w.NoteText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#89DCEB")
                $w.Container.BringIntoView()
                $this.RecountTotals()
            }
        })
    }

    [void]UpdatePhase([string]$phaseName, [int]$phaseIndex, [int]$totalPhases, [double]$pct) {
        $this.Window.Dispatcher.Invoke([Action]{
            $this.txtCurrentPhase.Text = "正在执行: $phaseName"
            $this.txtPhaseIndicator.Text = "阶段 $phaseIndex / $totalPhases"
            $this.pbTotal.Value = [math]::Min(100.0, [math]::Max(0.0, $pct))
            $this.txtProgressPct.Text = "{0:N0}%" -f $pct
        })
    }

    [void]UpdateStep([string]$stepText) {
        $this.Window.Dispatcher.Invoke([Action]{
            $this.txtCurrentStep.Text = "$stepText"
        })
    }

    [void]RecordStepResult([string]$dataset, [string]$status, [int]$rows, [string]$elapsed, [string]$note) {
        $this.Window.Dispatcher.Invoke([Action]{
            if ($this.RowMap.ContainsKey($dataset)) {
                $w = $this.RowMap[$dataset]
                $w.CurrentStatus = $status

                if ($status -eq "success" -or $status -eq "OK") {
                    $w.StatusBadge.Background = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#1E382B")
                    $w.BadgeText.Text = "成功"
                    $w.BadgeText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#A6E3A1")
                    if ($rows -gt 0) {
                        $w.RowsText.Text = "+$rows 行"
                        $w.RowsText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#A6E3A1")
                    } else {
                        $w.RowsText.Text = "0 行"
                        $w.RowsText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#A6ADC8")
                    }
                    $w.TimeText.Text = if ($elapsed) { $elapsed } else { "--" }
                    $w.TimeText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#CDD6F4")
                    $w.NoteText.Text = if ($note) { $note } else { "同步完成" }
                    $w.NoteText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#A6ADC8")
                } elseif ($status -eq "skip") {
                    $w.StatusBadge.Background = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#313244")
                    $w.BadgeText.Text = "跳过"
                    $w.BadgeText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#A6ADC8")
                    $w.RowsText.Text = "--"
                    $w.RowsText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#6C7086")
                    $w.TimeText.Text = "<1s"
                    $w.NoteText.Text = if ($note) { $note } else { "同周期无变动自动跳过" }
                    $w.NoteText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#6C7086")
                } elseif ($status -eq "warning") {
                    $w.StatusBadge.Background = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#3C3224")
                    $w.BadgeText.Text = "提示"
                    $w.BadgeText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#F9E2AF")
                    $w.RowsText.Text = if ($rows -gt 0) { "+$rows 行" } else { "0 行" }
                    $w.TimeText.Text = if ($elapsed) { $elapsed } else { "--" }
                    $w.NoteText.Text = if ($note) { $note } else { "有提示信息" }
                    $w.NoteText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#F9E2AF")
                } else {
                    $w.StatusBadge.Background = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#3C2028")
                    $w.BadgeText.Text = "失败"
                    $w.BadgeText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#F38BA8")
                    $w.RowsText.Text = "0 行"
                    $w.RowsText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#F38BA8")
                    $w.TimeText.Text = if ($elapsed) { $elapsed } else { "--" }
                    $w.NoteText.Text = if ($note) { $note } else { "执行异常" }
                    $w.NoteText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#F38BA8")
                }
                $w.Container.BringIntoView()
            }
            $this.RecountTotals()
        })
    }

    [void]ShowCompletionSummary([bool]$isSuccess, [string]$message) {
        $this.Stopwatch.Stop()
        $this.Timer.Stop()

        $this.Window.Dispatcher.Invoke([Action]{
            $this.pbTotal.Value = 100.0
            $this.txtProgressPct.Text = "100%"
            $this.txtCurrentPhase.Text = "流水线执行完毕"
            $this.txtCurrentStep.Text = $message
            $this.btnClose.IsEnabled = $true
            $this.txtFooterHint.Text = "每日数据同步已结束，点击[确定关闭]退出"

            if ($isSuccess) {
                $this.badgeOverallStatus.Background = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#1E382B")
                $this.txtOverallStatus.Text = "✅ 全量同步成功"
                $this.txtOverallStatus.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#A6E3A1")
            } else {
                $this.badgeOverallStatus.Background = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#3C2028")
                $this.txtOverallStatus.Text = "⚠️ 存在失败项"
                $this.txtOverallStatus.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#F38BA8")
            }

            $this.RecountTotals()
        })
    }
}

function New-CneDailyGuiApp {
    [CmdletBinding()]
    param(
        [string]$TradeDate = (Get-Date -Format "yyyy-MM-dd"),
        [string]$LogPath = ""
    )
    return [CneGuiApp]::new($TradeDate, $LogPath)
}

Export-ModuleMember -Function New-CneDailyGuiApp
