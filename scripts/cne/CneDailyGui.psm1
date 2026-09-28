using namespace System.Windows
using namespace System.Windows.Controls
using namespace System.Windows.Threading
using namespace System.Windows.Media

<#
.SYNOPSIS
  CNE 每日数据同步 - 现代化可视化进度与对账看板
#>

Add-Type -AssemblyName PresentationFramework, PresentationCore, WindowsBase, System.Drawing, System.Windows.Forms

$Global:CneDatasetNames = @{
    "instruments"                  = "全A标的清单"
    "trading_calendar"             = "交易日历"
    "trading_status"               = "停复牌/ST状态"
    "stock_st"                     = "风险警示板ST名单"
    "trading_status_st"            = "ST证据日更续签"
    "corporate_actions"            = "除权除息/送转"
    "tushare_wide_daily"           = "Tushare宽表行情"
    "daily_bars"                   = "A股日K线"
    "index_bars"                   = "主要指数日K线"
    "fund_bars"                    = "场内基金/ETF行情"
    "etf_bars"                     = "ETF日K线"
    "compact"                      = "数据合并入湖"
    "derive_adj_factors"           = "后复权因子计算"
    "derive_industry_index"        = "行业指数收益推导"
    "fund_flow"                    = "个股资金流向"
    "northbound_holdings"          = "陆股通持股季报"
    "northbound_flows"             = "北向资金流(已停产)"
    "margin_trading"               = "融资融券余额"
    "valuation_metrics"            = "估值指标(PE/PB/市值)"
    "sector_members"               = "板块概念成分股"
    "announcement_index"           = "巨潮公告索引"
    "fund_nav"                     = "公募基金净值"
    "index_bars_external"          = "基准指数行情"
    "dragon_tiger"                 = "龙虎榜交易明细"
    "block_trades"                 = "大宗交易明细"
    "dividend"                     = "分红送转披露"
    "namechange"                   = "股票曾用名变更"
    "share_float_external"         = "限售解禁数据"
    "stk_surv"                     = "机构调研活动"
    "fund_fees"                    = "公募基金费率"
    "financial_statement_items"    = "财报科目长表"
    "earnings_disclosure_schedule" = "预约披露时间表"
    "index_constituents"           = "指数成分股月度快照"
    "industry_members"             = "申万行业分类快照"
    "share_structure"              = "股本结构变动"
    "shareholder_counts"           = "股东户数数据"
    "balancesheet"                 = "资产负债表"
    "income"                       = "利润表"
    "cashflow"                     = "现金流量表"
    "fina_indicator"               = "财务核心指标"
    "report_rc"                    = "研报盈利预测"
    "macro_indicators"             = "宏观经济指标"
    "market_breadth"               = "市场宽度(涨跌家数)"
    "share_unlock_schedule"        = "限售解禁计划"
    "regulatory_events"            = "监管处罚公告"
    "commodity_bars"               = "商品期货主连K线"
    "institutional_holdings"       = "机构持股汇总"
    "analyst_consensus"            = "分析师一致预期"
    "hot_rank"                     = "东财人气热榜"
    "sector_bars"                  = "板块K线(同花顺)"
    "sector_fund_flow"             = "板块资金流向"
    "news_headlines"               = "新闻电报提要"
    "flash_news_wire"              = "7x24快讯"
    "sentiment_articles"           = "个股新闻舆情打分"
    "sentiment_scores"             = "市场情绪综合得分"
}

$Global:CneGuiXaml = @"
<Window xmlns="http://schemas.microsoft.com/winfx/2006/xaml/presentation"
        xmlns:x="http://schemas.microsoft.com/winfx/2006/xaml"
        Title="CNE 数据湖 · 每日同步" Width="720" Height="550"
        WindowStartupLocation="CenterScreen" ResizeMode="CanMinimize"
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
            <RowDefinition Height="Auto"/>
            <RowDefinition Height="*"/>
            <RowDefinition Height="Auto"/>
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
                        <TextBlock Text="CNE 数据湖 · 每日流水线更新" Style="{StaticResource HeaderTitle}"/>
                    </StackPanel>
                    <TextBlock Name="txtSubHeader" Text="准备执行数据同步..." Style="{StaticResource SubTitle}" Margin="26,2,0,0"/>
                </StackPanel>
                <Border Grid.Column="1" Background="#313244" CornerRadius="4" Padding="10,4" VerticalAlignment="Center">
                    <TextBlock Name="txtTimer" Text="用时: 00:00" Foreground="#F9E2AF" FontSize="12" FontFamily="Consolas, Segoe UI"/>
                </Border>
            </Grid>
        </Border>

        <!-- 中部主体：运行态视图 (RunningPanel) 与 总结报告视图 (SummaryPanel) 切换 -->
        <Grid Grid.Row="1">
            <!-- 运行中视图 -->
            <Border Name="panelRunning" Background="#252538" CornerRadius="6" Padding="16" BorderBrush="#313244" BorderThickness="1" Visibility="Visible">
                <Grid>
                    <Grid.RowDefinitions>
                        <RowDefinition Height="Auto"/>
                        <RowDefinition Height="Auto"/>
                        <RowDefinition Height="Auto"/>
                        <RowDefinition Height="*"/>
                    </Grid.RowDefinitions>

                    <!-- 当前波次大徽章 -->
                    <StackPanel Grid.Row="0" Margin="0,4,0,12">
                        <TextBlock Name="txtCurrentPhase" Text="正在执行: Core (核心行情与基础参考)" FontSize="14" FontWeight="SemiBold" Foreground="#89B4FA"/>
                        <TextBlock Name="txtCurrentStep" Text="当前操作: 正在初始化..." FontSize="12" Foreground="#A6ADC8" Margin="0,4,0,0"/>
                    </StackPanel>

                    <!-- 总进度条 -->
                    <Grid Grid.Row="1" Margin="0,0,0,14">
                        <ProgressBar Name="pbTotal" Height="10" Minimum="0" Maximum="100" Value="0"/>
                    </Grid>

                    <!-- 百分比指示与阶段分布 -->
                    <Grid Grid.Row="2" Margin="0,0,0,10">
                        <Grid.ColumnDefinitions>
                            <ColumnDefinition Width="Auto"/>
                            <ColumnDefinition Width="*"/>
                            <ColumnDefinition Width="Auto"/>
                        </Grid.ColumnDefinitions>
                        <TextBlock Name="txtPhaseIndicator" Grid.Column="0" Text="阶段 1/8" Foreground="#89DCEB" FontSize="11"/>
                        <TextBlock Name="txtProgressPct" Grid.Column="2" Text="0%" Foreground="#89B4FA" FontWeight="Bold" FontSize="12"/>
                    </Grid>

                    <!-- 动态状态流转卡片 -->
                    <Border Grid.Row="3" Background="#181825" CornerRadius="4" Padding="10" BorderBrush="#313244" BorderThickness="1">
                        <ScrollViewer VerticalScrollBarVisibility="Auto">
                            <StackPanel Name="panelStepLog">
                                <TextBlock Text="[系统就绪] 等待任务启动..." Foreground="#6C7086" FontSize="11" FontFamily="Consolas"/>
                            </StackPanel>
                        </ScrollViewer>
                    </Border>
                </Grid>
            </Border>

            <!-- 运行完成报告视图 (默认隐藏) -->
            <Border Name="panelSummary" Background="#252538" CornerRadius="6" Padding="16" BorderBrush="#313244" BorderThickness="1" Visibility="Collapsed">
                <Grid>
                    <Grid.RowDefinitions>
                        <RowDefinition Height="Auto"/>
                        <RowDefinition Height="*"/>
                    </Grid.RowDefinitions>

                    <!-- 完成状态横幅 -->
                    <Border Name="bannerStatus" Grid.Row="0" Background="#1E382B" CornerRadius="4" Padding="12,10" Margin="0,0,0,10" BorderBrush="#A6E3A1" BorderThickness="1">
                        <StackPanel>
                            <StackPanel Orientation="Horizontal" VerticalAlignment="Center">
                                <TextBlock Name="txtBannerIcon" Text="✅" FontSize="16" Margin="0,0,8,0"/>
                                <TextBlock Name="txtBannerTitle" Text="每日数据湖同步完成！" FontSize="14" FontWeight="Bold" Foreground="#A6E3A1"/>
                            </StackPanel>
                            <TextBlock Name="txtBannerDesc" Text="全波次执行成功，总耗时 00:00。" FontSize="11" Foreground="#A6ADC8" Margin="24,2,0,0"/>
                        </StackPanel>
                    </Border>

                    <!-- 对账结果清单 -->
                    <Border Grid.Row="1" Background="#181825" CornerRadius="4" Padding="8" BorderBrush="#313244" BorderThickness="1">
                        <ScrollViewer VerticalScrollBarVisibility="Auto">
                            <StackPanel Name="panelSummaryList">
                                <!-- 动态插入对账行 -->
                            </StackPanel>
                        </ScrollViewer>
                    </Border>
                </Grid>
            </Border>
        </Grid>

        <!-- 底部操作按钮栏 -->
        <Grid Grid.Row="2" Margin="0,12,0,0">
            <Grid.ColumnDefinitions>
                <ColumnDefinition Width="Auto"/>
                <ColumnDefinition Width="*"/>
                <ColumnDefinition Width="Auto"/>
                <ColumnDefinition Width="Auto"/>
            </Grid.ColumnDefinitions>

            <TextBlock Name="txtFooterHint" Grid.Column="0" Text="流水线运行中，请勿强制关闭计算机" VerticalAlignment="Center" FontSize="11" Foreground="#6C7086"/>

            <Button Name="btnViewLog" Grid.Column="2" Content="查看完整日志" Style="{StaticResource ActionButton}" Margin="0,0,8,0" Visibility="Collapsed"/>
            <Button Name="btnClose" Grid.Column="3" Content="确定关闭" Style="{StaticResource PrimaryButton}" IsEnabled="False"/>
        </Grid>
    </Grid>
</Window>
"@

class CneGuiApp {
    [System.Windows.Window]$Window
    [System.Windows.Controls.TextBlock]$txtSubHeader
    [System.Windows.Controls.TextBlock]$txtTimer
    [System.Windows.Controls.TextBlock]$txtCurrentPhase
    [System.Windows.Controls.TextBlock]$txtCurrentStep
    [System.Windows.Controls.TextBlock]$txtPhaseIndicator
    [System.Windows.Controls.TextBlock]$txtProgressPct
    [System.Windows.Controls.TextBlock]$txtBannerIcon
    [System.Windows.Controls.TextBlock]$txtBannerTitle
    [System.Windows.Controls.TextBlock]$txtBannerDesc
    [System.Windows.Controls.TextBlock]$txtFooterHint
    [System.Windows.Controls.ProgressBar]$pbTotal
    [System.Windows.Controls.Border]$panelRunning
    [System.Windows.Controls.Border]$panelSummary
    [System.Windows.Controls.Border]$bannerStatus
    [System.Windows.Controls.StackPanel]$panelStepLog
    [System.Windows.Controls.StackPanel]$panelSummaryList
    [System.Windows.Controls.Button]$btnViewLog
    [System.Windows.Controls.Button]$btnClose

    [System.Diagnostics.Stopwatch]$Stopwatch
    [System.Windows.Threading.DispatcherTimer]$Timer
    [string]$LogFilePath = ""
    [System.Collections.Generic.List[psobject]]$DatasetResults

    CneGuiApp([string]$tradeDate, [string]$logPath) {
        $this.LogFilePath = $logPath
        $this.DatasetResults = [System.Collections.Generic.List[psobject]]::new()

        $reader = [System.Xml.XmlReader]::Create([System.IO.StringReader]::new($Global:CneGuiXaml))
        $this.Window = [System.Windows.Markup.XamlReader]::Load($reader)

        $this.txtSubHeader      = $this.Window.FindName("txtSubHeader")
        $this.txtTimer          = $this.Window.FindName("txtTimer")
        $this.txtCurrentPhase   = $this.Window.FindName("txtCurrentPhase")
        $this.txtCurrentStep    = $this.Window.FindName("txtCurrentStep")
        $this.txtPhaseIndicator = $this.Window.FindName("txtPhaseIndicator")
        $this.txtProgressPct    = $this.Window.FindName("txtProgressPct")
        $this.txtBannerIcon     = $this.Window.FindName("txtBannerIcon")
        $this.txtBannerTitle    = $this.Window.FindName("txtBannerTitle")
        $this.txtBannerDesc     = $this.Window.FindName("txtBannerDesc")
        $this.txtFooterHint     = $this.Window.FindName("txtFooterHint")

        $this.pbTotal           = $this.Window.FindName("pbTotal")
        $this.panelRunning      = $this.Window.FindName("panelRunning")
        $this.panelSummary      = $this.Window.FindName("panelSummary")
        $this.bannerStatus      = $this.Window.FindName("bannerStatus")
        $this.panelStepLog      = $this.Window.FindName("panelStepLog")
        $this.panelSummaryList  = $this.Window.FindName("panelSummaryList")

        $this.btnViewLog        = $this.Window.FindName("btnViewLog")
        $this.btnClose          = $this.Window.FindName("btnClose")

        $this.txtSubHeader.Text = "交易日锚点: $tradeDate | 调度环境: 本地数据湖"

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
            $this.txtCurrentStep.Text = "当前操作: $stepText"
            
            # 滚动日志条目 (最多保留 40 条)
            if ($this.panelStepLog.Children.Count -ge 40) {
                $this.panelStepLog.Children.RemoveAt(0)
            }
            $tb = [System.Windows.Controls.TextBlock]::new()
            $tb.Text = "[$(Get-Date -Format 'HH:mm:ss')] $stepText"
            $tb.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#A6ADC8")
            $tb.FontSize = 11
            $tb.FontFamily = [System.Windows.Media.FontFamily]::new("Consolas")
            $this.panelStepLog.Children.Add($tb)
        })
    }

    [void]RecordStepResult([string]$dataset, [string]$status, [int]$rows, [string]$elapsed, [string]$note) {
        $res = [pscustomobject]@{
            Dataset = $dataset
            Status  = $status
            Rows    = $rows
            Elapsed = $elapsed
            Note    = $note
        }
        $this.DatasetResults.Add($res)
    }

    [void]ShowCompletionSummary([bool]$isSuccess, [string]$message) {
        $this.Stopwatch.Stop()
        $this.Timer.Stop()

        $this.Window.Dispatcher.Invoke([Action]{
            $this.panelRunning.Visibility = [System.Windows.Visibility]::Collapsed
            $this.panelSummary.Visibility = [System.Windows.Visibility]::Visible
            $this.btnClose.IsEnabled = $true
            $this.btnViewLog.Visibility = [System.Windows.Visibility]::Visible
            $this.txtFooterHint.Text = '数据湖同步完成，点击[确定关闭]退出'

            # 支持无人值守或自测自动关闭环境变量
            if ($env:CNE_GUI_AUTO_CLOSE_SEC) {
                $seconds = [int]$env:CNE_GUI_AUTO_CLOSE_SEC
                if ($seconds -gt 0) {
                    $autoTimer = [System.Windows.Threading.DispatcherTimer]::new()
                    $autoTimer.Interval = [TimeSpan]::FromSeconds(1)
                    $autoTimer.Tag = @{
                        Remaining = $seconds
                        Button    = $this.btnClose
                        Window    = $this.Window
                    }
                    $autoTimer.add_Tick({
                        param($sender, $e)
                        $ctx = $sender.Tag
                        $ctx.Remaining--
                        if ($ctx.Remaining -le 0) {
                            $sender.Stop()
                            $ctx.Window.Close()
                        } else {
                            $ctx.Button.Content = "确定关闭 ($($ctx.Remaining)s)"
                        }
                    })
                    $this.btnClose.Content = "确定关闭 ($($seconds)s)"
                    $autoTimer.Start()
                }
            }

            if ($isSuccess) {
                $this.bannerStatus.Background = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#1E382B")
                $this.bannerStatus.BorderBrush = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#A6E3A1")
                $this.txtBannerIcon.Text = "✅"
                $this.txtBannerTitle.Text = "每日数据湖同步完成！"
                $this.txtBannerTitle.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#A6E3A1")
                $this.txtBannerDesc.Text = "全部波次执行成功，总耗时 $($this.txtTimer.Text)。"
            } else {
                $this.bannerStatus.Background = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#3C2028")
                $this.bannerStatus.BorderBrush = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#F38BA8")
                $this.txtBannerIcon.Text = "⚠️"
                $this.txtBannerTitle.Text = "数据同步完成（部分波次有警告或失败）"
                $this.txtBannerTitle.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#F38BA8")
                $this.txtBannerDesc.Text = $message
            }

            # 填充对账表格
            $this.panelSummaryList.Children.Clear()

            # 类别分组表头
            $headerGrid = [System.Windows.Controls.Grid]::new()
            $headerGrid.Margin = [System.Windows.Thickness]::new(6, 2, 6, 6)
            $c1 = [System.Windows.Controls.ColumnDefinition]::new(); $c1.Width = [System.Windows.GridLength]::new(55, [System.Windows.GridUnitType]::Pixel)
            $c2 = [System.Windows.Controls.ColumnDefinition]::new(); $c2.Width = [System.Windows.GridLength]::new(210, [System.Windows.GridUnitType]::Pixel)
            $c3 = [System.Windows.Controls.ColumnDefinition]::new(); $c3.Width = [System.Windows.GridLength]::new(95, [System.Windows.GridUnitType]::Pixel)
            $c4 = [System.Windows.Controls.ColumnDefinition]::new(); $c4.Width = [System.Windows.GridLength]::new(70, [System.Windows.GridUnitType]::Pixel)
            $c5 = [System.Windows.Controls.ColumnDefinition]::new(); $c5.Width = [System.Windows.GridLength]::new(1, [System.Windows.GridUnitType]::Star)
            $headerGrid.ColumnDefinitions.Add($c1); $headerGrid.ColumnDefinitions.Add($c2)
            $headerGrid.ColumnDefinitions.Add($c3); $headerGrid.ColumnDefinitions.Add($c4); $headerGrid.ColumnDefinitions.Add($c5)

            $hStatus = [System.Windows.Controls.TextBlock]::new(); $hStatus.Text = "状态"; $hStatus.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#6C7086"); $hStatus.FontSize = 11; [System.Windows.Controls.Grid]::SetColumn($hStatus, 0)
            $hName = [System.Windows.Controls.TextBlock]::new(); $hName.Text = "数据集 (中文名称)"; $hName.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#6C7086"); $hName.FontSize = 11; [System.Windows.Controls.Grid]::SetColumn($hName, 1)
            $hRows = [System.Windows.Controls.TextBlock]::new(); $hRows.Text = "同步行数"; $hRows.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#6C7086"); $hRows.FontSize = 11; [System.Windows.Controls.Grid]::SetColumn($hRows, 2)
            $hTime = [System.Windows.Controls.TextBlock]::new(); $hTime.Text = "用时"; $hTime.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#6C7086"); $hTime.FontSize = 11; [System.Windows.Controls.Grid]::SetColumn($hTime, 3)
            $hNote = [System.Windows.Controls.TextBlock]::new(); $hNote.Text = "执行说明 / 详情"; $hNote.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#6C7086"); $hNote.FontSize = 11; [System.Windows.Controls.Grid]::SetColumn($hNote, 4)

            $headerGrid.Children.Add($hStatus); $headerGrid.Children.Add($hName)
            $headerGrid.Children.Add($hRows); $headerGrid.Children.Add($hTime); $headerGrid.Children.Add($hNote)
            $this.panelSummaryList.Children.Add($headerGrid)

            $sep = [System.Windows.Controls.Separator]::new()
            $sep.Background = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#313244")
            $sep.Margin = [System.Windows.Thickness]::new(0, 0, 0, 4)
            $this.panelSummaryList.Children.Add($sep)

            foreach ($item in $this.DatasetResults) {
                $rowGrid = [System.Windows.Controls.Grid]::new()
                $rowGrid.Margin = [System.Windows.Thickness]::new(6, 4, 6, 4)
                $r1 = [System.Windows.Controls.ColumnDefinition]::new(); $r1.Width = [System.Windows.GridLength]::new(55, [System.Windows.GridUnitType]::Pixel)
                $r2 = [System.Windows.Controls.ColumnDefinition]::new(); $r2.Width = [System.Windows.GridLength]::new(210, [System.Windows.GridUnitType]::Pixel)
                $r3 = [System.Windows.Controls.ColumnDefinition]::new(); $r3.Width = [System.Windows.GridLength]::new(95, [System.Windows.GridUnitType]::Pixel)
                $r4 = [System.Windows.Controls.ColumnDefinition]::new(); $r4.Width = [System.Windows.GridLength]::new(70, [System.Windows.GridUnitType]::Pixel)
                $r5 = [System.Windows.Controls.ColumnDefinition]::new(); $r5.Width = [System.Windows.GridLength]::new(1, [System.Windows.GridUnitType]::Star)
                $rowGrid.ColumnDefinitions.Add($r1); $rowGrid.ColumnDefinitions.Add($r2)
                $rowGrid.ColumnDefinitions.Add($r3); $rowGrid.ColumnDefinitions.Add($r4); $rowGrid.ColumnDefinitions.Add($r5)

                $statusBadge = [System.Windows.Controls.Border]::new()
                $statusBadge.CornerRadius = [System.Windows.CornerRadius]::new(3)
                $statusBadge.Padding = [System.Windows.Thickness]::new(4, 1, 4, 1)
                $statusBadge.HorizontalAlignment = [System.Windows.HorizontalAlignment]::Left
                $badgeText = [System.Windows.Controls.TextBlock]::new()
                $badgeText.FontSize = 10
                $badgeText.FontWeight = [System.Windows.FontWeights]::SemiBold

                if ($item.Status -eq "success" -or $item.Status -eq "OK") {
                    $statusBadge.Background = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#1E382B")
                    $badgeText.Text = "成功"
                    $badgeText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#A6E3A1")
                } elseif ($item.Status -eq "skip") {
                    $statusBadge.Background = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#313244")
                    $badgeText.Text = "跳过"
                    $badgeText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#A6ADC8")
                } elseif ($item.Status -eq "warning") {
                    $statusBadge.Background = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#3C3224")
                    $badgeText.Text = "提示"
                    $badgeText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#F9E2AF")
                } else {
                    $statusBadge.Background = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#3C2028")
                    $badgeText.Text = "失败"
                    $badgeText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#F38BA8")
                }
                $statusBadge.Child = $badgeText
                [System.Windows.Controls.Grid]::SetColumn($statusBadge, 0)

                # 中文名称增强
                $cn = $Global:CneDatasetNames[$item.Dataset]
                $displayName = if ($cn) { "$($item.Dataset) ($cn)" } else { $item.Dataset }

                $dsText = [System.Windows.Controls.TextBlock]::new()
                $dsText.Text = $displayName
                $dsText.FontWeight = [System.Windows.FontWeights]::Medium
                $dsText.FontSize = 11
                $dsText.TextTrimming = [System.Windows.TextTrimming]::CharacterEllipsis
                [System.Windows.Controls.Grid]::SetColumn($dsText, 1)

                $rowText = [System.Windows.Controls.TextBlock]::new()
                if ($item.Rows -gt 0) {
                    $rowText.Text = "+$($item.Rows) 行"
                    $rowText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#A6E3A1")
                } else {
                    $rowText.Text = "--"
                    $rowText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#6C7086")
                }
                $rowText.FontSize = 11
                [System.Windows.Controls.Grid]::SetColumn($rowText, 2)

                $timeText = [System.Windows.Controls.TextBlock]::new()
                $timeText.Text = if ($item.Elapsed) { $item.Elapsed } else { "--" }
                $timeText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#A6ADC8")
                $timeText.FontSize = 11
                [System.Windows.Controls.Grid]::SetColumn($timeText, 3)

                $noteText = [System.Windows.Controls.TextBlock]::new()
                $noteText.Text = if ($item.Note) { $item.Note } else { "" }
                $noteText.Foreground = [System.Windows.Media.BrushConverter]::new().ConvertFromString("#A6ADC8")
                $noteText.FontSize = 10
                $noteText.TextTrimming = [System.Windows.TextTrimming]::CharacterEllipsis
                [System.Windows.Controls.Grid]::SetColumn($noteText, 4)

                $rowGrid.Children.Add($statusBadge); $rowGrid.Children.Add($dsText)
                $rowGrid.Children.Add($rowText); $rowGrid.Children.Add($timeText); $rowGrid.Children.Add($noteText)
                $this.panelSummaryList.Children.Add($rowGrid)
            }
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
