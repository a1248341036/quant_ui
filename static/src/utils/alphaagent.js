/**
 * AlphaAgent 页面纯函数工具集
 *
 * 从原 static/src/views/AlphaAgent.vue 单体组件中抽出：
 * 只做数据变换/格式化，不持有状态、不发请求，供各子组件与 store 复用。
 */

export function emptyUsage() {
  return { calls: 0, input_tokens: 0, output_tokens: 0, cache_input_tokens: 0, cache_creation_input_tokens: 0 }
}

export function addUsage(total, event) {
  return {
    calls: (total.calls || 0) + 1,
    input_tokens: (total.input_tokens || 0) + Number(event.input_tokens || 0),
    output_tokens: (total.output_tokens || 0) + Number(event.output_tokens || 0),
    cache_input_tokens: (total.cache_input_tokens || 0) + Number(event.cache_input_tokens || 0),
    cache_creation_input_tokens: (total.cache_creation_input_tokens || 0) + Number(event.cache_creation_input_tokens || 0),
  }
}

export function usageFromEvents(events) {
  let total = emptyUsage()
  for (const event of events || []) {
    if (event.event === 'usage_total') total = { ...total, ...event }
    else if (event.event === 'usage') total = addUsage(total, event)
  }
  return total
}

export function parseArgs(raw) {
  if (!raw) return {}
  try { return typeof raw === 'string' ? JSON.parse(raw) : raw } catch (e) { return {} }
}

export function dynamicThinking(text, prefix = '思考') {
  const value = String(text || '').replace(/```[\s\S]*?```/g, '').replace(/\s+/g, ' ').trim()
  if (!value) return prefix + '…'
  const clean = value.replace(/^#+\s*/, '').replace(/^[-*]\s*/, '')
  return prefix + ' · ' + (clean.length > 140 ? clean.slice(0, 140) + '…' : clean)
}

export function fmt4(v) {
  const n = Number(v)
  return isNaN(n) ? String(v) : Math.abs(n) < 1 ? n.toFixed(4) : n.toFixed(2)
}

export function fmtNum(v) {
  if (v === null || v === undefined || Number.isNaN(Number(v))) return '—'
  return Number(v).toFixed(4)
}

export function toolLabel(name) {
  return ({
    evaluate_factor: 'Profile 评估',
    eval_on_train_set: '训练集评估',
    eval_on_val_set: '验证集评估',
    submit_factor: '提交因子',
  }[name] || name || '工具')
}

export function reviewLabel(verdict) {
  return ({ approve: '允许提交', revise: '需要修订', reject: '拒绝提交' }[verdict] || '拒绝提交')
}

export function memoryVerdictLabel(verdict) {
  return ({
    production_approved: '正式入库',
    candidate_approved: '候选保留',
    validated: '验证有效',
    promising: '训练有潜力',
    rejected: '明确否定',
    revise_required: '需修订',
    weak: '证据不足',
  }[verdict] || '待评估')
}

export function metricLabel(key) {
  return ({
    ic: 'IC',
    icir: 'ICIR',
    rank_ic: 'RankIC',
    factor_coverage: '覆盖率',
    coverage: '覆盖率',
    long_group_annual_excess_return: '多头年化超额',
    winsorized_abs_ic_decay: '截尾IC衰减',
    annualized_return: '年化收益',
    annualized_excess_return: '年化超额',
    sharpe: '夏普',
    excess_sharpe: '超额夏普',
    max_drawdown: '最大回撤',
    annual_turnover: '年换手',
    daily_overlap: '日重叠',
    monotonicity: '单调性',
  }[key] || key)
}

// 提取分位组合（纯多头）指标，供展示与导出复用；无数据返回 null
export function quantilePortfolioMetrics(result) {
  const qp = result?.metrics?.quantile_portfolio
  if (!qp || typeof qp !== 'object' || qp.available === false) return null
  const num = v => (v == null || Number.isNaN(Number(v))) ? null : Number(v)
  return {
    top_group_annualized_return: num(qp.top_group_annualized_return),
    top_group_annualized_excess_return: num(qp.top_group_annualized_excess_return),
    top_group_sharpe: num(qp.top_group_sharpe),
    top_group_excess_sharpe: num(qp.top_group_excess_sharpe),
    top_group_max_drawdown: num(qp.top_group_max_drawdown),
    n_groups: qp.n_groups,
    direction: qp.direction,
    long_side: qp.long_side,
  }
}

export function formatMetricValue(value) {
  const n = Number(value)
  return isNaN(n) ? String(value) : Math.abs(n) < 1 ? n.toFixed(4) : n.toFixed(2)
}

export function formatBacktestMetric(value, type) {
  const n = Number(value)
  if (isNaN(n)) return '—'
  if (type === 'pct') {
    // 百分比格式：0.1234 → 12.34%
    return (n * 100).toFixed(2) + '%'
  } else if (type === 'ratio') {
    // 比率格式：1.234 → 1.23
    return n.toFixed(2)
  } else {
    // 默认格式
    return Math.abs(n) < 1 ? n.toFixed(4) : n.toFixed(2)
  }
}

export function labelShort(col) {
  if (!col) return '—'
  const map = {
    label_1d_open_to_open: '1d O2O',
    label_1d_close_to_close: '1d C2C',
    label_10d_close_to_close: '10d C2C',
    label_20d_close_to_close: '20d C2C',
  }
  return map[col] || col.replace('label_', '')
}

export function freqShort(freq) {
  // 展示为具体交易日节奏：daily=每 1 个交易日；weekly=每周五（名义 5 天）；
  // monthly=自然月一次（名义约 20 天，实际随月份 19~23 浮动）——精确口径见 freqCadence
  return { daily: '1天', weekly: '5天', monthly: '≈20天' }[freq] || freq || '—'
}

export function freqCadence(freq) {
  return {
    daily: '门禁调仓节奏：每日调仓（间隔 1 个交易日）',
    weekly: '门禁调仓节奏：每周五信号、次日开盘调仓（名义每 5 个交易日，节假日顺延）',
    monthly: '门禁调仓节奏：每月末信号、次月初调仓（自然月一次，约 20 个交易日，随节假日浮动）',
  }[freq] || '门禁调仓频率'
}

export function freqSourceHint(source) {
  return {
    recorded: '',
    run_spec: '',
    derived_run_spec: '（按所在 run 的门禁配置回填）',
    derived: '（按评估标签推导）',
  }[source] ?? '（按所在 run 的门禁配置回填）'
}

export function statusLabel(status) {
  return ({ starting: '准备中', running: '运行中', stopping: '停止中', completed: '已完成', failed: '失败', interrupted: '已中断' }[status] || status || '未开始')
}

export function formatTime(value) {
  if (!value) return ''
  const d = new Date(value)
  if (isNaN(d.getTime())) return String(value).slice(0, 16).replace('T', ' ')
  const pad = n => String(n).padStart(2, '0')
  return d.getFullYear() + '-' + pad(d.getMonth() + 1) + '-' + pad(d.getDate()) + ' ' + pad(d.getHours()) + ':' + pad(d.getMinutes())
}

export function fmtTime(iso) {
  const t = Date.parse(iso || '')
  if (!Number.isFinite(t)) return '—'
  const d = new Date(t)
  const p = n => String(n).padStart(2, '0')
  return d.getFullYear() + '-' + p(d.getMonth() + 1) + '-' + p(d.getDate()) +
         ' ' + p(d.getHours()) + ':' + p(d.getMinutes())
}

export function runTitle(run) {
  if (run.title) return run.title.slice(0, 64)
  if (run.user_message) return run.user_message.slice(0, 32)
  return '因子研究任务 ' + String(run.run_id || '').slice(0, 8)
}

export function formatTokens(value) {
  const n = Number(value || 0)
  return n >= 1000 ? (n / 1000).toFixed(n >= 10000 ? 0 : 1) + 'k' : String(n)
}

export function icClass(v) {
  if (v == null) return ''
  return v >= 0 ? 'ic-pos' : 'ic-neg'
}

export function toolResultMessage(row, key) {
  const result = row.result || {}
  const summary = result.summary || result.metrics?.cross_sectional_core || {}
  const args = parseArgs(row.arguments_raw || '')
  const profile = result.profile?.profile_id || args.profile_id || ''
  const topn = result.metrics?.topn_portfolio
  const engineBt = result.engine_backtest
  const metrics = [
    profile ? 'Profile ' + profile : '',
    summary.ic != null ? 'IC ' + fmt4(summary.ic) : '',
    summary.rank_ic != null ? 'RankIC ' + fmt4(summary.rank_ic) : '',
    summary.icir != null ? 'ICIR ' + fmt4(summary.icir) : '',
    summary.factor_coverage != null ? 'Coverage ' + fmt4(summary.factor_coverage) : '',
    result.metrics?.long_group_annual_excess_return != null ? '多头年化超额 ' + fmt4(result.metrics.long_group_annual_excess_return) : '',
    result.metrics?.winsorized_abs_ic_decay != null ? '截尾IC衰减 ' + fmt4(result.metrics.winsorized_abs_ic_decay) : '',
    result.stored != null ? 'stored ' + result.stored : '',
    result.candidate_stored ? '候选池已保存' : '',
    result.candidate_state ? '状态 ' + result.candidate_state : '',
    result.rebalance_freq ? '调仓 ' + result.rebalance_freq : '',
    result.rule_results?.length ? '规则 ' + result.rule_results.filter(x => x.passed).length + '/' + result.rule_results.length : '',
  ].filter(Boolean)

  // 构建完整引擎回测展示数据（evaluate 的 topn_portfolio 或 submit 的 engine_backtest）
  let backtest = null
  if (engineBt) {
    const m = engineBt.metrics || {}
    backtest = {
      freq: engineBt.freq || 'daily',
      top_n: engineBt.top_n || 30,
      direction: engineBt.direction || 1,
      annual: m.annual_return,
      excess: m.excess_annual,
      sharpe: m.sharpe,
      drawdown: m.max_drawdown,
      overlap: m.daily_overlap,
      winRate: m.win_rate,
      gateReasons: engineBt.fail_reasons || [],
    }
  } else if (topn && typeof topn === 'object' && topn.available !== false) {
    backtest = {
      freq: topn.rebalance || 'daily',
      top_n: topn.top_n || 30,
      direction: topn.direction || 1,
      annual: topn.annualized_return,
      excess: topn.annualized_excess_return,
      sharpe: topn.sharpe,
      drawdown: topn.max_drawdown,
      overlap: topn.daily_overlap,
      winRate: topn.win_rate,
      byFreq: topn.by_freq || null,
      gateReasons: [],
    }
  }

  return {
    key: key + '-' + (row.tool_call_id || row.name || Math.random()),
    kind: 'tool_result',
    name: row.name || 'tool',
    factorName: args.factor_name || '',
    expression: args.multi_line_expr || '',
    ok: result.ok !== false || result.candidate_stored === true,
    elapsed: row.elapsed_seconds != null ? Number(row.elapsed_seconds).toFixed(1) : '',
    metrics,
    backtest,
    text: result.error || result.skipped_reason || '',
  }
}

// 事件流 → 会话时间线消息（纯函数；跳过心跳/usage 等内部事件）
export function buildTimeline(events) {
  const out = []
  for (const [index, event] of (events || []).entries()) {
    const key = event.ts || String(index)
    if (event.event === 'heartbeat' || event.event === 'stream_start' || event.event === 'usage' || event.event === 'reviewer_usage' || event.event === 'usage_total') continue
    if (event.event === 'user_message') {
      out.push({ key, kind: 'user', text: event.content || '' })
    } else if (event.event === 'agent_thinking') {
      out.push({ key, kind: 'thinking', label: '思考摘要', text: event.content || '' })
    } else if (event.event === 'assistant_tool_call') {
      const args = parseArgs(event.arguments_raw)
      out.push({ key, kind: 'tool_call', name: event.name || 'tool', factorName: args.factor_name || '', expression: args.multi_line_expr || '', text: event.arguments_raw || '' })
    } else if (event.event === 'assistant_message') {
      out.push({ key, kind: 'assistant', text: event.content || '' })
    } else if (event.event === 'assistant') {
      if (event.reasoning) out.push({ key: key + '-r', kind: 'thinking', label: '思考摘要', text: event.reasoning })
      if (event.content) out.push({ key: key + '-a', kind: 'assistant', text: event.content })
      for (const call of event.tool_calls || []) {
        const args = parseArgs(call.function?.arguments || '')
        out.push({ key: key + '-t-' + (call.id || call.function?.name || ''), kind: 'tool_call', name: call.function?.name || call.name || 'tool', factorName: args.factor_name || '', expression: args.multi_line_expr || '', text: call.function?.arguments || '' })
      }
    } else if (event.event === 'tool_results') {
      for (const result of event.results || []) out.push(toolResultMessage(result, key))
    } else if (event.event === 'run_summary') {
      const tools = event.tool_calls || {}
      out.push({ key, kind: 'system', text: '运行总结 · 工具调用 ' + (tools.count || 0) + ' 次 · 成功 ' + (tools.ok || 0) + ' 次' })
    } else if (event.event === 'session_start') {
      out.push({ key, kind: 'system', text: '会话开始 · ' + (event.model || '-') })
    } else if (event.event === 'session_end') {
      out.push({ key, kind: 'system', text: '会话结束 · ' + (event.reason || '-') })
    } else if (event.event === 'research_memory_retrieved') {
      out.push({ key, kind: 'system', text: `动态检索长期记忆 ${event.entry_count || 0} 条` })
    } else if (event.event === 'nudge') {
      out.push({ key, kind: 'system', text: 'Agent 继续推进研究' })
    } else if (event.event === 'continuation_queued') {
      out.push({ key, kind: 'system', text: '已追加指令，等待当前模型调用或工具批次结束后继续' })
      out.push({ key: key + '-user', kind: 'user', text: event.content || '' })
    } else if (event.event === 'continuation_accepted') {
      out.push({ key, kind: 'system', text: 'Agent 已接收追加指令，开始下一轮研究' })
    } else if (event.event === 'continuation_started') {
      out.push({ key, kind: 'system', text: '已从这段研究历史恢复上下文，继续新的 Agent 回合' })
      out.push({ key: key + '-user', kind: 'user', text: event.content || '' })
    } else if (event.event === 'branch_started') {
      out.push({ key, kind: 'system', text: '已从此处新建分支，分支会继承此前研究上下文' })
      out.push({ key: key + '-user', kind: 'user', text: event.content || '' })
    } else if (event.event === 'reviewer_start') {
      out.push({ key, kind: 'system', text: 'FactorReviewer 开始独立审查候选因子' })
    } else if (event.event === 'reviewer_thinking') {
      out.push({ key, kind: 'reviewer_thinking', text: event.content || '' })
    } else if (event.event === 'reviewer_message') {
      out.push({ key, kind: 'assistant', text: 'FactorReviewer：' + (event.content || '') })
    } else if (event.event === 'factor_review') {
      out.push({
        key,
        kind: 'review',
        verdict: event.verdict || 'reject',
        novelty: event.novelty || 'low',
        canonical: event.canonical_form || '未分类',
        reasons: Array.isArray(event.reasons) ? event.reasons : [],
        changes: Array.isArray(event.required_changes) ? event.required_changes : [],
      })
    }
  }
  return out
}

// 头部"当前活动"一行文案（取最后一个有效事件）
export function computeCurrentActivity(status, pendingMessages, events) {
  if (status === 'stopping') return '正在停止 Agent…'
  if (pendingMessages) return '已排队 ' + pendingMessages + ' 条追加指令，等待当前步骤结束…'
  if (!(events || []).length) return '正在加载数据并初始化 Agent…'
  const last = [...events].reverse().find(e => !['heartbeat', 'stream_start', 'usage', 'reviewer_usage', 'usage_total'].includes(e.event)) || {}
  if (last.event === 'session_start') return '研究会话已建立 · 模型 ' + (last.model || '当前模型')
  if (last.event === 'research_memory_retrieved') return '已按最新进展检索 ' + (last.entry_count || 0) + ' 条研究记忆'
  if (last.event === 'agent_thinking') return dynamicThinking(last.content)
  if (last.event === 'assistant_tool_call') {
    const args = parseArgs(last.arguments_raw)
    const name = last.name || '研究工具'
    const factor = args.factor_name ? ' · ' + args.factor_name : ''
    return '调用 ' + name + factor
  }
  if (last.event === 'tool_results') {
    const names = (last.results || []).map(x => x.name).filter(Boolean)
    const unique = [...new Set(names)]
    return '已完成 ' + (unique.length ? unique.join('、') : '工具调用') + ' · 正在比较结果'
  }
  if (last.event === 'assistant_message') return dynamicThinking(last.content, 'Agent 输出')
  if (last.event === 'reviewer_start') return 'FactorReviewer 正在独立审查候选的新颖性与稳健性'
  if (last.event === 'reviewer_thinking') return dynamicThinking(last.content, 'FactorReviewer 审查')
  if (last.event === 'reviewer_message') return dynamicThinking(last.content, 'FactorReviewer 输出')
  if (last.event === 'nudge') return 'Agent 正在根据当前结果继续推进研究'
  return last.event ? '事件：' + last.event : '正在研究…'
}

// 打字行动画文案
export function computeLiveActivity(events, currentActivityValue) {
  if (!(events || []).length) return '正在加载数据并初始化 Agent…'
  const last = [...events].reverse().find(e => !['heartbeat', 'stream_start', 'usage', 'reviewer_usage', 'usage_total'].includes(e.event)) || {}
  const turn = last.turn != null ? ' [Turn ' + (last.turn + 1) + ']' : ''
  if (last.event === 'llm_request') return '正在调用模型 · ' + (last.model || '') + ' · 上下文 ' + (last.message_count || '?') + ' 条消息' + turn
  if (last.event === 'assistant') {
    const tcs = last.tool_calls || []
    if (tcs.length) {
      const names = tcs.map(c => { try { const args = JSON.parse(c.function?.arguments || '{}'); return c.function?.name + (args.factor_name ? '(' + args.factor_name + ')' : '') } catch(e) { return c.function?.name || '' } }).filter(Boolean)
      return '发起工具调用: ' + names.join(', ') + turn
    }
    if (last.reasoning) return dynamicThinking(last.reasoning, '思考') + turn
    if (last.content) return dynamicThinking(last.content, '输出') + turn
    return '等待模型响应…' + turn
  }
  if (last.event === 'tool_results') {
    const results = last.results || []
    const ok = results.filter(r => r.ok).length
    return '工具执行完毕 · ' + ok + '/' + results.length + ' 通过' + turn
  }
  if (last.event === 'session_start') return '会话已建立 · 模型 ' + (last.model || '')
  if (last.event === 'research_memory_retrieved') return '已检索 ' + (last.entry_count || 0) + ' 条研究记忆'
  if (last.event === 'nudge') return '继续推进研究（未达最低工具调用轮数）'
  if (currentActivityValue) return currentActivityValue
  return '正在研究…'
}

// 汇总历史列表要展示的指标（train/val IC、夏普等）
export function labSummary(results) {
  const train = results?.train_screen?.metrics?.cross_sectional_core || {}
  const val = results?.validation?.metrics?.cross_sectional_core || {}
  const qp = quantilePortfolioMetrics(results?.validation) || quantilePortfolioMetrics(results?.train_screen) || {}
  return {
    train_ic: train.ic,
    train_icir: train.icir,
    val_ic: val.ic,
    val_icir: val.icir,
    sharpe: qp.top_group_sharpe,
    annualized_return: qp.top_group_annualized_return,
    annualized_excess_return: qp.top_group_annualized_excess_return,
    passed: Object.values(results || {}).some(r => r.ok && r.passed),
  }
}
