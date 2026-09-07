/**
 * 通用 CSV 导出：所有页面的导出按钮统一走这里（逃逸/BOM/下载一次实现）。
 *
 * downloadCsv(filename, columns, rows)
 * - columns: string[] 表头
 * - rows:    any[][] 数据行（与 columns 对齐；null/undefined 输出空串）
 */

export function csvEscape(v) {
  const s = v === null || v === undefined ? '' : String(v)
  return /[",\n\r]/.test(s) ? '"' + s.replace(/"/g, '""') + '"' : s
}

export function downloadCsv(filename, columns, rows) {
  const lines = [
    columns.map(csvEscape).join(','),
    ...rows.map(r => r.map(csvEscape).join(',')),
  ]
  const blob = new Blob(['\ufeff' + lines.join('\r\n')], { type: 'text/csv;charset=utf-8' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  document.body.appendChild(a)
  a.click()
  document.body.removeChild(a)
  URL.revokeObjectURL(a.href)
}
