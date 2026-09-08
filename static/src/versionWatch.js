// 前端旧标签页守卫
//
// 问题：dist 频繁重建后，长期开着的浏览器标签仍在运行旧 bundle（可能带着当时
// 还没修的 bug），路由懒加载的旧 chunk 也可能已不存在——用户感知为"前端打不开/白屏"。
// 方案：启动时记下当前 bundle 名，之后定期 + 页面回到前台时拉取 /index.html
// （no-store，后端已对 html 关缓存）比对 bundle 名；发现新版本或动态 import
// 失败时，在右下角提示刷新（不自动 reload，避免打断正在填的表单）。

const BUNDLE_NAME_RE = /assets\/(index-[^"']+\.js)/
const CHUNK_FAIL_RE = /failed to fetch dynamically imported module|importing a module script failed|error loading dynamically imported module/i

let myBundle = null
let checking = false
let toastEl = null

function currentBundleName() {
  const el = document.querySelector('script[type="module"][src*="/assets/index-"]')
  const m = el && (el.getAttribute('src') || '').match(/index-[^/]+\.js/)
  return m ? m[0] : null
}

export function showReloadToast() {
  if (toastEl) return
  toastEl = document.createElement('div')
  toastEl.style.cssText =
    'position:fixed;right:20px;bottom:20px;z-index:99999;display:flex;align-items:center;gap:12px;' +
    'padding:12px 16px;background:#1f2937;color:#fff;border-radius:8px;' +
    'box-shadow:0 6px 24px rgba(0,0,0,.25);font-size:14px;'
  const text = document.createElement('span')
  text.textContent = '检测到前端已更新，当前页面运行的是旧版本'
  const btn = document.createElement('button')
  btn.textContent = '立即刷新'
  btn.style.cssText =
    'border:0;padding:6px 14px;border-radius:6px;background:#4f8cff;color:#fff;cursor:pointer;font-size:14px;'
  btn.onclick = () => location.reload()
  const close = document.createElement('span')
  close.textContent = '×'
  close.style.cssText = 'cursor:pointer;opacity:.7;font-size:18px;line-height:1;'
  close.onclick = () => {
    toastEl.remove()
    toastEl = null
  }
  toastEl.append(text, btn, close)
  document.body.appendChild(toastEl)
}

async function checkForNewVersion() {
  if (checking || !myBundle) return
  checking = true
  try {
    const res = await fetch('/index.html?_vw=' + Date.now(), { cache: 'no-store' })
    if (!res.ok) return
    const m = (await res.text()).match(BUNDLE_NAME_RE)
    if (m && m[1] !== myBundle) showReloadToast()
  } catch (e) {
    // 后端重启中或网络异常：本轮忽略，等下一次触发
  } finally {
    checking = false
  }
}

export function startVersionWatch() {
  myBundle = currentBundleName()
  if (!myBundle) return // dev 模式（/src/main.js）没有 hashed bundle，不启用
  window.addEventListener('unhandledrejection', (e) => {
    const msg = String((e && e.reason && (e.reason.message || e.reason)) || '')
    if (CHUNK_FAIL_RE.test(msg)) showReloadToast()
  })
  document.addEventListener('visibilitychange', () => {
    if (!document.hidden) checkForNewVersion()
  })
  window.addEventListener('focus', checkForNewVersion)
  setInterval(checkForNewVersion, 120000)
  // 调试入口：浏览器控制台可手动触发比对 / 提示框
  window.__feVersionCheck = checkForNewVersion
  window.__feShowReloadToast = showReloadToast
}
