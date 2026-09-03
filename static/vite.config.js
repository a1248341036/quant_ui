import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

export default defineConfig({
  plugins: [vue()],
  // 开发时 Vite dev server 在 5173，API 代理到后端 17891
  server: {
    port: 5173,
    proxy: {
      '/api': 'http://127.0.0.1:17891',
      '/strategy_helpers.md': 'http://127.0.0.1:17891',
    },
  },
  // 构建产物输出到 dist/，供 FastAPI StaticFiles 挂载
  build: {
    outDir: 'dist',
    // 不清空输出目录：vite 默认构建前先清空，清空窗口会让正在运行的后端
    // StaticFiles 对新 index.html 引用的 assets 返回 404（页面白屏）。
    // 关掉后新哈希文件覆盖写入、旧哈希残留（旧页面继续可用），服务不中断。
    emptyOutDir: false,
  },
  // public 目录中的文件会被原样复制到 dist/
  publicDir: 'public',
})
