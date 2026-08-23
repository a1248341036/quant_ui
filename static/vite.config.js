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
    emptyOutDir: true,
  },
  // public 目录中的文件会被原样复制到 dist/
  publicDir: 'public',
})
