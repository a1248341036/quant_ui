import { createApp } from 'vue'
import * as echarts from 'echarts'

import App from './App.vue'
import './style.css'

// 把 echarts 挂到 window 上，使 utils/charts.js 可以直接使用
window.echarts = echarts

// 注册全局组件
import StrategySelect from './components/StrategySelect.vue'
import CodeEditor from './components/CodeEditor.vue'

const app = createApp(App)
app.component('strategy-select', StrategySelect)
app.component('code-editor', CodeEditor)
app.mount('#app')
