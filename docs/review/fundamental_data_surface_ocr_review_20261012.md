Review complete: 2 finding(s) across 11 selected item(s).

─── static/src/store/alphaagent.js:169-172 ───
[bug · medium] inferredMode 现在由 rebalance_freq 决定，但前端只有 toggleFocusFacet 会触发
syncInferredMode——用户改动「调仓频率」下拉（AgentThread.vue 直接 v-model 绑定）时 form.research_mode
不会随之切换到新的子档。后果：门槛弹窗标题用 effectiveModeLabel（已显示为新的「中周期档 label_5d」），但
openThresholdModal/saveThresholds/resetResearchSpec 落位的键是 form.research_mode（仍是旧的 technical）——用户以为在改
weekly 档门槛，实际写进了旧档位 spec 文件，且新档位门槛完全未生效。建议在 rebalance_freq 变化时同样走 syncInferredMode（如在 store 里 watch
该字段或组件 @change 触发），保持「徽标显示档位 = 实际编辑档位 = 启动档位」三者一致。



─── static/src/store/alphaagent.js:204-207 ───
[maintainability · low] 与本次 loadResearchModes 的改动矛盾：researchModes 现在全量登记了 internal
子档（fundamental_daily/fundamental_weekly/technical_* 都在列表里），上方 `researchModes.find(m => m.value ===
mode)` 的 hit 分支会先命中这些子档并返回后端 label，SUB_LABEL 兜底实际上不可达；注释「不在 research-modes
列表里」也已经过时。建议删除该兜底分支（或仅保留为加载失败时的防御）并同步修正注释，避免后人误读。



LLM retry report summary: 1 of 100 requests affected -- 1 request recovered after retry

Review planning (1 request):
- core/research_modes.py,scripts/harness_mine.py,scripts/harness_submit.py,scripts/promote_candidates.py,scripts/resubmit_promising.py,scripts/resubmit_promising_factors.py: provider error (HTTP 500) -> provider error (HTTP 500) -> succeeded

Per-attempt detail: --format json (retry_report).
