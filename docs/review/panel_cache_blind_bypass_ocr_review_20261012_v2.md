Review complete: 1 finding(s) across 3 selected item(s).

─── alphaagent/data/adapters/cnequity.py:386-388 ───
[test · low] 本次最核心的修复（盲测隔离：缓存命中后按请求区间裁剪，防止 wide 缓存把盲测段回流给 tuning）在仓库现有测试中没有任何覆盖（全仓搜索 cnequity/缓存相关
test 无匹配，仅 model.py 的 scheme_compare 修复有回归测试 test_all_nan_oos_rows_do_not_crash）。建议补充一个针对
<code>_find_cached_panel</code>/<code>load_panel_from_cne</code> 的回归用例：构造覆盖 [2020-01-01, 2026-09-30]
的宽缓存后请求 end=2024-12-31，断言返回面板的 datetime 层 max ≤ 2024-12-31，防止该盲测旁路再次回归。


