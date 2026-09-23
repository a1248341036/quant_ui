# 新增 DSL 算子完整链路

> 本文档回答"加一个算子要在哪些地方动、为什么"。以 2026-09-23 新增
> `SIGNAL_BLEND`（信号积分）为实例，后续加算子照此清单逐项核对。
> 核心结论：**注册是自动的**（按命名约定收集），真正要决策的是
> 目录分组、指纹分类、守卫豁免、测试四件事。

## 链路总览

```
operators.py 实现（唯一必改）
   │  registry.collect_module_operators 自动收集大写 callable（零注册）
   ▼
求值命名空间（eval.py → registry.build_operator_namespace）
   ├─ guard.wrap_lookahead_guard  自动包负整数守卫（默认拦截，豁免才放行）
   ├─ monitor.wrap_operator_namespace 自动包耗时监控（零登记）
   ▼
模型可见性（prompt 注入）
   ├─ catalog.py _CATALOG_GROUPS   目录分组（决定模型能否看到签名）
   ├─ operator_catalog.py          面族前缀（仅专属某数据面的算子）
   └─ behavior_rules.py            行为准则（可选：教模型怎么用）
   ▼
指纹/分类（熔断与记忆）
   ├─ ast.py SMOOTHING_OPS / NORMALIZE_OPS / _SIGNAL_OP_FAMILY
   ├─ _prefilter.py _SMOOTHING_OPS / _NORMALIZE_OPS / _BASIC_OPS
   └─ retrieval.py _SHELL_OPS / _COLD_OPS
   ▼
数据面（facets.py，仅专属面算子需要）
   ▼
测试（tests/test_dsl_*.py）
```

## 触点清单

| # | 文件 | 位置 | 作用 | SIGNAL_BLEND 是否改 |
|---|---|---|---|---|
| 1 | `alphaagent/dsl/core/operators.py` | 任意位置 | **算子实现**（模块级大写函数）。`registry.collect_module_operators` 按 `^[A-Z][A-Z0-9_]*$` + callable 自动收集，**无需注册表** | ✅ 新增 `SIGNAL_BLEND(df, lam=0.5)` |
| 2 | `alphaagent/dsl/catalog.py` | `_CATALOG_GROUPS`（50-65 行） | 算子目录分组。未命中任何组的算子折叠进"基础件"一行（模型看不到签名）。**新算子按机制归组** | ✅ 加入"时序基础"组 |
| 3 | `alphaagent/dsl/core/guard.py` | `_NEG_INT_EXEMPT`（37-53 行） | 负整数参数守卫。**新算子默认被守卫**（拦截负整数实参防未来函数）；参数允许负整数才加豁免 | ❌ 不改（λ 是 float，负整数 λ 被拦合理） |
| 4 | `alphaagent/dsl/core/ast.py` | `SMOOTHING_OPS`（99-102）/ `NORMALIZE_OPS`（104-108）/ `_SIGNAL_OP_FAMILY`（110-140） | 平滑/归一化指纹集合 + 信号族分类。**进 SMOOTHING_OPS = 被同质化熔断当平滑变体** | ❌ 不进 SMOOTHING_OPS（信号积分是推荐的替代方案，进了会被熔断误拦） |
| 5 | `alphaagent/factor/mining/tools/_prefilter.py` | `_SMOOTHING_OPS`（97-99）/ `_NORMALIZE_OPS`（101-105）/ `_BASIC_OPS`（107-110） | dispatch 层同质化熔断的信号根指纹。与 ast.py 同源 | ❌ 不进（同 #4） |
| 6 | `alphaagent/factor/mining/memory/retrieval.py` | `_SHELL_OPS`（30-35）/ `_COLD_OPS`（39-49） | 套壳算子（不参与信号算子分布判断）/ 冷门算子（软引导推荐） | ✅ 加入 `_SHELL_OPS`（信号积分是套壳变换） |
| 7 | `alphaagent/factor/mining/prompt/modules/operator_catalog.py` | `_FACET_FAMILY_PREFIXES`（15-20） | 聚焦面时注入完整签名的算子族前缀。**仅专属某数据面的算子**（如 CHIP_/CROWD_）需要 | ❌ 不改（通用变换，触面由输入列决定） |
| 8 | `alphaagent/factor/mining/prompt/modules/behavior_rules.py` | rule 2 等 | 行为准则教学。**可选**：教模型新算子的使用场景与参数语义 | ✅ rule 2 信号积分 DSL 写法改为推荐 `SIGNAL_BLEND(signal, λ)` |
| 9 | `alphaagent/factor/facets.py` | `FACET_DEFS`（19-34） | 数据面识别（按列/算子前缀匹配）。**仅专属面算子**需要登记 | ❌ 不改（通用） |
| 10 | `tests/test_dsl_*.py` | 新建或追加 | 语义/边界/目录可见性测试 | ✅ 新建 `tests/test_dsl_signal_blend.py`（5 用例） |

## 自动机制（无需登记）

- **注册**：`registry.collect_module_operators` 扫描模块 `dir()`，大写 callable 自动进命名空间。
- **监控**：`monitor.wrap_operator_namespace` 自动包计时代理（`dsl-monitor` API 可见）。
- **守卫**：`guard.wrap_lookahead_guard` 自动包负整数拦截（`_NEG_INT_EXEMPT` 以外全部默认守卫）。
- **目录**：`catalog.operator_catalog_markdown` 从命名空间自动渲染，签名由 `inspect.signature` 瘦身（`_slim_signature`）。

## 决策点（新算子必答四问）

1. **归哪个目录组？** `catalog.py _CATALOG_GROUPS` 有序规则首个命中生效。不归组 = 模型看不到签名（折叠进基础件一行）。
2. **算平滑吗？** 进 `ast.py SMOOTHING_OPS` + `_prefilter.py _SMOOTHING_OPS` = 同质化熔断把"同信号根 + 该算子"当平滑变体拦截。**降换手类算子要区分**：被推荐的替代方案（如信号积分）不进，被禁止的堆平滑（如末位 EMA 长窗）进。
3. **参数允许负整数吗？** 默认守卫拦截。窗口/滞后/位移类参数保持默认；算术/截面/阈值类参数加 `_NEG_INT_EXEMPT`。
4. **专属某数据面吗？** 是 → `operator_catalog.py _FACET_FAMILY_PREFIXES` + `facets.py FACET_DEFS` 登记；否 → 跳过（触面由输入列决定）。

## 测试要求

- **语义**：与手写等价式逐位一致（`np.testing.assert_allclose(..., equal_nan=True)`）。
- **边界**：参数极值（λ=0/1）、越界拒绝（λ=1.5 → `MultiLineFactorEvalError`）。
- **目录可见性**：`operator_catalog_markdown(tier="full")` 含新算子签名。
- **回归**：`tests/test_dsl_operator_consistency.py`（快/慢路径一致，仅加速算子）、`tests/test_dsl_operator_perf.py`（性能预算，仅慢算子）、`tests/test_dsl_slow_patterns.py`（静态拦截，仅逐品种循环算子）。

## 加速（可选，仅慢算子）

逐品种滚动/状态机类算子（PRICE_GAP_*/CHIP_*/CROWD_* 等）才需要：
`accel.py` 加 Numba 边界内核 → operators.py 快路径接线 → 回落路径保留 →
三重门禁（slow_patterns 静态拦截 + perf 预算 + consistency 逐位一致）。
简单逐元素/滞后组合算子（如 SIGNAL_BLEND）不需要。

## 本次实例：SIGNAL_BLEND

- 实现：`operators.py` `SIGNAL_BLEND(df, lam=0.5)` = `λ·F + (1-λ)·DELAY(F,1)`，λ∈[0,1] 校验，λ=1 恒等 / λ=0 纯滞后特例（避免 `0.0*NaN` 污染首日）。
- 动机：behavior_rules rule 2 已教信号积分手写式（`ADD(0.6*signal, 0.4*DELAY(signal,1))`），但实测 run 191 个表达式中 λ 积分 **0 次使用**——心算 λ + 组合写法门槛高，封装成算子降低使用成本。
- 决策：目录归"时序基础"组；**不进** SMOOTHING_OPS（避免同质化熔断误拦推荐方案）；进 `_SHELL_OPS`（套壳变换）；guard 默认守卫（λ 负整数被拦合理）。
- 验证：5 个测试全过 + DSL 回归 79 passed。