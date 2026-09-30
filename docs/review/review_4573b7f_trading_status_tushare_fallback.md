# Review 报告：`4573b7f`（trading_status Tushare stock_st 兜底）

**审查范围**：`4573b7f` 单 commit，2 文件 +67/-2
**分支**：`fix/audit-v5-findings` 之上（父提交 `e60a754`，即本会话的审计 P0-P3 修复）
**作者**：zhoubw，2026-09-23 16:53

---

## 一、改动概述

EastMoney push2 clist 对海外/云 IP 出口被 WAF 断连（`Server disconnected`），导致 `trading_status` 日频 step 直接失败。本 commit 在 `fetch_trading_status` 的 eastmoney 失败路径后加 Tushare `stock_st` fallback：

- `client.py`：新增 `_fetch_trading_status_tushare`，按日全市场 ST 名单，非 ST 标的标 `normal`+`is_trading=True`，行带 `source=tushare`
- `reference.py`：`step_trading_status` 保留实际 source（`eastmoney` 或 `tushare`），不再无条件标 eastmoney

---

## 二、修复正确性评估

### ✅ 正确

| 项 | 评估 |
|---|---|
| fallback 触发条件 `if config is not None` | 正确——`step_trading_status` 生产路径必传 config；`config=None`（测试/直接调用）跳过 fallback，`test_trading_status_raises_without_allow_mock` 仍通过 |
| fallback 失败时拼进 `reason` 走 `_fail_or_mock` | 正确——tushare 未配置（token 空）或 API 异常时优雅降级到原抛错路径 |
| `source=tushare` provenance 保留逻辑 | 正确——`observed == {"tushare"}` 严格集合相等判断，eastmoney 主路径（无 source 列）保持 `source="eastmoney"` |
| `with_provenance(df.drop("source"), source=...)` | 正确——先删 fallback 注入的 source 列再用统一 source 重注入，与 `schemas.py:1629` 的"source 列已存在则不覆盖"语义一致 |
| 完整性校验通过 | 正确——`_fetch_trading_status_tushare` 遍历全部 `symbols` 生成行，`observed_symbols == expected_symbols`，`reference.py:301` 的 missing/unexpected 校验不触发 |
| `unique(subset=["symbol","trade_date"], keep="last")` | 正确——与 eastmoney 主路径（`trading_status.py:151`）同防御 |
| 51 个相关单测全过 | 实测 `test_eastmoney_trading_status_adapter` + `test_m3_steps` + `test_tdx_mock_gate` + `test_trading_status_st_daily` 共 51 用例通过（commit 说 27，实际更多） |

### ⚠️ 已知局限（docstring 已声明，非缺陷）

**fallback 不覆盖停牌**：`stock_st` 只返回 ST 名单，停牌且非 ST 的票被标 `normal`+`is_trading=True`，下游 `universe.py:231`/`reader.py:440` 不会排除它们。

- 影响：fallback 路径下停牌票可能进入回测面板
- 兜底：`daily_bars` 停牌日无成交，引擎层有成交量/涨跌停过滤
- `source=tushare` 标记仅用于 provenance 审计，**下游 `universe.py`/`reader.py` 不检查 source 字段**，无法据此区分"真正常"和"fallback 未知停牌"
- 评估：可接受——fallback 是 eastmoney 不可用时的降级，停牌漏过是数据源固有局限，且 docstring 明确声明

---

## 三、问题清单

### 🟡 P2：fallback 路径零测试覆盖

**文件**：`CNEquity/src/cnequity/adapters/tdx_protocol/client.py:919-953`（`_fetch_trading_status_tushare` + fallback 块）

**问题**：commit 声称"27 个相关单测通过"，但既有测试全部 mock `fetch_trading_status_eastmoney` 或 `reference.fetch_trading_status`，**没有任何用例覆盖新增的 tushare fallback 路径**：

- 无测试验证 eastmoney 失败 → tushare fallback 成功 → 返回 `source=tushare` 行
- 无测试验证 tushare 也失败 → 走 `_fail_or_mock` 抛 `TdxSourceError`
- 无测试验证 `step_trading_status` 的 `observed == {"tushare"}` 分支（source 保留逻辑）
- 无测试验证 tushare token 未配置时 fallback 优雅降级

**影响**：fallback 逻辑的回归无门禁保护。未来改 `fetch_trading_status` 主路径或 `_get_pro` 时，fallback 可能静默失效。

**修复建议**：补 3-4 个用例：
1. mock eastmoney 抛错 + mock `_get_pro`/`_fetch_with_retry` 返回 ST 名单 → 验证 fallback 返回 `source=tushare`、ST 票 `status="st"`、非 ST `status="normal"`
2. mock eastmoney 抛错 + mock tushare 返回空 → 验证走 `_fail_or_mock` 抛 `TdxSourceError`
3. mock eastmoney 抛错 + mock `_get_pro` 抛 RuntimeError（token 未配置）→ 验证走 `_fail_or_mock`
4. `step_trading_status` 端到端：mock `fetch_trading_status` 返回带 `source=tushare` 列的 df → 验证落盘 df 的 `source=="tushare"`

### 🟢 P3：`ts_code` 列名隐式依赖

**文件**：`client.py:938` — `raw.get_column("ts_code")`

**问题**：`_fetch_trading_status_tushare` 直接调 `pro.stock_st()` 拿原始 DataFrame，硬依赖列名 `ts_code`。`tushare_stock_st.py:57` 有同样的 `ts_code` 假设（带 rename 防御），但本处无防御——若 tushare 改列名会 `ColumnNotFoundError`。

**影响**：低——tushare API 列名稳定，且 `tushare_stock_st.py` 已建立 `ts_code` 先例
**评估**：可接受，无需改

---

## 四、提交信息准确性

| 项 | 评估 |
|---|---|
| 改动描述 | 准确——client.py 加 fallback、reference.py 保留 source |
| "27 个相关单测通过" | **不准确**——实际 51 个通过（4 个测试文件），但 fallback 路径本身零覆盖（见 P2） |
| "8444 行，204 ST" | 无法独立验证（需 tushare token + 实时网络），但数字结构合理 |

---

## 五、总结

| 维度 | 评估 |
|---|---|
| 修复正确性 | 核心逻辑正确——fallback 触发条件、provenance 保留、完整性校验、优雅降级均无缺陷 |
| 已知局限 | fallback 不覆盖停牌（docstring 声明，daily_bars 兜底，可接受） |
| 测试覆盖 | **fallback 路径零覆盖**（P2）——既有测试全 mock 掉了真实 fallback 逻辑 |
| 提交信息 | 基本准确，"27 个单测"实际 51 个但 fallback 无覆盖 |
| 分支纪律 | 合规——落在 `fix/audit-v5-findings` 之上，未直接在 main |

**verdict**：`merge_after_fixes`

修复价值明确（eastmoney WAF 断连是真实生产问题，tushare 独立源 fallback 合理），核心实现无缺陷。唯一可修项是 P2 fallback 路径测试覆盖——建议补 3-4 个用例后合并。
