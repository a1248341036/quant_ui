# OCR Review：vol targeting + sell-down-to-target 功能链

- 日期：2026-10-09
- 对象：`main..feat/sell-down-to-target`（vol targeting 缩放层 + sell-down 双向调仓，7 commits 中代码 2 + 文档 2 + 本修复 1）
- 工具：`ocr review --audience agent`，原始输出 `_tmp_ocr_raw.md`（已清理）
- 结果：**2 findings（1 medium + 1 low），逐条交叉验证属实，0 误报，已全部修复**（commit ec8853f）

## Finding 1（medium）· paper 透传 `or 默认值` 吞合法 0

- 位置：`core/paper/rebalance.py`（当时 L577-579）
- 证据：`vol_target_lo=float(risk.get("vol_target_lo", 0.3) or 0.3)` ——用户显式传
  `vol_target_lo=0`（取消缩放下限）或 `vol_target_window=0` 会被静默替换为默认值；
  同函数 `weight_band`（L581-582）已是 `is not None` 语义，同函数两制不一致。
- 验证：属实。`or` 模式对 0/0.0/"" 等 falsy 合法值全部误伤。
- 修复：三行统一为 `float(risk["vol_target_lo"]) if risk.get("vol_target_lo") is not None else 0.3` 模式。

## Finding 2（low）· σ_p 无效时 clip 会抬升弱市降仓

- 位置：`core/selection.py` build_targets 权重出口（当时 L252-255）
- 证据：σ_p 无效 → `vol_scale_for` 恒返回 1.0（安全退化语义），但原代码仍执行
  `np.clip(scale * 1.0, lo, hi)`。若弱市 `regime_scale=0.2 < lo=0.3`，clip 会把降仓
  强制抬到 0.3——与 vol_scale_for 注释"σ_p 无效→恒等/不放大仓位"直接矛盾。
- 验证：属实。边界场景（弱市 regime + σ_p 不可用叠加），但违背声明的退化语义。
- 修复：`if vol_scale != 1.0:` 才执行整体 clip；补单测
  `test_build_targets_sigma_invalid_keeps_regime_scale_no_clip_lift`
  （σ_p=NaN + regime_scale=0.2 → targets=0.5×0.2=0.1，修复前会得 0.15）。

## 未覆盖项的人工补审

OCR 未发现 high/critical；两处超出 OCR 视角的人工确认：
1. `core/execution.py` 减仓段与买入段增量预算的符号边界（负增量 → 买 0）已有专门单测
   （`test_negative_budget_buys_zero_when_above_target_within_band`），OCR 未重复报。
2. `simulate.py` 4 行 getattr 接线（Lead 批准超清单改动）：getattr 兜底保证旧 config
   对象安全默认关，与 buffer_ratio 同款模式，无新增风险。

## 修复验证

- `tests/test_vol_targeting.py`（20 例）+ `tests/test_sell_down.py`（19 例）全绿
- 修复提交：ec8853f（feat/sell-down-to-target）

## 附：模拟盘工程验证（同日，仅执行行为、不构成策略证据）

- 两验证户（VT25-周频 top10 / VT15-日频 top5，capital 5000，start 2025-02-05），
  参数经 risk_config → rebalance.py → run_backtest 全链生效。
- 仓位利用率动态响应（vol targeting 生效证据）：周频户 mean 0.597 / p50 0.649 /
  区间 0.15~0.999；日频户 mean 0.468 / p50 0.453 / 区间 0.14~0.96（15% 档目标 ~50% 附近）。
  月末仓位随波动状态大幅调节（周频 0.15~0.97），非恒等、非满仓。
- sell-down 修剪单 0 笔（样本期无跳空推仓出带——与对照报告"修剪是低频通道、
  主通道是买入端增量预算"一致）；拒单（日频 1358 笔）以小资金预算过小/整手摩擦为主，
  属预期执行粒度限制。
- 两户期间收益（+2.7% / +0.9%）仅作链路健康记录，**不作为任何策略优劣证据**
  （收益证据的唯一样本外 = 盲测段单次终审，纪律不变）。
