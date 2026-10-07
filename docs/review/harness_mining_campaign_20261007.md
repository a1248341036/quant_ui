# Harness 自驱动挖掘战役 · 2026-10-07

> 目标（用户指令）：**不用 agentscope**，由宿主 agent（DeepSeek Harness）直接充当挖掘 LLM，
> 复用仓库现有基建（panel / DSL / 评估服务 / 交付门 / 候选池），自主完成一轮真实挖掘。

## 0. 一句话结论

4 轮共 **66 次评估**（全部走与 agentscope 同源的 `StockEvalService` + 真源门槛），
产出 **2 个通过全部 stage_one 门槛并真实入库候选池**的因子（`candidate_stored=true`）：
候选池 **42 → 44**。样本外（test 段 2025-01~2026-09）IC 仍显著（0.0589 / |-0.0410|）；
生产晋升（stage_two）被"与库内相关性 > 0.5"拦下；engine_gate 净值回测未过。

## 1. 工具（本次新建，均可复跑）

| 脚本 | 作用 |
|---|---|
| `scripts/harness_mine.py` | 批量评估自研表达式：建会话 → 逐因子 train 评估 → **用真源 `DeliveryCriteria` 自动逐门判 PASS/FAIL**（IC/ICIR/coverage/cs_autocorr/换手），可选 `--val` 追加样本外评估；结果落 JSONL |
| `scripts/harness_submit.py` | 把因子走**真实交付链路**（`FactorSubmitService.submit`：stage_one → 盲测 → 正交 → engine_gate），成功后写入候选池注册表 |

两者与 agentscope 路径**同源**：同一 `StockEvalService`、同一 `effective_research_spec`、
同一 `DeliveryCriteria`（门槛数值零硬编码），构造参数（`max_cs_corr=0.8` / `similar_top_k=3` /
`overwrite=False` / `research_mode`）与 `agentscope_run` 一致。

## 2. 逐轮结果

| 轮 | 设计思路 | 因子数 | 过门 | 最佳点 |
|---|---|---|---|---|
| 1 | 高 ρ_f 慢信号根（筹码/拥挤/VPIN/互信息/股东/融资） | 22 | 0 | 换手极低（`chip_comgap120` turn 0.153、ρ_f 0.986）但 **1 日 IC≈0**（0.002–0.013） |
| 2 | 已知有 IC 的机制 + 强降换手（vwap 偏离/CLV/非流动性） | 18 | 0 | `vwapdev EMA10`：IC **0.0235** ✓ ICIR **0.292** ✓ 但换手 **0.840** ✗；`amt_to_cap 20d`：IC 0.0227 ✓ 换手 **0.157** ✓ ρ_f 0.995 ✓ 但 ICIR **0.184** ✗ |
| 3 | ICIR 攻坚（窗宽扫描 + 首次合成，**符号未对齐**） | 13 | 0 | `vwapdev EMA20`：IC 0.0216 ✓ ICIR 0.274 ✗ 换手 0.676 ✗；合成因符号对冲 IC 塌到 0.002 |
| 4 | **月度档 + 符号对齐合成** | 13 | **2** | 见下表 |

## 3. 两个入库因子

表达式（多行 DSL，末行为输出）：

```text
# hx_atc_illiq_prem_monthly —— 慢注意力 × 非流动性溢价（符号对齐合成）
atc   = CS_ZSCORE(NEG(TS_MEAN(DIVIDE($amount, $float_cap), 20)))
illiq = CS_ZSCORE(RANK(CS_WINSORIZE(TS_MEAN(DIVIDE(ABS($ret), $amount), 20), 0.02, 0.98)))
CS_ZSCORE(ADD(atc, illiq))

# hx_atc_cap_neu_monthly —— 20 日均成交额/流通市值（市值中性残余）
atc = TS_MEAN(DIVIDE($amount, $float_cap), 20)
CS_NEUTRALIZE(CS_ZSCORE(atc), CS_BUCKET(LOG($float_cap), 10))
```

交付判定（`technical_monthly`，label_20d + monthly，门槛 IC≥0.053 / ICIR≥0.65 / 换手≤0.80）：

| 因子 | train IC / ICIR | val IC（保留比） | 全窗口 IC/ICIR（n_days） | test 段 IC/ICIR（保留比，方向一致） | stage_one | stage_two | 候选池 |
|---|---|---|---|---|---|---|---|
| `hx_atc_illiq_prem_monthly` | +0.0859 / 0.7971 | +0.0688（0.80） | +0.0811 / 0.6574（60） | **+0.0589 / 0.3646（0.69）** ✓ | **PASS** | FAIL（库内相关性） | **已入库** |
| `hx_atc_cap_neu_monthly` | −0.0800 / −0.7627 | −0.0639（0.80） | −0.0717 / −0.6591（61） | **−0.0410 / −0.2400（0.55）** ✓ | **PASS** | FAIL（库内相关性） | **已入库** |

- 盲测（blind_test）两项均 **PASS**；`candidate_storage=registry_only`。
- **stage_two 失败原因**：与候选池已有因子最大截面相关 **0.7515 / 0.8629**（> `criteria.max_abs_corr=0.5`；
  近邻 `to_res_range_cap_wnsz_wma8`、以及彼此之间）→ 注意力/流动性机制在库内已饱和。
- **engine_gate 失败项**：`excess_annual, excess_sharpe, max_drawdown, hold_overlap, high_turnover`
  （月频真实调仓净值口径；该门阈值 `min_excess_annual=0.03 / sharpe=0.5 / max_dd=0.3 / max_avg_daily_turnover=0.65`）。
- 复核入口：`artifacts/alphaagent/factorzoo/candidate_main/mining_candidate_registry.json`
  （`hx_atc_*` 两条，`status=candidate`、`review_status=pending_review`）与
  `.../candidate_main/expressions/hx_atc_*.dsl`。

## 4. 关键结论

1. **换手 ↔ IC 是硬权衡，且门槛组合决定了可挖区间**：日频档（label_1d + freq=daily，换手门 0.50）
   要求 ρ_f ≳ 0.93；而实测在该持久度上 ICIR 上限约 **0.27**（`vwapdev EMA20`：ICIR 0.274 但换手 0.676），
   高持久慢根（ρ_f 0.99+）ICIR 仅 **0.18**（`amt_to_cap`）。**当前 0 产的直接原因是"日频档门槛组合
   落在可实现前沿之外"**，而月度档（门 0.053/0.65/0.80）**可达**（本次 2/13 过门）。
2. **历史入库样本印证同一判断**：42 个既有候选中，多为 `technical` + **freq=weekly（旧解耦组合，换手门 0.65）**，
   且入库 ICIR 普遍 **0.21–0.27（低于现行 0.28 线）** ⇒ 2026-10-04 改为 `label_1d ⇒ freq=daily`
   之后，**现行主档比历史任何入库样本都更严**。
3. **慢信号根换手天然低但常无日频 alpha**：筹码熵/VPIN/互信息/股东/融资类 ρ_f 0.93–0.999、换手 0.01–0.4，
   但 1 日 IC 0.002–0.013 ⇒ "为降换手而选慢根"会同时牺牲 IC，需要**在已知有 IC 的机制上加降换手手段**。
4. **合成必须符号对齐**：`ADD(A, B)` 前须按各自 IC 方向对齐，否则互相抵消（第 3 轮 IC 0.006 vs 第 4 轮对齐后 0.086）。
5. **正交性是第二道真正的墙**：过了统计门/盲测后，与库内相关性 >0.5 仍会被 stage_two 拦下
   ⇒ 继续挖掘应优先**未被库内覆盖的机制/字段族**。

## 5. 下一步（待定）

- 继续挖掘：换用库内稀缺的信息源（研报预期修正 `ac_*`/`rc_*`、公告事件 `ann_*`、股东结构 `th_*`、
  分红 `div_*`、指数/风格 `idx_*`）并以结构化交互与慢结构组合，目标**同时过 stage_two**。
- 决策项：日频档（0.50 换手门 × 0.02/0.28）是否需要与月度/周频档重新对齐选择率（本次实测给出前沿证据）。
- 这两个候选是否提交 review / 晋升，由用户决定（现为 `pending_review`）。
