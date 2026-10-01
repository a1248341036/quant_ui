# OCR 审查 + 被拒因子清单（2026-10-01）

> 审查目标：`refactor/verdict-train-passed` vs `main`（2 提交：verdict 改名 + Reviewer 关闭）
> 工具：`ocr review --audience agent --background-file …`（26 文件、6 条评论、~970K tokens、1m58s）
> 原始输出：`.tmp_ocr_verdict.txt`

---

## 一、OCR 结论：6 条（3 medium 真实缺陷 / 3 low 可维护性）

**逐条我已回源核实（均属实）**，按严重度排列。

### Medium ①  新 verdict 缺 CSS → 前端配色丢失
`static/src/components/alphaagent/ResearchSummary.vue:274`（新增 `'train_passed'`）驱动
`'memv-' + verdict` 类名（同文件 35/39 行），但样式表只定义了旧名：
```
static/src/styles/alphaagent.css:67   .memory-candidate_approved, .memory-promising {…}
                                   :99   .memv-promising { color/bg … }
                                   :137  .summary-verdict-dot.memv-promising {…}
                                   :149  .verdict-filter-btn.active.memv-promising {…}
```
→ 新 verdict 在筛选栏、ResearchMemoryBank、AgentThread、MemoryDetailModal 全部丢失琥珀色配色（`MemoryDetailModal` 走 `'memory-' + verdict`，缺 `.memory-train_passed`）。
**修法**：4 处选择器各并列加上 `train_passed`（与 `promising` 同规则）。

### Medium ②  重放脚本去重优先级未归一 → 与"改名不改历史口径"相悖
`scripts/resubmit_promising_factors.py:101`：
```python
if prev is None or (prev["verdict"] != "train_passed" and r["verdict"] == "train_passed"):
```
SQL 已兼容两代字面量，但**去重优先级仍只比新名**。历史库里 `promising` 行不再压过 `rejected` 行，
与该脚本 60-62 行注释及本分支定调冲突。
**修法**：改用 `constants.normalize_verdict()` 归一后再比较。

### Medium ③  我新写的正则兜底可能捞回**错误**判定
`alphaagent/factor/mining/agent/factor_reviewer.py:320-322`：
```python
verdict_m = FactorReviewer._VERDICT_RE.search(text)   # 取“首个”匹配
```
`re.search` 只取首个匹配，无法区分"顶层 verdict 字段"与"正文/reasons 里二次提到的历史判定"。
若前文出现 `"verdict":"approve"` 的引用而顶层实为 reject（或反之），会**伪造**判定——
伪造 approve 会让因子前移，伪造 reject 会硬拦，比直接降级 revise 更隐蔽，也与"不阻断"意图相悖。
**修法（保守）**：收集全部匹配，**只有一个不同取值时才采信**；出现多个不同取值一律降级 `revise`。

### Low（可维护性，当前字面量正确）
- `memory/retrieval.py:661-663` 与 `~1076`：verdict 列表**硬编码重复**了 `POSITIVE_VERDICTS_READ`，
  与"正集收敛到常量层单一真源"的初衷背离 → 建议用 `sorted(POSITIVE_VERDICTS_READ)` 生成。
- `memory/schema.py:20`、`memory/ingestion.py:16`：`POSITIVE_VERDICTS_READ` **导入但全文未使用**（死 import，
  改名前后都是死的）→ 建议删除（已确认无 re-export）。

---

## 二、被 Reviewer 拒稿的因子（今夜 8 个 run 全量）

`factor_review` 事件共 **27 次**：**revise 18 / reject 9**。

| # | run | turn | 因子 | novelty | val IC | val ICIR | 拒因性质 |
|---|---|---|---|---|---|---|---|
| 1 | 27db5d627268 | 0 | `rq035_npacct_ind2` | low | −0.0051 | −0.0679 | **统计不达标** |
| 2 | 27db5d627268 | 0 | `rq035_yoy2nd_ue` | low | −0.0105 | −0.0905 | **统计不达标**（且方向翻转） |
| 3 | 27db5d627268 | 0 | `rq035_roe_yoy_accel` | medium | −0.0076 | −0.0564 | **统计不达标** |
| 4 | 27db5d627268 | 0 | `rq035_gm_accel` | low | +0.0006 | +0.0058 | **统计不达标** |
| 5 | 251534daa26c | 2 | `rq045_illiq_pvc_diversion` | low | +0.0059 | +0.0570 | **统计不达标** + 无契约加法组合 |
| 6 | fcc9a4618fe2 | 3 | `rq048_holder_conc_rev5_group_val` | low | — | — | **解析兜底**（模型原意亦 reject） |
| 7 | fcc9a4618fe2 | 4 | `rq048_vwap_ovlead_diverge_rank_sizeval` | low | +0.0011 | +0.0400 | **统计不达标** |
| 8 | fcc9a4618fe2 | 4 | `rq048_vwap_ovlead_diverge_rank_val` | low | +0.0033 | +0.0670 | **统计不达标** |
| 9 | fcc9a4618fe2 | 5 | `rq048_vwap_bias_industry_neu` | low | +0.0216* | +0.2949* | **IC/ICIR 过关，但引擎经济性失败** |

\* 取自该因子的 auto-val（`val_verification.passed=true`，门槛 `min_val_abs_ic=0.015`）。

### 两个"非统计"案例的实情

**#6 `rq048_holder_conc_rev5_group_val`**（`source=reviewer_parse_guard`）
```python
rev5 = NEG(TS_PCTCHANGE($adj_close, 5))
conc = CS_BUCKET(NEG($holder_count_chg_pct), 5)
g = CS_GROUP_RANK(rev5, conc)
ind_n = CS_NEUTRALIZE(g, $industry_sw_l1)
CS_ZSCORE(CS_NEUTRALIZE(ind_n, CS_BUCKET(LOG($float_cap), 10)))
```
触发解析兜底的原因是模型在字符串值里写了未转义引号（`split="val"`）→ `json.loads` 在 char 425 报
`Expecting ',' delimiter`。但从 raw 里可看到**模型本意也是 `"verdict":"reject"`**
（canonical「短期反转(5日)按股东数变化分组 + 行业/市值中性化」）→ **结局不变**；
本次修复的价值是「恢复理由透明 + 消除未来 fail-closed 风险」，不是救回这一个。

**#9 `rq048_vwap_bias_industry_neu`**
```python
vb = DIVIDE(SUBTRACT($adj_close, $adj_vwap), $adj_vwap)
vb_s = WMA(vb, 10); vb_c = CS_WINSORIZE(vb_s, 0.01, 0.99)
ind = CS_NEUTRALIZE(vb_c, $industry_sw_l1); res = CS_RESIDUALIZE(ind, LOG($float_cap))
NEG(CS_ZSCORE(res))
```
Reviewer 给了**两条**理由：①新颖性（"教科书短期反转/VWAP 乖离的保序尺度变换重包装"）；
②**样本外与可交易性**：val 单调性劣化（spearman 0.469 vs train 0.957）、val top-k 超额转负
（k=5 `net_excess_ann=−0.156`）、**engine 预演失败（annual_return −0.28 / sharpe −0.54）**。
→ 并非"纯新颖性误杀"。

### 结论（修正此前表述）
今夜 9 个 reject 中，**没有一个是"统计与引擎经济性都过关、被新颖性单独杀掉"的**：
7 个 val IC/ICIR 远低于门槛（0.0006~0.0105 / 0.006~0.09），1 个为解析兜底且模型原意亦 reject，
1 个 IC/ICIR 过关但引擎经济性明确失败。新颖性措辞总是与实质性失败同时出现。
→ **产出瓶颈仍在统计/引擎门槛，而非 Reviewer 的新颖性判定**；Reviewer 关闭的收益主要是
省掉每次审查的 LLM 开销与消除 `factor_review_reject_blocked` 硬拦风险，而非"解放被误杀的因子"。

> 如需把历史误杀捞回，`scripts/resubmit_promising_factors.py` 支持按 `verdict` 重放
> （含 `rejected` 捞回路径）——但需先修上面的 Medium ②（去重优先级归一）。
