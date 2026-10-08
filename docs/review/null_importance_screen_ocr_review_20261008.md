# Null Importance 因子预筛选 OCR Review（feat/factor-ml-screening @ 39a9278）

- 日期：2026-10-08
- 范围：`git show 39a9278`（相对父提交 1dd0fa6 的唯一新提交）
- 文件：`alphaagent/factor/stacking/screening.py`（新增 228 行）、`scripts/train_ml_composite.py`（+92）、`tests/test_null_importance_screen.py`（新增 122 行）
- 工具：open-code-review（ocr）→ 6 条评论（2 medium / 4 low）；**全部经人工逐条验证**，无 high，无误报。

## 背景

组合链路此前"先全上、训练完再挑"：置换贡献是事后诊断（全因子污染下评估），gain
importance 被分布属性虚高污染。本提交在训练前加 Stage 0 Null Importance 预筛选：
打乱标签 N 次训练 LightGBM-RF 对照，逐因子 `score=log(actual/(1+null_75分位))`，
本质为逐因子置换检验；窗口 = [panel 起点, mining_end)，验证/盲测段不参与（时间隔离契约）。

## 结论摘要

- **2 条 medium 均确认属实**，都落在同一根因：**筛选结果未做"剩余特征数/审计原因"
  的兜底校验**，在小样本（`n_valid<200`）或全量剔除时会把"未能执行筛选"误当
  "正常筛选后全踢"，或在空特征上继续训练崩溃/退化。
- 4 条 low 全部属实（均为入口校验/审计契约类，风险集中在 CLI 参数组合与复用路径）。
- 新增测试覆盖了设计意图的 5 个断言面，质量良好；但**未覆盖**本 review 的 medium
  场景（筛选后 <2 特征的崩溃路径、insufficient_window reasons 透传），修复后应补测。

## Medium

### M1. `apply_screening` 后缺剩余特征数防护 → 空特征继续训练崩溃/退化

- **证据**：`scripts/train_ml_composite.py:570-571`

  ```python
  apply_screening(dataset, screening_rows)
  print(f"筛选后特征 {len(dataset.feature_names)} 个进入训练")
  ```

  之后直接进入 ⑤ walk-forward 训练，无 `<2` 校验。对比同文件既有守卫：
  - 枚举阶段 `train_ml_composite.py:238-240`（`len(entries) < 2 → sys.exit(1)`）；
  - mRMR 块 `train_ml_composite.py:476-478`（`len(dataset.feature_names) < 2 → sys.exit(1)`）。
- **触发路径（真实）**：`screening.py:128-134` 在 `n_valid < 200` 或 `K == 0` 时**必然全判 fail**
  （宁缺毋滥，`passed=False`）。此时 `--screen`（非 `--screen-only`）继续训练：
  - 0 列 → Ridge/LightGBM 裸 ValueError traceback（无提示）；
  - 1 列 → 单因子退化组合静默产出。
- **最小修法**：`apply_screening(dataset, screening_rows)` 之后补：

  ```python
  if len(dataset.feature_names) < 2:
      print(f"Null Importance 筛选后有效特征仅 {len(dataset.feature_names)} 个（<2），"
            f"继续训练会崩溃或产出退化组合。建议调低 --min-score 或核对 screening.json 后重试。")
      sys.exit(1)
  ```

- **补测**：`tests/test_null_importance_screen.py` 增加"全量 fail 后 apply_screening +
  剩余 <2"的 CLI/单元路径断言。

### M2. `apply_screening` 丢弃审计原因：把"筛选未执行"伪装成"全踢"

- **证据**：`alphaagent/factor/stacking/screening.py:219-228`（dropped 拼接）只用
  score/actual/null_p75 拼 reason；而 `screening.py:128-134` 样本不足提前返回时每行带
  `reason="insufficient_window_samples=N"`（actual/null_p75/score 全 None）：

  ```python
  # screening.py:130-133
  {"name": n, "actual": None, "null_p75": None, "score": None,
   "passed": False, "reason": f"insufficient_window_samples={n_valid}"}
  ```

  → 落盘 `dataset.dropped` 变成 `null_importance_score=None (actual=None, null_p75=None)`，
  后续记忆/审计无法区分"全体因子因样本不足被迫判 fail"与"正常筛选后连噪声本底都过不了"
  两种截然不同的结论，容易对因子库误判。
- **最小修法**：`screening.py:223-224` 优先透传 rows 自带 reason：

  ```python
  "reason": rows_by_name[n].get("reason") or (
      f"null_importance_score={rows_by_name[n]['score']}"
      f" (actual={rows_by_name[n]['actual']}, null_p75={rows_by_name[n]['null_p75']})"
  ),
  ```

- **补测**：`test_apply_screening_drops_and_audits` 增补"rows 带 insufficient_window_samples
  reason 时 dropped 原样透传"断言。

## Low（属实，择要）

### L1. `--screen`/`--screen-only` 与 `--recommend-k>0` 组合 → 筛选被静默跳过
- **证据**：`train_ml_composite.py:473-510`：`recommend_k > 0` 时 mRMR 块无条件
  `sys.exit(0)`（line 510），其后的 ④c 筛选块（line 516+）根本不执行，无任何提示。
- **最小修法**：在 `recommend_k > 0` 分支前打印 warn："推荐模式不训练，--screen/--screen-only 不会执行"，
  或将两者声明为互斥参数。

### L2. 复用 `--screening-report` 时 `screening_summary` 恒为 None，report.json 审计缺失
- **证据**：`train_ml_composite.py:225-236`（复用名单只过滤 entries），
  `screening_summary` 初始 None（line 215），`report.json` 的 `"screening"` 字段（line 1005）为 null；
  且被过滤因子在枚举阶段即被剔除，不进 `dataset.dropped` —— 审计上无法区分
  "从未入选"与"被 Null Importance 剔除"。
- **最小修法**：读取报告时同步填充 `screening_summary`（n_total/n_passed/rows/来源路径）。

### L3. `--null-runs` 入口无校验，传 0/1/3 裸 ValueError traceback
- **证据**：`train_ml_composite.py:524` 未校验；`screening.py:120-121` 抛
  `ValueError("n_runs 至少 5...")`，但此时已 emit ml_stage 事件，报错无友好提示。
- **最小修法**：CLI 解析处钳制 `n_runs`（如 `<5 → sys.exit(1)` 给出指引），或捕获后友好报错。

### L4. `screen_dataset` 对空 panel 无前置防护
- **证据**：`screening.py:186-187`：panel 空时 `dts.min()` 抛
  `ValueError: zero-size array to reduction operation minimum`，报错与筛选语义无关；
  `K==0`（冗余过滤后无特征）时 `null_importance_scores` 返回空 rows，`apply_screening` 把
  特征矩阵清零继续下行（与 M1 同链）。
- **最小修法**：取 `dts.min()` 前显式校验 panel 非空并给明确错误（或返回空白名单）。

## 未覆盖文件补查结论

OCR 只选了 2 个核心文件；`tests/test_null_importance_screen.py` 已人工补读：

- 5 例覆盖：判别力（真信号过/噪声与 style 被压下 + 排序）、确定性（种子固定逐字段一致）、
  阈值单调性、窗口不足宁缺毋滥（reason 标注）、裁剪审计（library 取自裁剪前映射不错位）。
  断言精确（含字段集合 pin），质量良好。
- **缺口**：如上 M1/M2 补测；另 `test_structure_and_determinism` 的字段集合断言
  （`set(r1) == {...}`）与 `reason` 键的兼容性——正常窗口 rows 无 reason 键、不足窗口有，
  当前两用例各自成立不受影响，但将来加字段会触发该断言失败（属 pin 行为，可接受，注明即可）。

## 建议

1. M1/M2 属"筛选决策可被静默绕过/失真"的判定侧问题，建议在分支合入前修复（均为 3-5 行改动的
  最小修法），并补两条回归测试。
2. L1-L4 为入口校验/审计契约类，可随分支合入后择机处理；L4 与 M1 同链，建议与 M1 一并修。
3. 修复走分支提交（用户定合并时机）；本结论已写入 `docs/review/`。