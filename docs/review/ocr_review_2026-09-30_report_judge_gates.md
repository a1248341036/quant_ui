# OCR review 修复记录：研报判定网关与阶段提示词（2026-09-30）

## 背景

用 `ocr review`（open-code-review，skill `open-code-review`）审了两支未合并分支：

| 目标 | 范围 | 结果 |
|---|---|---|
| `feat/panel-v9-landing`（面板侧） | `main..feat/panel-v9-landing`（2 提交 / 6 文件） | **0 findings**（4 文件入选，两个新增测试文件未入选） |
| `feat/report-prompt-plugins`（报告侧） | `76d609c..feat/report-prompt-plugins`（35 提交 / 21 文件） | **44 findings / 18 文件** |

关键事实：报告侧那 44 条里，**多数在 main 线上已存在**（同一行号），即审的是已上线的报告模式代码。
本分支修的是其中"判定链路静默失败"这一类（与 2026-09-30 判定链修复同一族病）。

原始产物（不入库）：`logs/ocr_report_prompt_plugins.md`、`logs/ocr_panel_v9_landing.md`。

## 本分支修了什么

1. **`_dispatch.py`**
   - `_auto_val_verify`：插入的 try/except 原先排在 docstring **之前** → `"""..."""` 沦为死语句、
     `__doc__` 为空。docstring 归位到 `def` 之后。
   - `dispatch()`：网关提取原先在 **JSON 字符串解析之前**（`arguments` 为字符串时永远取不到），
     且**不重置**（上一次 dispatch 的旧网关会压掉本轮新网关）。现在解析后提取，无携带时显式置 `None`。
   - `_report_reproduce_judge()`：网关解析改**显式 `None` 哨兵**（`or` 链无法区分"未设置/空"，
     旧实例属性可压掉本轮全局网关）。
   - 判定通过后不再 `gate["phase"] = "diverge"`（原地改会污染全局网关对象 → 后续新题的复现判定
     被静默跳过），改为经 `set_run_gate({**gate, "phase": "diverge"})` 写回新字典并同步实例属性。
   - `int(gate.get("lock_rounds") or 3)` 加 `try/except`（非法值不再丢弃整次判定）。

2. **`report_channels.py`**：`get_run_gate()` 返回**副本**（根因修复：原返回全局字典本体，
   任何调用方原地改都会永久污染进程内网关）。

3. **`agentscope_run.py`**
   - 第二网关块：`reproduce_factor_of` 原先只在 `_rag_phase == "diverge"` 分支里 import，
     复现轮走到该块即 `NameError` 且被 `except Exception: pass` 吞掉 → 网关静默停在旧状态（fail-open）。
     现显式导入 + 失败留 `logging.warning`。
   - RAG 检索处 `research_mode` 默认值 `"technical"` → `"report"`（与全文件其余位置一致；
     原值在 `config.research_mode` 缺失时查错索引，RAG 恒为空）。
   - 新增：研报模式按状态机给到的 `_rag_phase` **重建系统提示词**（见下条）。

4. **`prompts.py`**：`build_system_prompt` / `build_system_prompt_with_report` 新增
   `report_phase` 参数。原实现写死 `resolve_report_phase(spec, turn=0)` → 恒为 `reproduce`，
   发散轮也按复现阶段装配知识通道（生产配置 `knowledge_mode_by_phase = {reproduce: mechanism_cards,
   diverge: report_rag}`）→ 发散轮错误地只给机制卡。同时把 `except Exception` 收窄为 `ImportError`
   （原写法会吞掉真正的编程错误，如签名变更的 `TypeError`）。

5. **`question_queue.py`**
   - `load_mechanism_cards()` 只缓存**非空**结果：原先把 `[]` 也固化 → 首次调用时文件不存在，
     本进程后续永远读不到机制卡。
   - 机制卡硬门槛补 `best_score > 0`：否则任何标题含"多因子模型"这类通用词的卡，只要该词出现在
     课题文本里就会被配上，绕过分数阈值 —— 与本块"宁可不给卡"的意图相悖。

6. **`prediction.py`**：方向中性主型（`monotonic` / `monotone` / `linear`）+ `expected_sign=-1`
   原先恒归一为 `monotonic_increasing` → 形态对账（`actual.shape == expected_shape`）拿 decreasing
   的实际形态去比 increasing，**正确预测被误判未 confirmed**。现在按 sign 落方向；显式方向
   （如 `monotonic_decreasing` + sign=1）不被过度纠正。

7. **脚本**
   - `batch_mineru_parse.py`：超时改**杀进程树**（原只杀父进程，MinerU 的 parse-server 子进程
     继续占显存）；**非零退出码不采信产物**（原逻辑只看文件 >1000 字节 → 错误页/半成品被永久 skip）；
     md 后处理改**原子写**；MinerU 路径支持 `--mineru` / `MINERU_EXE`；扫描 PDF 改按输出根目录排除
     （原 `"parsed" not in p.parts` 与实际目录名 `parsed_mineru` 永不相等）；元数据每行带 `run` 批次号。
   - `mineru_progress.ps1`：加 `#Requires -Version 7.0`（脚本用了 PS7 的 `??`，PS 5.1 下是解析错误）；
     `Get-Content` 加 `-Tail 500`；PDF 过滤同上修正。
   - `resume_mineru_parse.ps1`：启动前把上一轮 `batch_mineru_recent.*` 轮转为带时间戳文件
     （`Start-Process` 重定向是覆盖写，原写法会清空上一轮日志）；新增 `-Mineru` 参数并导出 `MINERU_EXE`。

## 明确**不修**（含误报）

- `question_state.py:99-112`（OCR 标 high）：称"无条件 `continue` 会让后面所有新题都选不到" ——
  **误报**。`continue` 只跳过当前迭代，循环会继续处理后面的新题。
- `agentscope_tools.py:534`（OCR 标 high）：称"`args["_report_gate"]` 注入是死代码" —— **误报**，
  `_dispatch.py::dispatch()` 确实读取该键（同一份报告的另一条 finding 也这么写）。
- `agentscope_run.py` 里 `locals().get("_report_phase_from_state")`（OCR 建议改直接引用）：**保留原写法**。
  该名字只在 `enable_question_queue` 分支内被赋值，直接引用在关闭课题队列时会 `NameError`。
- `question_queue.py:29` `parents[4]` 根路径假设、`ov_store` 硬编码 Windows 路径与每次 `git rev-parse`
  子进程、`_PARSED_INDEX_CACHE` 半扫描写入、`clean = lambda`（E731）—— 留待后续，低风险且改动面更大。

## 验证

- `tests/test_report_judge_gates.py`（新增 6 条）：网关副本隔离 / `set_run_gate` 仍生效 /
  方向中性主型按 sign 归一 / 显式方向不被覆盖 / 非单调形态不受影响 / `_auto_val_verify` 有 docstring。
- 相关回归：`test_prediction_reconciliation` `test_question_queue` `test_report_knowledge_modes`
  `test_question_field_gate` `test_system_prompt_modules` `test_prompt_consistency_audit`
  `test_mining_skills` `test_profile_evaluation_tool` 等 —— 74 passed / 0 failed。
- 更宽子集（含 `test_auto_mode_and_freq` `test_alphaagent_run_recovery` `test_archive_all_runs`
  `test_factor_expr_tools`）出现 6 个失败，已核对**在主干上同样失败**（既有问题，非本次引入）：
  `test_facet_focus_scope`×2、`test_mining_ablation_hardening`×1、`test_yield_improvements`×3。

## 待办（未在本分支处理）

- `_FUNDAMENTAL_SENTINEL_COLUMNS` 未含 `funda_ebitda`（缓存命中校验只看 12 个哨兵列），
  已放到面板分支 `feat/panel-v9-landing` 处理。
- `feat/report-mode` 链（判定链路/机制卡硬门槛/发散父本诊断）与 main 上的注入改动动的是同一个
  `question_queue.py`，将来合并需手工解冲突，不能整分支 merge。
