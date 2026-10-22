# scripts/cne — CNE 数据湖相关作业脚本区

依赖 CNE 数据湖日更产物（`docker/run_cne_daily.sh` 流水线）的作业统一放这里。
纯数据同步类任务（`sync_daily_to_cne.py` 等）由 `data_status/catalog.json` 登记管理，
仍在 `scripts/` 顶层；本目录放"吃 CNE 数据做下游计算"的作业。

## 当前作业

| 脚本 | 调度 | 说明 |
|---|---|---|
| `run_cne_daily.ps1` | Windows 计划任务 `QuantUIDataSync` 每日 21:30（容器内对应 `docker/run_cne_daily.sh`） | CNE 日更流水线 Windows 版 |
| `run_cne_stale_retry.ps1` | Windows 计划任务 `QuantUIDataStaleRetry` 每日 23:00 | 晚间 stale 补抓（排在 21:30 流水线之后 1.5h，补滞后/失败批次；全量对齐时 no-op） |
| `run_paper_daily.ps1` | Windows 计划任务 `QuantUIPaperRun` 每日 21:50 | 模拟盘每日执行（21:30 数据同步之后；后端离线时走离线回放，同一交易日幂等跳过） |
| `stacking_job.py` | 容器 supercronic 16:40（panel 重建之后） | ML 组合训练自门禁包装：跨入新月才真正训练，否则秒退 |

## stacking_job 门禁逻辑

`train_ml_composite.py`（walk-forward ML 组合）的干净 OOS 随日历月增长，
日频重跑无意义。包装层检查 `artifacts/alphaagent/stacking/` 最新
`report.json` 的 `panel_end`：落在当前月 → SKIP；跨入新月 / `--force` →
执行训练并摘录逐折 OOS IC 与 engine_gate 结论到 cron 日志。

手动操作：

```powershell
# 看门禁判定（不执行）
.venv\Scripts\python.exe scripts\cne\stacking_job.py --check
# 强制重跑
.venv\Scripts\python.exe scripts\cne\stacking_job.py --force
# 透传训练参数
.venv\Scripts\python.exe scripts\cne\stacking_job.py -- --model ridge
```
