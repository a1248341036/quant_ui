# 研报模式整夜运行统计（自动生成 2026-09-30 05:18）

## 各 run 产出

| run | 时间 | eval | promising | submit | 复现轮 | 发散轮 |
|---|---|---|---|---|---|---|
| de6c4089e46a | 09-30 05:12 | 32 | 0 | 1 | 1 | 0 |
| dc2411c3f84f | 09-30 05:11 | 420 | 2 | 4 | 10 | 0 |
| 89ad34655187 | 09-30 04:03 | 477 | 14 | 3 | 10 | 0 |
| 794bfef83938 | 09-30 02:53 | 297 | 8 | 2 | 1 | 3 |
| 88c31a68d941 | 09-30 00:40 | 43 | 0 | 1 | 1 | 0 |
| 7aaad97f19c7 | 09-30 00:33 | 21 | 0 | 1 | 0 | 1 |
| f08b3292ea92 | 09-30 00:26 | 32 | 0 | 1 | 0 | 0 |
| 2d9ddd100828 | 09-30 00:21 | 311 | 14 | 3 | 1 | 0 |
| a4bb32a981c0 | 09-29 23:41 | 92 | 0 | 1 | 4 | 0 |
| f9326c000709 | 09-29 23:27 | 143 | 0 | 1 | 0 | 0 |
| b8f004391d5c | 09-29 23:14 | 32 | 0 | 1 | 1 | 0 |
| 13790110a0db | 09-29 23:07 | 0 | 0 | 1 | 0 | 0 |

## 课题状态机

- 记录数: 43
- reproduce_pending: 24
- abandoned: 11
- reproduce_ok: 8

### 复现成功清单
- RQ_712 → rq712_vwap_soft_gate_sz  [ic=0.0342 icir=0.3597 cov=0.932]
- RQ_002 → rq002_mom60_grouprank_ind  [shape=confirmedic=0.0094 icir=-0.0853 cov=0.844]

## 机制卡
- 共 810 张

## 分支
- feat/report-mode tip c8051f3 docs(report): 研报模式整夜运行报告（闭环达成 + 未达成项 + 分支提交清单）
- 未 push: 24 个

## 监控日志尾部
- [09-30 04:54:22] 心跳: run=dc2411c3f84f status=running events=2237 last=user_message
- [09-30 05:04:31] 心跳: run=dc2411c3f84f status=running events=2720 last=metrics_snapshot
- [09-30 05:12:38] run 结束(dc2411c3f84f): status=completed | outcome=interrupted | funnel={"unique_train_evaluated": 337, "unique_val_evaluated": 4, "candidate_stored": 0, "production_stored": 0, "train_to_val_rate": 0.0119, "val_to_production_rate": 0.0}
- [09-30 05:12:38]   本次运行时长 68.6 分钟
- [09-30 05:12:39] 已启动 run: de6c4089e46a  (focus=自由探索)
- [09-30 05:14:41] 心跳: run=de6c4089e46a status=running events=8 last=metrics_snapshot
