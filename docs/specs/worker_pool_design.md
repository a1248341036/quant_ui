# 算力工作池方案（Compute Worker Pool）

> 状态：设计稿（Draft）  
> 作者：AI 协作  
> 日期：2026-09-14  

---

## 一、 现状诊断

### 1.1 当前进程架构

```
FastAPI 主进程 (17891)
├── Web API 路由 / SSE 推流 / 研究记忆读写
├── 因子实验室评估 → evaluate_single_factor()
│   └── 每次请求临时创建 StockEvalService → 创建 Session → 加载 panel → 评估 → 释放
│       （LRU SessionManager 缓存 ≤3 份 Session，避免重复加载面板）
└── 挖掘 run 管理 → start_run()
    └── subprocess.Popen(run_alphaagent.py)  ← 独立子进程
        ├── 自行加载 panel（mmap 命中时秒级 attach，否则 30-60s 冷构建）
        ├── 进程内 StockEvalService（ThreadPoolExecutor 并发 + Semaphore(6)）
        └── engine_gate 回测（core.engine 完整约束回测）
```

### 1.2 已有的共享层（不需要重复建设）

| 层 | 实现 | 效果 |
| --- | --- | --- |
| **行情面板 mmap** | `panel_mmap.py`：Arrow IPC + `pa.memory_map` | 数值列跨进程零拷贝共享物理页；第 2..N 进程 attach ≈ 秒级 |
| **因子值缓存** | `factor/cache.py`：进程内 LRU + 磁盘 `.npy` 持久化 | 同 expr+panel 指纹命中时跳过 DSL 求值（磁盘缓存可跨进程） |
| **session LRU** | `backend/services/session_manager.py`：LRUSessionCache(3) | Web 进程内相同参数的评估请求复用 panel |
| **准入控制** | `_admission_check()`：活跃 run 数上限 + 可用内存下限 | 防止前端连点几次 OOM |

### 1.3 现存痛点

| 痛点 | 根因 | 影响 |
| --- | --- | --- |
| **Web UI 卡顿** | 因子实验室评估（`eval_profile`）在 FastAPI 主进程线程池中跑重度 Pandas/NumPy 计算，GIL 锁竞争阻塞所有 HTTP 路由 | 评估期间前端"白屏"或接口超时 |
| **挖掘冷启动慢** | 每个 `subprocess.Popen(run_alphaagent.py)` 都要完整初始化 Python、import 全量依赖、构建/attach panel | 即使 mmap 命中，每次 spawn 仍需 5-10s 就绪 |
| **多 run 内存墙** | 尽管数值列 mmap 共享，每个子进程仍独立持有 instrument 字符串列（~0.4GB）+ DatetimeIndex（~64MB）+ split 缓存 + 因子值缓存 | 2 个 run ≈ 10GB，16GB 机器已到极限 |
| **评估渠道割裂** | Web 因子实验室、挖掘子进程、离线批量脚本三路各自创建独立的 `StockEvalService`，无全局并发统筹 | 偶发三路同时跑时 CPU 超卖、内存叠加 |
| **离线脚本重复加载** | `promote_candidates.py` / `rescreen_candidate_pool.py` 等脚本各自加载面板 | 每次跑都要冷启动几十秒 |

---

## 二、 目标与非目标

### 目标

1. **FastAPI 主进程永远轻量**：不再在 Web 进程中执行任何因子评估 / 回测计算
2. **全局统一并发配额**：Web 实验室、挖掘 run、离线脚本共用同一个 Worker 池，精确控制 CPU / 内存上限
3. **消灭挖掘冷启动**：Worker 常驻，panel 已 attach，收到任务即可开始计算
4. **单机零外部依赖**：不引入 Redis / RabbitMQ / Docker 等基础设施

### 非目标

- 不做分布式多机部署（当前单机场景）
- 不改变 LLM 交互循环的进程模型（LLM 循环仍在独立子进程中，只是评估计算改为 RPC 到 Worker）
- 不改变 DSL 算子实现、评估引擎 Profile 等核心逻辑

---

## 三、 架构设计

### 3.1 整体拓扑

```
┌──────────────────────────────────────────────────────────┐
│                    FastAPI 主进程 (17891)                 │
│  [Web API / SSE / 研究记忆 / 因子库 CRUD / run 管理]    │
│                                                          │
│  WorkerPoolClient ─────────────────────────────────┐     │
│    • submit_task(task) → Future                    │     │
│    • 异步/同步接口                                  │     │
└─────────────────────────────────────────────────────┼─────┘
                                                      │
              mp.Queue (task)  /  mp.Queue (result)    │
                                                      │
┌─────────────────────────────────────────────────────┼─────┐
│                  WorkerPoolManager                   │     │
│  • 启动 / 监控 / 重启 Worker 子进程                 │     │
│  • 全局并发配额（Semaphore）                        │     │
│  • 健康检查（heartbeat）                            │     │
└─────────────────────────────────────────────────────┼─────┘
                                                      │
         ┌────────────────────────┬───────────────────┘
         ▼                        ▼
┌─────────────────┐     ┌─────────────────┐
│  Worker 进程 1  │     │  Worker 进程 N  │
│                 │     │                 │
│  • panel mmap   │     │  • panel mmap   │    ← 共享物理页
│    (attach)     │     │    (attach)     │
│                 │     │                 │
│  • EvalEngine   │     │  • EvalEngine   │    ← 各自实例
│  • FactorCache  │     │  • FactorCache  │    ← 磁盘缓存共享
│  • engine_gate  │     │  • engine_gate  │
│  • jq_backtest  │     │  • jq_backtest  │
└─────────────────┘     └─────────────────┘

         ▲                        ▲
         │                        │
┌────────┴────────┐     ┌────────┴────────┐
│ 挖掘 run 子进程 │     │   离线脚本      │
│ (LLM 循环)      │     │ (promote/rescrn)|
│                 │     │                 │
│ WorkerPoolClient│     │ WorkerPoolClient│
│ (连接同一池)    │     │ (连接同一池)    │
└─────────────────┘     └─────────────────┘
```

### 3.2 核心组件

#### 3.2.1 `WorkerPoolManager`（调度层）

```python
# alphaagent/compute/pool.py

class WorkerPoolManager:
    """进程级算力工作池管理器。

    生命周期跟随 FastAPI 主进程：app startup 时启动 Worker 池，
    app shutdown 时优雅关闭。
    """

    def __init__(
        self,
        *,
        n_workers: int = 0,        # 0 = auto（物理核数 - 2，下限 2）
        panel_arrow_path: str = "", # 空 = 自动发现最新缓存
        max_queue_size: int = 64,
    ): ...

    def start(self) -> None:
        """启动 Worker 子进程池。"""

    def submit(self, task: ComputeTask) -> TaskFuture:
        """提交计算任务，返回 Future。"""

    def shutdown(self, timeout: float = 30) -> None:
        """优雅关闭：发送 poison pill → 等待 Worker 退出。"""
```

#### 3.2.2 `ComputeTask`（任务契约）

```python
# alphaagent/compute/task.py

@dataclass(frozen=True)
class ComputeTask:
    """跨进程序列化的计算任务描述。

    原则：只传最小必要参数（字符串/数值/小字典），
    大数据（panel/factor_values）在 Worker 内部按引用获取，
    绝不跨进程传输 DataFrame。
    """
    task_type: str          # "eval_factor" | "engine_gate" | "jq_backtest" | "batch_materialize"
    task_id: str            # UUID，用于结果关联
    priority: int = 0       # 0=普通, 1=高优先(Web实验室), -1=低优先(离线脚本)
    params: dict            # 按 task_type 不同的参数字典

    # eval_factor 参数示例：
    #   session_key: str        # panel 参数哈希（Worker 内部按 key 查已 attach 的 panel）
    #   multi_line_expr: str
    #   factor_name: str
    #   profile_id: str
    #   label_col: str
    #   label_quantile_n: int
    #   include_charts: bool
    #   split: str              # "train" | "val" | "full"
    #   split_range: tuple[str, str]

    # engine_gate 参数示例：
    #   session_key: str
    #   multi_line_expr: str
    #   val_start: str, val_end: str
    #   direction: int
    #   policy: dict

@dataclass
class TaskResult:
    task_id: str
    ok: bool
    result: dict | None = None
    error: str | None = None
    elapsed_ms: float = 0
```

#### 3.2.3 `Worker`（执行层）

```python
# alphaagent/compute/worker.py

class ComputeWorker:
    """常驻 Worker 进程的主循环。

    启动时：
    1. attach panel mmap（零拷贝）
    2. 初始化 EvaluationEngine + 默认 profiles
    3. 预热 Numba JIT 缓存（首次 import 触发编译）
    4. 进入任务消费循环
    """

    def __init__(
        self,
        task_queue: mp.Queue,
        result_queue: mp.Queue,
        panel_arrow_path: Path,
        worker_id: int,
    ): ...

    def run(self) -> None:
        """Worker 主循环（在子进程中执行）。"""
        self._attach_panel()      # mmap attach
        self._init_engine()       # EvaluationEngine + profiles
        self._warmup_numba()      # 确保 JIT 已编译
        while True:
            task = self.task_queue.get()
            if task is _POISON_PILL:
                break
            result = self._execute(task)
            self.result_queue.put(result)

    def _execute(self, task: ComputeTask) -> TaskResult:
        match task.task_type:
            case "eval_factor":
                return self._eval_factor(task)
            case "engine_gate":
                return self._engine_gate(task)
            case "jq_backtest":
                return self._jq_backtest(task)
            case "batch_materialize":
                return self._batch_materialize(task)
            case _:
                return TaskResult(task.task_id, ok=False, error=f"unknown_task_type:{task.task_type}")
```

### 3.3 Panel 会话管理（Worker 侧）

Worker 的 panel 生命周期管理是本方案的关键设计点。

```
Worker 进程启动
    │
    ├── 1. attach 默认 panel（最新 arrow 缓存，覆盖当前 train+val 窗口）
    │       → self._default_panel: pd.DataFrame  (mmap 零拷贝)
    │       → self._default_session_key: str
    │
    ├── 2. 收到 eval_factor 任务
    │       ├── task.params["session_key"] == self._default_session_key
    │       │   → 直接使用 self._default_panel（零开销）
    │       │
    │       └── task.params["session_key"] != self._default_session_key
    │           → 查 self._panel_cache (LRU ≤2)
    │           → 未命中 → 按参数加载新 panel（冷路径，极少触发）
    │
    └── 3. split 切片：按 (start, end) 缓存在 Worker 内部 _split_cache
```

**核心约束**：Worker 绝不完整序列化 panel 跨进程传输。所有调用方
传的是 `session_key`（参数哈希），Worker 自己管理 panel 的加载与切片。

### 3.4 通信协议

选择 `multiprocessing.Queue`（基于 OS 管道/FIFO）而非 Socket/HTTP：

| 对比项 | mp.Queue | Socket RPC |
| --- | --- | --- |
| 序列化 | pickle（dict/str/数值全支持） | 需自定义协议/protobuf |
| 延迟 | ~0.1ms（本地管道） | ~1ms（TCP loopback） |
| 复杂度 | 极低（标准库） | 需维护连接池/重连 |
| 缺点 | 仅单机 | 可跨机但目前不需要 |

**队列设计**：

```
task_queue:   PriorityQueue（按 priority 排序）
              高优先级（Web 实验室）插队低优先级（离线脚本）

result_queue: 普通 Queue
              Worker 写入 → Manager 的分发线程读取 → 通知对应 Future
```

### 3.5 挖掘 run 子进程的接入

当前挖掘 run 的评估调用链：

```
agentscope_tools.py → FactorEvalTools → StockEvalService._run_one()
                                         └── EvaluationEngine.evaluate()  ← 进程内直接调用
```

改造后：

```
agentscope_tools.py → FactorEvalTools → WorkerPoolClient.submit(eval_factor_task)
                                         └── mp.Queue → Worker 进程执行 → 结果回传
```

**关键变化**：挖掘子进程不再自行创建 `StockEvalService` 和 `SessionStore`，
不再在进程内执行 DSL 求值和指标计算。它只负责 LLM 交互循环和任务编排，
所有重计算通过 `WorkerPoolClient` 提交到共享工作池。

这意味着挖掘子进程的内存占用将从 ~5GB 骤降至 < 500MB
（仅 LLM 对话上下文 + 研究记忆 + JSON 元数据）。

### 3.6 Worker 池的生命周期

```
FastAPI app startup (lifespan)
    │
    ├── WorkerPoolManager.start()
    │   ├── 发现最新 panel arrow 缓存路径
    │   ├── fork N 个 Worker 子进程
    │   ├── 各 Worker 初始化（attach panel → init engine → warmup numba）
    │   └── 启动 result 分发线程（从 result_queue 取结果 → 通知 Future）
    │
    ├── 运行期
    │   ├── 健康检查线程：每 30s 检查 Worker 进程存活，死亡自动重启
    │   ├── panel 刷新：收到 /api/alphaagent/session-cache/evict 时
    │   │   通知所有 Worker 重新 attach panel（支持数据更新）
    │   └── 指标上报：Worker 汇报队列深度 / 平均处理时间 / 空闲率
    │
    └── FastAPI app shutdown
        └── WorkerPoolManager.shutdown()
            ├── 向 task_queue 投入 N 个 poison pill
            ├── 等待 Worker 进程退出（timeout 30s）
            └── 强制 kill 残留进程
```

---

## 四、 接入改造：渐进式三阶段

### 阶段一（P0）：Web 因子实验室卸载

**目标**：`POST /api/alphaagent/eval-factor` 和 `POST /api/alphaagent/backtest-factor`
不再在 FastAPI 主进程中执行计算。

**改造范围**：

```
backend/alphaagent_service.py
├── evaluate_single_factor()  →  改为提交 ComputeTask 到 Worker 池
├── backtest_single_factor()  →  同上
└── app lifespan              →  启动/关闭 WorkerPoolManager

alphaagent/compute/           ← 新增目录
├── __init__.py
├── pool.py                   # WorkerPoolManager
├── worker.py                 # ComputeWorker
├── task.py                   # ComputeTask / TaskResult
└── client.py                 # WorkerPoolClient（同步/异步接口）
```

**验收标准**：
- FastAPI 主进程在评估期间 CPU 占用 < 5%（GIL 完全释放）
- Web UI 在后台评估时不卡顿
- 评估结果与当前逐位一致（对照 `tests/test_metrics_fastpaths.py`）

### 阶段二（P1）：挖掘 run 评估卸载

**目标**：挖掘子进程的 `evaluate_factor` / `eval_on_train_set` / `engine_gate`
全部走 Worker 池。

**改造范围**：

```
alphaagent/factor/mining/eval/service.py
├── StockEvalService.__init__()  →  接受 WorkerPoolClient 注入
├── _run_one()                   →  改为提交 ComputeTask
├── _eval_train_two_stage()      →  两段式仍保留，但每段都走 Worker
└── _maybe_engine_preview()      →  engine_gate 也走 Worker

alphaagent/factor/mining/delivery/submit.py
├── _run_engine_gate()           →  改为提交 ComputeTask

backend/alphaagent_service.py
├── start_run()                  →  不再 subprocess.Popen 完整 python
│                                    改为 spawn 轻量 LLM 循环进程
│                                    （不加载 panel，仅连接 Worker 池）
```

**验收标准**：
- 挖掘子进程 USS < 500MB（不含 panel 独立拷贝）
- 同时 2 个挖掘 run + 1 个 Web 实验室评估，总内存 < 10GB
- 吞吐不低于当前水平（2.61 eval/s 基线）

### 阶段三（P2）：离线脚本与 JQ 回测接入

**目标**：所有离线脚本和 JQ 代码面板回测共用同一个 Worker 池。

**改造范围**：

```
scripts/promote_candidates.py      →  使用 WorkerPoolClient
scripts/rescreen_candidate_pool.py →  同上
scripts/blind_test_factors.py      →  同上
core/event_engine/jq/entry.py      →  回测执行部分走 Worker
```

**验收标准**：
- 离线脚本启动无冷加载延迟（Worker 已常驻 panel）
- JQ 回测崩溃不影响 FastAPI 主服务

---

## 五、 关键设计决策与取舍

### 5.1 为什么不用 Ray / Celery / Dramatiq

| 方案 | 拒绝理由 |
| --- | --- |
| **Ray** | 安装 + 集群启动重量级（worker init 需 Ray Plasma store），Windows 原生支持弱，对单机场景过度设计 |
| **Celery** | 强依赖消息中间件（Redis/RabbitMQ），序列化不支持 numpy 原生，与当前 `panel_mmap.py` 共享层对接需额外胶水 |
| **Dramatiq** | 同上，额外中间件依赖 |
| **concurrent.futures.ProcessPoolExecutor** | 每次提交都是 fork + pickle 全量参数，不支持常驻初始化（每次 fork 都要重新 attach panel + warmup numba） |

**选择 `multiprocessing.Process` + `Queue` 的理由**：
- Worker 常驻（一次 attach panel、一次 warmup numba，之后纯消费任务）
- 零外部依赖（标准库）
- 与现有 `panel_mmap.py`（Arrow IPC + `pa.memory_map`）无缝契合
- 单机场景下性能天花板（管道延迟 ~0.1ms）

### 5.2 为什么不直接用 `fork`（而是 `spawn`）

Windows 不支持 `fork`。项目运行在 Windows 上，`multiprocessing.Process`
默认使用 `spawn`（全新 Python 解释器）。这意味着：

- Worker 需要 import 全量依赖（一次性成本，启动时付完）
- 不能传 lambda / 闭包 / 未序列化对象跨进程
- 但 panel 通过 mmap 不受影响（OS 级文件映射，与 Python 进程模型无关）

### 5.3 任务粒度选择

**选择"单次评估"为最小任务单元**（即一次 `evaluate_factor` 调用 = 一个 `ComputeTask`），
不做更细粒度的拆分（如"DSL 求值"和"指标计算"分开）。

理由：
- 当前两段式海选已经实现了评估内部的短路优化
- 拆更细会导致中间结果跨进程传输（factor Series 几十 MB），抵消收益
- 任务粒度与 LLM `tool_calls` 粒度一致（一个 tool_call = 一个 eval），语义清晰

### 5.4 Worker 数量选择策略

```python
def _default_n_workers() -> int:
    """自动计算 Worker 数量。

    原则：
    - 留 2 核给 FastAPI 主进程 + LLM 循环进程 + 系统
    - 每个 Worker 至多占用 1 核的 GIL 段 + 多核的 Numba nogil 段
    - Numba nogil 内核会自行并行，Worker 间的并行收益主要来自
      不同评估任务的 GIL 段错开

    实测结论（2026-09-06 吞吐报告）：
    - 6 车道两段式 = 2.61 eval/s（当前最优）
    - 超过 6 车道后 Numba 线程争用+内存带宽饱和，吞吐不再增长
    """
    import os
    physical = os.cpu_count() or 4
    return max(2, min(physical - 2, 6))
```

### 5.5 错误隔离与 Worker 重启

```
Worker 进程异常情况           处理策略
─────────────────────────────────────────────────
单个任务 Python 异常          捕获 → TaskResult(ok=False) → Worker 继续
Numba 内核 access violation   Worker 进程崩溃 → Manager 检测到 → 重启新 Worker
内存泄漏（RSS > 阈值）       Manager 监控 → 优雅替换（等当前任务完成 → kill → 启新）
panel arrow 文件更新          Manager 广播 → Worker 重新 attach（不中断当前任务）
```

---

## 六、 接口契约详表

### 6.1 `eval_factor` 任务

```python
# 输入（ComputeTask.params）
{
    "session_key": "a3b4c5d6e7f8",     # panel 参数哈希（与 SessionManager._hash_params 同算法）
    "panel_spec": {                     # session_key 未命中时的加载规格（冷路径）
        "panel_path": "cne://",
        "start": "2020-01-01",
        "end": "2024-12-31",
        "include_fundamentals": True,
        "focus_facets": ["价量", "基本面"],
    },
    "multi_line_expr": "CS_RANK(TS_MEAN($close, 20))",
    "factor_name": "close_ma20_rank",
    "profile_id": "train_screen",       # | "train_screen_lite" | "validation"
    "label_col": "label_1d_open_to_open",
    "label_quantile_n": 10,
    "include_charts": False,
    "split": "train",
    "split_range": ["2020-01-01", "2022-12-31"],
}

# 输出（TaskResult.result）
# 与当前 EvaluationEngine.evaluate() 的返回值完全一致
```

### 6.2 `engine_gate` 任务

```python
# 输入
{
    "session_key": "a3b4c5d6e7f8",
    "multi_line_expr": "...",
    "val_start": "2023-01-01",
    "val_end": "2024-12-31",
    "direction": 1,
    "policy": { "freq": "weekly", "capital": 2000000, ... },
    "asset_type": "stock",
}

# 输出
# 与当前 run_engine_gate() 的返回值完全一致
```

### 6.3 `jq_backtest` 任务

```python
# 输入
{
    "code": "def initialize(context): ...\ndef handle_data(context, data): ...",
    "start_date": "2020-01-01",
    "end_date": "2024-12-31",
    "initial_cash": 100000,
    "benchmark": "000300.XSHG",
}

# 输出
# 与当前 run_jq_backtest() 的返回值一致
```

---

## 七、 性能预期与监控

### 7.1 预期收益

| 指标 | 当前 | 改造后 | 改善 |
| --- | --- | --- | --- |
| Web UI 在评估期间的响应延迟 | 5-30s（GIL 阻塞） | < 50ms | **100x** |
| 挖掘子进程 USS 内存 | ~5GB | < 500MB | **-90%** |
| 2 run + 1 实验室总内存 | ~15GB | ~8GB | **-47%** |
| 离线脚本启动时间 | 30-60s | < 2s | **15x** |
| 评估吞吐（eval/s） | 2.61 | ≥ 2.5（不低于现有） | 持平 |

### 7.2 监控端点

```
GET /api/compute/status
{
    "workers": {
        "total": 4,
        "alive": 4,
        "idle": 2,
        "busy": 2
    },
    "queue": {
        "pending": 3,
        "high_priority": 1,
        "low_priority": 0
    },
    "stats": {
        "total_tasks": 1247,
        "avg_eval_ms": 380,
        "p99_eval_ms": 1200,
        "worker_restarts": 0,
        "panel_reattach_count": 1
    }
}
```

---

## 八、 风险与缓解

| 风险 | 概率 | 缓解方案 |
| --- | --- | --- |
| Windows `spawn` 模式下 Worker 初始化慢 | 中 | 启动时一次性付完（~10s）；运行期零成本；可增加 `--preload` 标志在后台预启动 |
| mp.Queue pickle 失败（某些对象不可序列化） | 低 | 任务契约严格只传 str/int/float/dict；结果中的 numpy 数组转 list 再传 |
| Worker 泄漏导致 Numba 缓存不一致 | 极低 | Worker 重启时清理 `__pycache__`；可配置定时强制轮换 |
| 改造期间挖掘质量回归 | 中 | 阶段一只改 Web 实验室（影响面最小）；阶段二改挖掘时全量跑 `test_mining_eval_throughput.py` 对比 |

---

## 九、 文件清单（预计新增/修改）

```
新增：
  alphaagent/compute/__init__.py
  alphaagent/compute/pool.py          # WorkerPoolManager
  alphaagent/compute/worker.py        # ComputeWorker
  alphaagent/compute/task.py          # ComputeTask / TaskResult
  alphaagent/compute/client.py        # WorkerPoolClient
  alphaagent/compute/panel_store.py   # Worker 侧 panel 管理（mmap attach + LRU）
  tests/test_compute_pool.py          # 单元测试
  tests/test_compute_integration.py   # 集成测试（eval 结果一致性）

修改：
  backend/main.py                     # app lifespan 集成 WorkerPoolManager
  backend/alphaagent_service.py       # evaluate_single_factor → WorkerPoolClient
  alphaagent/factor/mining/eval/service.py  # StockEvalService 可选 Worker 模式
```
