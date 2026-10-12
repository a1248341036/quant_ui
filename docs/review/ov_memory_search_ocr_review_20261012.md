Review complete: 2 finding(s) across 3 selected item(s).

─── alphaagent/mcp_server/tools.py:424-425 ───
[performance · low] 每次 `ov_memory_search` 调用都会经 `_ov_store()` 新建 OVStore 实例，而 OVStore 的 OpenViking
客户端（SyncHTTPClient + `initialize()`）连接只在实例内部缓存（`_get_client` 的
`self._client`）——实例被丢弃后每次工具调用都会重新执行网络握手建连；OpenViking 不可用时每次调用还会重复尝试连接并打一条
warning（直驱会话高频查记忆时产生连接开销与日志噪音）。与 `_memory_store`（仅打开本地 SQLite）不同，OV 建连含网络初始化成本。建议在模块级对成功建连后的 OVStore
做单例缓存（失败时保留下次重试），避免逐调用重建。



─── alphaagent/mcp_server/tools.py:450-452 ───
[other · low] 这里的 `available: False` 降级分支在实际 OpenViking 宕机场景几乎不可达：`_ov_store()` 只做懒初始化构造，OVStore
构造本身不建连（真实建连在 `_get_client()` 里），所以 OV 服务不可用时 `store` 依然非 None，会走到下面的 `available: True` + 空 block
分支。结果是调用方无法从响应区分「OV 不可用」与「无相关记忆」，也就难以按 docstring 所述回落 SQLite memory_search。建议返回前先探测连通性（如
`store._get_client() is not None`）再设置 `available`，或至少把空 block 且未连通的情况显式标注出来。


