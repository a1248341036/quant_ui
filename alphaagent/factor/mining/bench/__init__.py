"""AlphaAgent 挖掘评测 Bench 模块。

包含：
- extended_metrics: A-F 分组高级指标解析
- config: 冻结评测配置加载与哈希
- runner: 评测执行、完成性与崩溃识别
- reporter: 评分、对比、台账与一页报告
"""
from __future__ import annotations

from alphaagent.factor.mining.bench.extended_metrics import compute_extended_metrics

__all__ = ["compute_extended_metrics"]
