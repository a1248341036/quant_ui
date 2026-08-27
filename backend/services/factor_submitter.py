"""因子提交流程：封装两阶段交付逻辑。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from alphaagent.factor.mining.service import StockEvalService


class FactorSubmitter:
    """因子提交器，处理因子的入库流程。"""

    def __init__(self, session_manager, factorlib_path: Path):
        """
        初始化提交器。

        Args:
            session_manager: SessionManager 实例
            factorlib_path: 因子库路径
        """
        self.session_manager = session_manager
        self.factorlib_path = factorlib_path
        self._submit_service: Any = None

    def submit_factor(
        self,
        session_id: str,
        multi_line_expr: str,
        factor_name: str,
        comment: str,
        category: str = "technical",
        rebalance_freq: str = "daily",
    ) -> dict:
        """
        提交因子入库。

        Args:
            session_id: 会话 ID
            multi_line_expr: 因子 DSL 表达式
            factor_name: 因子名称
            comment: 因子说明
            category: 因子类别
            rebalance_freq: 调仓频率

        Returns:
            提交结果（包含 stored, candidate_stored, error 等）
        """
        from core import factor_categories
        from alphaagent.factor.mining.submit import FactorSubmitService

        # 确定因子库路径
        lib_root = factor_categories.production_dir(category)

        # 创建提交服务
        submit_service = FactorSubmitService(
            service=None,  # 实际使用时需要传入 StockEvalService
            factorlib_path=lib_root,
        )

        # 执行提交
        return submit_service.submit(
            session_id=session_id,
            multi_line_expr=multi_line_expr,
            factor_name=factor_name,
            comment=comment,
            rebalance_freq=rebalance_freq,
        )

    def check_orthogonality(
        self, expr: str, panel_path: str, top_k: int = 5
    ) -> dict:
        """
        检查因子的正交性（与已有因子的相似度）。

        Args:
            expr: 因子 DSL 表达式
            panel_path: 数据源路径
            top_k: 返回最相似的因子数

        Returns:
            正交性检查结果（orthogonal, similar_factors, max_corr）
        """
        from alphaagent.dsl import eval_factor
        from alphaagent.data.adapters.cnequity import load_panel_from_cne
        from alphaagent.factor.align import align_series_to_panel
        from alphaagent.factor.metrics import spearman_ic
        from core import factor_categories
        from alphaagent.factor.zoo import FactorZoo

        # 加载 panel
        panel = load_panel_from_cne(start=None, end=None, universe_mask=False)

        # 评估新因子
        new_series = eval_factor(expr, panel)
        new_aligned = align_series_to_panel(new_series, panel)

        # 获取因子库
        prod_root = factor_categories.production_dir("technical")
        zoo = FactorZoo(prod_root)

        # 计算与已有因子的相似度
        similar = []
        for factor_id in zoo.list_factors():
            try:
                registry = zoo.get_factor_registry(factor_id)
                if not registry or "expr" not in registry:
                    continue

                old_expr = registry["expr"]
                old_series = eval_factor(old_expr, panel)
                old_aligned = align_series_to_panel(old_series, panel)

                # 计算截面 Spearman 相关
                corr = spearman_ic(new_aligned, old_aligned)
                if corr is not None and abs(corr) > 0.5:
                    similar.append(
                        {"factor_id": factor_id, "corr": float(abs(corr))}
                    )
            except Exception:
                continue

        # 按相似度排序
        similar.sort(key=lambda x: x["corr"], reverse=True)
        similar = similar[:top_k]

        return {
            "orthogonal": len(similar) == 0,
            "similar_factors": similar,
            "max_corr": similar[0]["corr"] if similar else 0.0,
        }
