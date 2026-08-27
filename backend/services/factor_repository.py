"""因子仓库服务：封装 FactorZoo，提供因子的增删改查。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphaagent.factor.zoo import FactorZoo


class FactorRepository:
    """因子仓库，管理因子库的读写操作。"""

    def __init__(self):
        """初始化因子仓库。"""
        self._zoo_cache: dict[str, FactorZoo] = {}

    def _get_zoo(self, library_path: Path) -> FactorZoo:
        """
        获取或创建 FactorZoo 实例。

        Args:
            library_path: 因子库路径

        Returns:
            FactorZoo 实例
        """
        path_str = str(library_path)
        if path_str not in self._zoo_cache:
            from alphaagent.factor.zoo import FactorZoo

            self._zoo_cache[path_str] = FactorZoo.open(library_path)
        return self._zoo_cache[path_str]

    def list_factors(
        self, library_path: Path, category: str = "technical"
    ) -> list[dict]:
        """
        列出因子库中的所有因子。

        Args:
            library_path: 因子库路径
            category: 因子类别（technical/fundamental 等）

        Returns:
            因子列表（包含基本信息）
        """
        try:
            zoo = self._get_zoo(library_path)
        except FileNotFoundError:
            return []

        factors = []
        for factor_id in zoo.catalog.list_factor_ids():
            try:
                meta = zoo.catalog.get(factor_id)
                factors.append(
                    {
                        "factor_id": factor_id,
                        "name": meta.name if meta else factor_id,
                        "created_at": meta.created_at if meta else None,
                        "status": str(meta.status) if meta else "active",
                    }
                )
            except Exception:
                continue

        return factors

    def get_factor_detail(self, factor_id: str, library_path: Path) -> dict:
        """
        获取因子详细信息。

        Args:
            factor_id: 因子 ID
            library_path: 因子库路径

        Returns:
            因子详细信息（包含表达式、指标等）
        """
        zoo = self._get_zoo(library_path)

        meta = zoo.catalog.get(factor_id)
        if meta is None:
            return {"error": "factor_not_found"}

        return {
            "factor_id": factor_id,
            "name": meta.name,
            "expr": meta.expr,
            "created_at": meta.created_at,
        }

    def delete_factor(self, factor_id: str, library_path: Path) -> bool:
        """
        删除因子。

        Args:
            factor_id: 因子 ID
            library_path: 因子库路径

        Returns:
            是否删除成功
        """
        zoo = self._get_zoo(library_path)

        try:
            zoo.delete_factor(factor_id)
            return True
        except Exception:
            return False

    def search_factors(
        self, library_path: Path, query: str, top_k: int = 10
    ) -> list[dict]:
        """
        搜索因子（基于表达式或元数据）。

        Args:
            library_path: 因子库路径
            query: 搜索关键词
            top_k: 返回结果数量

        Returns:
            匹配的因子列表
        """
        zoo = self._get_zoo(library_path)

        results = []
        for factor_id in zoo.catalog.list_factor_ids():
            try:
                meta = zoo.catalog.get(factor_id)
                expr = meta.expr if meta else ""
                name = meta.name if meta else factor_id

                # 简单关键词匹配
                if query.lower() in str(name).lower() or query.lower() in expr.lower():
                    results.append(
                        {
                            "factor_id": factor_id,
                            "name": name,
                            "expr": expr,
                        }
                    )

                if len(results) >= top_k:
                    break
            except Exception:
                continue

        return results
