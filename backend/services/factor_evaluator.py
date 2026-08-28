"""因子评估服务：封装 StockEvalService，提供统一的评估接口。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from alphaagent.factor.mining.service import StockEvalService
    from alphaagent.factor.mining.schemas import (
        SessionCreateRequest,
        EvalProfileRequest,
        EvalTrainRequest,
        EvalValRequest,
    )


class FactorEvaluator:
    """因子评估器，封装评估逻辑。"""

    def __init__(self, session_manager):
        """
        初始化评估器。

        Args:
            session_manager: SessionManager 实例
        """
        self.session_manager = session_manager
        self._service: StockEvalService | None = None
        self._service_params: str | None = None
        self._current_session_id: str | None = None

    def _get_service(self, req: SessionCreateRequest) -> StockEvalService:
        """
        获取或创建评估服务。

        Args:
            req: 会话创建请求

        Returns:
            StockEvalService 实例
        """
        from alphaagent.factor.mining.service import StockEvalService

        # 参数哈希用于检测是否需要重建服务
        param_hash = self.session_manager._hash_params(req)

        if self._service_params != param_hash or self._service is None:
            # 参数变化，重建服务
            self._service = StockEvalService()
            self._service_params = param_hash

        return self._service

    def create_session(self, req: SessionCreateRequest) -> dict:
        """
        创建评估会话。

        Args:
            req: 会话创建请求

        Returns:
            会话创建响应（包含 session_id 等信息）
        """
        # 通过 session_manager 创建会话（带 LRU 缓存）
        session = self.session_manager.get_or_create_session(req)

        # 获取服务并注册会话
        service = self._get_service(req)

        # 关键：LRU 缓存里的 session 必须同步注册进 service 自己的
        # SessionStore，否则 eval_profile 用 session_id 查 store 会报未知 session_id。
        service.sessions.register(session)

        # 保存当前 session_id
        self._current_session_id = session.session_id

        # 返回会话信息
        cols = list(session.panel.columns[:12])
        return {
            "session_id": session.session_id,
            "panel_rows": len(session.panel),
            "load_ms": float(session.meta.get("load_ms", 0)),
            "columns_sample": cols,
            "available_columns": list(session.panel.columns),
        }

    def eval_profile(self, req: EvalProfileRequest) -> dict:
        """
        按 Profile 评估因子。

        Args:
            req: 评估请求

        Returns:
            评估结果
        """
        # 如果没有指定 session_id，使用当前的
        if not req.session_id and self._current_session_id:
            req.session_id = self._current_session_id
            
        service = self._get_service_from_request(req)
        return service.eval_profile(req)

    def eval_train(self, req: EvalTrainRequest) -> dict:
        """
        训练集评估。

        Args:
            req: 评估请求

        Returns:
            评估结果
        """
        if not req.session_id and self._current_session_id:
            req.session_id = self._current_session_id
            
        service = self._get_service_from_request(req)
        return service.eval_train(req)

    def eval_val(self, req: EvalValRequest) -> dict:
        """
        验证集评估。

        Args:
            req: 评估请求

        Returns:
            评估结果
        """
        if not req.session_id and self._current_session_id:
            req.session_id = self._current_session_id
            
        service = self._get_service_from_request(req)
        return service.eval_val(req)

    def _get_service_from_request(self, req) -> StockEvalService:
        """
        从请求中获取服务实例。

        Args:
            req: 评估请求（包含 session_id）

        Returns:
            StockEvalService 实例
        """
        # 从 session_id 推断服务参数
        # 实际实现中，服务应该已经存在（由 create_session 创建）
        if self._service is None:
            raise ValueError("评估服务未初始化，请先调用 create_session")
        return self._service

    def record_candidate_review(
        self, session_id: str, candidate_id: str, review: dict
    ) -> dict | None:
        """
        记录候选因子评审结果。

        Args:
            session_id: 会话 ID
            candidate_id: 候选因子 ID
            review: 评审结果

        Returns:
            更新后的候选因子信息
        """
        if self._service is None:
            raise ValueError("评估服务未初始化")
        return self._service.record_candidate_review(session_id, candidate_id, review)
