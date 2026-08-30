# -*- coding: utf-8 -*-
"""AlphaAgent 研究记忆存储的组合根。

按职责拆分为 schema、入库、经验蒸馏、检索、advisory 和回填 mixin；
对外仍使用统一的 :class:`ResearchMemoryStore`。
"""

from __future__ import annotations

from .advisory import AdvisoryMixin
from .backfill import BackfillMixin
from .experience import ExperienceMixin
from .ingestion import IngestionMixin
from .retrieval import RetrievalMixin
from .schema import SchemaMixin


class ResearchMemoryStore(
    AdvisoryMixin,
    BackfillMixin,
    ExperienceMixin,
    IngestionMixin,
    RetrievalMixin,
    SchemaMixin,
):
    """跨 run 持久化研究记忆的统一门面。"""
