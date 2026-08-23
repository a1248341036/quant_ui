"""股票因子库：canonical index、memmap 存储、相似度。"""

from pathlib import Path

from alphaagent.factor.zoo.catalog import FactorCatalog
from alphaagent.factor.zoo.index import RowIndex, init_library, verify_index_hash
from alphaagent.factor.zoo.similarity import (
    SIMILARITY_KIND,
    SimilarityMatrix,
    cross_sectional_pearson_mean,
    cross_sectional_pearson_series,
)
from alphaagent.factor.zoo.types import (
    DEFAULT_BAR_INTERVAL,
    DEFAULT_DATASET,
    FactorLibraryPaths,
    FactorStatus,
    LibraryManifest,
    RowSlice,
)
from alphaagent.factor.zoo.zoo import FactorZoo

from alphaagent.core.paths import FACTORZOO_DIR

DEFAULT_FACTORLIB_ROOT = FACTORZOO_DIR

__all__ = [
    "DEFAULT_BAR_INTERVAL",
    "DEFAULT_DATASET",
    "DEFAULT_FACTORLIB_ROOT",
    "FactorCatalog",
    "FactorLibraryPaths",
    "FactorStatus",
    "FactorZoo",
    "LibraryManifest",
    "RowIndex",
    "RowSlice",
    "SimilarityMatrix",
    "SIMILARITY_KIND",
    "cross_sectional_pearson_mean",
    "cross_sectional_pearson_series",
    "init_library",
    "verify_index_hash",
]
