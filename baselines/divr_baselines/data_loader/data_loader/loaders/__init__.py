from .cached import CachedLoader
from .normal import NormalLoader
from .batch_ahead import BatchAheadLoader
from .feature_ahead import FeatureAheadLoader
from enum import Enum


class LoaderTypes(Enum):
    NORMAL = "NORMAL"
    CACHED = "CACHED"
    BATCH_AHEAD = "BATCH_AHEAD"
    FEATURE_AHEAD = "FEATURE_AHEAD"


__all__ = [
    "BatchAheadLoader",
    "CachedLoader",
    "FeatureAheadLoader",
    "NormalLoader",
    "LoaderTypes",
]
