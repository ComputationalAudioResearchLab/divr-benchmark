from .cached import CachedLoader
from .normal import NormalLoader
from .batch_ahead import BatchAheadLoader
from enum import Enum


class LoaderTypes(Enum):
    NORMAL = "NORMAL"
    CACHED = "CACHED"
    BATCH_AHEAD = "BATCH_AHEAD"


__all__ = ["BatchAheadLoader", "CachedLoader", "NormalLoader", "LoaderTypes"]
