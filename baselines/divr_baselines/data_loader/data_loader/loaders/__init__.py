from .cached import CachedLoader
from .normal import NormalLoader
from enum import Enum


class LoaderTypes(Enum):
    NORMAL = "NORMAL"
    CACHED = "CACHED"


__all__ = ["CachedLoader", "NormalLoader", "LoaderTypes"]
