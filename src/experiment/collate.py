from abc import abstractmethod
import numpy as np
from typing import Dict, List, Protocol

InputFeatures = Dict[str, np.ndarray]


class CollateFunc(Protocol):
    @abstractmethod
    def __call__(self, input: InputFeatures, **kwargs) -> np.ndarray:
        raise NotImplementedError()


class CollateFuncFactory:
    @classmethod
    def get_collate_func(cls, fn_name: str, **kwargs):
        return getattr(cls, fn_name)

    @classmethod
    def cat(cls, input: InputFeatures, cat_order: List[str], **kwargs) -> np.ndarray:
        return np.concatenate([input[key] for key in cat_order])
