from __future__ import annotations
import math
import numpy as np
import torch
from typing import Dict, Tuple, List, Union, Iterable

from .dtypes import AudioBatch, InputTensors, LabelTensor

DataPoint = Union[
    Tuple[InputTensors, LabelTensor, List[str]],
    Tuple[InputTensors, LabelTensor],
]


class BaseDataLoader:
    feature_size: int
    unique_diagnosis: Dict[int, List[str]]
    num_unique_diagnosis: Dict[int, int]
    levels_map: Dict[int, List[List[int]]]
    train_class_weights: Dict[int, int]

    def __init__(
        self,
        random_seed: int,
        diag_levels: List[int],
        batch_size: int,
        task,
    ) -> None:
        self.unique_diagnosis = {}
        self.num_unique_diagnosis = {}
        self.levels_map = {}
        self.train_class_weights = {}
        np.random.seed(random_seed)
        max_diag_level = max(diag_levels)
        max_level_diags = task.unique_diagnosis()
        for diag_level in diag_levels:
            cur_level_diags = task.unique_diagnosis(level=diag_level)
            level_map = [[] for _ in range(len(cur_level_diags))]
            for diag_name in max_level_diags:
                diag_idx = task.diag_name_to_index(diag_name, max_diag_level)
                diag = task.index_to_diag(diag_idx, max_diag_level)
                diag_at_level = diag.at_level(diag_level)
                cur_diag_idx = task.diag_to_index(diag_at_level, diag_level)
                level_map[cur_diag_idx] += [diag_idx]
            self.levels_map[diag_level] = level_map
            self.unique_diagnosis[diag_level] = cur_level_diags
            self.num_unique_diagnosis[diag_level] = len(cur_level_diags)
            self.train_class_weights[diag_level] = task.train_class_weights(
                level=diag_level
            )
        self.max_diag_level = max_diag_level
        self.__batch_size = batch_size
        self.__task = task

    @property
    def num_classes(self):
        return self.num_unique_diagnosis[self.max_diag_level]

    def idx_to_diag_name(self, idx: int, level: int) -> str:
        return self.__task.index_to_diag(idx, level).name

    def __len__(self) -> int:
        raise NotImplementedError()

    @torch.no_grad()
    def __getitem__(self, idx: int) -> DataPoint:
        raise NotImplementedError()

    def train(self) -> Iterable[DataPoint]:
        raise NotImplementedError()

    def eval(self) -> Iterable[DataPoint]:
        raise NotImplementedError()

    def test(self) -> Iterable[DataPoint]:
        raise NotImplementedError()

    def collate_function(self, batch: AudioBatch) -> InputTensors:
        raise NotImplementedError()

    def _num_batches(self, num_points: int) -> int:
        return math.ceil((num_points / self.__batch_size))
