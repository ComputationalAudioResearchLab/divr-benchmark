import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Union
from .base import Base
from ..dtypes import InputTensors, LabelTensor
from divr_benchmark import Benchmark, TrainPoint, TestPoint


class NormalLoader(Base):
    cache_enabled = False

    def __init__(
        self,
        benchmark_path: Path,
        benchmark_version: str,
        stream: int,
        task: int,
        device: torch.device,
        batch_size: int,
        random_seed: int,
        shuffle_train: bool,
        feature_init,
        feature_function,
        feature_size,
    ) -> None:
        np.random.seed(random_seed)
        self.feature_init = feature_init
        self.feature_function = feature_function
        self.feature_size = feature_size
        self.device = device
        self.feature_init()
        self._batch_size = batch_size
        self._shuffle_train = shuffle_train
        self.__load_data(
            benchmark_path=benchmark_path,
            benchmark_version=benchmark_version,
            stream=stream,
            task=task,
        )

    def test(self):
        self._indices = self._test_indices
        self._points = self._test_points
        self._data_len = len(self._points) // self._batch_size
        self._getitem = self.test_getitem

    @torch.no_grad()
    def tv_getitem(self, idx: int) -> Tuple[InputTensors, LabelTensor]:
        batch: List[TrainPoint] = self.__get_batch(idx)
        inputs: InputTensors = self.collate_function([b.audio for b in batch])
        labels: LabelTensor = torch.tensor(
            [self.__task.diag_to_index(b.label) for b in batch],
            device=self.device,
            dtype=torch.long,
        )
        return (inputs, labels)

    @torch.no_grad()
    def test_getitem(self, idx) -> InputTensors:
        batch: List[TestPoint] = self.__get_batch(idx)
        inputs: InputTensors = self.collate_function([b.audio for b in batch])
        return inputs

    def __get_batch(self, idx: int) -> List[Union[TrainPoint, TestPoint]]:
        idx = self._indices[idx]
        start = idx * self._batch_size
        end = start + self._batch_size
        batch = self._points[start:end]
        return batch

    def __load_data(
        self,
        benchmark_path: Path,
        benchmark_version: str,
        stream: int,
        task: int,
    ) -> None:
        benchmark = Benchmark(
            storage_path=benchmark_path,
            version=benchmark_version,
        )
        btask = benchmark.task(stream=stream, task=task)
        self.audio_sample_rate = btask.audio_sample_rate
        self.unique_diagnosis = btask.unique_diagnosis
        self.num_unique_diagnosis = len(self.unique_diagnosis)
        self.__task = btask
        self._train_points = btask.train
        self._test_points = btask.test
        self._val_points = btask.val
        self._train_indices = np.arange(len(self._train_points) // self._batch_size)
        self._test_indices = np.arange(len(self._test_points) // self._batch_size)
        self._val_indices = np.arange(len(self._val_points) // self._batch_size)
