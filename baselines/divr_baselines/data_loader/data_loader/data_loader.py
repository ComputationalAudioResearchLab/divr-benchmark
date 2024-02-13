from __future__ import annotations
import torch
from pathlib import Path
from .loaders import BatchAheadLoader, CachedLoader, NormalLoader, LoaderTypes
from .dtypes import InputArrays, InputTensors


class DataLoader:
    feature_size: int

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
        loader_type: LoaderTypes,
        cache_base_path: Path,
        cache_key: str,
    ) -> None:
        self.device = device

        if loader_type == LoaderTypes.CACHED:
            self.__impl = CachedLoader(
                benchmark_path=benchmark_path,
                benchmark_version=benchmark_version,
                stream=stream,
                task=task,
                device=device,
                batch_size=batch_size,
                random_seed=random_seed,
                shuffle_train=shuffle_train,
                cache_base_path=cache_base_path,
                cache_key=cache_key,
                feature_init=self.feature_init,
                feature_function=self.feature_function,
                feature_size=self.feature_size,
            )
        elif loader_type == LoaderTypes.NORMAL:
            self.__impl = NormalLoader(
                benchmark_path=benchmark_path,
                benchmark_version=benchmark_version,
                stream=stream,
                task=task,
                device=device,
                batch_size=batch_size,
                random_seed=random_seed,
                shuffle_train=shuffle_train,
                feature_init=self.feature_init,
                feature_function=self.feature_function,
                feature_size=self.feature_size,
            )
            self.audio_sample_rate = self.__impl.audio_sample_rate
        elif loader_type == LoaderTypes.BATCH_AHEAD:
            self.__impl = BatchAheadLoader(
                benchmark_path=benchmark_path,
                benchmark_version=benchmark_version,
                stream=stream,
                task=task,
                device=device,
                batch_size=batch_size,
                random_seed=random_seed,
                shuffle_train=shuffle_train,
                feature_init=self.feature_init,
                feature_size=self.feature_size,
            )
            self.audio_sample_rate = self.__impl.audio_sample_rate
        else:
            raise ValueError(f"Invalid loader_type {loader_type} selected")
        self.unique_diagnosis = self.__impl.unique_diagnosis
        self.num_unique_diagnosis = self.__impl.num_unique_diagnosis

    def __len__(self) -> int:
        return self.__impl.__len__()

    @torch.no_grad()
    def __getitem__(self, idx: int):
        return self.__impl.__getitem__(idx)

    def train(self) -> DataLoader:
        self.__impl.train()
        return self

    def eval(self) -> DataLoader:
        self.__impl.eval()
        return self

    def test(self) -> DataLoader:
        self.__impl.test()
        return self

    def feature_init(self) -> None:
        pass

    def feature_function(self, batch: InputArrays) -> InputTensors:
        raise NotImplementedError()
