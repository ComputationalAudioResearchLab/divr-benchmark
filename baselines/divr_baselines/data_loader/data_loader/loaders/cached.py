import h5py
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Tuple
from .base import Base
from ..dtypes import CacheSet, InputTensors, LabelTensor
from divr_benchmark import Benchmark


class CachedLoader(Base):
    cache_enabled = True
    _points: CacheSet

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
        cache_base_path: Path,
        cache_key: str,
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
            cache_base_path=cache_base_path,
            cache_key=cache_key,
        )

    @torch.no_grad()
    def tv_getitem(self, idx: int) -> Tuple[InputTensors, LabelTensor]:
        idx = self._indices[idx]
        start = idx * self._batch_size
        end = start + self._batch_size
        data = self._points.inputs[start:end]
        shapes = self._points.shapes[start:end]
        labels = self._points.labels[start:end]
        batch_size = len(shapes)
        max_audio_len = np.concatenate(shapes).max()
        max_audios = len(max(shapes, key=len))
        feature = np.zeros((batch_size, max_audios, max_audio_len, self.feature_size))
        feature_lens = np.zeros((batch_size, max_audios))
        for idx, (shape, row) in enumerate(zip(shapes, data)):
            audios_in_session = len(shape)
            audio_len = max(shape)
            data_point = row.reshape(audios_in_session, audio_len, self.feature_size)
            feature[idx, :audios_in_session, :audio_len, :] = data_point
            feature_lens[idx, :audios_in_session] = shape
        feature = torch.tensor(feature, device=self.device, dtype=torch.float32)
        feature_lens = torch.tensor(feature_lens, device=self.device, dtype=torch.long)
        labels = torch.tensor(labels, device=self.device, dtype=torch.long)
        inputs = (feature, feature_lens)
        return (inputs, labels)

    def __load_data(
        self,
        benchmark_path: Path,
        benchmark_version: str,
        stream: int,
        task: int,
        cache_base_path: Path,
        cache_key: str,
    ) -> None:
        cache_path = Path(f"{cache_base_path}/cache/{cache_key}.hdf5")
        if not cache_path.is_file():
            self.__create_cache(
                cache_path=cache_path,
                benchmark_path=benchmark_path,
                benchmark_version=benchmark_version,
                stream=stream,
                task=task,
            )
        self.__load_cache(cache_path=cache_path)

    def __load_cache(self, cache_path):
        print("cache exists")
        cache = h5py.File(cache_path, "r")
        train_len = cache["train"].shape[0]
        self.unique_diagnosis = cache.attrs["unique_diagnosis"]
        self.num_unique_diagnosis = len(self.unique_diagnosis)
        self._train_points = CacheSet(
            inputs=cache["train"],
            shapes=cache["train_shapes"],
            labels=cache["train_labels"],
        )
        self._train_indices = np.arange(train_len // self._batch_size)
        val_len = cache["val"].shape[0]
        self._val_points = CacheSet(
            inputs=cache["val"],
            shapes=cache["val_shapes"],
            labels=cache["val_labels"],
        )
        self._val_indices = np.arange(val_len // self._batch_size)

    def __create_cache(
        self,
        cache_path: Path,
        benchmark_path: Path,
        benchmark_version: str,
        stream: int,
        task: int,
    ) -> None:
        print(f"cache does not exist, creating at {cache_path}")
        cache_path.parent.mkdir(exist_ok=True, parents=True)
        benchmark = Benchmark(
            storage_path=benchmark_path,
            version=benchmark_version,
        )
        btask = benchmark.task(stream=stream, task=task)
        # audio_sample_rate is needed for some features
        self.audio_sample_rate = btask.audio_sample_rate
        cache = h5py.File(cache_path, "w")
        cache.attrs["unique_diagnosis"] = btask.unique_diagnosis
        vfloat = h5py.vlen_dtype(np.float32)
        vint = h5py.vlen_dtype(int)

        def create_dset(key, task_data):
            dset = cache.create_dataset(
                name=key,
                shape=(len(task_data),),
                dtype=vfloat,
            )
            dset_shapes = cache.create_dataset(
                name=f"{key}_shapes",
                shape=(len(task_data)),
                dtype=vint,
            )
            dlabels = cache.create_dataset(
                name=f"{key}_labels",
                shape=(len(task_data)),
                dtype=int,
            )
            for idx, point in enumerate(
                tqdm(task_data, desc=f"caching {key}", leave=True)
            ):
                feature, feature_len = self.collate_function([point.audio])
                feature = feature[0].cpu().numpy()
                feature_len = feature_len[0].cpu().numpy()
                dset_shapes[idx] = feature_len.reshape(-1)
                dset[idx] = feature.reshape(-1)
                dlabels[idx] = btask.diag_to_index(point.label)

        create_dset("train", btask.train)
        create_dset("val", btask.val)
        cache.close()
