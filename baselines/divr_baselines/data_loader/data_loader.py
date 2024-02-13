from __future__ import annotations
from dataclasses import dataclass
import h5py
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple, Union
from divr_benchmark import Benchmark, Task, TestPoint, TrainPoint

AudioBatch = List[List[np.ndarray]]
"""
AudioBatch = collection of audios in a given batch

[
    [
        session_audio_1,
        session_audio_2,
        session_audio_3,
        ... for session
    ]
    ... for batch
]
"""

InputArrays = Tuple[np.ndarray, np.ndarray]
"""
InputArrays[0]
    - contains audio data
    - shape = [B, F, S]
---
InputArrays[1]
    - contains length of each audio in batch
    - shape = [B, F]

---
- B = Batch size
- F = Number of files in a session
- S = Sequence length
---
"""

InputTensors = Tuple[torch.Tensor, torch.Tensor]
"""
InputTensors[0]
    - contains feature data
    - type = torch.FloatTensor
    - shape = [B, F, S, H]
---
InputTensors[1]
    - contains length of each audio in batch
    - type = torch.LongTensor
    - shape = [B, F]

---
- B = Batch size
- F = Number of files in a session
- S = Sequence length
- H = Feature length
---
"""

LabelTensor = torch.Tensor
"""
LabelTensor
    - type = torch.LongTensor
    - shape = [B]
---
 - B = Batch size
"""


@dataclass
class CacheSet:
    inputs: h5py.Dataset
    shapes: h5py.Dataset
    labels: h5py.Dataset


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
        cache_enabled: bool,
        cache_base_path: Path,
        cache_key: str,
    ) -> None:
        np.random.seed(random_seed)
        self.device = device
        self.feature_init()
        self.__cache_enabled = cache_enabled
        self.__batch_size = batch_size
        self.__shuffle_train = shuffle_train
        self.__load_data(
            benchmark_path=benchmark_path,
            benchmark_version=benchmark_version,
            stream=stream,
            task=task,
            cache_base_path=cache_base_path,
            cache_key=cache_key,
        )

    def __len__(self) -> int:
        return self.__data_len

    @torch.no_grad()
    def __getitem__(self, idx: int):
        return self.__getitem(idx)

    def train(self) -> DataLoader:
        if self.__shuffle_train:
            np.random.shuffle(self.__train_indices)
        self.__indices = self.__train_indices
        self.__points = self.__train_points
        self.__data_len = len(self.__indices)
        if self.__cache_enabled:
            self.__getitem = self.__tv_getitem_cached
        else:
            self.__getitem = self.__tv_getitem
        return self

    def eval(self) -> DataLoader:
        self.__indices = self.__val_indices
        self.__points = self.__val_points
        self.__data_len = len(self.__indices)
        if self.__cache_enabled:
            self.__getitem = self.__tv_getitem_cached
        else:
            self.__getitem = self.__tv_getitem
        return self

    def test(self) -> DataLoader:
        self.__indices = self.__test_indices
        self.__points = self.__test_points
        self.__data_len = len(self.__points) // self.__batch_size
        self.__getitem = self.__test_getitem
        return self

    def feature_init(self) -> None:
        pass

    def feature_function(self, batch: InputArrays) -> InputTensors:
        raise NotImplementedError()

    @torch.no_grad()
    def __tv_getitem_cached(self, idx: int) -> Tuple[InputTensors, LabelTensor]:
        idx = self.__indices[idx]
        start = idx * self.__batch_size
        end = start + self.__batch_size
        data = self.__points.inputs[start:end]
        shapes = self.__points.shapes[start:end]
        labels = self.__points.labels[start:end]
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

    @torch.no_grad()
    def __tv_getitem(self, idx: int) -> Tuple[InputTensors, LabelTensor]:
        batch: List[TrainPoint] = self.__get_batch(idx)
        inputs: InputTensors = self.__collate_function([b.audio for b in batch])
        labels: LabelTensor = torch.tensor(
            [self.__task.diag_to_index(b.label) for b in batch],
            device=self.device,
            dtype=torch.long,
        )
        return (inputs, labels)

    @torch.no_grad()
    def __test_getitem(self, idx) -> InputTensors:
        batch: List[TestPoint] = self.__get_batch(idx)
        inputs: InputTensors = self.__collate_function([b.audio for b in batch])
        return inputs

    def __collate_function(self, batch: AudioBatch) -> InputTensors:
        batch_len = len(batch)
        max_num_audios = 0
        max_audio_len = 0
        for audios in batch:
            len_audios = len(audios)
            if len_audios > max_num_audios:
                max_num_audios = len_audios
            for audio in audios:
                audio_len = audio.shape[0]
                if audio_len > max_audio_len:
                    max_audio_len = audio_len
        audio_tensor = np.zeros((batch_len, max_num_audios, max_audio_len))
        audio_lens = np.zeros((batch_len, max_num_audios), dtype=int)
        for batch_idx, audios in enumerate(batch):
            for audio_idx, audio in enumerate(audios):
                audio_len = audio.shape[0]
                audio_tensor[batch_idx, audio_idx, :audio_len] = audio
                audio_lens[batch_idx, audio_idx] = audio_len

        return self.feature_function((audio_tensor, audio_lens))

    def __get_batch(self, idx: int) -> List[Union[TrainPoint, TestPoint]]:
        idx = self.__indices[idx]
        start = idx * self.__batch_size
        end = start + self.__batch_size
        batch = self.__points[start:end]
        return batch

    def __load_data(
        self,
        benchmark_path: Path,
        benchmark_version: str,
        stream: int,
        task: int,
        cache_base_path: Path,
        cache_key: str,
    ) -> None:
        if not self.__cache_enabled:
            btask = self.__load_task(
                benchmark_path=benchmark_path,
                benchmark_version=benchmark_version,
                stream=stream,
                task=task,
            )
            self.audio_sample_rate = btask.audio_sample_rate
            self.unique_diagnosis = btask.unique_diagnosis
            self.num_unique_diagnosis = len(self.unique_diagnosis)
            self.__task = btask
            self.__train_points = btask.train
            self.__test_points = btask.test
            self.__val_points = btask.val
            self.__train_indices = np.arange(
                len(self.__train_points) // self.__batch_size
            )
            self.__test_indices = np.arange(
                len(self.__test_points) // self.__batch_size
            )
            self.__val_indices = np.arange(len(self.__val_points) // self.__batch_size)
        else:
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
        self.__train_points = CacheSet(
            inputs=cache["train"],
            shapes=cache["train_shapes"],
            labels=cache["train_labels"],
        )
        self.__train_indices = np.arange(train_len // self.__batch_size)
        val_len = cache["val"].shape[0]
        self.__val_points = CacheSet(
            inputs=cache["val"],
            shapes=cache["val_shapes"],
            labels=cache["val_labels"],
        )
        self.__val_indices = np.arange(val_len // self.__batch_size)

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
        btask = self.__load_task(
            benchmark_path=benchmark_path,
            benchmark_version=benchmark_version,
            stream=stream,
            task=task,
        )
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
                feature, feature_len = self.__collate_function([point.audio])
                feature = feature[0].cpu().numpy()
                feature_len = feature_len[0].cpu().numpy()
                dset_shapes[idx] = feature_len.reshape(-1)
                dset[idx] = feature.reshape(-1)
                dlabels[idx] = btask.diag_to_index(point.label)

        create_dset("train", btask.train)
        create_dset("val", btask.val)
        cache.close()

    def __load_task(
        self,
        benchmark_path: Path,
        benchmark_version: str,
        stream: int,
        task: int,
    ) -> Task:
        benchmark = Benchmark(
            storage_path=benchmark_path,
            version=benchmark_version,
        )
        return benchmark.task(stream=stream, task=task)
