from __future__ import annotations
import torch
import shelve
import numpy as np
from tqdm import tqdm
from pathlib import Path
from divr_diagnosis import Diagnosis
from typing import Tuple, List, Union, TypedDict

from .dtypes import AudioBatch, InputTensors, LabelTensor
from .base_data_loader import BaseDataLoader
from ..model.feature import Feature
from .extra_db import ExtraDB


class CachePoint(TypedDict):
    id: str
    features: List[np.ndarray]
    label: Diagnosis


class CachedDataLoader(BaseDataLoader):

    __cache_key_train = "train"
    __cache_key_val = "val"
    __cache_key_test = "test"

    def __init__(
        self,
        random_seed: int,
        shuffle_train: bool,
        batch_size: int,
        device: torch.device,
        diag_levels: List[int],
        task,
        feature_function: Feature | None,
        return_ids: bool,
        cache_path: Path,
        extra_db: ExtraDB,
        percent_injection: int,
    ) -> None:
        super().__init__(
            random_seed=random_seed,
            diag_levels=diag_levels,
            task=task,
            batch_size=batch_size,
            extra_db=extra_db,
            percent_injection=percent_injection,
        )
        cache_path.mkdir(parents=True, exist_ok=True)
        self.__cache = shelve.open(str(cache_path))
        self.__diag_levels = diag_levels
        self.__batch_size = batch_size
        self.__task = task
        self.__device = device
        if feature_function is None:
            self.__feature_function = self.__noop_feature_function
            self.feature_size = 1
        else:
            self.__feature_function = feature_function
            self.feature_size = feature_function.feature_size
        self.__shuffle_train = shuffle_train
        self.__return_ids = return_ids
        self.__create_cache(task.train, self.__cache_key_train)
        self.__create_cache(task.val, self.__cache_key_val)
        self.__create_cache(task.test, self.__cache_key_test)
        self.__train_points = self.__prepare_points_for_indexing(task.train)
        self.__val_points = self.__prepare_points_for_indexing(task.val)
        self.__test_points = self.__prepare_points_for_indexing(task.test)

    def __prepare_points_for_indexing(self, points: list) -> np.ndarray:
        return np.array([{"id": point.id, "label": point.label} for point in points])

    def __del__(self):
        self.__cache.close()

    def __len__(self) -> int:
        return self._data_len

    @torch.no_grad()
    def __getitem__(self, idx: int) -> Union[
        Tuple[InputTensors, LabelTensor, List[str]],
        Tuple[InputTensors, LabelTensor],
    ]:
        if idx >= self._data_len:
            raise StopIteration()
        batch = self.__get_batch(idx)
        inputs: InputTensors = self.collate_function([b["features"] for b in batch])
        labels = torch.tensor(
            [
                [
                    self.__task.diag_to_index(b["label"], level)
                    for level in self.__diag_levels
                ]
                for b in batch
            ],
            device=self.__device,
            dtype=torch.long,
        )
        if self.__return_ids:
            ids = [b["id"] for b in batch]
            return (inputs, labels, ids)
        else:
            return (inputs, labels)

    def __get_batch(self, idx: int) -> List[CachePoint]:
        start = idx * self.__batch_size
        end = start + self.__batch_size
        return [
            {
                "features": self.__cache[f"{self.__cache_key}:{point['id']}"],
                "id": point["id"],
                "label": point["label"],
            }
            for point in self.__points[start:end]
        ]

    def train(self) -> CachedDataLoader:
        if self.__shuffle_train:
            np.random.shuffle(self.__train_points)
        self.__points = self.__train_points
        self.__cache_key = self.__cache_key_train
        self._data_len = self._num_batches(len(self.__points))
        return self

    def eval(self) -> CachedDataLoader:
        self.__points = self.__val_points
        self.__cache_key = self.__cache_key_val
        self._data_len = self._num_batches(len(self.__points))
        return self

    def test(self) -> CachedDataLoader:
        self.__points = self.__test_points
        self.__cache_key = self.__cache_key_test
        self._data_len = self._num_batches(len(self.__points))
        return self

    def collate_function(self, batch: AudioBatch) -> InputTensors:
        batch_len = len(batch)
        max_num_features = 0
        max_feature_len = 0
        max_feature_size = 0
        for features in batch:
            len_features = len(features)
            if len_features > max_num_features:
                max_num_features = len_features
            for feature in features:
                feature_len, feature_size = feature.shape
                if feature_len > max_feature_len:
                    max_feature_len = feature_len
                if feature_size > max_feature_size:
                    max_feature_size = feature_size
        feature_tensor = np.zeros(
            (batch_len, max_num_features, max_feature_len, max_feature_size)
        )
        feature_lens = np.zeros((batch_len, max_num_features), dtype=int)
        for batch_idx, features in enumerate(batch):
            for feature_idx, feature in enumerate(features):
                feature_len, feature_size = feature.shape
                feature_tensor[
                    batch_idx,
                    feature_idx,
                    :feature_len,
                    :feature_size,
                ] = feature
                feature_lens[batch_idx, feature_idx] = feature_len
        feature_tensor = torch.tensor(
            feature_tensor,
            dtype=torch.float,
            device=self.__device,
        )
        feature_lens = torch.tensor(
            feature_lens,
            dtype=torch.long,
            device=self.__device,
        )
        return (feature_tensor, feature_lens)

    def __noop_feature_function(self, batch) -> InputTensors:
        batch_inputs, batch_lens = batch
        audio_tensor = torch.tensor(
            batch_inputs,
            dtype=torch.float32,
            device=self.__device,
        )
        audio_lens = torch.tensor(
            batch_lens,
            dtype=torch.long,
            device=self.__device,
        )
        return (audio_tensor, audio_lens)

    def __create_cache(self, points, cache_key: str) -> None:
        for i, point in tqdm(
            enumerate(points),
            total=len(points),
            desc=f"caching {cache_key}",
            leave=False,
        ):
            key = f"{cache_key}:{point.id}"
            if key not in self.__cache:
                self.__cache[key] = [
                    self.__feature_function.individual_np(audio)
                    for audio in point.audio
                ]
