from __future__ import annotations
import torch
import numpy as np
from typing import Tuple, List, Union

from .dtypes import AudioBatch, InputTensors, LabelTensor
from .base_data_loader import BaseDataLoader
from ..model.feature import Feature


class DataLoader(BaseDataLoader):

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
        test_only: bool,
        allow_inter_level_comparison: bool,
    ) -> None:
        super().__init__(
            random_seed=random_seed,
            diag_levels=diag_levels,
            task=task,
            batch_size=batch_size,
            test_only=test_only,
            allow_inter_level_comparison=allow_inter_level_comparison,
        )
        self.__train_points = np.array(task.train)
        self.__val_points = np.array(task.val)
        self.__test_points = np.array(task.test)
        self.__train_indices = np.arange(len(self.__train_points))
        self.__test_indices = np.arange(len(self.__test_points))
        self.__val_indices = np.arange(len(self.__val_points))
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
        inputs: InputTensors = self.collate_function([b.audio for b in batch])
        labels = torch.tensor(
            [
                [
                    self.__task.diag_to_index(b.label, level)
                    for level in self.diag_levels
                ]
                for b in batch
            ],
            device=self.__device,
            dtype=torch.long,
        )
        if self.__return_ids:
            ids = [b.id for b in batch]
            return (inputs, labels, ids)
        else:
            return (inputs, labels)

    def __get_batch(self, idx: int):
        start = idx * self.__batch_size
        end = start + self.__batch_size
        batch = self._points[self.__indices[start:end]]
        return batch

    def train(self) -> DataLoader:
        if self.__shuffle_train:
            np.random.shuffle(self.__train_indices)
        self.__indices = self.__train_indices
        self._points = self.__train_points
        self._data_len = self._num_batches(len(self.__indices))
        return self

    def eval(self) -> DataLoader:
        self.__indices = self.__val_indices
        self._points = self.__val_points
        self._data_len = self._num_batches(len(self.__indices))
        return self

    def test(self) -> DataLoader:
        self.__indices = self.__test_indices
        self._points = self.__test_points
        self._data_len = self._num_batches(len(self.__indices))
        return self

    def collate_function(self, batch: AudioBatch) -> InputTensors:
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
            if len(audios) < 1:
                print(batch)
                raise ValueError(f"batch point {batch_idx} without audios")
            for audio_idx, audio in enumerate(audios):
                audio_len = audio.shape[0]
                if audio_len < 1:
                    print(batch)
                    raise ValueError("0 length audio detected")
                audio_tensor[batch_idx, audio_idx, :audio_len] = audio
                audio_lens[batch_idx, audio_idx] = audio_len
        return self.__feature_function((audio_tensor, audio_lens))

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
