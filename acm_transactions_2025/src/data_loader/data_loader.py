from __future__ import annotations
import torch
import numpy as np
from typing import Tuple, List, Union

from .dtypes import AudioBatch, InputTensors, LabelTensor


class DataLoader:

    def __init__(
        self,
        random_seed: int,
        shuffle_train: bool,
        batch_size: int,
        device: torch.device,
        diag_levels: List[int],
        task,
        feature_function,
        return_ids: bool,
    ) -> None:
        np.random.seed(random_seed)
        self.__train_points = task.train
        self.__val_points = task.val
        self.__test_points = task.test
        self.unique_diagnosis = {}
        self.num_unique_diagnosis = {}
        self.train_class_weights = {}
        self.levels_map = {0: [[0, 1], [2, 3]], 1: [[0], [1], [2], [3]]}
        self.levels_map = {}
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
        self.__train_indices = np.arange(len(self.__train_points) // batch_size)
        self.__test_indices = np.arange(len(self.__test_points) // batch_size)
        self.__val_indices = np.arange(len(self.__val_points) // batch_size)
        self.__diag_levels = diag_levels
        self.max_diag_level = max_diag_level
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

    def idx_to_diag_name(self, idx: int, level: int) -> str:
        return self.__task.index_to_diag(idx, level).name

    def __len__(self) -> int:
        return self._data_len

    @torch.no_grad()
    def __getitem__(self, idx: int) -> Union[
        Tuple[InputTensors, LabelTensor, List[str]],
        Tuple[InputTensors, LabelTensor],
    ]:
        batch = self.__get_batch(idx)
        inputs: InputTensors = self.collate_function([b.audio for b in batch])
        labels = torch.tensor(
            [
                [
                    self.__task.diag_to_index(b.label, level)
                    for level in self.__diag_levels
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
        idx = self._indices[idx]
        start = idx * self.__batch_size
        end = start + self.__batch_size
        batch = self._points[start:end]
        return batch

    def train(self) -> DataLoader:
        if self.__shuffle_train:
            np.random.shuffle(self.__train_indices)
        self._indices = self.__train_indices
        self._points = self.__train_points
        self._data_len = len(self._indices)
        return self

    def eval(self) -> DataLoader:
        self._indices = self.__val_indices
        self._points = self.__val_points
        self._data_len = len(self._indices)
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
            for audio_idx, audio in enumerate(audios):
                audio_len = audio.shape[0]
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
