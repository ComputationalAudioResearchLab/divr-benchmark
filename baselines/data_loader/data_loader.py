from __future__ import annotations
import torch
import numpy as np
from abc import ABC
from typing import List, Tuple, Union
from divr_benchmark import Task, TestPoint, TrainPoint

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


class DataLoader(ABC):
    def __init__(
        self,
        task: Task,
        device: torch.device,
        batch_size: int,
        random_seed: int,
        shuffle_train: bool,
    ) -> None:
        np.random.seed(random_seed)
        self.audio_sample_rate = task.audio_sample_rate
        self.device = device
        self.__task = task
        self.__train_points = task.train
        self.__train_indices = np.arange(len(self.__train_points) // batch_size)
        self.__test_points = task.test
        self.__test_indices = np.arange(len(self.__test_points) // batch_size)
        self.__val_points = task.val
        self.__val_indices = np.arange(len(self.__val_points) // batch_size)
        self.__batch_size = batch_size
        self.__shuffle_train = shuffle_train
        self.feature_init()

    def __len__(self) -> int:
        return self.__data_len

    def __getitem__(self, idx: int):
        return self.__getitem(idx)

    def train(self) -> DataLoader:
        if self.__shuffle_train:
            np.random.shuffle(self.__train_indices)
        self.__indices = self.__train_indices
        self.__points = self.__train_points
        self.__data_len = len(self.__points) // self.__batch_size
        self.__getitem = self.__tv_getitem
        return self

    def val(self) -> DataLoader:
        self.__indices = self.__val_indices
        self.__points = self.__val_points
        self.__data_len = len(self.__points) // self.__batch_size
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

    def __tv_getitem(self, idx: int) -> Tuple[InputTensors, LabelTensor]:
        batch: List[TrainPoint] = self.__get_batch(idx)
        inputs: InputTensors = self.__collate_function([b.audio for b in batch])
        labels: LabelTensor = torch.LongTensor(
            [self.__task.diag_to_index(b.label) for b in batch]
        ).to(self.device)
        return (inputs, labels)

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
