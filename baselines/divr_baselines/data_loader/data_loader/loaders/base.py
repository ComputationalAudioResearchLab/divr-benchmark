import torch
import numpy as np
from typing import Any, List
from ..dtypes import AudioBatch, InputTensors


class Base:
    cache_enabled: bool
    feature_function: Any
    tv_getitem: Any
    _shuffle_train: bool
    _train_points: List
    _val_points: List
    _train_indices: np.ndarray
    _val_indices: np.ndarray
    _batch_size: int
    _points: List

    def __len__(self) -> int:
        return self._data_len

    @torch.no_grad()
    def __getitem__(self, idx: int):
        return self._getitem(idx)

    def train(self):
        if self._shuffle_train:
            np.random.shuffle(self._train_indices)
        self._indices = self._train_indices
        self._points = self._train_points
        self._data_len = len(self._indices)
        self._getitem = self.tv_getitem

    def eval(self):
        self._indices = self._val_indices
        self._points = self._val_points
        self._data_len = len(self._indices)
        self._getitem = self.tv_getitem

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

        return self.feature_function((audio_tensor, audio_lens))
