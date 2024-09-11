from __future__ import annotations
from typing import List, Tuple
import collections.abc
import torch
import librosa
import numpy as np
from torch import nn
from pathlib import Path
from .dtypes import Database, DataPoint, Dataset, AudioBatch, InputTensors


class DataLoader(collections.abc.Sequence[DataPoint]):
    unique_diagnosis: List[str]
    class_counts: torch.LongTensor
    """
    total count of a classes in training data
    """

    def __init__(
        self,
        data_root   : Path,
        sample_rate: int,
        device: torch.device,
        batch_size: int,
        random_seed: int,
        shuffle_train: bool,
        database: Database,
        max_audio_samples: int|None = None,
    ) -> None:
        np.random.seed(random_seed)
        self.__sample_rate = sample_rate
        self.__device = device
        self.__batch_size = batch_size
        self.__shuffle_train = shuffle_train
        self.__load_database(data_root=data_root, database=database, max_audio_samples=max_audio_samples)

    def __len__(self) -> int:
        return self.__data_len

    @torch.no_grad()
    def __getitem__(self, idx: int) -> DataPoint:
        idx = self.__indices[idx]
        start = idx * self.__batch_size
        end = start + self.__batch_size
        inputs = self.__inputs[start:end]
        inputs = self.__collate_function(inputs)
        labels = self.__labels[start:end]
        return (inputs, labels)

    def train(self) -> DataLoader:
        if self.__shuffle_train:
            np.random.shuffle(self.__train_indices)
        self.__indices = self.__train_indices
        self.__inputs = self.__train_inputs
        self.__labels = self.__train_labels
        self.__data_len = len(self.__indices)
        return self

    def eval(self) -> DataLoader:
        self.__indices = self.__val_indices
        self.__inputs = self.__val_inputs
        self.__labels = self.__val_labels
        self.__data_len = len(self.__indices)
        return self

    def test(self) -> DataLoader:
        self.__indices = self.__test_indices
        self.__inputs = self.__test_inputs
        self.__labels = self.__test_labels
        self.__data_len = len(self.__indices)
        return self

    def __collate_function(self, batch: AudioBatch) -> InputTensors:
        batch_len = len(batch)
        max_audio_len = 0
        for audio in batch:
            audio_len = audio.shape[0]
            if audio_len > max_audio_len:
                max_audio_len = audio_len
        audio_tensor = np.zeros((batch_len, max_audio_len))
        audio_lens = np.zeros((batch_len), dtype=int)
        for batch_idx, audio in enumerate(batch):
            audio_len = audio.shape[0]
            audio_tensor[batch_idx, :audio_len] = audio
            audio_lens[batch_idx] = audio_len
        audio_tensor = torch.tensor(
            audio_tensor, device=self.__device, dtype=torch.float32
        )
        audio_lens = torch.tensor(audio_lens, device=self.__device, dtype=torch.long)
        return (audio_tensor, audio_lens)

    def __load_database(self, data_root: Path, database: Database, max_audio_samples: int|None) -> None:
        train, val, test = database
        self.__train_indices, self.__train_inputs, train_labels = self.__load_dataset(
            data_root,
            train,
            max_audio_samples
        )
        self.__val_indices, self.__val_inputs, val_labels = self.__load_dataset(
            data_root,
            val,
            max_audio_samples
        )
        self.__test_indices, self.__test_inputs, test_labels = self.__load_dataset(
            data_root,
            test,
            max_audio_samples
        )
        self.__transform_labels(
            train_labels=train_labels,
            val_labels=val_labels,
            test_labels=test_labels,
        )
        self.class_counts = self.__train_labels.unique(return_counts=True)[1]

    def __load_dataset(
        self, data_root: Path, dataset: Dataset, max_audio_samples: int|None
    ) -> Tuple[List[int], AudioBatch, List[str]]:
        total_data = len(dataset)
        indices = list(range(total_data // self.__batch_size))
        audios: AudioBatch = []
        labels: List[str] = []
        for _, _, label, audio_file_path in dataset:
            audios += [self.__load_audio(f"{data_root}/{audio_file_path}", max_audio_samples)]
            labels += [label]
        return (indices, audios, labels)

    def __transform_labels(
        self, train_labels: List[str], val_labels: List[str], test_labels: List[str]
    ) -> None:
        unique_labels = set(train_labels + val_labels + test_labels)
        self.__labels_map = dict(enumerate(sorted(unique_labels, key=lambda x: x)))
        self.__labels_map_reversed = dict(
            [(v, k) for k, v in self.__labels_map.items()]
        )
        self.unique_diagnosis = list(self.__labels_map.values())
        self.__train_labels = torch.tensor(
            [self.__labels_map_reversed[x] for x in train_labels],
            device=self.__device,
            dtype=torch.long,
        )
        self.__val_labels = torch.tensor(
            [self.__labels_map_reversed[x] for x in val_labels],
            device=self.__device,
            dtype=torch.long,
        )
        self.__test_labels = torch.tensor(
            [self.__labels_map_reversed[x] for x in test_labels],
            device=self.__device,
            dtype=torch.long,
        )

    def __load_audio(self, audio_file_path: str, max_audio_samples: int|None) -> np.ndarray:
        audio, _ = librosa.load(path=audio_file_path, sr=self.__sample_rate)
        if max_audio_samples is not None:
            audio = audio[:max_audio_samples]
        return self.__normalize(audio)

    def __normalize(self, audio: np.ndarray) -> np.ndarray:
        mean = np.mean(audio, axis=0)
        stddev = np.std(audio, axis=0)
        normalized_audio = (audio - mean) / stddev
        return normalized_audio


class DataLoaderWithFeature(DataLoader):

    def __init__(
        self,
        data_root: Path,
        feature: nn.Module,
        sample_rate: int,
        device: torch.device,
        batch_size: int,
        random_seed: int,
        shuffle_train: bool,
        database: Database,
        max_audio_samples: int|None = None,
    ) -> None:
        super().__init__(
            data_root=data_root,
            sample_rate=sample_rate,
            device=device,
            batch_size=batch_size,
            random_seed=random_seed,
            shuffle_train=shuffle_train,
            database=database,
            max_audio_samples=max_audio_samples
        )
        self.feature = feature

    @torch.no_grad()
    def __getitem__(self, idx: int) -> DataPoint:
        inputs, labels = super().__getitem__(idx=idx)
        inputs = self.feature(inputs)
        return (inputs, labels)



class RandomAudioDataLoader(collections.abc.Sequence[DataPoint]):
    unique_diagnosis: List[str]
    class_counts: torch.LongTensor
    """
    total count of a classes in training data
    """

    def __init__(
        self,
        data_root: Path,
        sample_rate: int,
        device: torch.device,
        batch_size: int,
        random_seed: int,
        shuffle_train: bool,
        database: Database,
        max_audio_samples: int,
    ) -> None:
        np.random.seed(random_seed)
        self.__sample_rate = sample_rate
        self.__device = device
        self.__batch_size = batch_size
        self.__shuffle_train = shuffle_train
        self.__max_audio_samples = max_audio_samples
        self.__load_database(data_root=data_root, database=database)

    def __len__(self) -> int:
        return self.__data_len

    @torch.no_grad()
    def __getitem__(self, idx: int) -> DataPoint:
        idx = self.__indices[idx]
        start = idx * self.__batch_size
        end = start + self.__batch_size
        inputs = self.__inputs[start:end]
        inputs = self.__collate_function(inputs)
        labels = self.__labels[start:end]
        return (inputs, labels)

    def train(self) -> DataLoader:
        if self.__shuffle_train:
            np.random.shuffle(self.__train_indices)
        self.__indices = self.__train_indices
        self.__inputs = self.__train_inputs
        self.__labels = self.__train_labels
        self.__data_len = len(self.__indices)
        return self

    def eval(self) -> DataLoader:
        self.__indices = self.__val_indices
        self.__inputs = self.__val_inputs
        self.__labels = self.__val_labels
        self.__data_len = len(self.__indices)
        return self

    def test(self) -> DataLoader:
        self.__indices = self.__test_indices
        self.__inputs = self.__test_inputs
        self.__labels = self.__test_labels
        self.__data_len = len(self.__indices)
        return self

    def __collate_function(self, batch: AudioBatch) -> InputTensors:
        batch_len = len(batch)
        audio_tensor = np.zeros((batch_len, self.__max_audio_samples))
        audio_lens = np.zeros((batch_len), dtype=int)
        for batch_idx, audio in enumerate(batch):
            audio_len = audio.shape[0]
            random_start = np.random.randint(0, max(1, audio_len - self.__max_audio_samples))
            audio_len = min(audio_len, self.__max_audio_samples)
            audio_tensor[batch_idx, :audio_len] = audio[random_start:random_start+audio_len]
            audio_lens[batch_idx] = audio_len
        audio_tensor = torch.tensor(
            audio_tensor, device=self.__device, dtype=torch.float32
        )
        audio_lens = torch.tensor(audio_lens, device=self.__device, dtype=torch.long)
        return (audio_tensor, audio_lens)

    def __load_database(self, data_root: Path, database: Database) -> None:
        train, val, test = database
        self.__train_indices, self.__train_inputs, train_labels = self.__load_dataset(
            data_root,
            train,
        )
        self.__val_indices, self.__val_inputs, val_labels = self.__load_dataset(
            data_root,
            val,
        )
        self.__test_indices, self.__test_inputs, test_labels = self.__load_dataset(
            data_root,
            test,
        )
        self.__transform_labels(
            train_labels=train_labels,
            val_labels=val_labels,
            test_labels=test_labels,
        )
        self.class_counts = self.__train_labels.unique(return_counts=True)[1]

    def __load_dataset(self, data_root: Path, dataset: Dataset) -> Tuple[List[int], AudioBatch, List[str]]:
        total_data = len(dataset)
        indices = list(range(total_data // self.__batch_size))
        audios: AudioBatch = []
        labels: List[str] = []
        for _, _, label, audio_file_path in dataset:
            audios += [self.__load_audio(f"{data_root}/{audio_file_path}")]
            labels += [label]
        return (indices, audios, labels)

    def __transform_labels(
        self, train_labels: List[str], val_labels: List[str], test_labels: List[str]
    ) -> None:
        unique_labels = set(train_labels + val_labels + test_labels)
        self.__labels_map = dict(enumerate(sorted(unique_labels, key=lambda x: x)))
        self.__labels_map_reversed = dict(
            [(v, k) for k, v in self.__labels_map.items()]
        )
        self.unique_diagnosis = list(self.__labels_map.values())
        self.__train_labels = torch.tensor(
            [self.__labels_map_reversed[x] for x in train_labels],
            device=self.__device,
            dtype=torch.long,
        )
        self.__val_labels = torch.tensor(
            [self.__labels_map_reversed[x] for x in val_labels],
            device=self.__device,
            dtype=torch.long,
        )
        self.__test_labels = torch.tensor(
            [self.__labels_map_reversed[x] for x in test_labels],
            device=self.__device,
            dtype=torch.long,
        )

    def __load_audio(self, audio_file_path: str) -> np.ndarray:
        audio, _ = librosa.load(path=audio_file_path, sr=self.__sample_rate)
        return self.__normalize(audio)

    def __normalize(self, audio: np.ndarray) -> np.ndarray:
        mean = np.mean(audio, axis=0)
        stddev = np.std(audio, axis=0)
        normalized_audio = (audio - mean) / stddev
        return normalized_audio


class RandomAudioDataLoaderWithFeature(RandomAudioDataLoader):

    def __init__(
        self,
        data_root: Path,
        feature: nn.Module,
        sample_rate: int,
        device: torch.device,
        batch_size: int,
        random_seed: int,
        shuffle_train: bool,
        database: Database,
        max_audio_samples: int,
    ) -> None:
        super().__init__(
            data_root=data_root,
            sample_rate=sample_rate,
            device=device,
            batch_size=batch_size,
            random_seed=random_seed,
            shuffle_train=shuffle_train,
            database=database,
            max_audio_samples=max_audio_samples
        )
        self.feature = feature

    @torch.no_grad()
    def __getitem__(self, idx: int) -> DataPoint:
        inputs, labels = super().__getitem__(idx=idx)
        inputs = self.feature(inputs)
        return (inputs, labels)
