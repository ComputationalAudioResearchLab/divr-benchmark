import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Tuple
from .base import Base
from ..dtypes import InputTensors, LabelTensor
from divr_benchmark import Benchmark


class FeatureAheadLoader(Base):
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
        feature_size,
    ) -> None:
        np.random.seed(random_seed)
        self.model = feature_init()
        self.feature_size = feature_size
        self.device = device
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
        inputs, labels = self._points[idx]
        (audio_tensor, audio_lens) = inputs
        inputs = (audio_tensor.to(self.device), audio_lens)
        return (inputs, labels)

    @torch.no_grad()
    def test_getitem(self, idx) -> InputTensors:
        inputs, _ = self._points[idx]
        (audio_tensor, audio_lens) = inputs
        inputs = (audio_tensor.to(self.device), audio_lens)
        return inputs

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
        self._train_points = self.__load_tv_batches("train", btask.train, btask)
        self._test_points = self.__load_test_batches(btask.test, btask)
        self._val_points = self.__load_tv_batches("val", btask.val, btask)
        self._train_indices = np.arange(len(self._train_points))
        self._test_indices = np.arange(len(self._test_points))
        self._val_indices = np.arange(len(self._val_points))

    def __load_tv_batches(self, key, pointset, btask) -> List:
        total_points = len(pointset)
        total_batches = total_points // self._batch_size
        batches = []
        for batch_idx in tqdm(range(total_batches), desc=f"precomputing {key}"):
            start = batch_idx * self._batch_size
            end = start + self._batch_size
            batch = pointset[start:end]
            inputs = self.__collate_function(batch)
            labels = self.__prepare_label(batch, btask)
            batches.append((inputs, labels))
        return batches

    def __load_test_batches(self, pointset, btask) -> List:
        total_points = len(pointset)
        total_batches = total_points // self._batch_size
        batches = []
        for batch_idx in tqdm(range(total_batches), desc="precomputing test"):
            start = batch_idx * self._batch_size
            end = start + self._batch_size
            batch = pointset[start:end]
            inputs = self.__collate_function(batch)
            batches.append((inputs, None))
        return batches

    def __collate_function(self, batch):
        batch_len = len(batch)
        max_num_audios = 0
        max_audio_len = 0
        for point in batch:
            audios = point.audio
            len_audios = len(audios)
            if len_audios > max_num_audios:
                max_num_audios = len_audios
            for audio in audios:
                audio_len = audio.shape[0]
                if audio_len > max_audio_len:
                    max_audio_len = audio_len
        audio_tensor = np.zeros((batch_len, max_num_audios, max_audio_len))
        audio_lens = np.zeros((batch_len, max_num_audios), dtype=int)
        for batch_idx, point in enumerate(batch):
            for audio_idx, audio in enumerate(point.audio):
                audio_len = audio.shape[0]
                audio_tensor[batch_idx, audio_idx, :audio_len] = audio
                audio_lens[batch_idx, audio_idx] = audio_len
        audio_tensor = torch.tensor(
            audio_tensor,
            device=self.device,
            dtype=torch.float32,
        )
        audio_lens = torch.tensor(
            audio_lens,
            device=self.device,
            dtype=torch.long,
        )
        features = self.__feature_function((audio_tensor, audio_lens))
        return features

    @torch.no_grad()
    def __feature_function(self, batch):
        batch_inputs, batch_lens = batch
        batch_size, max_audios_in_session, max_audio_len = batch_inputs.shape
        audios = batch_inputs.reshape(batch_size * max_audios_in_session, max_audio_len)
        audio_lens = batch_lens.reshape(batch_size * max_audios_in_session)
        all_hs, all_hs_len = self.model(audios, audio_lens)
        feature = torch.cat(all_hs, dim=2)
        _, max_feature_len, feature_hidden_len = feature.shape
        feature = feature.reshape(
            (batch_size, max_audios_in_session, max_feature_len, feature_hidden_len)
        )
        feature_lens = all_hs_len[0].reshape((batch_size, max_audios_in_session))
        return feature.to(device="cpu", non_blocking=True), feature_lens

    def __prepare_label(self, batch, btask):
        return torch.tensor(
            [btask.diag_to_index(b.label) for b in batch],
            device=self.device,
            dtype=torch.long,
        )
