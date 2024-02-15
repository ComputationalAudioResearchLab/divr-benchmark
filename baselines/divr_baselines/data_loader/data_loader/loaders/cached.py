import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
import multiprocessing as mp
from typing import List, Tuple, Iterator
from .batch_ahead import BatchAheadLoader
from .rds import RDS
from ..dtypes import InputTensors, LabelTensor
from divr_benchmark import Benchmark

TVIterator = Iterator[Tuple[int, InputTensors, LabelTensor]]
TestIterator = Iterator[Tuple[int, InputTensors]]


class QDataSet:
    __rds = RDS()
    __q = mp.Queue(maxsize=2)

    def __init__(self, max_loaded_batches: int, max_processes: int, points) -> None:
        # self.__q = mp.Queue(maxsize=max_loaded_batches)
        # self.__pool = mp.Pool(processes=max_processes)
        # self.__pool.apply_async(self._q_loader, points)
        self.__proc = mp.Process(target=self._q_loader, args=(points[0],))
        self.__proc.start()

    def __del__(self) -> None:
        # self.__pool.close()
        print("closing up QDataset")
        self.__proc.join()
        self.__q.close()

    def _q_loader(self, point):
        print("Starting qloader")
        audio_file, len_file, label_file = point
        audio = self.__rds.load_tensor(file_path=audio_file, device=torch.device("cpu"))
        lens = self.__rds.load_tensor(file_path=len_file, device=torch.device("cpu"))
        labels = self.__rds.load_tensor(
            file_path=label_file, device=torch.device("cpu")
        )
        print(f"putting {audio_file} in queue")
        self.__q.put(((audio, lens), labels))
        print(f"put {audio_file} in queue")

    def get(self):
        print(f"getting some from queue")
        return self.__q.get()


class CachedLoader(BatchAheadLoader):
    cache_enabled = True
    _points: List[Tuple[Path, Path, Path]]
    __rds = RDS()
    cpu = torch.device("cpu")

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
        self.__max_loaded_batches = 2
        self.__max_processes = 2

    # def train(self):
    #     super().train()
    #     self.__qset = QDataSet(
    #         max_loaded_batches=self.__max_loaded_batches,
    #         max_processes=self.__max_processes,
    #         points=self._train_points,
    #     )

    # def eval(self):
    #     super().eval()
    #     self.__qset = QDataSet(
    #         max_loaded_batches=self.__max_loaded_batches,
    #         max_processes=self.__max_processes,
    #         points=self._val_points,
    #     )

    # @torch.no_grad()
    # def tv_getitem(self, idx: int) -> Tuple[InputTensors, LabelTensor]:
    #     print("getting from tv_getitem")
    #     value = self.__qset.get()
    #     print("got value from queue")
    #     return value

    @torch.no_grad()
    def tv_getitem(self, idx: int) -> Tuple[InputTensors, LabelTensor]:
        audio_file, len_file, label_file = self._points[idx]
        audio = self.__rds.load_tensor(file_path=audio_file, device=self.device)
        lens = self.__rds.load_tensor(file_path=len_file, device=self.device)
        labels = self.__rds.load_tensor(file_path=label_file, device=self.device)
        return ((audio, lens), labels)

    def __load_data(
        self,
        benchmark_path: Path,
        benchmark_version: str,
        stream: int,
        task: int,
        cache_base_path: Path,
        cache_key: str,
    ) -> None:
        cache_path = Path(f"{cache_base_path}/cache/{cache_key}")
        metadata_path = Path(f"{cache_path}/metadata.json")
        if not metadata_path.is_file():
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
        metadata = self.__rds.load_json(file_path=Path(f"{cache_path}/metadata.json"))
        # batch_size = metadata["batch_size"]
        total_train = metadata["total_train"]
        total_val = metadata["total_val"]
        self.unique_diagnosis = metadata["unique_diagnosis"]
        self.audio_sample_rate = metadata["audio_sample_rate"]
        self._train_indices = np.arange(total_train)
        self._val_indices = np.arange(total_val)
        self.num_unique_diagnosis = len(self.unique_diagnosis)

        def gen_points(key: str):
            return lambda idx: (
                f"{cache_path}/{key}/{idx}_audio.pt",
                f"{cache_path}/{key}/{idx}_lens.pt",
                f"{cache_path}/{key}/{idx}_labels.pt",
            )

        self._train_points = list(map(gen_points("train"), range(total_train)))
        self._val_points = list(map(gen_points("val"), range(total_val)))

    def __create_cache(
        self,
        cache_path: Path,
        benchmark_path: Path,
        benchmark_version: str,
        stream: int,
        task: int,
    ) -> None:
        print(f"cache does not exist, creating at {cache_path}")
        benchmark = Benchmark(
            storage_path=benchmark_path,
            version=benchmark_version,
        )
        btask = benchmark.task(stream=stream, task=task)
        train_cache_path = Path(f"{cache_path}/train")
        val_cache_path = Path(f"{cache_path}/val")
        total_train = self.__cache_tv(
            cache_path=train_cache_path,
            iterator=self.__load_tv_batches(
                train_cache_path, "train", btask.train, btask
            ),
        )
        total_val = self.__cache_tv(
            cache_path=val_cache_path,
            iterator=self.__load_tv_batches(val_cache_path, "val", btask.val, btask),
        )
        metadata = {
            "batch_size": self._batch_size,
            "total_train": total_train,
            "total_val": total_val,
            "unique_diagnosis": btask.unique_diagnosis,
            "audio_sample_rate": btask.audio_sample_rate,
        }
        self.__rds.save_json(
            object=metadata, file_path=Path(f"{cache_path}/metadata.json")
        )
        del btask

    def __cache_tv(self, cache_path: Path, iterator: TVIterator):
        cache_path.mkdir(exist_ok=True, parents=True)
        total = 0

        for (
            batch_idx,
            (audio_tensor, audio_lens),
            label_tensor,
        ) in iterator:
            total += 1
            self.__rds.save_tensor(
                audio_tensor,
                Path(f"{cache_path}/{batch_idx}_audio.pt"),
            )
            self.__rds.save_tensor(
                audio_lens,
                Path(f"{cache_path}/{batch_idx}_lens.pt"),
            )
            self.__rds.save_tensor(
                label_tensor,
                Path(f"{cache_path}/{batch_idx}_labels.pt"),
            )
        return total

    def __load_tv_batches(self, cache_path, key, pointset, btask) -> TVIterator:
        total_points = len(pointset)
        total_batches = total_points // self._batch_size
        resume_idx = self.__resume_idx(cache_path)
        print(f"resuming caching from idx: {resume_idx}")
        for batch_idx in tqdm(
            range(resume_idx, total_batches),
            desc=f"precomputing {key}",
        ):
            start = batch_idx * self._batch_size
            end = start + self._batch_size
            batch = pointset[start:end]
            inputs = self.__collate_function(batch)
            inputs = self.feature_function(inputs)
            labels = self.__prepare_label(batch, btask)
            yield (batch_idx, inputs, labels)

    def __resume_idx(self, cache_path: Path) -> int:
        if not any(cache_path.glob("*_labels.pt")):
            # no labels generated yet
            return 0
        return max(
            map(
                lambda x: int(x.stem.replace("_labels", "")),
                cache_path.glob("*_labels.pt"),
            )
        )

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
        audio_tensor = torch.tensor(audio_tensor, dtype=torch.float32)
        audio_lens = torch.tensor(
            audio_lens,
            device=self.device,
            dtype=torch.long,
        )
        return (audio_tensor, audio_lens)

    def __prepare_label(self, batch, btask):
        return torch.tensor(
            [btask.diag_to_index(b.label) for b in batch],
            device=self.device,
            dtype=torch.long,
        )
