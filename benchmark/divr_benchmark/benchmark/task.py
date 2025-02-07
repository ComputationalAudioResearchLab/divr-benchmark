import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from divr_diagnosis import Diagnosis, DiagnosisMap

from .result import Result
from .audio_loader import AudioLoader


@dataclass
class TrainPoint:
    audio: List[np.ndarray]
    label: Diagnosis


@dataclass
class TestPoint:
    id: str
    audio: List[np.ndarray]


@dataclass
class DataPoint:
    id: str
    audio: List[np.ndarray]
    label: Diagnosis

    def to_testpoint(self) -> TestPoint:
        return TestPoint(
            id=self.id,
            audio=self.audio,
        )

    def to_trainpoint(self) -> TrainPoint:
        return TrainPoint(
            audio=self.audio,
            label=self.label,
        )

    def satisfies(self, prediction: str) -> bool:
        return self.label.satisfies(prediction)


class Task:

    __train: List[DataPoint]
    __val: List[DataPoint]
    __test: Dict[str, DataPoint]

    def __init__(
        self,
        diagnosis_map: DiagnosisMap,
        audio_loader: AudioLoader,
        train: Path,
        val: Path,
        test: Path,
        quiet: bool,
        diag_level: int | None,
        load_audios: bool,
    ) -> None:
        self.__diagnosis_map = diagnosis_map
        self.__audio_loader = audio_loader
        self.__train = self.__load_file(
            data_file=train,
            key="train",
            quiet=quiet,
            diag_level=diag_level,
            load_audios=load_audios,
        )
        self.__val = self.__load_file(
            data_file=val,
            key="val",
            quiet=quiet,
            diag_level=diag_level,
            load_audios=load_audios,
        )
        self.__test = dict(
            [
                (v.id, v)
                for v in self.__load_file(
                    data_file=test,
                    key="test",
                    quiet=quiet,
                    diag_level=diag_level,
                    load_audios=load_audios,
                )
            ]
        )
        self.audio_sample_rate = audio_loader.sample_rate
        self.__diagnosis_indices = self.__count_diagnosis()
        self.__max_diag_level = list(self.__diagnosis_indices.keys())[-1]
        self.__normal = diagnosis_map.get("normal")

    @property
    def max_diag_level(self) -> int:
        return self.__max_diag_level

    @property
    def sample_rate(self) -> int:
        return self.__audio_loader.sample_rate

    def normal_index(self, level) -> int:
        _, diag_indices_reversed = self.__diagnosis_indices[level]
        return diag_indices_reversed[self.__normal.name]

    def unique_diagnosis(self, level: int | None = None) -> List[str]:
        if level is None:
            level = self.__max_diag_level
        _, diag_indices_reversed = self.__diagnosis_indices[level]
        return list(diag_indices_reversed.keys())

    def index_to_diag(self, index: int, level: int | None = None) -> Diagnosis:
        if level is None:
            level = self.__max_diag_level
        diag_indices, _ = self.__diagnosis_indices[level]
        return diag_indices[index]

    def diag_to_index(self, diag: Diagnosis, level: int | None = None) -> int:
        if level is None:
            level = self.__max_diag_level
        _, diag_indices_reversed = self.__diagnosis_indices[level]
        return diag_indices_reversed[diag.at_level(level).name]

    def diag_name_to_index(self, diag_name: str, level: int | None = None) -> int:
        if level is None:
            level = self.__max_diag_level
        _, diag_indices_reversed = self.__diagnosis_indices[level]
        diag = self.__diagnosis_map.get(name=diag_name)
        return diag_indices_reversed[diag.at_level(level).name]

    def train_class_weights(self, level: int | None = None) -> List[float]:
        """
        Returns a list of (total_samples / class_samples)
        """
        if level is None:
            level = self.__max_diag_level
        diags = np.array([t.label.at_level(level).name for t in self.__train])
        class_weights = [0.0] * len(self.unique_diagnosis(level=level))
        _, diag_indices_reversed = self.__diagnosis_indices[level]
        for diag_name, count in zip(*np.unique(diags, return_counts=True)):
            idx = diag_indices_reversed[diag_name]
            class_weights[idx] = count
        total_sessions = sum(class_weights)
        for i in range(len(class_weights)):
            class_weights[i] = total_sessions / class_weights[i]
        return class_weights

    @property
    def train(self) -> List[DataPoint]:
        return self.__train

    @property
    def val(self) -> List[DataPoint]:
        return self.__val

    @property
    def test(self) -> List[DataPoint]:
        return list(self.__test.values())

    def test_label(self, id: str) -> Diagnosis:
        return self.__test[id].label

    def score(self, predictions: Dict[str, int]) -> Result:
        """
        predictions: Dict[test_id, index_of_diag]
        """
        results: List[Tuple[Diagnosis, Diagnosis]] = []
        for test_id, predicted_diagnosis in predictions.items():
            predicted_diagnosis = self.index_to_diag(predicted_diagnosis)
            actual_diagnosis = self.__test[test_id].label
            results += [(actual_diagnosis, predicted_diagnosis)]
        return Result(data=results)

    def __load_file(
        self,
        data_file: Path,
        key: str,
        quiet: bool,
        diag_level: int | None,
        load_audios: bool,
    ) -> List[DataPoint]:
        with open(data_file, "r") as df:
            data = yaml.load(df, Loader=yaml.FullLoader)
        dataset: List[DataPoint] = []
        if not quiet:
            iterator = tqdm(data.items(), desc=f"Loading {key} files", leave=False)
        else:
            iterator = data.items()
        for key, val in iterator:
            label = self.__diagnosis_map.get(val["label"])
            if diag_level is not None:
                label = label.at_level(diag_level)
            if load_audios:
                audio = self.__audio_loader(val["audio_keys"])
                if len(audio) < 1:
                    raise ValueError(f"Invalid data point (no audio): {key}")
            else:
                audio = []
            dataset.append(DataPoint(id=key, audio=audio, label=label))
        return dataset

    def __count_diagnosis(
        self,
    ) -> Dict[int, Tuple[Dict[int, Diagnosis], Dict[str, int]]]:
        train_diags = [d.label for d in self.__train]
        test_diags = [d.label for d in self.__test.values()]
        val_diags = [d.label for d in self.__val]
        unique_diagnosis = set(train_diags + test_diags + val_diags)
        max_level = max([d.level for d in unique_diagnosis])
        counts = {}
        for level in range(max_level + 1):
            diags_at_level = set([d.at_level(level) for d in unique_diagnosis])
            counts[level] = [
                dict(enumerate(sorted(diags_at_level, key=lambda x: x.name))),
                dict(
                    [
                        (diag.name, idx)
                        for idx, diag in enumerate(
                            sorted(diags_at_level, key=lambda x: x.name)
                        )
                    ]
                ),
            ]
        return counts
