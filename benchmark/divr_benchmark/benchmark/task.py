import yaml
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from ..diagnosis import Diagnosis, DiagnosisMap
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
    ) -> None:
        self.__diagnosis_map = diagnosis_map
        self.__audio_loader = audio_loader
        self.__train = self.__load_file(
            data_file=train,
            key="train",
            quiet=quiet,
        )
        self.__val = self.__load_file(
            data_file=val,
            key="val",
            quiet=quiet,
        )
        self.__test = dict(
            [
                (v.id, v)
                for v in self.__load_file(
                    data_file=test,
                    key="test",
                    quiet=quiet,
                )
            ]
        )
        self.audio_sample_rate = audio_loader.sample_rate
        self.__diagnosis_index = self.__count_diagnosis()
        self.__diagnosis_index_reversed = dict(
            [(v.name, k) for k, v in self.__diagnosis_index.items()]
        )

    @property
    def unique_diagnosis(self) -> List[str]:
        return list(self.__diagnosis_index_reversed.keys())

    def index_to_diag(self, index: int) -> Diagnosis:
        return self.__diagnosis_index[index]

    def diag_to_index(self, diag: Diagnosis) -> int:
        return self.__diagnosis_index_reversed[diag.name]

    @property
    def train(self) -> List[TrainPoint]:
        return [x.to_trainpoint() for x in self.__train]

    @property
    def val(self) -> List[TrainPoint]:
        return [x.to_trainpoint() for x in self.__val]

    @property
    def test(self) -> List[TestPoint]:
        return [x.to_testpoint() for x in self.__test.values()]

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

    def __load_file(self, data_file: Path, key: str, quiet: bool) -> List[DataPoint]:
        with open(data_file, "r") as df:
            data = yaml.load(df, Loader=yaml.FullLoader)
        dataset: List[DataPoint] = []
        if not quiet:
            iterator = tqdm(data.items(), desc=f"Loading {key} files", leave=True)
        else:
            iterator = data.items()
        for key, val in iterator:
            label = self.__diagnosis_map.get(val["label"])
            audio = self.__audio_loader(val["audio_keys"])
            dataset.append(DataPoint(id=key, audio=audio, label=label))
        return dataset

    def __count_diagnosis(self) -> Dict[int, Diagnosis]:
        train_diags = [d.label for d in self.__train]
        test_diags = [d.label for d in self.__test.values()]
        val_diags = [d.label for d in self.__val]
        unique_diagnosis = set(train_diags + test_diags + val_diags)
        return dict(enumerate(sorted(unique_diagnosis, key=lambda x: x.name)))
