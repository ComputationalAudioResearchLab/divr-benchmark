import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List
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
    ) -> None:
        self.__diagnosis_map = diagnosis_map
        self.__audio_loader = audio_loader
        self.__train = self.__load_file(train)
        self.__val = self.__load_file(val)
        self.__test = dict([(v.id, v) for v in self.__load_file(test)])

    @property
    def train(self) -> List[TrainPoint]:
        return [x.to_trainpoint() for x in self.__train]

    @property
    def val(self) -> List[TrainPoint]:
        return [x.to_trainpoint() for x in self.__val]

    @property
    def test(self) -> List[TestPoint]:
        return [x.to_testpoint() for x in self.__test.values()]

    def score(self, predictions: Dict[str, str]) -> Result:
        correct = 0
        incorrect = 0
        for test_id, predicted_diagnosis in predictions:
            if self.__test[test_id].label.satisfies(predicted_diagnosis):
                correct += 1
            else:
                incorrect += 1
        return Result(
            correct=correct,
            incorrect=incorrect,
        )

    def __load_file(self, data_file: Path) -> List[DataPoint]:
        with open(data_file, "r") as df:
            data = yaml.load(df, Loader=yaml.FullLoader)
        dataset: List[DataPoint] = []
        for key, val in data.items():
            label = self.__diagnosis_map.get(val["label"])
            audio = self.__audio_loader(val["audio_keys"])
            dataset.append(DataPoint(id=key, audio=audio, label=label))
        return dataset

    def load_audio(self, audio_key: str) -> np.ndarray:
        raise NotImplementedError()
