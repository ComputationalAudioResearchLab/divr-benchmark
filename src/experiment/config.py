import json
from pathlib import Path
from typing import List, Optional
from src.diagnosis import Diagnosis
from dataclasses import dataclass, fields
from src.preprocess.processed import ProcessedSession, ProcessedFile


@dataclass(init=False)
class FeatureBase:
    def __init__(self, **kwargs):
        names = set([f.name for f in fields(self)])
        for k, v in kwargs.items():
            if k in names:
                setattr(self, k, v)


@dataclass(init=False)
class FeatureMFCC(FeatureBase):
    number_of_coefficients: int
    sampling_frequency: float
    window_length: float
    time_step: float
    first_filter_frequency: float
    max_filter_frequency: float
    distance_between_filters: float


@dataclass(init=False)
class FeatureJitter(FeatureBase):
    sampling_frequency: int
    pitch_floor: float
    pitch_ceiling: float
    start_time: float
    shortest_period: float
    longest_period: float
    max_period_factor: float


@dataclass(init=False)
class FeatureShimmer(FeatureBase):
    sampling_frequency: int
    pitch_floor: float
    pitch_ceiling: float
    start_time: float
    shortest_period: float
    longest_period: float
    max_period_factor: float
    max_amplitude_factor: float


@dataclass()
class HyperParams:
    sampling_frequency: int
    mfcc: Optional[FeatureMFCC]
    jitter: Optional[FeatureJitter]
    shimmer: Optional[FeatureShimmer]


@dataclass()
class ExperimentConfig:
    key: str
    random_seed: int
    model_path: Path
    tensorboard_path: Path
    train_data: List[ProcessedSession]
    val_data: List[ProcessedSession]
    test_data: List[ProcessedSession]
    hyper_params: HyperParams


class ConfigFactory:
    feature_map = {
        "mfcc": FeatureMFCC,
        "jitter": FeatureJitter,
        "shimmer": FeatureShimmer,
    }

    def load_experiment_config(self, data) -> ExperimentConfig:
        return ExperimentConfig(
            key=data["key"],
            random_seed=data["random_seed"],
            model_path=Path(data["model_path"]),
            tensorboard_path=Path(data["tensorboard_path"]),
            train_data=sum(
                list(map(self.load_processed_sessions, data["train_data"])), []
            ),
            val_data=sum(list(map(self.load_processed_sessions, data["val_data"])), []),
            test_data=sum(
                list(map(self.load_processed_sessions, data["test_data"])), []
            ),
            hyper_params=self.load_hyper_params(data["hyper_params"]),
        )

    def load_processed_sessions(self, file_path) -> List[ProcessedSession]:
        with open(file_path, "r") as json_file:
            data = json.load(json_file)
            return list(map(self.load_processed_session, data))

    def load_processed_session(self, json_data):
        return ProcessedSession(
            id=json_data["id"],
            age=json_data["age"],
            gender=json_data["gender"],
            diagnosis=list(map(self.load_diagnosis, json_data["diagnosis"])),
            files=list(map(self.load_processed_file, json_data["files"])),
        )

    def load_diagnosis(self, data) -> Diagnosis:
        parent = None
        if data["parent"] is not None:
            parent = self.load_diagnosis(data["parent"])
        return Diagnosis(
            name=data["name"],
            parent=parent,
        )

    def load_processed_file(self, data):
        return ProcessedFile(path=Path(data["path"]))

    def load_hyper_params(self, hyper_params) -> HyperParams:
        data = {}
        common_params = hyper_params["common"]
        data["sampling_frequency"] = common_params["sampling_frequency"]
        for key, val in self.feature_map.items():
            if key in hyper_params:
                key_params = hyper_params[key]
                params = {
                    **common_params,
                    **(key_params if key_params is not None else {}),
                }
                data[key] = val(**params)
        return HyperParams(**data)
