import json
from tqdm import tqdm
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from typing import List, Tuple, Dict
from src.experiment.features import features, load_features, FeatureMap
from src.preprocess.processed import ProcessedSession
from src.experiment.collate import CollateFuncFactory
from src.diagnosis import DiagnosisMap


class Data:
    def __init__(
        self,
        diagnosis_level: int,
        balance_dataset: bool,
        data_files: List[Path],
        collate_fn: Dict,
        feature_params,
        allowed_labels: List[str] | None = None,
        **kwargs,
    ) -> None:
        self.diagnosis_map = DiagnosisMap()
        self.diagnosis_level = diagnosis_level
        self.balance_dataset = balance_dataset
        self.sessions = self.__load_processed_sessions(data_files, allowed_labels)
        self.features = self.__load_features(feature_params)
        self.collate_fn_args = collate_fn
        self.collate_fn = CollateFuncFactory.get_collate_func(**collate_fn)

    def __load_processed_sessions(
        self, file_paths, allowed_labels: List[str] | None
    ) -> List[ProcessedSession]:
        sessions: List[ProcessedSession] = []
        for file_path in file_paths:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
                sessions += list(map(ProcessedSession.from_json, data))

        if allowed_labels is None:
            return sessions

        filtered_sessions = []
        for session in sessions:
            all_diagnosis = []
            for diag in session.diagnosis:
                all_diagnosis += diag.to_list()
            for label in allowed_labels:
                if label in all_diagnosis:
                    # sometimes people have multiple diagnosis,
                    # we want to make sure we only keep the one we are interested in
                    session.diagnosis = [self.diagnosis_map.get(label)]
                    filtered_sessions += [session]
        return filtered_sessions

    def __load_features(self, feature_params) -> FeatureMap:
        data = {}
        common_params = feature_params["common"]
        data["sampling_frequency"] = common_params["sampling_frequency"]
        for key, val in features.items():
            if key in feature_params:
                key_params = feature_params[key]
                params = {
                    **common_params,
                    **(key_params if key_params is not None else {}),
                }
                data[key] = val(**params)
        return data

    def load(
        self,
    ) -> None:
        self.X, self.Y = self.__load_data("data", sessions=self.sessions)

    def __load_data(
        self,
        key: str,
        sessions: List[ProcessedSession],
    ) -> Tuple[np.ndarray, np.ndarray]:
        files = []
        labels = []
        for session in sessions:
            files += [file.path for file in session.files]
            label = self.diagnosis_map.most_severe(
                session.diagnosis, level=self.diagnosis_level
            )
            labels += [label] * len(session.files)
        labels = np.array(labels)
        files = np.array(files)
        if self.balance_dataset:
            files, labels = self.__balance_data(files, labels)
        with Pool(10) as pool:
            input_data = [
                (file_path, self.features, self.collate_fn, self.collate_fn_args)
                for file_path in files
            ]
            X = list(
                tqdm(
                    # pool.imap returns ordered results
                    # if any other parallelization is used then the order
                    # needs to be considered
                    pool.imap(load_features, input_data),
                    total=len(files),
                    desc=f"Loading {key}",
                )
            )
            return (np.stack(X), labels)

    def __balance_data(self, files, labels):
        classes, counts = np.unique(labels, return_counts=True)
        min_count = min(counts)
        keep_indices = np.array([], dtype=int)
        for cls in classes:
            indices = (labels == cls).nonzero()[0][:min_count]
            keep_indices = np.concatenate((keep_indices, indices))
        keep_indices = np.sort(keep_indices)
        labels = labels[keep_indices]
        files = files[keep_indices]
        return (files, labels)
