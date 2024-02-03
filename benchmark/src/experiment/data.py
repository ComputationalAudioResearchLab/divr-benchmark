import json
from tqdm import tqdm
from pathlib import Path
import numpy as np
from multiprocessing import Pool
from typing import List, Tuple, Dict
from .features import features, load_features, FeatureMap
from src.preprocess.processed import ProcessedSession
from .collate import CollateFuncFactory
from src.diagnosis import DiagnosisMap


class Data:
    def __init__(
        self,
        diagnosis_level: int,
        balance_dataset: bool,
        batch_size: int,
        shuffle_train: bool,
        cross_validate: bool,
        preprocessed_train_paths: List[Path],
        preprocessed_val_paths: List[Path],
        collate_fn: Dict,
        feature_params,
        **kwargs,
    ) -> None:
        self.diagnosis_level = diagnosis_level
        self.balance_dataset = balance_dataset
        self.cross_validate = cross_validate
        self.train_sessions = self.__load_processed_sessions(preprocessed_train_paths)
        self.val_sessions = self.__load_processed_sessions(preprocessed_val_paths)
        self.features = self.__load_features(feature_params)
        self.collate_fn_args = collate_fn
        self.collate_fn = CollateFuncFactory.get_collate_func(**collate_fn)
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train
        self.diagnosis_map = DiagnosisMap()

    def __load_processed_sessions(self, file_paths) -> List[ProcessedSession]:
        sessions = []
        for file_path in file_paths:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
                sessions += list(map(ProcessedSession.from_json, data))
        return sessions

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
        self.train_X, self.train_Y = self.__load_data(
            "train", sessions=self.train_sessions
        )
        self.train_X_len = self.train_X.shape[0] // self.batch_size
        self.train_X_indices = np.arange(self.train_X_len)
        self.val_X, self.val_Y = self.__load_data("val", sessions=self.val_sessions)
        self.val_X_len = self.val_X.shape[0] // self.batch_size
        self.val_X_indices = np.arange(self.val_X_len)

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

    def __len__(self):
        return self.__data_len

    def __getitem__(self, idx):
        idx = self.__indices[idx]
        start = idx * self.batch_size
        end = start + self.batch_size
        X = self.__X[start:end]
        Y = self.__Y[start:end]
        return (X, Y)

    def train(self):
        self.__data_len = self.train_X_len
        self.__X = self.train_X
        self.__Y = self.train_Y
        self.__indices = self.train_X_indices
        if self.shuffle_train:
            np.random.shuffle(self.__indices)

    def eval(self):
        self.__data_len = self.val_X_len
        self.__X = self.val_X
        self.__Y = self.val_Y
        self.__indices = self.val_X_indices
