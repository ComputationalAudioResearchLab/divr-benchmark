import itertools
import json
from multiprocessing import Pool
import torch
import pickle
import numpy as np
from typing import List, Dict
from pathlib import Path
from diagnosis import Diagnosis, DiagnosisMap
from tqdm import tqdm
from sklearn import svm
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class Classifier:
    ssl_models = [
        "apc_960hr",
        "vq_apc_960hr",
        "npc_960hr",
        "modified_cpc",
        "decoar_layers",
        "decoar2",
        "wav2vec_large",
        "vq_wav2vec_gumbel",
        "vq_wav2vec_kmeans",
        "vq_wav2vec_kmeans_roberta",
        "wav2vec2_large_lv60_cv_swbd_fsh",
        "xlsr_53",
        "xls_r_2b",
        "hubert_base",
        "distilhubert_base",
        "hubert_base_robust_mgr",
        "unispeech_sat_large",
        "wavlm_large",
        "data2vec_base_960",
    ]
    model_configs = [
        dict(zip(("C", "degree", "kernel"), config))
        for config in itertools.product([1.0], [5, 10, 20], ["rbf", "poly"])
    ]

    def __init__(
        self,
        balance_dataset: bool,
        output_folder: Path,
        diagnosis_level: int,
    ) -> None:
        self.balance_dataset = balance_dataset
        self.diagnosis_level = diagnosis_level
        self.diagnosis_map = DiagnosisMap()
        self.output_folder = output_folder
        self.models_folder = Path(f"{output_folder}/models")
        self.results_folder = Path(f"{output_folder}/results")
        self.models_folder.mkdir(parents=True, exist_ok=True)
        self.results_folder.mkdir(parents=True, exist_ok=True)

    def run(self, key, train_files: List[Path], val_files: List[Path]):
        train_dataset = self.load_dataset(files=train_files)
        val_dataset = self.load_dataset(files=val_files)
        for model_name in tqdm(self.ssl_models, "on feature", position=0):
            train_X, train_Y, train_diag = self.load_features(model_name, train_dataset)
            val_X, val_Y, val_diag = self.load_features(model_name, val_dataset)
            for config in tqdm(self.model_configs, "on config", position=1):
                config_key = "_".join(map(str, config.values()))
                full_key = f"{key}.{model_name}.{config_key}"
                model = svm.SVC(**config)
                model.fit(train_X, train_Y)
                self.save(model, file_name=f"{self.models_folder}/{full_key}.svm")
                pred_train_Y = model.predict(train_X)
                pred_val_Y = model.predict(val_X)
                self.save_metrics(
                    full_key,
                    train_Y,
                    pred_train_Y,
                    train_diag,
                    val_Y,
                    pred_val_Y,
                    val_diag,
                )

    def save_metrics(
        self, key, train_Y, pred_train_Y, train_diag, val_Y, pred_val_Y, val_diag
    ):
        try:
            train_metrics = self.metrics(train_Y, pred_train_Y, train_diag)
            val_metrics = self.metrics(val_Y, pred_val_Y, val_diag)
            results_file = f"{self.results_folder}/{key}.log"
            with open(results_file, "w") as results:
                self.write_metrics(
                    results_file=results, key="Train", metrics=train_metrics
                )
                self.write_metrics(results_file=results, key="Val", metrics=val_metrics)
        except Exception:
            print("Retrying save metrics")
            self.save_metrics(
                key, train_Y, pred_train_Y, train_diag, val_Y, pred_val_Y, val_diag
            )

    def write_metrics(self, results_file, key, metrics):
        (
            accuracy,
            balanced_accuracy,
            precision,
            recall,
            f1,
            (confusion, confusion_labels),
            (target_Y, pred_Y, diags),
        ) = metrics
        results_file.write(f"{key} >>\n")
        results_file.write(f"   accuracy = {accuracy}\n")
        results_file.write(f"   balanced_accuracy = {balanced_accuracy}\n")
        results_file.write(f"   precision = {precision}\n")
        results_file.write(f"   recall = {recall}\n")
        results_file.write(f"   f1 = {f1}\n")
        results_file.write(f"   confusion_labels = {confusion_labels}\n")
        results_file.write("    confusion >\n")
        results_file.write(f"       {confusion}\n")
        results_file.write("    full eval results >\n")
        for t, p, diag in zip(target_Y, pred_Y, diags):
            results_file.write(
                f"        {t},{p},{list(map(Diagnosis.to_list, diag))}\n"
            )

    def metrics(self, target_Y, pred_Y, diags):
        accuracy = accuracy_score(target_Y, pred_Y)
        balanced_accuracy = balanced_accuracy_score(target_Y, pred_Y)
        precision = precision_score(
            target_Y, pred_Y, average="weighted", zero_division=0
        )
        recall = recall_score(target_Y, pred_Y, average="weighted", zero_division=0)
        f1 = f1_score(target_Y, pred_Y, average="weighted", zero_division=0)
        confusion = confusion_matrix(target_Y, pred_Y)
        all_keys = np.unique(np.concatenate((target_Y, pred_Y))).tolist()
        display_labels = [
            self.diagnosis_map.from_int(i).name for i in np.unique(all_keys)
        ]
        return (
            accuracy,
            balanced_accuracy,
            precision,
            recall,
            f1,
            (confusion, display_labels),
            (target_Y, pred_Y, diags),
        )

    def save(self, model, file_name) -> None:
        try:
            with open(file_name, "wb") as checkpoint_file:
                pickle.dump(model, checkpoint_file)
        except Exception:
            print("Retrying save model")
            self.save(model, file_name)

    def load_dataset(self, files: List[Path]):
        dataset = []
        for data_file in files:
            with open(data_file, "r") as config_data:
                config = json.load(config_data)
                for i, session in tqdm(
                    enumerate(config), total=len(config), desc="loading data"
                ):
                    diagnosis = self.load_diagnosis(session=session)
                    for session_file in session["files"]:
                        file_base_path = session_file["path"]
                        dataset += [(diagnosis, file_base_path)]
        if not self.balance_dataset:
            return dataset

        all_diags = np.array([data[0][0] for data in dataset])
        unique_diags, counts = np.unique(all_diags, return_counts=True)
        min_count = np.min(counts)
        final_indices = []
        for diag in unique_diags:
            selected_idx = np.where(all_diags == diag)[0][:min_count]
            final_indices = np.concatenate((final_indices, selected_idx))
        final_indices = np.sort(final_indices).astype(int).tolist()
        balanced_dataset = [dataset[i] for i in final_indices]
        return balanced_dataset

    def load_features(self, model_name, dataset):
        X = []
        Y = []
        full_diag = []
        for diagnosis, file_base_path in tqdm(dataset, desc="loading features"):
            data_path = f"{file_base_path}.{model_name}.pt"
            feature = self.load_or_retry(data_path)
            frame_wise_avg_feature = feature.mean(dim=0)
            X += [frame_wise_avg_feature]
            Y += [self.diagnosis_map.to_int(diagnosis[0])]
            full_diag += [diagnosis[1]]
        X = torch.stack(X)
        Y = np.array(Y)
        return X, Y, full_diag

    def load_or_retry(self, data_path):
        try:
            return torch.load(data_path)
        except Exception:
            print("Retrying load feature")
            return self.load_or_retry(data_path)

    def load_diagnosis(self, session: Dict):
        diagnosis_list = list(map(Diagnosis.from_json, session["diagnosis"]))
        return (
            self.diagnosis_map.from_int(
                self.diagnosis_map.most_severe(
                    diagnosis_list,
                    level=self.diagnosis_level,
                )
            ).name,
            diagnosis_list,
        )


if __name__ == "__main__":
    classifier = Classifier(
        diagnosis_level=0,
        balance_dataset=True,
        output_folder=Path("/home/storage/data/nn_latents_full/16000/svm"),
    )
    classifier.run(
        key="voiced",
        train_files=[
            Path("/home/storage/data/nn_latents_full/16000/features/voiced_train.json"),
        ],
        val_files=[
            Path("/home/storage/data/nn_latents_full/16000/features/voiced_val.json"),
        ],
    )
    classifier.run(
        key="svd_a",
        train_files=[
            Path(
                "/home/storage/data/nn_latents_full/16000/features/svd_a_n_train.json"
            ),
        ],
        val_files=[
            Path("/home/storage/data/nn_latents_full/16000/features/svd_a_n_val.json"),
        ],
    )
    classifier.run(
        key="svd_i",
        train_files=[
            Path(
                "/home/storage/data/nn_latents_full/16000/features/svd_i_n_train.json"
            ),
        ],
        val_files=[
            Path("/home/storage/data/nn_latents_full/16000/features/svd_i_n_val.json"),
        ],
    )
    classifier.run(
        key="svd_u",
        train_files=[
            Path(
                "/home/storage/data/nn_latents_full/16000/features/svd_u_n_train.json"
            ),
        ],
        val_files=[
            Path("/home/storage/data/nn_latents_full/16000/features/svd_u_n_val.json"),
        ],
    )
    classifier.run(
        key="svd_aiu",
        train_files=[
            Path(
                "/home/storage/data/nn_latents_full/16000/features/svd_a_n_train.json"
            ),
            Path(
                "/home/storage/data/nn_latents_full/16000/features/svd_i_n_train.json"
            ),
            Path(
                "/home/storage/data/nn_latents_full/16000/features/svd_u_n_train.json"
            ),
        ],
        val_files=[
            Path("/home/storage/data/nn_latents_full/16000/features/svd_a_n_val.json"),
            Path("/home/storage/data/nn_latents_full/16000/features/svd_i_n_val.json"),
            Path("/home/storage/data/nn_latents_full/16000/features/svd_u_n_val.json"),
        ],
    )
