import itertools
import json
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
    base_models = [
        ["mel_mu"],
        ["mel_std"],
        ["mfcc_mu"],
        ["mfcc_std"],
        ["mfcc_mu", "mfcc_std"],
        ["mel_mu", "mel_std"],
        ["mel_mu", "mel_std", "mfcc_mu", "mfcc_std"],
    ]
    model_configs = [
        dict(zip(("C", "degree", "kernel"), config))
        for config in itertools.product([1.0], [5, 10, 20], ["rbf"])
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
        for features in tqdm(self.base_models, "on feature", position=0):
            model_name = "_".join(features)
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
        train_metrics = self.metrics(train_Y, pred_train_Y, train_diag)
        val_metrics = self.metrics(val_Y, pred_val_Y, val_diag)
        results_file = f"{self.results_folder}/{key}.log"
        with open(results_file, "w") as results:
            self.write_metrics(results_file=results, key="Train", metrics=train_metrics)
            self.write_metrics(results_file=results, key="Val", metrics=val_metrics)

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
        with open(file_name, "wb") as checkpoint_file:
            pickle.dump(model, checkpoint_file)

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
        for diagnosis, file_base_path in tqdm(dataset, "desc loading feature"):
            data_path = f"{file_base_path}.{model_name}.pt"
            feature = torch.load(data_path)
            X += [feature]
            Y += [self.diagnosis_map.to_int(diagnosis[0])]
            full_diag += [diagnosis[1]]
        X = torch.stack(X)
        Y = np.array(Y)
        return X, Y, full_diag

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
        output_folder=Path("/home/storage/data/baseline_latents/16000/svm"),
    )
    # classifier.run(
    #     key="voiced",
    #     train_files=[
    #         Path(
    #             "/home/storage/data/baseline_latents/16000/features/voiced_train.json"
    #         ),
    #     ],
    #     val_files=[
    #         Path("/home/storage/data/baseline_latents/16000/features/voiced_val.json"),
    #     ],
    # )
    # classifier.run(
    #     key="svd_a",
    #     train_files=[
    #         Path(
    #             "/home/storage/data/baseline_latents/16000/features/svd_a_n_train.json"
    #         ),
    #     ],
    #     val_files=[
    #         Path("/home/storage/data/baseline_latents/16000/features/svd_a_n_val.json"),
    #     ],
    # )
    classifier.run(
        key="svd_i",
        train_files=[
            Path(
                "/home/storage/data/baseline_latents/16000/features/svd_i_n_train.json"
            ),
        ],
        val_files=[
            Path("/home/storage/data/baseline_latents/16000/features/svd_i_n_val.json"),
        ],
    )
    classifier.run(
        key="svd_u",
        train_files=[
            Path(
                "/home/storage/data/baseline_latents/16000/features/svd_u_n_train.json"
            ),
        ],
        val_files=[
            Path("/home/storage/data/baseline_latents/16000/features/svd_u_n_val.json"),
        ],
    )
    # classifier.run(
    #     key="svd_aiu",
    #     train_files=[
    #         Path(
    #             "/home/storage/data/baseline_latents/16000/features/svd_a_n_train.json"
    #         ),
    #         Path(
    #             "/home/storage/data/baseline_latents/16000/features/svd_i_n_train.json"
    #         ),
    #         Path(
    #             "/home/storage/data/baseline_latents/16000/features/svd_u_n_train.json"
    #         ),
    #     ],
    #     val_files=[
    #         Path("/home/storage/data/baseline_latents/16000/features/svd_a_n_val.json"),
    #         Path("/home/storage/data/baseline_latents/16000/features/svd_i_n_val.json"),
    #         Path("/home/storage/data/baseline_latents/16000/features/svd_u_n_val.json"),
    #     ],
    # )
