import json
import pickle
import numpy as np
from typing import List, Dict
from pathlib import Path
from tqdm import tqdm
from torch import nn
import torch
from diagnosis import Diagnosis, DiagnosisMap
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


class NNModel(nn.Module):
    def __init__(
        self,
        exp_type: str,
        sample_rate: str,
        model_type: str,
        model_tag: str,
        device: torch.device,
    ):
        super().__init__()
        checkpoint_path = f"/home/storage/data/{exp_type}/{sample_rate}/{model_type}/models/{model_tag}.pt"
        state_dict = torch.load(checkpoint_path, map_location=device)
        weight_keys = [key for key in state_dict if ".weight" in key]
        num_layers = len(weight_keys)
        latent_dim, input_dim = state_dict[weight_keys[0]].shape
        output_dim, _ = state_dict[weight_keys[-1]].shape
        if num_layers == 1:
            layers = [nn.Linear(input_dim, output_dim)]
        else:
            layers = [nn.Linear(input_dim, latent_dim), nn.ReLU()]
            for _ in range(num_layers - 2):
                layers += [nn.Linear(latent_dim, latent_dim), nn.ReLU()]
            layers += [nn.Linear(latent_dim, output_dim)]
        self.model = nn.Sequential(*layers)
        self.to(device)
        self.load_state_dict(state_dict)
        self.device = device

    def forward(self, input):
        return self.model(input)

    @torch.no_grad()
    def predict(self, X: np.ndarray) -> np.ndarray:
        torchX = torch.tensor(X).to(self.device)
        return self(torchX).argmax(dim=1).cpu().numpy()


class SVMModel:
    def __init__(
        self,
        exp_type: str,
        sample_rate: str,
        model_type: str,
        model_tag: str,
        device: torch.device,
    ):
        checkpoint_path = f"/home/storage/data/{exp_type}/{sample_rate}/{model_type}/models/{model_tag}.svm"
        with open(checkpoint_path, "rb") as checkpoint_file:
            self.model = pickle.load(checkpoint_file)
        # print(self.model, self.model.n_features_in_, len(self.model.classes_))

    def predict(self, X):
        return self.model.predict(X)


class CrossTester:
    model_types = {"nn": NNModel, "svm": SVMModel}
    dataset_confs = {}
    dataset_features = {}

    def __init__(
        self,
        balance_dataset: bool,
        diagnosis_level: int,
        output_root: str,
        device: torch.device,
    ) -> None:
        self.balance_dataset = balance_dataset
        self.diagnosis_level = diagnosis_level
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.results_root = Path(f"{self.output_root}/results")
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.diagnosis_map = DiagnosisMap()

    def load_dataset(self, files: List[str]):
        dataset = []
        for data_file in files:
            with open(data_file, "r") as config_data:
                config = json.load(config_data)
                for i, session in tqdm(
                    enumerate(config),
                    total=len(config),
                    desc="loading data",
                    position=0,
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

    def load_features(self, model_name, dataset):
        X = []
        Y = []
        full_diag = []
        for diagnosis, file_base_path in tqdm(
            dataset, desc="loading features", position=2
        ):
            data_path = f"{file_base_path}.{model_name}.pt"
            feature = self.load_or_retry(data_path)
            # print(feature.shape)
            # exit()
            if len(feature.shape) == 2:
                X += [feature.mean(dim=0)]
            else:
                X += [feature]
            Y += [self.diagnosis_map.to_int(diagnosis[0])]
            full_diag += [diagnosis[1]]
        X = np.stack(X)
        Y = np.array(Y)
        return X, Y, full_diag

    def load_or_retry(self, data_path):
        try:
            return torch.load(data_path)
        except Exception:
            print("Retrying load feature")
            return self.load_or_retry(data_path)

    def run(self, best_logs: List[str]):
        for log_file in tqdm(best_logs, "on model", position=1):
            (
                exp_type,
                sample_rate,
                model_type,
                model_tag,
                dataset_tag,
                model_name,
                model_conf,
            ) = self.resolve_log_tags(log_file)
            model = self.model_types[model_type](
                exp_type=exp_type,
                sample_rate=sample_rate,
                model_type=model_type,
                model_tag=model_tag,
                device=self.device,
            )
            dataset_conf_tag = (
                f"/home/storage/data/{exp_type}/{sample_rate}/features/svd"
            )
            if dataset_conf_tag not in self.dataset_confs:
                self.dataset_confs[dataset_conf_tag] = self.load_dataset(
                    files=[
                        f"{dataset_conf_tag}_i_n_val.json",
                    ]
                )
            dataset = self.dataset_confs[dataset_conf_tag]
            full_feature_tag = f"/{exp_type}/{sample_rate}/{model_name}"
            if full_feature_tag not in self.dataset_features:
                self.dataset_features[full_feature_tag] = self.load_features(
                    model_name, dataset
                )
            X, Y, full_diag = self.dataset_features[full_feature_tag]
            pred_Y = model.predict(X)
            self.save_metrics(
                key=f"{exp_type}.{sample_rate}.{model_type}.{model_tag}",
                Y=Y,
                pred_Y=pred_Y,
                full_diag=full_diag,
            )

    def resolve_log_tags(self, log_file: str):
        log_file_parts = log_file.split("/")
        exp_type = log_file_parts[4]
        sample_rate = log_file_parts[5]
        model_type = log_file_parts[6]
        model_tag = log_file_parts[8].removesuffix(".log")
        dataset, model_name, model_conf = model_tag.split(".", maxsplit=2)
        model_conf = model_conf.split("_")
        return (
            exp_type,
            sample_rate,
            model_type,
            model_tag,
            dataset,
            model_name,
            model_conf,
        )

    def save_metrics(self, key, Y, pred_Y, full_diag):
        try:
            metrics = self.metrics(Y, pred_Y, full_diag)
            results_file = f"{self.results_root}/{key}.log"
            with open(results_file, "w") as results:
                self.write_metrics(results_file=results, key="Val", metrics=metrics)
        except Exception:
            print("Retrying save metrics")
            self.save_metrics(key, Y, pred_Y, full_diag)

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


if __name__ == "__main__":
    curdir = Path(__file__).resolve().parent
    tester = CrossTester(
        balance_dataset=True,
        diagnosis_level=0,
        output_root="/home/storage/data/cross_tester/voiced_to_svd_i",
        device=torch.device("cuda"),
    )
    df = pd.read_csv(f"{curdir}/crunched_accuracies.csv")
    best_logs = [log_file for log_file in df["log_file"] if "voiced" in log_file]
    tester.run(best_logs=best_logs)
