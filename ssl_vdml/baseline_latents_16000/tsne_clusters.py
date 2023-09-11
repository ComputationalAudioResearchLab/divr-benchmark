import json
import torch
import numpy as np
from sklearn.manifold import TSNE
from typing import List, Dict
from pathlib import Path
from diagnosis import Diagnosis, DiagnosisMap
from tqdm import tqdm
import matplotlib.pyplot as plt


class Cluster:
    base_models = [
        ["mel_mu"],
        ["mel_std"],
        ["mfcc_mu"],
        ["mfcc_std"],
        ["mfcc_mu", "mfcc_std"],
        ["mel_mu", "mel_std"],
        ["mel_mu", "mel_std", "mfcc_mu", "mfcc_std"],
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
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.tsne = TSNE(n_components=2, perplexity=42, learning_rate="auto")

    def run(self, key, files: List[Path]):
        dataset = self.load_dataset(files)

        for features in tqdm(self.base_models, "clustering"):
            model_name = "_".join(features)
            self.cluster(key, model_name, dataset)

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

        all_diags = np.array([data[0] for data in dataset])
        unique_diags, counts = np.unique(all_diags, return_counts=True)
        min_count = np.min(counts)
        final_indices = []
        for diag in unique_diags:
            selected_idx = np.where(all_diags == diag)[0][:min_count]
            final_indices = np.concatenate((final_indices, selected_idx))
        final_indices = np.sort(final_indices).astype(int).tolist()
        balanced_dataset = [dataset[i] for i in final_indices]
        return balanced_dataset

    def cluster(self, key, model_name, dataset):
        X = []
        Y = []
        for diagnosis, file_base_path in dataset:
            data_path = f"{file_base_path}.{model_name}.pt"
            feature = torch.load(data_path)
            X += [feature]
            Y += [diagnosis]
        X = np.stack(X, axis=0)
        Y = np.array(Y)
        X_tsne = self.tsne.fit_transform(X)
        fig, ax = plt.subplots(
            1, 1, figsize=(20, 20), constrained_layout=True, sharex="col"
        )
        legends = np.unique(Y)
        for y in np.unique(Y):
            mask = Y == y
            ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=y)
        ax.set_title(f"{model_name} - TSNE")
        ax.legend(legends)
        self.save_or_retry(fig, f"{self.output_folder}/{key}.{model_name}.png")
        plt.close()

    def save_or_retry(self, fig, file_path):
        try:
            fig.savefig(file_path, bbox_inches="tight")
        except Exception:
            self.save_or_retry(fig, file_path)

    def load_diagnosis(self, session: Dict):
        diagnosis_list = list(map(Diagnosis.from_json, session["diagnosis"]))
        return self.diagnosis_map.from_int(
            self.diagnosis_map.most_severe(
                diagnosis_list,
                level=self.diagnosis_level,
            )
        ).name


if __name__ == "__main__":
    cluster = Cluster(
        balance_dataset=True,
        output_folder=Path("/home/storage/data/baseline_latents/16000/tsne"),
        diagnosis_level=0,
    )
    cluster.run(
        key="voiced",
        files=[
            Path(
                "/home/storage/data/baseline_latents/16000/features/voiced_train.json"
            ),
            Path("/home/storage/data/baseline_latents/16000/features/voiced_val.json"),
            Path("/home/storage/data/baseline_latents/16000/features/voiced_test.json"),
        ],
    )
    cluster.run(
        key="svd_a",
        files=[
            Path(
                "/home/storage/data/baseline_latents/16000/features/svd_a_n_train.json"
            ),
            Path("/home/storage/data/baseline_latents/16000/features/svd_a_n_val.json"),
            Path(
                "/home/storage/data/baseline_latents/16000/features/svd_a_n_test.json"
            ),
        ],
    )
    cluster.run(
        key="svd_i",
        files=[
            Path(
                "/home/storage/data/baseline_latents/16000/features/svd_i_n_train.json"
            ),
            Path("/home/storage/data/baseline_latents/16000/features/svd_i_n_val.json"),
            Path(
                "/home/storage/data/baseline_latents/16000/features/svd_i_n_test.json"
            ),
        ],
    )
    cluster.run(
        key="svd_u",
        files=[
            Path(
                "/home/storage/data/baseline_latents/16000/features/svd_i_n_train.json"
            ),
            Path("/home/storage/data/baseline_latents/16000/features/svd_i_n_val.json"),
            Path(
                "/home/storage/data/baseline_latents/16000/features/svd_i_n_test.json"
            ),
        ],
    )
    cluster.run(
        key="svd_aiu",
        files=[
            Path(
                "/home/storage/data/baseline_latents/16000/features/svd_a_n_train.json"
            ),
            Path(
                "/home/storage/data/baseline_latents/16000/features/svd_i_n_train.json"
            ),
            Path(
                "/home/storage/data/baseline_latents/16000/features/svd_u_n_train.json"
            ),
            Path("/home/storage/data/baseline_latents/16000/features/svd_a_n_val.json"),
            Path("/home/storage/data/baseline_latents/16000/features/svd_i_n_val.json"),
            Path("/home/storage/data/baseline_latents/16000/features/svd_u_n_val.json"),
            Path(
                "/home/storage/data/baseline_latents/16000/features/svd_a_n_test.json"
            ),
            Path(
                "/home/storage/data/baseline_latents/16000/features/svd_i_n_test.json"
            ),
            Path(
                "/home/storage/data/baseline_latents/16000/features/svd_u_n_test.json"
            ),
        ],
    )
