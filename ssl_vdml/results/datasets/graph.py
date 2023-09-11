import json
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict
from diagnosis import Diagnosis, DiagnosisMap
import matplotlib.pyplot as plt
import seaborn as sns


class Grapher:
    def __init__(
        self,
        balance_dataset: bool,
        diagnosis_level: int,
    ) -> None:
        self.balance_dataset = balance_dataset
        self.diagnosis_level = diagnosis_level
        self.diagnosis_map = DiagnosisMap()

    def run(self, key: str, train_files: List[Path], val_files: List[Path]):
        train_dataset = self.split_age_diagnosis(self.load_dataset(files=train_files))
        val_dataset = self.split_age_diagnosis(self.load_dataset(files=val_files))
        fig, ax = plt.subplots(
            1, 2, figsize=(7, 3), constrained_layout=True, sharey="row"
        )
        sns.barplot(
            data=train_dataset, x="diagnosis", y="counts", hue="gender", ax=ax[0]
        )
        ax[0].set_title("train dataset")
        sns.barplot(data=val_dataset, x="diagnosis", y="counts", hue="gender", ax=ax[1])
        ax[0].get_legend().remove()
        ax[1].set_title("val dataset")
        ax[0].set_xlabel(None)
        ax[1].set_xlabel(None)
        fig.suptitle(f"Dataset: {key}")
        fig.savefig(f"dataset_{key}.png")

    def split_age_diagnosis(self, dataset):
        grouped = dataset.groupby(["diagnosis", "gender"]).size()
        print(grouped)
        return grouped.reset_index(name="counts")

    def load_dataset(self, files: List[Path]):
        dataset = []
        for data_file in files:
            with open(data_file, "r") as config_data:
                config = json.load(config_data)
                for i, session in tqdm(
                    enumerate(config), total=len(config), desc="loading data"
                ):
                    diagnosis = self.load_diagnosis(session=session)
                    gender = session["gender"]
                    for session_file in session["files"]:
                        dataset += [{"diagnosis": diagnosis, "gender": gender}]
        if not self.balance_dataset:
            return dataset

        all_diags = np.array([data["diagnosis"] for data in dataset])
        unique_diags, counts = np.unique(all_diags, return_counts=True)
        min_count = np.min(counts)
        final_indices = []
        for diag in unique_diags:
            selected_idx = np.where(all_diags == diag)[0][:min_count]
            final_indices = np.concatenate((final_indices, selected_idx))
        final_indices = np.sort(final_indices).astype(int).tolist()
        balanced_dataset = [dataset[i] for i in final_indices]
        return pd.DataFrame(balanced_dataset)

    def load_diagnosis(self, session: Dict):
        diagnosis_list = list(map(Diagnosis.from_json, session["diagnosis"]))
        return self.diagnosis_map.from_int(
            self.diagnosis_map.most_severe(
                diagnosis_list,
                level=self.diagnosis_level,
            )
        ).name


if __name__ == "__main__":
    grapher = Grapher(balance_dataset=True, diagnosis_level=0)
    grapher.run(
        key="svd",
        train_files=[
            Path(
                "/home/storage/data/baseline_latents/16000/features/svd_a_n_train.json"
            ),
            Path(
                "/home/storage/data/baseline_latents/16000/features/svd_i_n_train.json"
            ),
            Path(
                "/home/storage/data/baseline_latents/16000/features/svd_u_n_train.json"
            ),
        ],
        val_files=[
            Path("/home/storage/data/baseline_latents/16000/features/svd_a_n_val.json"),
            Path("/home/storage/data/baseline_latents/16000/features/svd_i_n_val.json"),
            Path("/home/storage/data/baseline_latents/16000/features/svd_u_n_val.json"),
        ],
    )
    grapher.run(
        key="voiced",
        train_files=[
            Path(
                "/home/storage/data/baseline_latents/16000/features/voiced_train.json"
            ),
        ],
        val_files=[
            Path("/home/storage/data/baseline_latents/16000/features/voiced_val.json"),
        ],
    )
