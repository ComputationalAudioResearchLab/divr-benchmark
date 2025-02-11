import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.metrics import confusion_matrix
from .experiments import Runner


class Analyser:

    def __init__(self, results_path: Path) -> None:
        self.__results_path = results_path
        self.__collated_self_path = f"{results_path}/collated_results_self.csv"

    def analyse_self(self):
        df = pd.read_csv(self.__collated_self_path)
        single_task_df = df[df["num_diags"] == 1]
        best_epoch_per_feature = (
            single_task_df.sort_values(
                by=[
                    "max_diag_level",
                    "task_key",
                    "feature",
                    "0_acc_balanced",
                    "1_acc_balanced",
                    "2_acc_balanced",
                    "3_acc_balanced",
                    "4_acc_balanced",
                ],
                ascending=[True, True, True, False, False, False, False, False],
            )
            .groupby(by=["max_diag_level", "task_key", "feature"])
            .head(n=1)
        )
        # top N models per task
        top_n = 2
        best_models_per_task = (
            best_epoch_per_feature.sort_values(
                by=[
                    "max_diag_level",
                    "task_key",
                    "0_acc_balanced",
                    "1_acc_balanced",
                    "2_acc_balanced",
                    "3_acc_balanced",
                    "4_acc_balanced",
                ],
                ascending=[True, True, False, False, False, False, False],
            )
            .groupby(by=["max_diag_level", "task_key"])
            .head(n=top_n)
        )
        best_models_per_task.to_csv(
            f"{self.__results_path}/best_single_task_results.csv", index=False
        )

    def collate_self(self):
        results_path = Path(f"{self.__results_path}/self")
        results = sorted(list(results_path.rglob("*.csv")))
        all_results = []

        for result_file in tqdm(results, desc="collating results"):
            exp_key = result_file.parent.stem
            (
                task_key,
                diag_levels,
                feature_cls,
                num_epochs,
                batch_size,
                trainer_cls,
                lr,
            ) = Runner._exp[exp_key]
            row = {
                "exp_key": exp_key,
                "epoch": result_file.stem,
                "task_key": task_key,
                "num_diags": len(diag_levels),
                "max_diag_level": max(diag_levels),
                "feature": feature_cls.__name__,
            }
            df = pd.read_csv(result_file)
            col_names = df.columns
            to_process = []
            if "actual" in col_names:
                level = row["exp_key"].rsplit("_", maxsplit=1)[1]
                actual = df["actual"]
                predicted = df["predicted"]
                to_process += [(level, actual, predicted)]
            else:
                for level in range(4):
                    if f"actual_{level}" in col_names:
                        actual = df[f"actual_{level}"]
                        predicted = df[f"predicted_{level}"]
                        to_process += [(level, actual, predicted)]
            for level, actual, predicted in to_process:
                labels, per_class_accuracy, balanced_accuracy = self.balanced_accuracy(
                    actual, predicted
                )
                row[f"{level}_acc_balanced"] = balanced_accuracy
                for label, class_acc in zip(labels, per_class_accuracy):
                    row[f"{level}_acc_{label}"] = class_acc
            all_results += [row]
        all_results = pd.DataFrame.from_records(all_results)
        all_results = all_results.sort_values(
            by=[
                "4_acc_balanced",
                "3_acc_balanced",
                "2_acc_balanced",
                "1_acc_balanced",
                "0_acc_balanced",
                "exp_key",
                "epoch",
            ],
            ascending=False,
        )
        leading_cols = [
            "exp_key",
            "epoch",
            "task_key",
            "feature",
            "num_diags",
            "max_diag_level",
            "0_acc_balanced",
            "1_acc_balanced",
            "2_acc_balanced",
            "3_acc_balanced",
            "4_acc_balanced",
        ]
        col_names = all_results.columns.to_list()
        for col_name in leading_cols:
            col_names.remove(col_name)
        all_results = all_results[leading_cols + col_names]
        all_results.to_csv(self.__collated_self_path, index=False)

    def balanced_accuracy(self, actual, predicted):
        class_weights = actual.value_counts()
        labels = class_weights.index.to_list()
        confusion = confusion_matrix(
            y_true=actual,
            y_pred=predicted,
            labels=labels,
        )
        total_per_class = np.maximum(1, confusion.sum(axis=1))
        corrects = confusion.diagonal()
        per_class_accuracy = corrects / total_per_class
        balanced_accuracy = per_class_accuracy.mean()
        return labels, per_class_accuracy, balanced_accuracy
