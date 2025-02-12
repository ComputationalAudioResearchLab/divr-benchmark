import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from .experiments import Runner


class Analyser:

    def __init__(self, results_path: Path) -> None:
        self.__results_path = results_path
        self.__collated_self_path = f"{results_path}/collated_results_self.csv"
        self.__collated_cross_path = f"{results_path}/collated_results_cross.csv"

    def plot_results_self(self):
        acm_path = (
            "/home/storage/acm_transactions_2025/results/collated_results_self.csv"
        )
        int_path = "/home/storage/interspeech_2025/results/collated_results_self.csv"
        df_acm = pd.read_csv(acm_path)
        df_int = pd.read_csv(int_path)
        print(df_acm)
        print(df_int)

        df = pd.concat([df_acm, df_int]).reset_index()
        print(df)
        num_diags_selection = df["num_diags"] == 1
        diag_level_selection = df["max_diag_level"].isin([0, 1, 4])
        feature_selection = df["feature"].isin(["MFCCDD", "Wav2Vec", "UnispeechSAT"])
        input_task_selection = df["task_key"].isin(["a_n", "phrase"])
        selection = (
            num_diags_selection
            & diag_level_selection
            & feature_selection
            & input_task_selection
        )
        selected_df = df[selection]
        print(selected_df)
        selected_df["injection"] = (
            df[["extra_db", "percent_injection"]]
            .fillna("")
            .astype(str)
            .agg("-".join, axis=1)
        )
        selected_df["accuracy"] = (
            selected_df["0_acc_balanced"]
            .combine_first(selected_df["1_acc_balanced"])
            .combine_first(selected_df["2_acc_balanced"])
            .combine_first(selected_df["3_acc_balanced"])
            .combine_first(selected_df["4_acc_balanced"])
        )
        grouping = ["max_diag_level", "task_key", "feature"]
        selected_df = (
            selected_df.sort_values(by=["accuracy"])
            .groupby(by=grouping + ["injection"])
            .head(n=1)
        )
        G = selected_df.sort_values(by=["injection"]).groupby(by=grouping)[
            ["injection", "accuracy"]
        ]
        selected_df.sort_values(by=grouping + ["accuracy"], ascending=False)[
            grouping + ["injection", "epoch", "accuracy"]
        ].to_csv(
            f"{self.__results_path}/self_results.csv",
            index=False,
        )
        total_groups = len(G)
        fig, ax = plt.subplots(
            total_groups,
            1,
            figsize=(10, total_groups * 3),
            constrained_layout=True,
            sharex="col",
        )
        for idx, (g_idx, group) in enumerate(G):
            sns.barplot(data=group, x="injection", y="accuracy", ax=ax[idx])
            ax[idx].set_ylabel(g_idx)

            print(
                g_idx,
                group.sort_values(by="accuracy", ascending=False)
                .head(n=1)[["injection", "accuracy"]]
                .to_dict(),
            )
        ax[idx].set_xticklabels(ax[idx].get_xticklabels(), rotation=90)
        fig.savefig(f"{self.__results_path}/result_self.png")

    def plot_results_cross(self):
        acm_path = (
            "/home/storage/acm_transactions_2025/results/collated_results_cross.csv"
        )
        int_path = "/home/storage/interspeech_2025/results/collated_results_cross.csv"
        df_acm = pd.read_csv(acm_path)
        df_int = pd.read_csv(int_path)
        print(df_acm)
        print(df_int)

        df = pd.concat([df_acm, df_int]).reset_index()
        print(df)
        diag_level_selection = df["max_diag_level"].isin([0, 1, 4])
        task_selection = df["cross_task_key"].isin(
            ["cross_test_meei", "cross_test_voiced"]
        )
        feature_selection = df["feature"].isin(["MFCCDD", "Wav2Vec", "UnispeechSAT"])
        input_task_selection = df["input_task_key"].isin(["a_n", "phrase"])
        selection = (
            diag_level_selection
            & task_selection
            & feature_selection
            & input_task_selection
        )
        selected_df = df[selection]
        print(selected_df)
        selected_df["injection"] = (
            df[["extra_db", "percent_injection"]]
            .fillna("")
            .astype(str)
            .agg("-".join, axis=1)
        )
        selected_df["accuracy"] = (
            selected_df["0_acc_balanced"]
            .combine_first(selected_df["1_acc_balanced"])
            .combine_first(selected_df["2_acc_balanced"])
            .combine_first(selected_df["3_acc_balanced"])
            .combine_first(selected_df["4_acc_balanced"])
        )
        grouping = ["cross_task_key", "max_diag_level", "input_task_key", "feature"]
        G = selected_df.sort_values(by=["injection"]).groupby(by=grouping)[
            ["injection", "accuracy"]
        ]
        total_groups = len(G)
        fig, ax = plt.subplots(
            total_groups,
            1,
            figsize=(10, total_groups * 3),
            constrained_layout=True,
            sharex="col",
        )
        for idx, (g_idx, group) in enumerate(G):
            sns.barplot(data=group, x="injection", y="accuracy", ax=ax[idx])
            ax[idx].set_ylabel(g_idx)

            print(
                g_idx,
                group.sort_values(by="accuracy", ascending=False)
                .head(n=1)[["injection", "accuracy"]]
                .to_dict(),
            )
        ax[idx].set_xticklabels(ax[idx].get_xticklabels(), rotation=90)
        fig.savefig(f"{self.__results_path}/result.png")

    def collate_cross(self):
        results_path = Path(f"{self.__results_path}/cross")
        results = sorted(list(results_path.rglob("*.csv")))
        all_results = []

        for result_file in tqdm(results, desc="collating results"):
            cross_task_key = result_file.parent.stem
            exp_key = result_file.parent.parent.stem
            (
                task_key,
                diag_levels,
                feature_cls,
                num_epochs,
                batch_size,
                percent_injection,
                extra_db,
            ) = Runner._exp[exp_key]
            row = {
                "cross_task_key": cross_task_key,
                "exp_key": exp_key,
                "epoch": result_file.stem,
                "input_task_key": task_key,
                "num_diags": len(diag_levels),
                "max_diag_level": max(diag_levels),
                "feature": feature_cls.__name__,
                "extra_db": extra_db.__name__,
                "percent_injection": percent_injection,
            }
            df = pd.read_csv(result_file)
            col_names = df.columns
            to_process = []
            if "actual" in col_names:
                level = row["exp_key"].split("_")[2]
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
                ok_mask = actual != "unclassified"
                actual = actual[ok_mask]
                predicted = predicted[ok_mask]
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
                "cross_task_key",
                "max_diag_level",
                "4_acc_balanced",
                "1_acc_balanced",
                "0_acc_balanced",
                "exp_key",
                "epoch",
            ],
            ascending=False,
        )
        leading_cols = [
            "cross_task_key",
            "max_diag_level",
            "exp_key",
            "extra_db",
            "percent_injection",
            "epoch",
            "0_acc_balanced",
            "1_acc_balanced",
            "4_acc_balanced",
            "input_task_key",
            "feature",
            "num_diags",
        ]
        col_names = all_results.columns.to_list()
        for col_name in leading_cols:
            col_names.remove(col_name)
        all_results = all_results[leading_cols + col_names]
        all_results.to_csv(self.__collated_cross_path, index=False)

    def analyse_self(self):
        df = pd.read_csv(self.__collated_self_path)
        single_task_df = df[df["num_diags"] == 1]
        best_epoch_per_feature = (
            single_task_df.sort_values(
                by=[
                    "exp_key",
                    "0_acc_balanced",
                    "1_acc_balanced",
                    "4_acc_balanced",
                ],
                ascending=[True, False, False, False],
            )
            .groupby(by=["exp_key"])
            .head(n=1)
        )
        best_epoch_per_feature.to_csv(
            f"{self.__results_path}/best_epoch_self.csv", index=False
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
                percent_injection,
                extra_db,
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
                level = row["exp_key"].split("_")[2]
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
            "4_acc_balanced",
        ]
        col_names = all_results.columns.to_list()
        for col_name in leading_cols:
            col_names.remove(col_name)
        all_results = all_results[leading_cols + col_names]
        all_results.to_csv(self.__collated_self_path, index=False)

    def balanced_accuracy(self, actual, predicted):
        actual_labels = actual.value_counts().index.to_list()
        predicted_labels = predicted.value_counts().index.to_list()
        labels = list(set(actual_labels + predicted_labels))
        confusion = confusion_matrix(
            y_true=actual,
            y_pred=predicted,
            labels=labels,
        )
        total_per_class = np.maximum(1, confusion.sum(axis=1))
        corrects = confusion.diagonal()
        selected_totals = []
        selected_corrects = []
        selected_labels = []
        for idx, label in enumerate(labels):
            if label in actual_labels:
                selected_totals += [total_per_class[idx]]
                selected_corrects += [corrects[idx]]
                selected_labels += [labels[idx]]
        selected_corrects = np.array(selected_corrects)
        selected_totals = np.array(selected_totals)
        per_class_accuracy = selected_corrects / selected_totals
        balanced_accuracy = per_class_accuracy.mean()
        return selected_labels, per_class_accuracy, balanced_accuracy
