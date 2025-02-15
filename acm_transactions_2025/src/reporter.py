import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Literal
from sklearn.metrics import confusion_matrix
from .tasks_generator import TaskGenerator

class Reporter:

    def __init__(self, results_path: Path, task_generator: TaskGenerator) -> None:
        self.__results_path = results_path
        self.__task_generator = task_generator

    @classmethod
    def reports(cls):
        prefix = "report_"
        return Literal[tuple([k for k in cls.__dict__ if k.startswith(prefix)])]

    def report(self, method_name):
        self.__getattribute__(method_name)()

    def report_same_diag_different_class_system(self) -> None:
        """
        Comparing classification performance on one set of data with different
        classification systems.
        daSilvaMoura_2024:
            inputs: recurrent_paralysis, laryngitis, hyperfunctional_dysphonia, reinkes_edema, dysphonia, contact_pachydermia, cordectomy, functional_dysphonia, psychogenic_dysphonia, vocal_fold_polyp, normal

            daSilvaMoura:
                functional:
                    dysphonia
                    functional_dysphonia
                    hyperfunctional_dysphonia
                    psychogenic_dysphonia
                organic:
                    contact_pachydermia
                    cordectomy
                    laryngitis
                    recurrent_paralysis
                organofunctional:
                    reinkes_edema
                    vocal_fold_polyp
            USVAC:
                unclassified:
                    dysphonia
                    contact_pachydermia
                    cordectomy
                functional:
                    functional_dysphonia:
                        functional_dysphonia
                        psychogenic_dysphonia
                muscle_tension:
                    hyperfunctional_dysphonia
                organic:
                    organic_inflammatory:
                        organic_inflammatory_infective:
                            laryngitis
                    organic_neuro_muscular:
                        organic_neuro_muscular_peripheral_nervous_disorder:
                            recurrent_paralysis
                    organic_structural:
                        organic_structural_epithelial_propria:
                            reinkes_edema
                            vocal_fold_polyp
        Zaim_2023
            inputs: psychogenic_dysphonia, laryngitis, vocal_fold_polyp, normal
            Zaim:
                non_structural:
                    psychogenic_dysphonia
                structural:
                    laryngitis
                    vocal_fold_polyp
            daSilvaMoura:
                functional:
                    psychogenic_dysphonia
                organic:
                    laryngitis
                organofunctional:
                    vocal_fold_polyp
            USVAC:
                functional:
                    functional_dysphonia:
                        psychogenic_dysphonia
                organic:
                    organic_inflammatory:
                        organic_inflammatory_infective:
                            laryngitis
                    organic_structural:
                        organic_structural_epithelial_propria:
                            vocal_fold_polyp
        """
        df = pd.read_csv(f"{self.__results_path}/collated_results_self.csv")
        df["accuracy"] = (
            df["0_acc_balanced"]
            .combine_first(df["1_acc_balanced"])
            .combine_first(df["2_acc_balanced"])
            .combine_first(df["3_acc_balanced"])
            .combine_first(df["4_acc_balanced"])
        )
        # mask_cross_sys = df["exp_key"].str.startswith("USVAC-") | df[
        #     "exp_key"
        # ].str.startswith("daSilvaMoura-")
        mask_cross_sys = ~df["exp_key"].str.endswith("-with-unclassified")
        mask_tasks = df["task_key"].str.startswith("daSilvaMoura_2024-") | df[
            "task_key"
        ].str.startswith("Zaim_2023-")
        num_diags_mask = df["num_diag_levels"] == 1
        # df = df[mask_cross_sys & mask_tasks]
        df = df[mask_tasks & num_diags_mask & mask_cross_sys]
        df = df.sort_values(by=["exp_key", "accuracy"], ascending=False)
        df = df.groupby(by=["exp_key"]).head(1)
        df = df.sort_values(
            by=["task_key", "max_diag_level", "feature", "accuracy"], ascending=False
        ).reset_index(drop=True)
        leading_cols = [
            "exp_key",
            "task_key",
            "accuracy",
            "epoch",
            "max_diag_level",
            "feature",
            "num_diag_levels",
            "0_acc_balanced",
            "1_acc_balanced",
            "2_acc_balanced",
            "3_acc_balanced",
            "4_acc_balanced",
        ]
        col_names = df.columns.to_list()
        for col_name in leading_cols:
            col_names.remove(col_name)
        all_results = df[leading_cols + col_names]
        print(all_results)
        all_results.to_csv(
            f"{self.__results_path}/report_same_diag_different_class_system.csv",
            index=False,
        )
        groups = df.groupby(by=["task_key", "max_diag_level", "feature"], sort=False)
        with open(
            f"{self.__results_path}/report_same_diag_different_class_system.log", "w"
        ) as log:
            for group_idx, group in groups:
                log.write(repr(group_idx))
                log.write("\n")
                for row_idx, row in group.iterrows():
                    exp_key = row["exp_key"]
                    epoch = row["epoch"]
                    data_path = f"{self.__results_path}/self/{exp_key}/{epoch}.csv"
                    df = pd.read_csv(data_path)
                    labels, per_class_accuracy, balanced_accuracy, confusion_matrix = (
                        self.confusion(actual=df["actual"], predicted=df["predicted"])
                    )
                    log.write(f"\t{data_path}\n")
                    mat_str = repr(confusion_matrix)
                    log.write("\n".join([f"\t{line}" for line in mat_str.split("\n")]))
                    log.write("\n\t")
                    log.write(repr(dict(zip(labels, np.round(per_class_accuracy, 2)))))
                    log.write(f"\n\t{balanced_accuracy}\n\n")
                log.write("\n")

                
    def report_superset(self) -> None:
        """
        Comparing classification performance on the superset set of data with different
        classification systems.
        """
        df = pd.read_csv(f"{self.__results_path}/collated_results_self.csv")
        df["accuracy"] = (
            df["0_acc_balanced"]
            .combine_first(df["1_acc_balanced"])
            .combine_first(df["2_acc_balanced"])
            .combine_first(df["3_acc_balanced"])
            .combine_first(df["4_acc_balanced"])
        )
        mask_cross_sys = ~(
            df["exp_key"].str.endswith("-with-unclassified") |
            df["exp_key"].str.startswith("superset-Zaim_2023") |
            df["exp_key"].str.startswith("superset-Compton_2022")
        )
        mask_features = df["feature"].isin(["Wav2Vec", "UnispeechSAT", "MFCCDD"])
        mask_tasks = df["task_key"].isin(["a_n", "phrase", "all"])
        num_diag_levels_mask = df["num_diag_levels"] == 1
        max_diag_level_mask = df["max_diag_level"].isin([1, 2])
        df = df[mask_features & mask_tasks & num_diag_levels_mask & max_diag_level_mask & mask_cross_sys]
        df = df.sort_values(by=["exp_key", "accuracy"], ascending=False)
        df = df.groupby(by=["exp_key"]).head(1)
        df = df.sort_values(
            by=["task_key", "feature", "accuracy", "max_diag_level"], ascending=False
        ).reset_index(drop=True)
        leading_cols = [
            "exp_key",
            "task_key",
            "feature",
            "accuracy",
            "epoch",
            "max_diag_level",
            "num_diag_levels",
            "0_acc_balanced",
            "1_acc_balanced",
            "2_acc_balanced",
            "3_acc_balanced",
            "4_acc_balanced",
        ]
        col_names = df.columns.to_list()
        for col_name in leading_cols:
            col_names.remove(col_name)
        all_results = df[leading_cols + col_names]
        print(all_results)
        all_results.to_csv(
            f"{self.__results_path}/report_superset.csv",
            index=False,
        )
        groups = df.groupby(by=["task_key", "max_diag_level", "feature"], sort=False)
        results = {}
        for group_idx, group in tqdm(groups, total=len(groups), desc="Analysing"):
            task_key = group_idx[0]
            diagnosis_map = self.__task_generator.get_diagnosis_map('USVAC_2025', allow_unmapped=False)
            task = self.__task_generator.load_task(
                task=task_key,
                diag_level=None,
                diagnosis_map=diagnosis_map,
                load_audios=False,
            )
            for row_idx, row in group.iterrows():
                exp_key = row["exp_key"]
                epoch = row["epoch"]
                task_key = row['task_key']
                data_path = f"{self.__results_path}/self/{exp_key}/{epoch}.csv"
                df = pd.read_csv(data_path)[['actual', 'predicted', 'id']]
                df['diag'] = df['id'].apply(lambda x: task.test_label(id=x).name)
                df['correct'] = df['actual'] == df['predicted']
                res = df.groupby(by=['diag'])['correct'].apply(lambda x: (x==True).sum()/len(x))
                results[exp_key] = res.to_dict()
        results = pd.DataFrame.from_records(data=results).T
        results = results.reindex(all_results['exp_key'])
        # results = results.sort_values(by=list(results.columns), ascending=False)
        print(results.round(decimals=2).to_string())
        results.to_csv(f"{self.__results_path}/report_superset_analysis.csv")

    def report_input_tasks(self) -> None:
        # single task models here
        # combining input tasks improvement here
        pass

    def report_hierarchies(self) -> None:
        # difference by hierarchy
        pass

    def report_diag_levels(self) -> None:
        # difference in performance by level of hierarchy
        pass

    def report_data_availability(self) -> None:
        # difference in performance by availability of data
        pass

    def report_consensus(self) -> None:
        # difference in performance by consensus of classification
        pass

    def report_multi_task(self) -> None:
        # impact of multi task and multi crit
        # speed up in accuracy jump
        pass

    def confusion(self, actual, predicted):
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
        return labels, per_class_accuracy, balanced_accuracy, confusion
