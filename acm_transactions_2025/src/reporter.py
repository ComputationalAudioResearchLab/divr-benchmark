import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from typing import Literal
import matplotlib.pyplot as plt
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
        def categorize(exp_key: str):
            if exp_key.startswith('superset-CaRLab_2025'):
                return 'CaRLab_2025'
            if exp_key.startswith('superset-daSilvaMoura_2024'):
                return 'daSilvaMoura_2024'
            if exp_key.endswith('_1'):
                return 'USVAC_2025_1'
            if exp_key.endswith('_2'):
                return 'USVAC_2025_2'
            raise ValueError(f"Unexpted exp_key: {exp_key}")
        
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
        df['category'] = df['exp_key'].apply(categorize)
        df = df.sort_values(by=["exp_key", "accuracy"], ascending=False)
        df = df.groupby(by=["exp_key"]).head(1)
        df = df.sort_values(
            by=["task_key", "feature", "accuracy", "max_diag_level"], ascending=False
        ).reset_index(drop=True).set_index(keys=['exp_key'])
        def accuracy_delta(group: pd.DataFrame):
            carlab_result = group[group.index.str.startswith('superset-CaRLab_2025')].iloc[0]
            return group - carlab_result
        df['accuracy_delta'] = df.groupby(by=['task_key', 'feature'])['accuracy'].apply(accuracy_delta).reset_index().set_index(keys=['exp_key'])['accuracy']
        leading_cols = [
            "category",
            "task_key",
            "feature",
            "accuracy",
            "accuracy_delta",
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
        
        all_results.to_csv(
            f"{self.__results_path}/report_superset.csv",
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
                exp_key = row_idx
                epoch = row["epoch"]
                task_key = row['task_key']
                data_path = f"{self.__results_path}/self/{exp_key}/{epoch}.csv"
                df = pd.read_csv(data_path)[['actual', 'predicted', 'id']]
                df['diag'] = df['id'].apply(lambda x: task.test_label(id=x).name)
                df['correct'] = df['actual'] == df['predicted']
                res = df.groupby(by=['diag'])['correct'].apply(lambda x: (x==True).sum()/len(x))
                results[exp_key] = res.to_dict()
        results = pd.DataFrame.from_records(data=results).T
        results = results.reindex(all_results.index)
        results['task_key'] = all_results['task_key']
        results['feature'] = all_results['feature']
        results['category'] = all_results['category']
        print(results.round(decimals=2).to_string())
        results.to_csv(f"{self.__results_path}/report_superset_analysis.csv")
        def recall_delta(group: pd.DataFrame):
            carlab_result = group[group.index.str.startswith('superset-CaRLab_2025')].iloc[0]
            return group - carlab_result
        delta_results = results.drop(columns=['category']).groupby(by=['task_key', 'feature']).apply(recall_delta).reset_index().set_index(keys=['exp_key'])
        delta_results['category'] = all_results['category']
        delta_results.to_csv(f"{self.__results_path}/report_superset_delta_analysis.csv")

    def report_superset_fig_confusions(self):
        chosen_paths = {
            "(a)": f"{self.__results_path}/self/superset-CaRLab_2025-unispeechSAT_phrase_1/34.csv",
            "(b)": f"{self.__results_path}/self/superset-daSilvaMoura_2024-unispeechSAT_phrase_1/20.csv",
            "(c.1)": f"{self.__results_path}/self/unispeechSAT_phrase_1/5.csv",
            "(c.2)": f"{self.__results_path}/self/unispeechSAT_phrase_2/9.csv",
        }
        fig, ax = plt.subplots(1, 4, figsize=(24, 7), constrained_layout=True)
        for idx, (key, val) in enumerate(chosen_paths.items()):
            df = pd.read_csv(val)
            labels, _, _, confusion = self.confusion(
                actual=df['actual'],
                predicted=df['predicted'],
            )
            sns.heatmap(
                data=confusion,
                annot=True,
                ax=ax[idx],
                cbar=False,
                fmt='g',
                cmap="YlGnBu",
                annot_kws={"fontsize":16}
            )
            ax[idx].set_xticklabels(labels, rotation=90, fontsize=15)
            ax[idx].set_yticklabels(labels, rotation=0, fontsize=15)
            ax[idx].set_title(key, fontsize=18)
            ax[idx].set_xlabel('Predicted', fontsize=18)
        ax[0].set_ylabel('Actual', fontsize=18)
        fig_path = f"{self.__results_path}/superset_fig_confusions.png"
        fig.suptitle('Confusion matrix for different classification systems', fontsize=24)
        fig.align_xlabels()
        fig.savefig(fig_path, bbox_inches='tight')
        print(f"Figure saved at: {fig_path}")
    
    def report_superset_balanced_acc_over_tasks(self):
        df = pd.read_csv(
            f"{self.__results_path}/report_superset.csv",
        )[['category', 'task_key', 'feature', 'accuracy']]
        rename_categories = {
            'CaRLab_2025': '(a)',
            'daSilvaMoura_2024': '(b)',
            'USVAC_2025_1': '(c.1)',
            'USVAC_2025_2': '(c.2)',
        }
        df['category'] = df['category'].apply(rename_categories.get)
        df['accuracy'] = (df['accuracy'] * 100).round(decimals=2)
        print(df)
        fig, ax = plt.subplots(3, 1, figsize=(10, 6), constrained_layout=True, sharex='col')
        groups = df.groupby(by=['task_key'])
        total_groups = len(groups)
        for idx, ((task_key,), group) in enumerate(groups):
            sns.barplot(
                data=group,
                x='feature',
                y='accuracy',
                hue='category',
                hue_order=rename_categories.values(),
                ax=ax[idx],
                palette='GnBu',
                legend=idx == total_groups-1,
            )
            ax[idx].set_ylabel(task_key, fontsize=16)
            ax[idx].set_yticklabels(ax[idx].get_yticklabels(), fontsize=14)
            ax[idx].set_xlabel(None)
            ax[idx].set_ylim(35, 75)
            for c in ax[idx].containers:
                ax[idx].bar_label(c, fontsize=10)
        ax[-1].set_xticklabels(ax[idx].get_xticklabels(), fontsize=14)
        sns.move_legend(
            ax[-1], "lower center",
            bbox_to_anchor=(.5, -0.50),
            ncol=4, title=None, frameon=False, fontsize=14,
        )
        fig.suptitle("Balanced accuracy (%) for classification systems", fontsize=20)
        fig_path = f"{self.__results_path}/superset_balanced_acc_over_tasks.png"
        fig.savefig(fig_path, bbox_inches='tight')
        print(f"Figure saved at: {fig_path}")

    def report_superset_average_recall_diff(self): 
        df = pd.read_csv(
            f"{self.__results_path}/report_superset_delta_analysis.csv",
            index_col='exp_key',
        )
        non_agg_cols = ['task_key', 'feature', 'exp_key', 'category']
        agg_cols = [c for c in df.columns if c not in non_agg_cols]
        def prep(group: pd.Series) -> str:
            mean = group.mean()*100
            std = group.std()*100
            return f"{mean:.2f}±{std:.2f}"
        data = df.groupby(by='category')[agg_cols].agg(prep)
        data = data.T[[
            'CaRLab_2025',
            'daSilvaMoura_2024',
            'USVAC_2025_1',
            'USVAC_2025_2',
        ]].reset_index(names=['category'])
        print(data)
        out_path = f"{self.__results_path}/superset_average_recall_diff.csv"
        data.to_csv(out_path, index=False)
        print(f"Saved at: {out_path}")

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
        labels.remove('normal')
        labels = ['normal'] + sorted(labels)
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
