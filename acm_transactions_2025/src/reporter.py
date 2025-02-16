import numpy as np
import pandas as pd
import networkx as nx
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

    def report_autogen_hierarchy(self):
        df = pd.read_csv(f"{self.__results_path}/self/unispeechSAT_phrase_4/71.csv")
        labels, _, _, confusion = self.confusion(
            actual=df["actual"],
            predicted=df["predicted"],
        )
        print(labels)
        print(confusion)

        df = pd.read_csv(f"{self.__results_path}/autogen_hierarchy_dists.csv")
        all_nodes = sorted(set(df["from"].tolist() + df["to"].tolist()))
        print(all_nodes)
        edge_weights = np.arange(1, 8) / 7
        all_edges = []
        for node in all_nodes:
            mask = (df["from"] == node) | (df["to"] == node)
            edges = df[mask].sort_values(by=["mean_dist"], ascending=False)
            edges["weight"] = edge_weights
            for row_idx, row in edges.iterrows():
                node_from = row["from"]
                node_to = row["to"]
                if node_to == node:
                    node_from = row["to"]
                    node_to = row["from"]
                dist_mean = row["mean_dist"]
                dist_std = row["std_dist"]
                weight = row["weight"]
                label = f"{dist_mean:.2f}±{dist_std:.2f}"
                all_edges += [
                    (
                        node_from,
                        node_to,
                        weight,
                        label,
                    )
                ]
        all_edges = pd.DataFrame.from_records(
            all_edges, columns=["from", "to", "weight", "label"]
        )
        G = nx.DiGraph()
        for row_idx, row in all_edges.sort_values(by=["from", "weight"]).iterrows():
            G.add_edge(row["from"], row["to"], weight=row["weight"], label=row["label"])
        fig, ax = plt.subplots(
            2,
            1,
            figsize=(10, 18),
            constrained_layout=True,
            sharex=False,
            gridspec_kw={
                "height_ratios": [1, 1.5],
            },
        )
        pos = nx.circular_layout(G=G)
        sns.heatmap(
            data=confusion,
            cmap="YlGnBu",
            ax=ax[0],
            fmt="g",
            annot=True,
            annot_kws={"fontsize": 16},
        )
        ax[0].set_xticklabels(labels, rotation=90, fontsize=15)
        ax[0].set_yticklabels(labels, rotation=0, fontsize=15)
        ax[0].set_ylabel("Actual", fontsize=18)
        ax[0].set_xlabel("Predicted", fontsize=18)
        ax[0].set_title("(a) Confusion Matrix of best model", fontsize=20)
        cbar = plt.colormaps.get_cmap("Blues")
        ax[1].set_position([0.1, 0.1, 0.8, 0.35])
        edges = G.edges()
        colors = [cbar(G[u][v]["weight"] ** 2) for u, v in edges]
        # weights = [(G[u][v]['weight']**8) for u,v in edges]
        # weights = [1 if (G[u][v]['weight'] == 1) else 0 for u,v in edges]
        weights = [
            (3 if (G[u][v]["weight"]) > 0.9 else G[u][v]["weight"] ** 2)
            for u, v in edges
        ]
        # weights = [1 for u,v in edges]
        ax[1].margins(0.14)
        nx.draw(
            G=G,
            pos=pos,
            ax=ax[1],
            # edges=edges,
            edge_color=colors,
            width=weights,
            with_labels=True,
            font_size=14,
            node_size=1600,
            node_color="#CBF1F5",
        )
        ax[1].set_title("(b) Label confusion frequency network", fontsize=20)
        fig.add_artist(plt.Line2D([0, 1], [0.475, 0.475], color="#393E46", linewidth=1))
        fig_path = f"{self.__results_path}/autogen_hierarchy.png"
        fig.savefig(fig_path, bbox_inches="tight")
        print(f"Saved at {fig_path}")

    def report_superset(self) -> None:
        """
        Comparing classification performance on the superset set of data with different
        classification systems.
        """
        df = pd.read_csv(f"{self.__results_path}/collated_results_self.csv")

        def categorize(exp_key: str):
            if exp_key.startswith("superset-CaRLab_2025"):
                return "CaRLab_2025"
            if exp_key.startswith("superset-daSilvaMoura_2024"):
                return "daSilvaMoura_2024"
            if exp_key.endswith("_1"):
                return "USVAC_2025_1"
            if exp_key.endswith("_2"):
                return "USVAC_2025_2"
            raise ValueError(f"Unexpted exp_key: {exp_key}")

        df["accuracy"] = (
            df["0_acc_balanced"]
            .combine_first(df["1_acc_balanced"])
            .combine_first(df["2_acc_balanced"])
            .combine_first(df["3_acc_balanced"])
            .combine_first(df["4_acc_balanced"])
        )
        mask_cross_sys = ~(
            df["exp_key"].str.endswith("-with-unclassified")
            | df["exp_key"].str.startswith("superset-Zaim_2023")
            | df["exp_key"].str.startswith("superset-Compton_2022")
        )
        mask_features = df["feature"].isin(["Wav2Vec", "UnispeechSAT", "MFCCDD"])
        mask_tasks = df["task_key"].isin(["a_n", "phrase", "all"])
        num_diag_levels_mask = df["num_diag_levels"] == 1
        max_diag_level_mask = df["max_diag_level"].isin([1, 2])
        df = df[
            mask_features
            & mask_tasks
            & num_diag_levels_mask
            & max_diag_level_mask
            & mask_cross_sys
        ]
        df["category"] = df["exp_key"].apply(categorize)
        df = df.sort_values(by=["exp_key", "accuracy"], ascending=False)
        df = df.groupby(by=["exp_key"]).head(1)
        df = (
            df.sort_values(
                by=["task_key", "feature", "accuracy", "max_diag_level"],
                ascending=False,
            )
            .reset_index(drop=True)
            .set_index(keys=["exp_key"])
        )

        def accuracy_delta(group: pd.DataFrame):
            carlab_result = group[
                group.index.str.startswith("superset-CaRLab_2025")
            ].iloc[0]
            return group - carlab_result

        df["accuracy_delta"] = (
            df.groupby(by=["task_key", "feature"])["accuracy"]
            .apply(accuracy_delta)
            .reset_index()
            .set_index(keys=["exp_key"])["accuracy"]
        )
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
            diagnosis_map = self.__task_generator.get_diagnosis_map(
                "USVAC_2025", allow_unmapped=False
            )
            task = self.__task_generator.load_task(
                task=task_key,
                diag_level=None,
                diagnosis_map=diagnosis_map,
                load_audios=False,
            )
            for row_idx, row in group.iterrows():
                exp_key = row_idx
                epoch = row["epoch"]
                task_key = row["task_key"]
                data_path = f"{self.__results_path}/self/{exp_key}/{epoch}.csv"
                df = pd.read_csv(data_path)[["actual", "predicted", "id"]]
                df["diag"] = df["id"].apply(lambda x: task.test_label(id=x).name)
                df["correct"] = df["actual"] == df["predicted"]
                res = df.groupby(by=["diag"])["correct"].apply(
                    lambda x: (x == True).sum() / len(x)
                )
                results[exp_key] = res.to_dict()
        results = pd.DataFrame.from_records(data=results).T
        results = results.reindex(all_results.index)
        results["task_key"] = all_results["task_key"]
        results["feature"] = all_results["feature"]
        results["category"] = all_results["category"]
        print(results.round(decimals=2).to_string())
        results.to_csv(f"{self.__results_path}/report_superset_analysis.csv")

        def recall_delta(group: pd.DataFrame):
            carlab_result = group[
                group.index.str.startswith("superset-CaRLab_2025")
            ].iloc[0]
            return group - carlab_result

        delta_results = (
            results.drop(columns=["category"])
            .groupby(by=["task_key", "feature"])
            .apply(recall_delta)
            .reset_index()
            .set_index(keys=["exp_key"])
        )
        delta_results["category"] = all_results["category"]
        delta_results.to_csv(
            f"{self.__results_path}/report_superset_delta_analysis.csv"
        )

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
                actual=df["actual"],
                predicted=df["predicted"],
            )
            sns.heatmap(
                data=confusion,
                annot=True,
                ax=ax[idx],
                cbar=False,
                fmt="g",
                cmap="YlGnBu",
                annot_kws={"fontsize": 16},
            )
            ax[idx].set_xticklabels(labels, rotation=90, fontsize=15)
            ax[idx].set_yticklabels(labels, rotation=0, fontsize=15)
            ax[idx].set_title(key, fontsize=18)
            ax[idx].set_xlabel("Predicted", fontsize=18)
        ax[0].set_ylabel("Actual", fontsize=18)
        fig_path = f"{self.__results_path}/superset_fig_confusions.png"
        fig.suptitle(
            "Confusion matrix for different classification systems", fontsize=24
        )
        fig.align_xlabels()
        fig.savefig(fig_path, bbox_inches="tight")
        print(f"Figure saved at: {fig_path}")

    def report_superset_balanced_acc_over_tasks(self):
        df = pd.read_csv(
            f"{self.__results_path}/report_superset.csv",
        )[["category", "task_key", "feature", "accuracy"]]
        rename_categories = {
            "CaRLab_2025": "(a)",
            "daSilvaMoura_2024": "(b)",
            "USVAC_2025_1": "(c.1)",
            "USVAC_2025_2": "(c.2)",
        }
        df["category"] = df["category"].apply(rename_categories.get)
        df["accuracy"] = (df["accuracy"] * 100).round(decimals=2)
        print(df)
        fig, ax = plt.subplots(
            3, 1, figsize=(10, 6), constrained_layout=True, sharex="col"
        )
        groups = df.groupby(by=["task_key"])
        total_groups = len(groups)
        for idx, ((task_key,), group) in enumerate(groups):
            sns.barplot(
                data=group,
                x="feature",
                y="accuracy",
                hue="category",
                hue_order=rename_categories.values(),
                ax=ax[idx],
                palette="GnBu",
                legend=idx == total_groups - 1,
            )
            ax[idx].set_ylabel(task_key, fontsize=18)
            ax[idx].set_yticklabels(ax[idx].get_yticklabels(), fontsize=14)
            ax[idx].set_xlabel(None)
            ax[idx].set_ylim(35, 75)
            for c in ax[idx].containers:
                ax[idx].bar_label(c, fontsize=14)
        ax[-1].set_xticklabels(ax[idx].get_xticklabels(), fontsize=18)
        sns.move_legend(
            ax[-1],
            "lower center",
            bbox_to_anchor=(0.5, -0.6),
            ncol=4,
            title=None,
            frameon=False,
            fontsize=18,
        )
        fig.suptitle("Balanced accuracy (%) for classification systems", fontsize=20)
        fig_path = f"{self.__results_path}/superset_balanced_acc_over_tasks.png"
        fig.savefig(fig_path, bbox_inches="tight")
        print(f"Figure saved at: {fig_path}")

    def report_superset_average_recall_diff(self):
        df = pd.read_csv(
            f"{self.__results_path}/report_superset_delta_analysis.csv",
            index_col="exp_key",
        )
        non_agg_cols = ["task_key", "feature", "exp_key", "category"]
        agg_cols = [c for c in df.columns if c not in non_agg_cols]

        def prep(group: pd.Series) -> str:
            mean = round((group.mean() * 100))
            std = round((group.std() * 100))
            if mean < 0:
                return f"\\textcolor{{red}}{{${mean}\pm{std}$}}"
            else:
                return f"${mean}\pm{std}$"

        data = df.groupby(by="category")[agg_cols].agg(prep)
        data = data.T[
            [
                "daSilvaMoura_2024",
                "USVAC_2025_1",
                "USVAC_2025_2",
            ]
        ].reset_index(names=["category"])
        print(data)
        out_path = f"{self.__results_path}/superset_average_recall_diff.csv"
        data.to_csv(out_path, index=False)
        print(f"Saved at: {out_path}")

    def report_input_features(self) -> None:
        # single task models here
        # combining input features improvement here
        df = pd.read_csv(f"{self.__results_path}/collated_results_self.csv", header=0)
        mask = ~(
            df["exp_key"].str.endswith("-with-unclassified")
            | df["exp_key"].str.startswith("USVAC-")
            | df["exp_key"].str.startswith("daSilvaMoura-")
            | df["exp_key"].str.startswith("superset-")
        )
        df = df[mask]
        single_task_results = df[df["num_diag_levels"] == 1]
        single_task_results["accuracy"] = (
            single_task_results["0_acc_balanced"]
            .combine_first(single_task_results["1_acc_balanced"])
            .combine_first(single_task_results["2_acc_balanced"])
            .combine_first(single_task_results["3_acc_balanced"])
            .combine_first(single_task_results["4_acc_balanced"])
        )
        best_model_epoch = (
            single_task_results.sort_values(
                by=["feature", "max_diag_level", "accuracy"], ascending=False
            )
            .groupby("exp_key")
            .head(1)
        )

        def task_type(key: str):
            if "_all_" in key:
                return "all"
            if "_a_" in key:
                return "a_n"
            if "_i_" in key:
                return "i_n"
            if "_u_" in key:
                return "u_n"
            if "_phrase_" in key:
                return "phrase"
            raise ValueError(f"Unexpected key: {key}")

        def input_type(key: str):
            if "-" not in key:
                if "_all_" in key:
                    return "superset"
                if "_a_" in key:
                    return "superset"
                if "_i_" in key:
                    return "superset"
                if "_u_" in key:
                    return "superset"
                if "_phrase_" in key:
                    return "superset"
            parts = key.split("-")
            num_parts = len(parts)
            if num_parts == 2:
                return parts[0]
            raise ValueError(f"Unexpected key: {key}")

        best_model_epoch["task_type"] = best_model_epoch["exp_key"].apply(task_type)
        best_model_epoch["input_type"] = best_model_epoch["exp_key"].apply(input_type)
        best_model_epoch = best_model_epoch[
            best_model_epoch["input_type"] == "superset"
        ]
        best_model_epoch = best_model_epoch[best_model_epoch["task_type"] != "all"]
        data = (
            best_model_epoch.groupby(by=["feature", "max_diag_level"], sort=False)[
                ["accuracy"]
            ]
            .agg(
                mean_acc=("accuracy", "mean"),
                std_acc=("accuracy", "std"),
            )
            .reset_index()
        )
        out_path = f"{self.__results_path}/input_features.csv"
        fig_path = f"{self.__results_path}/input_features.png"
        data.to_csv(out_path)
        print(f"Saved at: {out_path}")
        data["mean_acc"] = (data["mean_acc"] * 100).round(decimals=2)
        data["std_acc"] = (data["std_acc"] * 100).round(decimals=2)
        groups = data.groupby(by=["max_diag_level"])
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(15, 10),
            constrained_layout=True,
        )
        best_model_epoch["accuracy"] = best_model_epoch["accuracy"] * 100
        sns.lineplot(
            data=best_model_epoch,
            x="feature",
            y="accuracy",
            # errorbar="sd",
            hue="max_diag_level",
            ax=ax,
            palette="rainbow",
        )
        ax.set_ylim(10, 90)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, rotation=90)
        ax.set_ylabel("Accuracy (%)", fontsize=20)
        ax.set_xlabel("Feature (%)", fontsize=20)
        ax.legend(
            loc="upper right",
            fontsize=16,
            title="Diag Level",
            title_fontsize=16,
        )
        fig.suptitle(
            "Classification accuracy for input features on different classification levels",
            fontsize=24,
        )
        fig.savefig(fig_path, bbox_inches="tight")
        print(f"Saved at: {fig_path}")

    def report_input_tasks(self) -> None:
        # single task models here
        # combining input tasks improvement here
        df = pd.read_csv(f"{self.__results_path}/collated_results_self.csv", header=0)
        mask = ~(
            df["exp_key"].str.endswith("-with-unclassified")
            | df["exp_key"].str.startswith("USVAC-")
            | df["exp_key"].str.startswith("daSilvaMoura-")
            | df["exp_key"].str.startswith("superset-")
        )
        df = df[mask]
        single_task_results = df[df["num_diag_levels"] == 1]
        single_task_results["accuracy"] = (
            single_task_results["0_acc_balanced"]
            .combine_first(single_task_results["1_acc_balanced"])
            .combine_first(single_task_results["2_acc_balanced"])
            .combine_first(single_task_results["3_acc_balanced"])
            .combine_first(single_task_results["4_acc_balanced"])
        )
        best_model_epoch = (
            single_task_results.sort_values(
                by=["feature", "max_diag_level", "accuracy"], ascending=False
            )
            .groupby("exp_key")
            .head(1)
        )

        def task_type(key: str):
            if "_all_" in key:
                return "all"
            if "_a_" in key:
                return "a_n"
            if "_i_" in key:
                return "i_n"
            if "_u_" in key:
                return "u_n"
            if "_phrase_" in key:
                return "phrase"
            raise ValueError(f"Unexpected key: {key}")

        def input_type(key: str):
            if "-" not in key:
                if "_all_" in key:
                    return "superset"
                if "_a_" in key:
                    return "superset"
                if "_i_" in key:
                    return "superset"
                if "_u_" in key:
                    return "superset"
                if "_phrase_" in key:
                    return "superset"
            parts = key.split("-")
            num_parts = len(parts)
            if num_parts == 2:
                return parts[0]
            raise ValueError(f"Unexpected key: {key}")

        best_model_epoch["task_type"] = best_model_epoch["exp_key"].apply(task_type)
        best_model_epoch["input_type"] = best_model_epoch["exp_key"].apply(input_type)
        best_model_epoch = best_model_epoch[
            best_model_epoch["input_type"] == "superset"
        ]
        data = (
            best_model_epoch.groupby(by=["feature", "max_diag_level"], sort=False)[
                ["input_type", "task_type", "exp_key", "accuracy"]
            ]
            .apply(lambda x: x)
            .reset_index()
            .drop(columns="level_2")
        )
        print(data)
        out_path = f"{self.__results_path}/input_tasks.csv"
        fig_path = f"{self.__results_path}/input_tasks.png"
        data.pivot(
            index=["max_diag_level", "feature"],
            columns=["task_type"],
            values="accuracy",
        ).drop(columns=["all"]).to_csv(out_path)
        print(f"Saved at: {out_path}")
        # num_features = len(data['feature'].unique())
        # num_levels = len(data['max_diag_level'].unique())
        data = data[data["task_type"] != "all"]
        data["accuracy"] = (data["accuracy"] * 100).round(decimals=2)
        groups = data.groupby(by=["max_diag_level"])
        total_groups = len(groups)
        fig, ax = plt.subplots(
            total_groups,
            1,
            figsize=(22, 15),
            constrained_layout=True,
            # sharey="row",
            sharex="col",
        )
        for idx, ((diag_level,), group) in enumerate(groups):
            min_acc = group["accuracy"].min()
            max_acc = group["accuracy"].max()
            sns.barplot(
                data=group,
                x="feature",
                y="accuracy",
                hue="task_type",
                ax=ax[idx],
                palette="GnBu",
                legend=idx == total_groups - 1,
            )
            ax[idx].set_ylabel(diag_level, fontsize=18, rotation=90)
            # ax[idx].set_yticklabels(ax[idx].get_yticklabels(), fontsize=14)
            ax[idx].set_xlabel(None)
            margin = 5
            ax[idx].set_ylim(min_acc - margin, max_acc + margin)
            for c in ax[idx].containers:
                ax[idx].bar_label(c, fontsize=14)
        ax[-1].set_xticklabels(ax[idx].get_xticklabels(), fontsize=18, rotation=90)
        sns.move_legend(
            ax[-1],
            "lower center",
            bbox_to_anchor=(0.5, -1.8),
            ncol=4,
            title=None,
            frameon=False,
            fontsize=18,
        )
        fig.savefig(fig_path, bbox_inches="tight")
        print(f"Saved at: {fig_path}")

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
        labels.remove("normal")
        labels = ["normal"] + sorted(labels)
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
