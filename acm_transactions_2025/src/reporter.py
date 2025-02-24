import yaml
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
from tqdm import tqdm
from pathlib import Path
from typing import Literal
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
from .tasks_generator import TaskGenerator


class Reporter:

    label_map = {
        "auto_a": "Aa",
        "auto_b": "Ab",
        "auto_c": "Ac",
        "functional": "F",
        "functional_dysphonia": "FD",
        "hyperfunctional_dysphonia": "HD",
        "inflammatory": "I",
        "laryngitis": "La",
        "leukoplakia": "Le",
        "muscle_tension": "MT",
        "mass_lesions": "ML",
        "normal": "N",
        "non_structural_dysphonia": "NS",
        "organic": "O",
        "organic_inflammatory": "OI",
        "organic_inflammatory_infective": "OII",
        "organic_neuro_muscular": "ON",
        "organic_neuro_muscular_peripheral_nervous_disorder": "ONP",
        "organofunctional": "OF",
        "organic_structural": "OS",
        "organic_structural_epithelial_propria": "OSE",
        "structural_dysphonia": "S",
        "pathological": "P",
        "psychogenic_dysphonia": "PD",
        "reinkes_edema": "RE",
        "recurrent_paralysis": "RP",
        "unclassified": "U",
        "vocal_fold_polyp": "VFP",
        # MEEI stuff
        "abductor_spasmodic_dysphonia": "AbD",
        "adductor_spasmodic_dysphonia": "AdD",
        "conversion_dysphonia": "CD",
        "cyst": "C",
        "episodic_functional_dysphonia": "EFD",
        "exudative_hyperkeratotic_lesions_of_epithelium": "HL",
        "hemmoragic_reinkes_edema": "RE",
        "hyperfunction": "HD",  # Matching with Hyperfunctional Dysphonia above
        "laryngeal_tuberculosis": "LT",
        "leukoplakia": "Le",
        "normal": "N",
        "paralysis": "RP",  # This is RP to match up with Recurrent Paralysis above
        "paresis": "Pa",
        "polypoid_degeneration_reinkes": "PR",
        "post_intubation_submucosal_edema_mild": "SEM",
        "presbyphonia": "Pr",
        "scarring": "Sc",
        "subglottal_mass": "SM",
        "subglottis_stenosis": "SS",
        "varix": "Va",
        "vocal_fold_edema": "VE",
        "vocal_fold_nodules": "VN",
        "vocal_tremor": "VT",
    }

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
            if exp_key.endswith("_4"):
                return "narrow"
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
        mask_features = df["feature"].isin(
            ["Wav2Vec", "UnispeechSAT", "MFCCDD", "MelSpec"]
        )
        mask_tasks = df["task_key"].isin(["a_n", "phrase", "all"])
        num_diag_levels_mask = df["num_diag_levels"] == 1
        max_diag_level_mask = df["max_diag_level"].isin([1, 2, 4])
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
        results["epoch"] = all_results["epoch"]
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

    def report_superset_average_recall_exact(self):
        df = pd.read_csv(
            f"{self.__results_path}/report_superset_analysis.csv",
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
                "narrow",
                "CaRLab_2025",
                "daSilvaMoura_2024",
                "USVAC_2025_1",
                "USVAC_2025_2",
            ]
        ].reset_index(names=["category"])
        print(data)
        out_path = f"{self.__results_path}/superset_average_recall_exact.csv"
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
        ax.set_xlabel("Feature", fontsize=20)
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
        best_model_epoch = best_model_epoch[best_model_epoch["task_type"] != "all"]
        best_model_epoch = best_model_epoch[
            best_model_epoch["feature"].isin(["Wav2Vec", "UnispeechSAT", "MFCCDD"])
        ]
        data = (
            best_model_epoch.groupby(by=["task_type", "max_diag_level"], sort=False)[
                ["accuracy"]
            ]
            .agg(
                mean_acc=("accuracy", "mean"),
                std_acc=("accuracy", "std"),
            )
            .reset_index()
        )
        out_path = f"{self.__results_path}/input_tasks.csv"
        fig_path = f"{self.__results_path}/input_tasks.png"
        data.to_csv(out_path)
        print(f"Saved at: {out_path}")
        data["mean_acc"] = (data["mean_acc"] * 100).round(decimals=2)
        data["std_acc"] = (data["std_acc"] * 100).round(decimals=2)
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(15, 10),
            constrained_layout=True,
        )
        best_model_epoch["accuracy"] = best_model_epoch["accuracy"] * 100
        order = {"phrase": 0, "a_n": 1, "i_n": 2, "u_n": 3}
        best_model_epoch["order"] = best_model_epoch["task_type"].apply(order.get)
        best_model_epoch = best_model_epoch.sort_values(by=["order"])
        sns.lineplot(
            data=best_model_epoch,
            x="task_type",
            y="accuracy",
            # errorbar="sd",
            hue="max_diag_level",
            ax=ax,
            palette="rainbow",
            sort=False,
        )
        ax.set_ylim(15, 95)
        ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=18, rotation=90)
        ax.set_ylabel("Accuracy (%)", fontsize=20)
        ax.set_xlabel("Vocal Task", fontsize=20)
        ax.legend(
            loc="upper right",
            fontsize=16,
            title="Diag Level",
            title_fontsize=16,
        )
        fig.suptitle(
            "Classification accuracy for vocal tasks on different classification levels",
            fontsize=24,
        )
        fig.savefig(fig_path, bbox_inches="tight")
        print(f"Saved at: {fig_path}")

    def report_hierarchies(self) -> None:
        # difference by hierarchy
        df = pd.read_csv(f"{self.__results_path}/collated_results_self.csv", header=0)
        df = df[df["feature"] == "UnispeechSAT"]
        df = df[df["task_key"].isin(["phrase"])]

        single_task_results = df[df["num_diag_levels"] == 1]
        single_task_results["accuracy"] = (
            single_task_results["0_acc_balanced"]
            .combine_first(single_task_results["1_acc_balanced"])
            .combine_first(single_task_results["2_acc_balanced"])
            .combine_first(single_task_results["3_acc_balanced"])
            .combine_first(single_task_results["4_acc_balanced"])
        )
        best_single_task_result = (
            single_task_results.sort_values(by="accuracy", ascending=False)
            .groupby(by=["exp_key"])
            .head(1)
        )
        usvac = best_single_task_result[
            ~best_single_task_result["exp_key"].str.startswith("superset-")
        ].reset_index(drop=True)
        cross_system = best_single_task_result[
            best_single_task_result["max_diag_level"] == 1
        ].reset_index(drop=True)

        exp_map = {
            "superset-CaRLab_2025-unispeechSAT_phrase_1": "CaRLab 2025",
            "unispeechSAT_phrase_1": "USVAC 2025",
            "superset-daSilvaMoura_2024-unispeechSAT_phrase_1": "da Silva Moura 2024",
            "superset-Compton_2022-unispeechSAT_phrase_1": "Compton 2022",
            "superset-Zaim_2023-unispeechSAT_phrase_1": "Za'im 2023",
        }

        fig = plt.figure(figsize=(35, 15), constrained_layout=True)
        subfigs = fig.subfigures(nrows=2, ncols=1, hspace=0.1)
        axs = [
            f.subplots(nrows=1, ncols=5, gridspec_kw={"wspace": 0.1, "hspace": 0})
            for f in subfigs
        ]
        # fig, axs = plt.subplots(2, 5, figsize=(20, 10), constrained_layout=True)
        for idx, row in usvac.iterrows():
            ax = axs[0][idx]
            exp_key = row["exp_key"]
            epoch = row["epoch"]
            exp_df = pd.read_csv(f"{self.__results_path}/self/{exp_key}/{epoch}.csv")
            labels, _, acc, confusion = self.confusion(
                actual=exp_df["actual"],
                predicted=exp_df["predicted"],
            )
            labels = [self.label_map[label] for label in labels]
            sns.heatmap(
                confusion,
                ax=ax,
                annot=True,
                fmt="g",
                cmap="YlGnBu",
                xticklabels=labels,
                yticklabels=labels,
                annot_kws={"fontsize": 20},
            )
            ax.tick_params(axis="both", labelsize=22)
            ax.set_title(f"Level {idx}", fontsize=24)
            ax.set_xlabel("Predicted", fontsize=22)
            ax.set_ylabel("Actual", fontsize=22)
            print(exp_key, round(acc * 100, 2))
        for idx, row in cross_system.iterrows():
            ax = axs[1][idx]
            exp_key = row["exp_key"]
            epoch = row["epoch"]
            exp_df = pd.read_csv(f"{self.__results_path}/self/{exp_key}/{epoch}.csv")
            labels, _, acc, confusion = self.confusion(
                actual=exp_df["actual"],
                predicted=exp_df["predicted"],
            )
            labels = [self.label_map[label] for label in labels]
            sns.heatmap(
                confusion,
                ax=ax,
                annot=True,
                fmt="g",
                cmap="YlGnBu",
                xticklabels=labels,
                yticklabels=labels,
                annot_kws={"fontsize": 20},
            )
            ax.tick_params(axis="both", labelsize=22)
            ax.set_title(f"{exp_map[exp_key]}", fontsize=24)
            ax.set_xlabel("Predicted", fontsize=22)
            ax.set_ylabel("Actual", fontsize=22)
            print(exp_key, round(acc * 100, 2))
        # axs[0][0].set_ylabel("Actual", fontsize=22)
        # axs[1][0].set_ylabel("Actual", fontsize=22)
        subfigs[0].suptitle(
            "(a) Classification confusion for different levels of USVAC 2025",
            fontsize=30,
        )
        fig.add_artist(plt.Line2D([0, 1], [0.505, 0.505], color="#393E46", linewidth=2))
        subfigs[1].suptitle(
            "(b) Classification confusion for different classification systems",
            fontsize=30,
        )
        fig_path = f"{self.__results_path}/hierarchies.png"
        fig.savefig(fig_path, bbox_inches="tight")
        print(f"Saved at: {fig_path}")
        # first line USVAC all levels
        # second line all systems level 1

    def report_best_others(self) -> None:
        df = pd.read_csv(f"{self.__results_path}/collated_results_self.csv", header=0)
        df = df[
            ~(
                df["exp_key"].str.startswith("USVAC-")
                | df["exp_key"].str.startswith("daSilvaMoura-")
            )
        ]
        df = df[df["feature"] == "UnispeechSAT"]
        df = df[
            df["task_key"].isin(
                ["Compton_2022-phrase", "Sztaho_2018-phrase", "Zaim_2023-phrase"]
            )
        ]
        single_task_results = df[
            (df["num_diag_levels"] == 1) & (df["max_diag_level"] == 2)
        ]
        single_task_results["accuracy"] = (
            single_task_results["0_acc_balanced"]
            .combine_first(single_task_results["1_acc_balanced"])
            .combine_first(single_task_results["2_acc_balanced"])
            .combine_first(single_task_results["3_acc_balanced"])
            .combine_first(single_task_results["4_acc_balanced"])
        )
        df = (
            single_task_results.sort_values(by=["accuracy"], ascending=False)
            .groupby(by="exp_key")
            .head(1)
            .dropna(axis="columns", how="all")
            .drop(
                columns=[
                    "task_key",
                    "feature",
                    "num_diag_levels",
                    "max_diag_level",
                    "2_acc_balanced",
                ]
            )
            .to_dict(orient="records")
        )
        df = [{k: v for k, v in row.items() if not pd.isna(v)} for row in df]
        print(df)

    def report_data_availability(self) -> None:
        # difference in performance by availability of data
        df = pd.read_csv(f"{self.__results_path}/report_superset_analysis.csv")
        df = df[(df["task_key"] == "phrase") & (df["feature"] == "UnispeechSAT")]
        df = df.drop(columns=["epoch", "feature", "task_key"]).melt(
            id_vars=["exp_key", "category"],
            var_name="val_type",
            value_name="val",
        )
        df["val"] = (df["val"] * 100).round(decimals=2)
        print(df)
        all_demographics = []
        file_base = "/home/workspace/acm_transactions_2025/src/tasks/a_n"
        for file_type in ["train", "val", "test"]:
            with open(
                f"{file_base}/{file_type}.demographics.yml", "r"
            ) as demographics_file:
                demographics = yaml.full_load(demographics_file)
                for diag, d_data in demographics.items():
                    for gender, g_data in d_data.items():
                        all_demographics += [[diag, gender, file_type, g_data["total"]]]
        all_demographics = pd.DataFrame.from_records(
            data=all_demographics, columns=["diag", "gender", "dataset", "total"]
        )
        all_demographics = (
            all_demographics.groupby(by=["diag", "dataset"])["total"]
            .sum()
            .reset_index()
        )
        print(all_demographics)

        fig_path = f"{self.__results_path}/data_availability.png"

        fig, ax = plt.subplots(
            2, 1, figsize=(15, 10), constrained_layout=True, sharex="col"
        )
        sns.barplot(
            data=df,
            x="val_type",
            y="val",
            hue="category",
            palette="YlGnBu",
            ax=ax[0],
        )
        ax[0].tick_params(axis="both", labelsize=18)
        ax[0].set_ylabel("Recall (%)", fontsize=20)
        sns.move_legend(
            ax[0],
            "upper center",
            bbox_to_anchor=(0.5, 1.17),
            ncol=5,
            title=None,
            frameon=False,
            fontsize=14,
        )
        sns.lineplot(
            data=all_demographics,
            x="diag",
            y="total",
            hue="dataset",
            ax=ax[1],
        )
        sns.move_legend(ax[1], "upper right", fontsize=14, title_fontsize=16)
        ax[1].set_ylabel("Total speakers", fontsize=20)
        ax[1].tick_params(axis="x", rotation=90)
        ax[1].tick_params(axis="both", labelsize=18)
        ax[1].set_xlabel("Diagnosis Label", fontsize=20)
        fig.suptitle(
            "Recall accuracy and data availability of different diagnosis", fontsize=26
        )
        fig.align_labels()
        fig.savefig(fig_path)
        print(f"Saved at : {fig_path}")

    def report_consensus(self) -> None:
        # difference in performance by consensus of classification

        diag_confidences = {
            "recurrent_paralysis": {
                3: {
                    "organic_neuro_muscular_peripheral_nervous_disorder": 0.57,
                    # "organic_neuro_muscular_central_nervous_disorder": 0.29,
                    # "unclassified": 0.14,
                },
                2: {
                    "organic_neuro_muscular": 0.86,
                    # "unclassified": 0.14,
                },
                1: {
                    "organic": 0.86,
                    # "unclassified": 0.14,
                },
            },
            "reinkes_edema": {
                3: {
                    "organic_structural_epithelial_propria": 0.86,
                    # "organic_inflammatory_non_infective": 0.14,
                },
                2: {
                    "organic_structural": 0.86,
                    # "organic_inflammatory": 0.14,
                },
                1: {"organic": 1.00},
            },
            "vocal_fold_polyp": {
                3: {"organic_structural_epithelial_propria": 1.0},
                2: {"organic_structural": 1.0},
                1: {"organic": 1.0},
            },
            "hyperfunctional_dysphonia": {
                3: {
                    "muscle_tension": 0.71,
                    # "unclassified": 0.29,
                },
                2: {
                    "muscle_tension": 0.71,
                    # "unclassified": 0.29,
                },
                1: {
                    "muscle_tension": 0.71,
                    # "unclassified": 0.29,
                },
            },
            "leukoplakia": {
                3: {
                    "organic_structural_epithelial_propria": 0.86,
                    # "organic_structural_structural_abnormality": 0.14,
                },
                2: {"organic_structural": 1.00},
                1: {"organic_structural": 1.00},
            },
            "laryngitis": {
                3: {
                    "organic_inflammatory_infective": 0.71,
                    # "organic_structural_epithelial_propria": 0.14,
                    # "unclassified": 0.14,
                },
                2: {
                    "organic_inflammatory": 0.71,
                    # "organic_structural": 0.14,
                    # "unclassified": 0.14,
                },
                1: {
                    "organic": 0.86,
                    # "unclassified": 0.14,
                },
            },
            "psychogenic_dysphonia": {
                3: {"functional_dysphonia": 1.0},
                2: {"functional_dysphonia": 1.0},
                1: {"functional": 1.0},
            },
            "functional_dysphonia": {
                3: {"functional_dysphonia": 1.0},
                2: {"functional_dysphonia": 1.0},
                1: {"functional": 1.0},
            },
            "normal": {
                3: {"normal": 1.0},
                2: {"normal": 1.0},
                1: {"normal": 1.0},
            },
        }
        confidence_at_level = {1: {}, 2: {}, 3: {}}
        for k, levels in diag_confidences.items():
            for level, diags in levels.items():
                for diag, confidence in diags.items():
                    if diag not in confidence_at_level[level]:
                        confidence_at_level[level][diag] = [confidence]
                    else:
                        confidence_at_level[level][diag] += [confidence]
        for level in [1, 2, 3]:
            for diag, vals in confidence_at_level[level].items():
                confidence_at_level[level][diag] = np.mean(vals)
        print(confidence_at_level)
        df = pd.read_csv(f"{self.__results_path}/collated_results_self.csv", header=0)
        df = df[~df["exp_key"].str.startswith("superset-")]
        df = df[df["feature"] == "UnispeechSAT"]
        df = df[df["task_key"].isin(["phrase"])]
        df = df[df["max_diag_level"].isin([1, 2, 3])]
        single_task_results = df[df["num_diag_levels"] == 1]
        single_task_results["accuracy"] = (
            single_task_results["0_acc_balanced"]
            .combine_first(single_task_results["1_acc_balanced"])
            .combine_first(single_task_results["2_acc_balanced"])
            .combine_first(single_task_results["3_acc_balanced"])
            .combine_first(single_task_results["4_acc_balanced"])
        )
        best_single_task_result = (
            (
                single_task_results.sort_values(by="accuracy", ascending=False)
                .groupby(by=["exp_key"])
                .head(1)
            )
            .dropna(axis="columns", how="all")
            .drop(
                columns=[
                    "epoch",
                    "task_key",
                    "feature",
                    "num_diag_levels",
                    "min_diag_level",
                    "3_acc_balanced",
                    "2_acc_balanced",
                    "1_acc_balanced",
                ]
            )
            .to_dict(orient="records")
        )
        df = [
            {k: v for k, v in row.items() if not pd.isna(v)}
            for row in best_single_task_result
        ]
        fig, axs = plt.subplots(3, 1, figsize=(15, 9), constrained_layout=True)

        for row in df:
            diag_level = row["max_diag_level"]
            keys = [
                str(k).removeprefix(f"{diag_level}_acc_")
                for k in row.keys()
                if str(k).startswith(f"{diag_level}_acc_")
            ]
            row_df = (
                pd.DataFrame(
                    data=[
                        (
                            self.label_map[k],
                            row[f"{diag_level}_acc_{k}"],
                            confidence_at_level[diag_level][k],
                        )
                        for k in keys
                    ],
                    columns=["label", "recall", "confidence"],
                )
                .sort_values(by=["confidence"], ascending=False)
                .melt(
                    id_vars=["label"],
                    var_name="val_type",
                    value_name="val",
                )
            )
            print(row_df)
            ax = axs[diag_level - 1]
            sns.lineplot(
                data=row_df,
                x="label",
                y="val",
                hue="val_type",
                ax=ax,
                legend=diag_level == 1,
            )
            ax.set_ylabel(f"Level {diag_level}", fontsize=22)
            ax.set_xlabel(None)
            ax.tick_params(axis="both", labelsize=20)
        axs[-1].set_xlabel("Classification Label", fontsize=20)

        sns.move_legend(
            axs[0],
            "upper center",
            bbox_to_anchor=(0.5, 1.4),
            ncol=2,
            title=None,
            frameon=False,
            fontsize=18,
        )
        fig_path = f"{self.__results_path}/report_consensus.png"
        fig.suptitle(
            "Comparison of Recall(%) and Confidence of classification(%) per label per level",
            fontsize=25,
        )
        fig.savefig(fig_path, bbox_inches="tight")
        print(f"Saved at: {fig_path}")

    def report_multi_task(self) -> None:
        # impact of multi task and multi crit
        # speed up in accuracy jump
        # We use UnispeechSAT here as it is the best model
        df = pd.read_csv(f"{self.__results_path}/collated_results_self.csv", header=0)
        df = df[~df["exp_key"].str.startswith("superset-")]
        df = df[df["feature"] == "UnispeechSAT"]
        df = df[df["task_key"].isin(["phrase"])]
        df = df[df["max_diag_level"] != 0]
        # df = df[df["num_diag_levels"] > 1]
        # print(df)
        # print(df["exp_key"].value_counts())
        # print(df.groupby(by=["max_diag_level"]).apply(lambda x: x))

        single_task_results = df[df["num_diag_levels"] == 1]
        single_task_results["accuracy"] = (
            single_task_results["0_acc_balanced"]
            .combine_first(single_task_results["1_acc_balanced"])
            .combine_first(single_task_results["2_acc_balanced"])
            .combine_first(single_task_results["3_acc_balanced"])
            .combine_first(single_task_results["4_acc_balanced"])
        )
        best_single_task_result = (
            single_task_results.sort_values(by="accuracy", ascending=False)
            .groupby(by=["exp_key"])
            .head(1)
        )

        # Only experiments that have 0 + max diag level
        # otherwise there's too much data to visualize
        multi_task_results = df[df["num_diag_levels"] > 1]
        # multi_task_results = multi_task_results[
        #     multi_task_results["exp_key"].str.contains("_0+")
        # ]

        out_path = f"{self.__results_path}/multi_task.csv"
        fig_path = f"{self.__results_path}/multi_task.png"

        fig, ax = plt.subplots(
            4, 1, figsize=(18, 12), constrained_layout=True, sharex="col"
        )
        # for _, row in best_single_task_result.iterrows():
        #     idx = row["max_diag_level"]
        #     acc = row["accuracy"]
        #     ax[idx].axhline(y=acc, linestyle="--", color="#D2665A")

        for _, exp in multi_task_results.groupby(by=["exp_key"]):
            idx = exp["max_diag_level"].iloc[0]
            sns.lineplot(
                data=exp,
                x="epoch",
                y=f"{idx}_acc_balanced",
                ax=ax[idx - 1],
                color="#A9B5DF99",
            )
        for _, exp in single_task_results.groupby(by=["exp_key"]):
            idx = exp["max_diag_level"].iloc[0]
            sns.lineplot(
                data=exp,
                x="epoch",
                y=f"{idx}_acc_balanced",
                ax=ax[idx - 1],
                color="#D2665A",
            )
            ax[idx - 1].tick_params(axis="both", labelsize=20)
            ax[idx - 1].set_ylabel(f"Level {idx}", fontsize=24)
            ax[idx - 1].grid(visible=True)
        ax[-1].set_xlabel("Epoch", fontsize=24)

        single_task = mpatches.Patch(color="#D2665A", label="Single-Task")
        multi_task = mpatches.Patch(color="#A9B5DF", label="Multi-Task")
        ax[-1].legend(
            handles=[single_task, multi_task],
            loc="lower center",
            fontsize=20,
            ncols=2,
        )
        fig.align_labels()
        fig.suptitle(
            "Classification Accuracy over training epochs for single-task vs multi-task experiments",
            fontsize=26,
        )
        fig.savefig(fig_path, bbox_inches="tight")
        print(f"Saved at: {fig_path}")

    def report_multi_task_2(self) -> None:
        # impact of multi task and multi crit
        df = pd.read_csv(f"{self.__results_path}/collated_results_self.csv", header=0)
        df = df[~df["exp_key"].str.startswith("superset-")]
        df = df[df["feature"] == "UnispeechSAT"]
        df = df[df["task_key"].isin(["phrase"])]
        df = df[df["num_diag_levels"].isin([1, 5])]
        df = df[df["max_diag_level"] == 4]
        df = (
            df.sort_values(by=["4_acc_balanced"], ascending=False)
            .groupby(by="exp_key")
            .head(1)
        )
        diagnosis_map = self.__task_generator.get_diagnosis_map(
            "USVAC_2025", allow_unmapped=False
        )

        test_levels = [0, 1, 2, 3, 4]

        def diag_to_level(diag, level):
            return diagnosis_map.get(diag).at_level(level).name

        for _, row in df[["exp_key", "epoch", "num_diag_levels"]].iterrows():
            exp_key = row["exp_key"]
            epoch = row["epoch"]
            num_diag_levels = row["num_diag_levels"]
            exp_df = pd.read_csv(f"{self.__results_path}/self/{exp_key}/{epoch}.csv")
            print(exp_key, epoch, num_diag_levels)
            if num_diag_levels == 1:
                for level in test_levels:
                    actual = exp_df["actual"].apply(lambda x: diag_to_level(x, level))
                    predicted = exp_df["predicted"].apply(
                        lambda x: diag_to_level(x, level)
                    )
                    _, _, acc, _ = self.confusion(
                        actual=actual,
                        predicted=predicted,
                    )
                    print(f"\t {level}: {acc*100:.2f}")
            else:
                for level in test_levels:
                    _, _, acc, _ = self.confusion(
                        actual=exp_df[f"actual_{level}"],
                        predicted=exp_df[f"predicted_{level}"],
                    )
                    print(f"\t {level}: {acc*100:.2f}")

    def report_cross_database(self) -> None:
        df = pd.read_csv(f"{self.__results_path}/report_superset_analysis.csv")
        df = df[df["feature"].isin(["MFCCDD", "UnispeechSAT", "Wav2Vec"])]
        df = df[df["exp_key"].str.contains("_phrase_")]
        # df = df[df["exp_key"].str.contains("_all_")]
        # df = df[df["exp_key"].str.contains("_a_")]

        cross_tests = [
            "cross_test_avfad",
            "cross_test_meei",
            # "cross_test_torgo",
            # "cross_test_uaspeech",
            "cross_test_uncommon_voice",
            "cross_test_voiced",
        ]
        dmap = self.__task_generator.get_diagnosis_map(
            task_key="USVAC_2025", allow_unmapped=False
        )
        dls = {
            ct: self.__task_generator.load_task(
                task=ct,
                diag_level=None,
                diagnosis_map=dmap,
                load_audios=False,
            )
            for ct in cross_tests
        }

        def id_to_label(ct, label_id):
            label = dls[ct].test_label(label_id).root.name
            if label == "without_dysarthria":
                return "normal"
            return label

        def predicted_root(label):
            if label == "normal":
                return "normal"
            return "pathological"

        all_results = []

        for _, row in df.iterrows():
            exp_key = row["exp_key"]
            epoch = row["epoch"]
            category = row["category"]
            feature = row["feature"]
            for ct in cross_tests:
                exp_df = pd.read_csv(
                    f"{self.__results_path}/cross/{exp_key}/{ct}/{epoch}.csv"
                )
                exp_df["actual"] = exp_df["id"].apply(lambda lid: id_to_label(ct, lid))
                exp_df["predicted"] = exp_df["predicted"].apply(predicted_root)
                labels, _, acc, confusion = self.confusion(
                    actual=exp_df["actual"],
                    predicted=exp_df["predicted"],
                )
                # print(exp_key, ct, acc)
                # print(exp_df.groupby(by="actual")["predicted"].value_counts())
                # print(labels)
                # print(confusion)
                all_results += [(category, feature, ct, acc)]
        all_results = pd.DataFrame(
            data=all_results, columns=["category", "feature", "ct", "acc"]
        )
        all_results["acc"] = (all_results["acc"] * 100).round(2)
        print(all_results)
        ct_idx = {
            "cross_test_meei": 0,
            "cross_test_uncommon_voice": 1,
            "cross_test_voiced": 2,
            "cross_test_avfad": 3,
        }
        ct_labels = {
            "cross_test_meei": "MEEI",
            "cross_test_uncommon_voice": "Uncommon Voice",
            "cross_test_voiced": "VOICED",
            "cross_test_avfad": "AVFAD",
        }
        fig, axs = plt.subplots(
            len(ct_idx),
            1,
            figsize=(20, 13),
            constrained_layout=True,
            sharex="col",
        )
        for (ct,), group in all_results.groupby(by=["ct"]):
            idx = ct_idx[ct]
            ax = axs[idx]
            sns.barplot(
                data=group,
                x="category",
                y="acc",
                hue="feature",
                palette="YlGnBu",
                ax=ax,
                legend=idx == 0,
            )
            for c in ax.containers:
                ax.bar_label(c, fontsize=22)
            margin = 5
            y_max = group["acc"].max() + margin
            y_min = group["acc"].min() - margin
            ax.set_ylim(y_min, y_max)
            ax.set_xlabel(None)
            ax.set_ylabel(ct_labels[ct], fontsize=26)
            ax.tick_params(axis="y", labelsize=22)
            ax.tick_params(axis="x", labelsize=26)

        sns.move_legend(
            axs[0],
            "upper center",
            bbox_to_anchor=(0.5, 1.25),
            ncol=3,
            title=None,
            frameon=False,
            fontsize=24,
        )
        fig_path = f"{self.__results_path}/cross_database.png"
        fig.suptitle(
            "Binary detection accuracy on out of domain databases", fontsize=34
        )
        fig.savefig(fig_path, bbox_inches="tight")
        print(f"Saved at: {fig_path}")

    def report_cross_meei(self) -> None:
        df = pd.read_csv(f"{self.__results_path}/report_superset_analysis.csv")
        df = df[df["feature"].isin(["MFCCDD", "UnispeechSAT", "Wav2Vec"])]
        df = df[df["exp_key"].str.contains("_all_")]
        # df = df[df["exp_key"].str.contains("_phrase_")]
        # df = df[df["category"] == "narrow"]

        cross_tests = ["cross_test_meei"]
        dmap = self.__task_generator.get_diagnosis_map(
            task_key="USVAC_2025", allow_unmapped=False
        )
        dls = {
            ct: self.__task_generator.load_task(
                task=ct,
                diag_level=None,
                diagnosis_map=dmap,
                load_audios=False,
            )
            for ct in cross_tests
        }

        all_results = []
        log_file_path = f"{self.__results_path}/cross_meei.log"
        log_file = open(log_file_path, "w")
        prev_category = None
        confusions = {}
        max_total = 0
        for _, row in df.sort_values(by=["category", "feature"]).iterrows():
            exp_key = row["exp_key"]
            epoch = row["epoch"]
            category = row["category"]
            feature = row["feature"]
            if category != prev_category:
                log_file.write(f"\n\n{category}:\n")
                prev_category = category
            for ct in cross_tests:
                exp_df = pd.read_csv(
                    f"{self.__results_path}/cross/{exp_key}/{ct}/{epoch}.csv"
                )
                exp_df["actual"] = exp_df["id"].apply(
                    lambda lid: self.label_map[dls[ct].test_label(lid).name]
                )
                exp_df["predicted"] = exp_df["predicted"].apply(self.label_map.get)
                # labels, _, acc, confusion = self.confusion(
                #     actual=exp_df["actual"],
                #     predicted=exp_df["predicted"],
                # )
                log_file.write(f"\t{exp_key}:\n")
                # print(exp_key, ct, acc)
                confusion = (
                    exp_df.groupby(by="actual")["predicted"]
                    .value_counts()
                    .reset_index()
                    .pivot(
                        index="predicted",
                        columns="actual",
                        values="count",
                    )
                    .fillna(0)
                )
                if feature == "MFCCDD":
                    confusions[category] = confusion
                    current_max_count: int = confusion.max(axis=None)
                    if current_max_count > max_total:
                        max_total = current_max_count
                cstr = confusion.to_string()
                log_file.write("\n".join([f"\t{l}" for l in cstr.split("\n")]))
                log_file.write("\n")
                # print(confusion)
                # print(labels)
                # print(confusion)
                # all_results += [(category, feature, ct, acc)]
        log_file.close()

        categorization = {
            "CaRLab_2025": {
                "HD": "Aa",
                "Le": "Ab",
                "N": "N",
                "RE": "Ac",
                "RP": "Ac",
                "VFP": "Ab",
            },
            "daSilvaMoura_2024": {
                "HD": "F",
                "Le": "OF",
                "N": "N",
                "RE": "OF",
                "RP": "O",
                "VFP": "OF",
            },
            "USVAC_2025_1": {
                "HD": "MT",
                "Le": "O",
                "N": "N",
                "RE": "O",
                "RP": "O",
                "VFP": "O",
            },
            "USVAC_2025_2": {
                "HD": "MT",
                "Le": "OS",
                "N": "N",
                "RE": "OS",
                "RP": "ON",
                "VFP": "OS",
            },
            "narrow": {
                "HD": "HD",
                "Le": "Le",
                "N": "N",
                "RE": "RE",
                "RP": "RP",
                "VFP": "VFP",
            },
        }
        recalls = {k: {} for k in categorization}
        for key, confusion in confusions.items():
            categories = categorization[key]
            for k, v in categories.items():
                actual_total = confusion[k].sum()
                correct_total = confusion[k][v]
                recall = correct_total / actual_total
                recalls[key][k] = round(recall * 100, 2)
        print("Recalls: ", recalls)
        print(
            "Average Recalls: ",
            {
                k: round(sum([t for t in v.values()]) / len(v), 2)
                for k, v in recalls.items()
            },
        )

        print(f"Saved at: {log_file_path}")
        key_idx = {
            "CaRLab_2025": 0,
            "daSilvaMoura_2024": 1,
            "USVAC_2025_1": 2,
            "USVAC_2025_2": 3,
            "narrow": 4,
        }
        fig, axs = plt.subplots(
            1,
            len(key_idx),
            figsize=(22, 10),
            constrained_layout=True,
            sharey="row",
            gridspec_kw={
                "width_ratios": [1, 1, 1, 1.5, 2],
                "wspace": 0.1,
                "hspace": 0.1,
            },
        )
        for key, conf in confusions.items():
            idx = key_idx[key]
            ax = axs[idx]
            sns.heatmap(
                data=conf.T,
                cmap="Blues",
                ax=ax,
                annot=True,
                annot_kws={"fontsize": 16},
                vmin=0,
                vmax=max_total,
                cbar=False,
            )
            ax.set_ylabel(None)
            ax.set_title(key, fontsize=22)
            ax.set_xlabel("Predicted", fontsize=22)
            ax.tick_params(axis="y", rotation=0)
            ax.tick_params(axis="x", rotation=90)
            ax.tick_params(axis="both", labelsize=22)

        axs[0].set_ylabel("Actual", fontsize=22)
        # cbar = fig.colorbar(ax[0].get_children()[0], ax=axes, orientation='vertical', fraction=0.02, pad=0.02)
        fig_path = f"{self.__results_path}/cross_meei_all.png"
        fig.suptitle(
            "Multi-class classification confusion on MEEI using different classification systems",
            fontsize=34,
        )
        fig.align_labels()
        fig.savefig(fig_path, bbox_inches="tight")
        print(f"Saved at: {fig_path}")

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
