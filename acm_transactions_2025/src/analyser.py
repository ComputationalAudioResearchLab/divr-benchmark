import itertools
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
        single_task_df = df[df["num_diag_levels"] == 1]
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
                "num_diag_levels": len(diag_levels),
                "min_diag_level": min(diag_levels),
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
                for level in range(5):
                    if f"actual_{level}" in col_names:
                        actual = df[f"actual_{level}"]
                        predicted = df[f"predicted_{level}"]
                        to_process += [(level, actual, predicted)]
            for level, actual, predicted in to_process:
                labels, per_class_accuracy, balanced_accuracy, _ = self.confusions(
                    actual, predicted
                )
                row[f"{level}_acc_balanced"] = balanced_accuracy
                for label, class_acc in zip(labels, per_class_accuracy):
                    row[f"{level}_acc_{label}"] = class_acc
            all_results += [row]
        all_results = pd.DataFrame.from_records(all_results)
        all_results = all_results.sort_values(
            by=[
                "task_key",
                "max_diag_level",
                "feature",
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
            "num_diag_levels",
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

    def autogen_hierarchies(self):
        test_keys = [
            "mfccdd_phrase_4",
            "mfccdd_a_4",
            "mfccdd_i_4",
            "mfccdd_u_4",
            "mfccdd_all_4",
            "wav2vec_phrase_4",
            "wav2vec_a_4",
            "wav2vec_i_4",
            "wav2vec_u_4",
            "wav2vec_all_4",
            "unispeechSAT_phrase_4",
            "unispeechSAT_a_4",
            "unispeechSAT_i_4",
            "unispeechSAT_u_4",
            "unispeechSAT_all_4",
        ]
        all_results = pd.read_csv(self.__collated_self_path)
        selected_results = all_results[all_results["exp_key"].isin(test_keys)]
        best_results = (
            selected_results.sort_values(by="4_acc_balanced", ascending=False)
            .groupby("exp_key")
            .head(1)
        )
        log = open(f"{self.__results_path}/autogen_hierarchies.log", "w")
        all_edges = []
        for ridx, row in best_results.iterrows():
            exp_key = row["exp_key"]
            epoch = row["epoch"]
            df = pd.read_csv(f"{self.__results_path}/self/{exp_key}/{epoch}.csv")
            labels, acc_per_class, balanced_acc, confusion = self.confusions(
                actual=df["actual"],
                predicted=df["predicted"],
            )
            log.write(f"{exp_key}[{balanced_acc}]\n")
            log.write(f"\t{dict(zip(labels, acc_per_class))}\n")
            log.write(
                "\n".join(f"\t{line}" for line in repr(confusion).split("\n")) + "\n"
            )
            optimal_hierarchy, mixed_labels = self.optimal_hierarchy(
                actual=df["actual"],
                predicted=df["predicted"],
            )
            log.write(f"\t{optimal_hierarchy}\n")
            hier = {}
            last_key: str
            for key, vals in optimal_hierarchy.items():
                ((mix_from, mix_to), acc) = vals
                if mix_from not in hier and mix_to not in hier:
                    hier[f"{key}"] = [mix_from, mix_to]
                elif mix_from in hier and mix_to in hier:
                    hier[key] = [hier[mix_from], hier[mix_to]]
                    del hier[mix_from]
                    del hier[mix_to]
                elif mix_from in hier:
                    hier[key] = [hier[mix_from], mix_to]
                    del hier[mix_from]
                elif mix_to in hier:
                    hier[key] = [hier[mix_to], mix_from]
                    del hier[mix_to]
                else:
                    raise ValueError("Unexpected scenario")
                last_key = key
            assert len(hier) == 1
            hier = hier[last_key]
            log.write(f"\t{hier}\n")
            edges = []
            hier_str = str(hier)

            for pair in itertools.combinations(mixed_labels, 2):
                A, B = pair
                A_idx = hier_str.index(A)
                B_idx = hier_str.index(B)
                if A_idx > B_idx:
                    start = B_idx
                    end = A_idx
                else:
                    start = A_idx
                    end = B_idx
                distance = hier_str[start:end].count("]")
                edges += [(A, B, distance)]
            all_edges += edges
            log.write(f"\t{edges}\n\n")
        df = pd.DataFrame.from_records(all_edges, columns=["from", "to", "distance"])
        df["distance"] = df["distance"] / df["distance"].max()
        df = df.groupby(by=["from", "to"]).agg(
            mean_dist=("distance", "mean"),
            std_dist=("distance", "std"),
            min_dist=("distance", "min"),
            max_dist=("distance", "max"),
        )
        df["dist_upper_bound"] = df["mean_dist"] + df["std_dist"]
        df = df.sort_values(by=["mean_dist"]).reset_index()
        df = df[
            [
                "from",
                "to",
                "mean_dist",
                "dist_upper_bound",
                "std_dist",
                "min_dist",
                "max_dist",
            ]
        ]
        df.to_csv(f"{self.__results_path}/autogen_hierarchy_dists.csv", index=False)
        log.write(df.to_string())
        log.close()

    def confusions(self, actual, predicted):
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

    def optimal_hierarchy(
        self, actual: "pd.Series[str]", predicted: "pd.Series[str]"
    ) -> tuple[dict[str, tuple[tuple[str, str], float]], list[str]]:
        actual = actual.copy()
        predicted = predicted.copy()
        class_weights = actual.value_counts()
        labels = class_weights.index.to_list()
        restricted = ["normal"]
        mixable = [label for label in labels if label not in restricted]
        mixed_labels = [l for l in mixable]
        hierachy = {}
        for idx in range(len(mixable) - 1):
            idx = str(idx)
            best_mixing, acc = self.best_class_merge(actual, predicted, mixable)
            hierachy[idx] = (best_mixing, acc)
            for mix in best_mixing:
                mixable.remove(mix)
                actual[actual == mix] = idx
                predicted[predicted == mix] = idx
            mixable += [idx]
        return hierachy, mixed_labels

    def best_class_merge(
        self, actual: "pd.Series[str]", predicted: "pd.Series[str]", mixable: list[str]
    ) -> tuple[tuple[str, str], float]:
        best_acc = 0
        best_mixing: tuple[str, str]
        for mix_from, mix_to in itertools.combinations(mixable, 2):
            new_actual = actual.copy()
            new_predicted = predicted.copy()
            new_actual[new_actual == mix_from] = mix_to
            new_predicted[new_predicted == mix_from] = mix_to
            _, _, acc, _ = self.confusions(actual=new_actual, predicted=new_predicted)
            if acc > best_acc:
                best_acc = acc
                best_mixing = (mix_from, mix_to)
        return best_mixing, best_acc
