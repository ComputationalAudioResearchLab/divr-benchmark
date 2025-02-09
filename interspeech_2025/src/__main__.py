import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from class_argparse import ClassArgParser
from sklearn.metrics import confusion_matrix

from . import env
from .tasks_generator import TaskGenerator
from .experiments import Runner


class Main(ClassArgParser):

    def __init__(self) -> None:
        super().__init__(name="Interspeech 2025")

    def train(
        self,
        exp_key: Runner.EXP_KEYS,  # type: ignore
        tboard_enabled: bool = True,
        use_cache_loader: bool = False,
        limit_vram: float = None,
    ):
        tasks_generator = TaskGenerator(
            research_data_path=env.RESEARCH_DATA_PATH,
            tasks_path=env.TASKS_PATH,
        )
        runner = Runner(
            tasks_generator=tasks_generator,
            cache_path=env.CACHE_PATH,
            results_path=env.RESULTS_PATH,
            research_data_path=env.RESEARCH_DATA_PATH,
        )
        runner.train(
            exp_key=exp_key,
            tboard_enabled=tboard_enabled,
            use_cache_loader=use_cache_loader,
            limit_vram=limit_vram,
        )

    def test(self, exp_key: Runner.EXP_KEYS, load_epoch: int):  # type: ignore
        tasks_generator = TaskGenerator(
            research_data_path=env.RESEARCH_DATA_PATH,
            tasks_path=env.TASKS_PATH,
        )
        runner = Runner(
            tasks_generator=tasks_generator,
            cache_path=env.CACHE_PATH,
            results_path=Path(f"{env.RESULTS_PATH}/test"),
            research_data_path=env.RESEARCH_DATA_PATH,
        )
        runner.test(exp_key=exp_key, load_epoch=load_epoch)

    def test_all(self):
        tasks_generator = TaskGenerator(
            research_data_path=env.RESEARCH_DATA_PATH,
            tasks_path=env.TASKS_PATH,
        )
        runner = Runner(
            tasks_generator=tasks_generator,
            cache_path=env.CACHE_PATH,
            results_path=Path(f"{env.RESULTS_PATH}/test"),
            research_data_path=env.RESEARCH_DATA_PATH,
        )

        checkpoints_path = Path(f"{env.CACHE_PATH}/checkpoints")
        checkpoints = sorted(list(checkpoints_path.rglob("*.h5")))
        pbar = tqdm(checkpoints, desc="testing")
        for checkpoint in pbar:
            exp_key = checkpoint.parent.stem
            epoch = int(checkpoint.stem)
            pbar.set_postfix({"exp_key": exp_key, "epoch": epoch})
            results_path = Path(
                f"{env.RESULTS_PATH}/test/{exp_key}/{epoch}/results.csv"
            )
            if not results_path.is_file():
                runner.test(exp_key=exp_key, load_epoch=epoch)

    def collate_test_results(self):
        results_path = Path(f"{env.RESULTS_PATH}/test")
        results = sorted(list(results_path.rglob("results.csv")))
        all_results = []

        for result_file in results:
            row = {
                "exp_key": result_file.parent.parent.stem,
                "epoch": result_file.parent.stem,
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
                row[f"{level}_acc_balanced"] = balanced_accuracy
                for label, class_acc in zip(labels, per_class_accuracy):
                    row[f"{level}_acc_{label}"] = class_acc
            all_results += [row]
        all_results = pd.DataFrame.from_records(all_results)
        all_results = all_results.sort_values(
            by=[
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
            "0_acc_balanced",
            "1_acc_balanced",
            "2_acc_balanced",
            "3_acc_balanced",
        ]
        col_names = all_results.columns.to_list()
        for col_name in leading_cols:
            col_names.remove(col_name)
        all_results = all_results[leading_cols + col_names]
        all_results.to_csv(f"{results_path}/collated_results.csv", index=False)


if __name__ == "__main__":
    Main()()
