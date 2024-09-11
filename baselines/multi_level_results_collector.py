import numpy as np
import pandas as pd
import sklearn.utils
from tqdm import tqdm
import sklearn.metrics
from pathlib import Path
from class_argparse import ClassArgParser
from divr_benchmark import DiagnosisMap


class ResultsCollector(ClassArgParser):

    base_path = Path("/home/workspace/baselines/data/divr_benchmark/results/S1")
    results_file = "/home/workspace/baselines/data/multi_level_collector_results.csv"
    diagnosis_map = DiagnosisMap.v1()

    def __init__(self) -> None:
        super().__init__(name="ResultsCollector")

    def run(self):
        results_files = self.base_path.rglob("results.csv")
        final_df = []
        for results_file in results_files:
            model_dir = results_file.parent
            feature_dir = model_dir.parent
            task_dir = feature_dir.parent
            stream_dir = task_dir.parent
            model_name = model_dir.stem
            feature_name = feature_dir.stem
            task_key = int(task_dir.stem.removeprefix("T"))
            stream_key = int(stream_dir.stem.removeprefix("S"))
            results = pd.read_csv(results_file)
            final_df += [
                {
                    "stream_key": stream_key,
                    "task_key": task_key,
                    "feature_name": feature_name,
                    "model_name": model_name,
                    "results": results,
                }
            ]
        final_df = pd.DataFrame.from_records(final_df)
        tqdm.pandas()
        accs = final_df["results"].progress_apply(self.multi_level_acc)
        final_df = pd.merge(
            left=final_df,
            right=accs,
            left_index=True,
            right_index=True,
        )
        final_df = final_df.drop("results", axis=1)
        final_df = final_df.sort_values(by=final_df.columns.to_list())
        acc_cols = [
            c
            for c in final_df.columns
            if (c.startswith("acc_") or c.startswith("conf_"))
        ]
        for acc_col in acc_cols:
            final_df[acc_col] = final_df[acc_col].apply(lambda x: f"{x*100:.2f}")
        final_df.to_csv(self.results_file, index=False)

    def multi_level_acc(self, df: pd.DataFrame) -> pd.Series:
        diags = df.map(lambda name: self.diagnosis_map.get(name))
        levels = diags.map(lambda diag: diag.level)
        max_level: int = levels.max(axis=None)
        levels = range(max_level, -1, -1)
        results = {}
        for level in levels:
            level_diags = diags.map(lambda diag: diag.at_level(level).name)
            level_acc = self.acc(level_diags)
            cil, cih = self.calc_confidence(level_diags)
            results[f"acc_{level}"] = level_acc
            results[f"conf_low_{level}"] = cil
            results[f"conf_high_{level}"] = cih
        return pd.Series(results)

    def acc(self, df: pd.DataFrame):
        return sklearn.metrics.balanced_accuracy_score(
            df["actual"],
            df["predicted"],
        )

    def calc_confidence(
        self,
        df: pd.DataFrame,
        alpha: int = 5,
        num_bootstraps: int = 1000,
    ):
        accs = []
        for n in range(num_bootstraps):
            resampled_df = self.balanced_resample(df, random_state=n)
            acc = self.acc(resampled_df)
            accs += [acc]
        low = np.percentile(accs, alpha / 2)
        high = np.percentile(accs, 100 - alpha / 2)
        return [low, high]

    def balanced_resample(self, df: pd.DataFrame, random_state: int):
        uniques = df.groupby(["actual"])["actual"]
        new_indices = np.array([])
        for indices in uniques.indices.values():
            new_indices = np.concatenate(
                (
                    new_indices,
                    sklearn.utils.resample(
                        indices,
                        replace=True,
                        n_samples=len(indices),
                        random_state=random_state,
                    ),
                )
            )
        return df.iloc[new_indices].reset_index(drop=True)


if __name__ == "__main__":
    ResultsCollector()()
