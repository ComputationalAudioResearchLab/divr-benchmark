import numpy as np
import pandas as pd
import sklearn.utils
from tqdm import tqdm
import sklearn.metrics
from pathlib import Path
from class_argparse import ClassArgParser


class ResultsCollector(ClassArgParser):

    base_path = Path("/home/workspace/baselines/data/divr_benchmark/results/S1")
    results_file = "/home/workspace/baselines/data/collector_results.csv"

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
        final_df["acc"] = final_df["results"].apply(self.acc)
        tqdm.pandas()
        conf = final_df["results"].progress_apply(self.calc_confidence)
        final_df["confidence_low"], final_df["confidence_high"] = zip(*conf)
        final_df = final_df.sort_values(
            by=["stream_key", "task_key", "feature_name", "model_name", "acc"]
        )
        final_df = final_df.drop("results", axis=1)
        final_df["acc"] = final_df["acc"].apply(lambda x: f"{x*100:.2f}")
        final_df["confidence_high"] = final_df["confidence_high"].apply(
            lambda x: f"{x*100:.2f}"
        )
        final_df["confidence_low"] = final_df["confidence_low"].apply(
            lambda x: f"{x*100:.2f}"
        )
        final_df.to_csv(self.results_file, index=False)

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
