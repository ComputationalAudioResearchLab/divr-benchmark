import numpy as np
import pandas as pd
from pathlib import Path
from yaml import safe_load
from class_argparse import ClassArgParser


class Collector(ClassArgParser):
    task_base_path = "/home/workspace/benchmark/divr_benchmark/tasks/v1/streams"

    def __init__(self) -> None:
        super().__init__(name="Demographics Collector")

    def run(self):
        tasks = {}
        for stream in Path(self.task_base_path).iterdir():
            if stream.is_dir():
                tasks[stream.stem] = []
                for task in stream.iterdir():
                    if task.is_dir():
                        tasks[stream.stem] += [task.stem]
        dfs = []
        for stream, _tasks in tasks.items():
            for task in _tasks:
                df = self.__read_task_demographics(stream=stream, task=task)
                dfs += [df]
        df = pd.concat(dfs)
        col_names = df.columns.to_list()
        col_names.remove("task")
        col_names.remove("stream")
        col_names.remove("dataset")
        col_names = ["stream", "task", "dataset"] + col_names
        df = df[col_names]
        df = df.sort_values(by=["stream", "task", "dataset", "sex", "keys"])
        df.to_csv("data/demographics.csv", index=False)

    def __read_task_demographics(self, stream: int, task: int):
        task_path = f"{self.task_base_path}/{stream}/{task}"
        names = ["train", "val", "test"]
        dfs = []
        for name in names:
            demo_file = Path(f"{task_path}/{name}.demographics.yml")
            if demo_file.is_file():
                dfs += [self.__read_demographics(dset_name=name, demo_file=demo_file)]
        df = pd.concat(dfs)
        df["task"] = int(task)
        df["stream"] = int(stream)
        return df

    def __read_demographics(self, dset_name, demo_file):
        keys = ["max", "mean", "min", "std", "total"]

        def exploder(cell):
            if pd.isna(cell):
                data = dict([(key, np.nan) for key in keys])
                return pd.Series(data)
            age_stats = cell["age_stats"]
            if age_stats is None:
                age_stats = dict([(key, np.nan) for key in keys])
            age_stats["total"] = cell["total"]
            return pd.Series(age_stats)

        def grouper(group):
            group.index = keys
            return group

        with open(demo_file, "r") as test_data_file:
            data = safe_load(test_data_file)
            df = pd.DataFrame.from_dict(data)
            df = df.map(exploder)
            df = df.explode(df.columns.to_list())
            df.index.names = ["sex"]
            df = df.reset_index()
            df = df.groupby("sex").apply(grouper, include_groups=False)
            df.index.names = ["sex", "keys"]
            df = df.reset_index()
            df["dataset"] = dset_name
            return df


if __name__ == "__main__":
    Collector()()
