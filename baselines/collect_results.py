import numpy as np
import pandas as pd
from pathlib import Path
from class_argparse import ClassArgParser


class Shell(ClassArgParser):

    def __init__(self):
        super().__init__(name="Result Collector")

    def run(self):
        # df = self.__load_results_data()
        # self.__calc_results(df)
        df = self.__load_confusion_data()
        self.__calc_results(df)

    def __calc_results(self, df):
        df = df.sort_values(by=["Stream", "Task", "Feature"])
        self.save_df(df=df.copy(), file_name="data/results.csv")
        self.save_df(
            df=df.copy()[["Stream", "Task", "Feature", "Accuracy", "F1"]],
            file_name="data/results_min.csv",
        )

        best_models = (
            df.groupby(by=["Stream", "Task"])
            .apply(self.__find_best_model, include_groups=False)
            .reset_index()
        )
        self.save_df(df=best_models, file_name="data/best_results.csv")

    def save_df(self, df, file_name):
        df["Accuracy"] = df["Accuracy"].apply(lambda x: f"{x:.02f}")
        df["F1"] = df["F1"].apply(lambda x: f"{x:.04f}")
        df.to_csv(file_name, index=False)

    def __find_best_model(self, group):
        metric_1 = group["F1"]
        metric_2 = group["Accuracy"]
        if metric_1.isnull().all():
            if metric_2.isnull().all():
                best_idx = metric_1.index[0]
            else:
                best_idx = metric_2.idxmax()
        else:
            best_idx = metric_1.idxmax()
        return group.loc[best_idx]

    def __load_results_data(self):
        start_path = Path("/home/workspace/baselines/data/divr_benchmark/results")
        result_files = list(start_path.rglob("result.log"))
        return pd.DataFrame.from_records(map(self.__load_result_file, result_files))

    def __load_confusion_data(self):
        start_path = Path("/home/workspace/baselines/data/divr_benchmark/results")
        confusion_files = list(start_path.rglob("confusion.csv"))
        return pd.DataFrame.from_records(
            map(self.__load_confusion_file, confusion_files)
        )

    def __load_result_file(self, file):
        model = file.parent.name
        batch = model.removeprefix("Simple")
        model = model.removesuffix(batch)
        if batch == "":
            batch = 1
        feature = file.parent.parent.name
        task = int(file.parent.parent.parent.name.removeprefix("T"))
        stream = int(file.parent.parent.parent.parent.name.removeprefix("S"))
        with open(file, "r") as result_file:
            result_data = result_file.readlines()
            accuracy = float(result_data[0].strip().split(":")[1]) * 100
            return {
                "Stream": stream,
                "Task": task,
                "Feature": feature,
                "Batch": batch,
                "Model": model,
                "Accuracy": accuracy,
            }

    def __load_confusion_file(self, file):
        model = file.parent.name
        batch = model.removeprefix("Simple")
        model = model.removesuffix(batch)
        if batch == "":
            batch = 1
        feature = file.parent.parent.name
        task = int(file.parent.parent.parent.name.removeprefix("T"))
        stream = int(file.parent.parent.parent.parent.name.removeprefix("S"))
        confusion_df = pd.read_csv(file, index_col=0)
        total_actual_per_class = np.sum(confusion_df.to_numpy(), axis=1)
        total_predicted_per_class = np.sum(confusion_df.to_numpy(), axis=0)
        correct_per_class = np.diag(confusion_df.to_numpy())
        recall_per_class = correct_per_class / total_actual_per_class
        precision_per_class = correct_per_class / total_predicted_per_class
        f1_per_class = (2 * precision_per_class * recall_per_class) / (
            precision_per_class + recall_per_class
        )
        acc_per_class = recall_per_class * 100
        mean_acc = acc_per_class.mean()
        mean_f1 = f1_per_class.mean()
        return {
            "Stream": stream,
            "Task": task,
            "Feature": feature,
            "Batch": batch,
            "Model": model,
            "Accuracy": mean_acc,
            "F1": mean_f1,
            "TotalActualPerClass": total_actual_per_class,
            "TotalPredictedPerClass": total_predicted_per_class,
            "CorrectPerClass": correct_per_class,
            "RecallPerClass": recall_per_class,
            "PrecisionPerClass": precision_per_class,
            "AccPerClass": acc_per_class,
            "F1PerClass": f1_per_class,
        }


if __name__ == "__main__":
    Shell()()
