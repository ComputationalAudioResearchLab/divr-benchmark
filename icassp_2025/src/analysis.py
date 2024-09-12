import itertools
from typing import Dict
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt

class Analyser:

    def __init__(self, cache_path: Path, results_path: Path) -> None:
        self.__cache_path = cache_path
        self.__cache_results_path = Path(f"{cache_path}/results")
        self.__results_path = results_path
        results_path.mkdir(parents=True, exist_ok=True)
    
    def run(self) -> None:
        result_files = list(self.__cache_results_path.rglob("results.csv"))
        all_accuracies = []
        for result_file_path in tqdm(result_files, desc="Analysing"):
            row = self.read_results_file(result_file_path)
            all_accuracies += [row]
        all_accuracies = pd.DataFrame.from_records(all_accuracies)
        self.plot_1(data=all_accuracies)
        self.plot_2(data=all_accuracies)
        self.plot_3(data=all_accuracies)
        self.plot_4(data=all_accuracies)
        self.plot_5(data=all_accuracies)
        self.save_all_results(data=all_accuracies)

    def plot_1(self, data: pd.DataFrame) -> None:
        df = data.query("exp_type == 'self' & model_name == 'unispeechSAT' & max_audio_len != max_audio_len")
        df = df.sort_values(by=["diagnosis_level", "task_type"])
        df_p = df.pivot(index='task_type', columns='diagnosis_level', values='top_1_accuracy')
        df_p = (df_p*100).round(decimals=2).sort_values(by=list(df_p.columns), ascending=False)
        df_p.to_csv(f"{self.__results_path}/plot_1.csv")
        # fig, ax = plt.subplots(1,1, figsize=(6, 4), constrained_layout=True)
        # for name, group in df.groupby('task_type'):
        #     group.plot(ax=ax, x='diagnosis_level', y='top_1_accuracy', label=name)
        # ax.set_xlabel("Diagnosis Level")
        # ax.set_ylabel("Top 1 Accuracy")
        # ax.set_xticks(range(4))
        # fig.savefig(f"{self.__results_path}/plot_1.png", bbox_inches='tight')

    def plot_2(self, data: pd.DataFrame) -> None:
        df = data.query("exp_type == 'self' & model_name == 'mfcc' & max_audio_len != max_audio_len")
        df = df.sort_values(by=["diagnosis_level", "task_type"])
        df_p = df.pivot(index='task_type', columns='diagnosis_level', values='top_1_accuracy')
        df_p = (df_p*100).round(decimals=2).sort_values(by=list(df_p.columns), ascending=False)
        df_p.to_csv(f"{self.__results_path}/plot_2.csv")
        
    def plot_3(self, data: pd.DataFrame) -> None:
        df = data.query("exp_type == 'self' & model_name == 'unispeechSAT' & max_audio_len == max_audio_len")
        df = df.sort_values(by=["diagnosis_level", "task_type"])
        df_p = df.pivot(index="max_audio_len", columns="diagnosis_level", values="top_1_accuracy")
        df_p = (df_p*100).round(decimals=2).sort_values(by=list(df_p.columns), ascending=False)
        df_p.to_csv(f"{self.__results_path}/plot_3.csv")
        
    def plot_4(self, data: pd.DataFrame) -> None:
        df = data.query("exp_type == 'self' & model_name == 'mfcc' & max_audio_len == max_audio_len")
        df = df.sort_values(by=["diagnosis_level", "task_type"])
        df_p = df.pivot(index="max_audio_len", columns="diagnosis_level", values="top_1_accuracy")
        df_p = (df_p*100).round(decimals=2).sort_values(by=list(df_p.columns), ascending=False)
        df_p.to_csv(f"{self.__results_path}/plot_4.csv")
        
    def plot_5(self, data: pd.DataFrame) -> None:
        data = data[data["exp_type"] == "self"]
        # data = data.explode('f1_per_class')
        vowel_tasks = data[data['task_type'].isin(['a','i','u'])]
        speech_tasks = data[data['task_type'].isin(['speech'])]

        def prepare_group(group):
            group = group.reset_index(drop=True)
            df = pd.json_normalize(group["f1_per_class"])
            df = pd.concat((df.idxmax(), df.max()), axis=1).reset_index()
            df.columns = ['label', 'orig_idx', 'max_f1']
            df["max_f1"] = df["max_f1"].round(decimals=2)
            all_cols = [x for x in group.columns if x != "f1_per_class"]
            df = pd.merge(
                group[all_cols],
                df,
                left_index=True,
                right_on='orig_idx',
            ).reset_index(drop=True)
            df = df.sort_values(by="label")
            return df.drop("orig_idx", axis=1)
        
        def fetch_actual(row):
            return row["total_actual_per_class_x"][row["label"]]

        df_speech = speech_tasks.groupby(by="diagnosis_level").apply(prepare_group).reset_index(drop=True)
        df_vowel = vowel_tasks.groupby(by="diagnosis_level").apply(prepare_group).reset_index(drop=True)
        merged = pd.merge(df_speech, df_vowel, on=['diagnosis_level', 'label'])
        merged["total_actual"] = merged[["label", "total_actual_per_class_x"]].apply(fetch_actual, axis=1)
        merged = merged[["diagnosis_level", "label", "task_type_x", "model_name_x", "task_type_y", "model_name_y", "max_f1_x","max_f1_y", "total_actual"]]
        merged["good_vowel"] = merged["max_f1_y"] >= merged["max_f1_x"]
        merged = merged.sort_values(by=["diagnosis_level", "total_actual"])
        merged.to_csv(f"{self.__results_path}/plot_5.csv",index=False)
        
    def save_all_results(self, data: pd.DataFrame) -> None:
        data = data.sort_values(by=["model_name", "diagnosis_level", "top_1_accuracy"], ascending=[True, True, False])
        data.to_csv(f"{self.__results_path}/all_results.csv", index=False)

    def read_results_file(self, file_path: Path) -> pd.Series:
        exp_folder = file_path.parent
        depth = len(exp_folder.parents) - len(self.__cache_results_path.parents)
        exp_key = exp_folder.name
        key_parts = exp_key.split("_")
        train_db = key_parts[0]
        task_type = key_parts[1]
        diagnosis_level = int(key_parts[2])
        model_name = key_parts[-1]
        if len(key_parts) == 5:
            max_audio_len = float(key_parts[3])
        else:
            max_audio_len = None
        if depth == 1:
            label = exp_key
            exp_type = "self"
        else:
            label = f"{exp_folder.parent.stem}_{exp_key}"
            exp_type = "cross"
        data = pd.read_csv(file_path)
        confusion = self.data_to_confusion(data)
        top_1_accuracy = self.top_1_accuracy(confusion=confusion)
        f1_per_class = self.f1_per_class(confusion=confusion)
        total_actual_per_class = self.total_actual_per_class(confusion=confusion)
        row = {
            "exp_type": exp_type,
            "train_db": train_db,
            "task_type": task_type,
            "diagnosis_level": diagnosis_level,
            "model_name": model_name,
            "max_audio_len": max_audio_len,
            "top_1_accuracy": top_1_accuracy,
            "f1_per_class": f1_per_class,
            "total_actual_per_class": total_actual_per_class,
        }
        return pd.Series(row)

    def top_1_accuracy(self, confusion: pd.DataFrame) -> float:
        confusion_np = confusion.to_numpy()
        total_per_class = np.maximum(1, confusion_np.sum(axis=1))
        corrects = confusion_np.diagonal()
        per_class_accuracy = corrects / total_per_class
        accuracy = per_class_accuracy.mean()
        return accuracy


    def total_actual_per_class(self, confusion: pd.DataFrame) -> Dict[str, float]:
        confusion_np = confusion.to_numpy()
        total_actual_per_class = np.maximum(1, confusion_np.sum(axis=1))
        return dict(zip(list(confusion.columns), total_actual_per_class))

    def f1_per_class(self, confusion: pd.DataFrame) -> Dict[str, float]:
        confusion_np = confusion.to_numpy()
        total_actual_per_class = np.maximum(1, confusion_np.sum(axis=1))
        total_predicted_per_class = np.maximum(1, confusion_np.sum(axis=0))
        corrects = confusion_np.diagonal()
        precision_per_class = corrects / total_actual_per_class
        recall_per_class = corrects / total_predicted_per_class
        den = (precision_per_class + recall_per_class)
        den[den == 0] = 1
        f1_per_class = 2  * (precision_per_class * recall_per_class) / den
        return dict(zip(list(confusion.columns), f1_per_class))

    def data_to_confusion(self, data: pd.DataFrame) -> pd.DataFrame:
        confusion: Dict[str, Dict[str, int]] = {}
        all_names = list(set(
            data["actual"].unique().tolist() +
            data["predicted"].unique().tolist()
        ))
        product_name_pair = itertools.product(all_names, all_names)
        for actual, predicted in product_name_pair:
            if actual not in confusion:
                confusion[actual] = {}
            actual = confusion[actual]
            if predicted not in actual:
                actual[predicted] = 0
        for _, row in data.iterrows():
            confusion[row["predicted"]][row["actual"]] += 1
        return pd.DataFrame(confusion).fillna(0).sort_index(axis=0).sort_index(axis=1)