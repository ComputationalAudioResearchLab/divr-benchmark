import re
import json
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class Analysis:
    def __init__(self, analysis_files_root: Path) -> None:
        assert analysis_files_root.is_dir()
        log_files = analysis_files_root.glob("logs/*.log")
        results_files = analysis_files_root.glob("results/*.log")
        self.analysis_files = list(zip(log_files, results_files))
        self.analysis_output = f"{analysis_files_root}/analysis.csv"

    def run(self):
        all_results = []
        for log_file, result_file in tqdm(self.analysis_files):
            config = self.get_config(log_file=log_file)
            results = self.read_results(result_file=result_file)
            if results is not None:
                assert config["key"] == results["key"]
                all_results += [
                    {
                        "diagnosis_level": config["data"]["diagnosis_level"],
                        "balance_dataset": config["data"]["balance_dataset"],
                        "sampling_rate": config["data"]["feature_params"]["common"][
                            "sampling_frequency"
                        ],
                        "svm_kernel": config["model"]["kernel"],
                        "svm_degree": config["model"]["degree"],
                        "mfcc_coefficients": config["data"]["feature_params"][
                            "mean_mfcc_praat"
                        ]["number_of_coefficients"],
                        "train_accuracy": results["train"]["accuracy"],
                        "train_precision": results["train"]["precision"],
                        "train_recall": results["train"]["recall"],
                        "train_f1": results["train"]["f1"],
                        "val_accuracy": results["val"]["accuracy"],
                        "val_precision": results["val"]["precision"],
                        "val_recall": results["val"]["recall"],
                        "val_f1": results["val"]["f1"],
                    }
                ]
        df = pd.DataFrame(all_results)
        df = df.sort_values(
            by=[
                "diagnosis_level",
                "balance_dataset",
                "sampling_rate",
                "svm_kernel",
                "svm_degree",
                "mfcc_coefficients",
            ]
        )
        df = df.reset_index(drop=True)
        df.to_csv(self.analysis_output, index=False)

    def get_config(self, log_file):
        with open(log_file, "r") as logs:
            matches = re.search(r"config: ({.*})", logs.read())
            assert matches is not None
            return json.loads(matches[1])

    def read_results(self, result_file):
        with open(result_file, "r") as results:
            lines = results.readlines()
            if len(lines) < 3:
                return None
            return {
                "key": self.parse_key(lines[0]),
                "train": self.parse_result_line(lines[1]),
                "val": self.parse_result_line(lines[2]),
            }

    def parse_key(self, key_line):
        matches = re.search(r"key=(.*)accuracy,", key_line)
        assert matches is not None
        return matches[1]

    def parse_result_line(self, result_line):
        accuracy, precision, recall, f1 = result_line.split(",")
        return {
            "accuracy": float(accuracy.strip()),
            "precision": float(precision.strip()),
            "recall": float(recall.strip()),
            "f1": float(f1.strip()),
        }
