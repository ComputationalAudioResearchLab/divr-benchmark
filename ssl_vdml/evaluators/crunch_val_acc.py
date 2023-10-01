import re
import pandas as pd
from tqdm import tqdm
from typing import List
from pathlib import Path


class Cruncher:
    def run(self, output_path: str, collection_roots: List[str]):
        all_log_files = []
        for root in tqdm(collection_roots, "Collecting log files"):
            all_log_files += list(Path(root).glob("*.log"))
        all_data = []
        for log_file in tqdm(all_log_files, "Crunching logs"):
            log_data = self.read_log(log_file)
            train_acc, val_acc = self.extract_train_val_accuracy(log_data)
            
            all_data += [(train_acc, val_acc, log_file)]
        df = pd.DataFrame(all_data, columns=["train_acc", "val_acc", "log_file"])
        df = df.sort_values(by=["val_acc", "train_acc"], ascending=False)
        df.to_csv(output_path, index=False, float_format="%1.6f")

    def extract_train_val_accuracy(self, log_data):
        train_matches = re.search(r"Train >>\n.*accuracy = (.*)", log_data)
        train_accuracy = float(train_matches[1])
        val_matches = re.search(r"Val >>\n.*accuracy = (.*)", log_data)
        val_accuracy = float(val_matches[1])
        return train_accuracy, val_accuracy

    def read_log(self, log_file):
        try:
            with open(log_file, "r") as log_file:
                return log_file.read()
        except Exception:
            return self.read_log(log_file)


if __name__ == "__main__":
    curdir = Path(__file__).resolve().parent
    Cruncher().run(
        output_path=f"{curdir}/crunched_accuracies.csv",
        collection_roots=[
            "/home/workspace/data/nn_latents_full/16000/nn/results-all",
            "/home/workspace/data/nn_latents[0][0]/16000/nn/results",
            "/home/workspace/data/nn_latents[1][0]/16000/nn/results",
            # "/home/storage/data/baseline_latents/16000/svm/results",
            # "/home/storage/data/baseline_latents/24000/nn/results",
            # "/home/storage/data/baseline_latents/24000/svm/results",
            # "/home/storage/data/nn_latents/16000/nn/results",
            # "/home/storage/data/nn_latents/16000/svm/results",
            # "/home/storage/data/nn_latents/24000/nn/results",
            # "/home/storage/data/nn_latents/24000/svm/results",
            # "/home/storage/data/nn_latents_full/16000/nn/results",
            # "/home/storage/data/nn_latents_full/16000/svm/results",
            # "/home/storage/data/nn_latents_full/24000/nn/results",
            # "/home/storage/data/nn_latents_full/24000/svm/results",
        ],
    )
