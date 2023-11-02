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
        output_path=f"{curdir}/wavlm_large_crunched_accuracies.csv",
        collection_roots=[
            "/home/workspace/data/wavlm_data2vec[10][0]_a_n/16000/nn/results",
            "/home/workspace/data/wavlm_decoar2[10][5]_a_n/16000/nn/results",
            "/home/workspace/data/wavlm_hubert[10][3]_a_n/16000/nn/results",
            # "/home/workspace/data/data2vec[0][1]_a_n/16000/nn/results",
            # "/home/workspace/data/decoar2[5][4]_a_n/16000/nn/results",
            # "/home/workspace/data/hubert[3][2]_a_n/16000/nn/results",
            # "/home/workspace/data/wav2vec_large[3][4]_a_n/16000/nn/results",
            # "/home/workspace/data/wavlm_large[10][5]_a_n/16000/nn/results",
            # "/home/workspace/data/wavlm_large_nn_latents_full/16000/nn/results",
            # "/home/workspace/data/wavlm_large_nn_latents[0][0]_a_n/16000/nn/results",
            # "/home/workspace/data/wavlm_large_nn_latents[1][0]_a_n/16000/nn/results",
            # "/home/workspace/data/wavlm_large_nn_latents[2][0]_a_n/16000/nn/results",
            # "/home/workspace/data/wavlm_large_nn_latents[3][0]_a_n/16000/nn/results",
            # "/home/workspace/data/wavlm_large_nn_latents[4][0]_a_n/16000/nn/results",
            # "/home/workspace/data/wavlm_large_nn_latents[5][0]_a_n/16000/nn/results",
            # "/home/workspace/data/wavlm_large_nn_latents[6][0]_a_n/16000/nn/results",
            # "/home/workspace/data/wavlm_large_nn_latents[7][0]_a_n/16000/nn/results",
            # "/home/workspace/data/wavlm_large_nn_latents[8][0]_a_n/16000/nn/results",
            # "/home/workspace/data/wavlm_large_nn_latents[9][0]_a_n/16000/nn/results",
            # "/home/workspace/data/wavlm_large_nn_latents[10][0]_a_n/16000/nn/results",
            # "/home/workspace/data/wavlm_large_nn_latents[11][0]_a_n/16000/nn/results",
            # "/home/workspace/data/wavlm_large_nn_latents[12][0]_a_n/16000/nn/results",
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
