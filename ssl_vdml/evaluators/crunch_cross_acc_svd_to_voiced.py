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
            acc = self.extract_accuracy(log_data)
            all_data += [(acc, log_file)]
        df = pd.DataFrame(all_data, columns=["acc", "log_file"])
        df = df.sort_values(by=["acc"], ascending=False)
        df.to_csv(output_path, index=False, float_format="%1.6f")

    def extract_accuracy(self, log_data):
        matches = re.search(r"Val >>\n.*accuracy = (.*)", log_data)
        accuracy = float(matches[1])
        return accuracy

    def read_log(self, log_file):
        try:
            with open(log_file, "r") as log_file:
                return log_file.read()
        except Exception:
            return self.read_log(log_file)


if __name__ == "__main__":
    curdir = Path(__file__).resolve().parent
    Cruncher().run(
        output_path=f"{curdir}/crunched_cross_accuracies_svd_to_voiced.csv",
        collection_roots=[
            "/home/storage/data/cross_tester/svd_to_voiced/results",
        ],
    )
