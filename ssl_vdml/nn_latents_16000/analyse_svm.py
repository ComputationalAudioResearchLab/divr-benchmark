import re
import json
from typing import List
from pathlib import Path
from tqdm import tqdm
from diagnosis import DiagnosisMap


class Analyser:
    def __init__(self):
        self.diagnosis_map = DiagnosisMap()

    def run(self, output_file: str, data_file_root: str):
        all_results = {}
        data_files = list(Path(data_file_root).glob("*.log"))
        for data_file in tqdm(data_files, "processing data"):
            (
                svd_type,
                sampling_rate,
                ssl_type,
                (C, degree, kernel),
            ) = self.get_file_tags(data_file)
            this_results = all_results
            if svd_type not in this_results:
                all_results[svd_type] = {}
            this_results = this_results[svd_type]
            if sampling_rate not in this_results:
                this_results[sampling_rate] = {}
            this_results = this_results[sampling_rate]
            if ssl_type not in this_results:
                this_results[ssl_type] = {}
            this_results = this_results[ssl_type]
            if kernel not in this_results:
                this_results[kernel] = {}
            this_results = this_results[kernel]
            if degree not in this_results:
                this_results[degree] = {}
            this_results = this_results[degree]
            if C not in this_results:
                this_results[C] = {}
            this_results = this_results[C]

            train_results, val_results = self.analyse_one_file(data_file=data_file)
            this_results["train"] = train_results
            this_results["val"] = val_results

        with open(output_file, "w") as output_file_ptr:
            json.dump(all_results, output_file_ptr, indent=2)

    def get_file_tags(self, data_file: Path):
        file_parts = str(data_file).split("/")
        sampling_rate = int(file_parts[5])
        file_name = file_parts[8]
        svd_type, ssl_type, svm_config = file_name.split(".", maxsplit=2)
        svm_config = svm_config.removesuffix(".log")
        C, degree, kernel = svm_config.split("_")
        return (svd_type, sampling_rate, ssl_type, (C, degree, kernel))

    def analyse_one_file(self, data_file: Path):
        train_results, val_results = self.read_data(data_file=data_file)
        train_results = self.bucket_results(train_results)
        val_results = self.bucket_results(val_results)
        return train_results, val_results

    def bucket_results(self, results):
        buckets = {}
        for result in results:
            target, pred, diags = result
            for diag in diags:
                for key in diag:
                    if key not in buckets:
                        buckets[key] = {"correct": 0, "incorrect": 0}
                    if target == pred:
                        buckets[key]["correct"] += 1
                    else:
                        buckets[key]["incorrect"] += 1

        def acc(dict_item):
            value = dict_item[1]
            return value["correct"] / (value["correct"] + value["incorrect"])

        buckets = dict(sorted(buckets.items(), key=acc, reverse=True))
        return buckets

    def read_data(self, data_file: Path):
        with open(data_file, "r") as data:
            data = data.read()
            train_matches = re.findall(
                r"Train >>(.|\n)*full eval results >((.|\n)*)Val", data
            )[0][1].split("\n")[1:-1]
            train_results = self.format_eval_results(train_matches)
            val_matches = re.findall(
                r"Val >>(.|\n)*full eval results >((.|\n)*)", data
            )[0][1].split("\n")[1:-1]
            val_results = self.format_eval_results(val_matches)
            return train_results, val_results

    def format_eval_results(self, results: List[str]):
        formatted_results = []
        for result in results:
            matches = re.findall(r"(\d+),(\d+),(\[\[.*\]\])", result)[0]
            if matches is None:
                print(result)
                exit()
            target, pred, diagnosis = matches
            target = self.diagnosis_map.from_int(int(target)).name
            pred = self.diagnosis_map.from_int(int(pred)).name
            diagnosis = json.loads(diagnosis.replace("'", '"'))
            formatted_results += [(target, pred, diagnosis)]
        return formatted_results


if __name__ == "__main__":
    curdir = Path(__file__).resolve().parent
    analyser = Analyser()
    analyser.run(
        output_file=f"{curdir}/results_svm.json",
        data_file_root="/home/storage/data/nn_latents/16000/svm/results",
    )
