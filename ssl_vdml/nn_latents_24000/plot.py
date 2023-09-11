import json
from pathlib import Path
import matplotlib.pyplot as plt


class Plotter:
    def __init__(self, sampling_rate: int, degree: int, C: float) -> None:
        self.sampling_rate = sampling_rate
        self.degree = degree
        self.C = C

    def run(self, root_dir: Path):
        results_file = f"{root_dir}/results.json"
        output_file = f"{root_dir}/plot_{self.degree}_{self.C}.png"
        with open(results_file, "r") as results_file_ptr:
            results = json.load(results_file_ptr)
        fig, ax = plt.subplots(
            4, 3, figsize=(32, 25), constrained_layout=True, sharex="col", sharey="row"
        )
        self.plot_data(svd_type="svd_a", kernel="rbf", results=results, ax=ax[0, 0])
        self.plot_data(svd_type="svd_a", kernel="linear", results=results, ax=ax[0, 1])
        self.plot_data(svd_type="svd_a", kernel="poly", results=results, ax=ax[0, 2])
        self.plot_data(svd_type="svd_i", kernel="rbf", results=results, ax=ax[1, 0])
        self.plot_data(svd_type="svd_i", kernel="linear", results=results, ax=ax[1, 1])
        self.plot_data(svd_type="svd_i", kernel="poly", results=results, ax=ax[1, 2])
        self.plot_data(svd_type="svd_u", kernel="rbf", results=results, ax=ax[2, 0])
        self.plot_data(svd_type="svd_u", kernel="linear", results=results, ax=ax[2, 1])
        self.plot_data(svd_type="svd_u", kernel="poly", results=results, ax=ax[2, 2])
        self.plot_data(svd_type="svd_aiu", kernel="rbf", results=results, ax=ax[3, 0])
        self.plot_data(
            svd_type="svd_aiu", kernel="linear", results=results, ax=ax[3, 1]
        )
        self.plot_data(svd_type="svd_aiu", kernel="poly", results=results, ax=ax[3, 2])
        fig.savefig(output_file)

    def plot_data(self, svd_type, kernel, results, ax):
        ax.set_title(f"{svd_type}: {kernel}")
        results = results[svd_type][str(self.sampling_rate)]
        for ssl_model in results:
            diag_results = results[ssl_model][kernel][str(self.degree)][str(self.C)][
                "val"
            ]
            bucket = {}
            for diag in diag_results:
                correct = diag_results[diag]["correct"]
                incorrect = diag_results[diag]["incorrect"]
                bucket[diag] = correct / (correct + incorrect)
            bucket = dict(sorted(bucket.items(), key=lambda x: x[0]))
            ax.set_xticks(ticks=range(len(bucket)), labels=bucket.keys(), rotation=90)
            ax.plot(bucket.values(), label=ssl_model)
        ax.legend()


if __name__ == "__main__":
    curdir = Path(__file__).resolve().parent
    Plotter(sampling_rate=24000, degree=20, C=1.0).run(root_dir=curdir)
