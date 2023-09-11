import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import List


class Grapher:
    def __init__(self):
        pass

    def run(self, logs: List[str], sample_rate):
        dfs = []
        for log in logs:
            dfs += [pd.read_csv(log)]
        df = pd.concat(dfs).reset_index()
        conf_df = df["log_file"].apply(self.resolve_log_tags).apply(pd.Series)
        conf_df.columns = [
            "exp_type",
            "feature_type",
            "sample_rate",
            "model_type",
            "train_data",
            "feature_name",
            "model_conf",
        ]
        full_df = pd.concat((df, conf_df), axis=1)
        full_df = full_df[full_df["sample_rate"] == sample_rate]
        full_df = full_df[full_df["exp_type"] != "voiced_to_svd_aiu"]
        # full_df = self.filter_model_confs(full_df)
        fig, ax = plt.subplots(
            1, 1, figsize=(7, 5), constrained_layout=True, sharey="row"
        )
        self.my_plot(full_df, min_acc=0.0, ax=ax)
        # self.my_plot(full_df, min_acc=0.60, ax=ax[1])
        fig.suptitle("Validation Accuracy / Feature Embeddings")
        plt.savefig("cross_accuracies.png")

    def my_plot(self, df, min_acc, ax):
        my_df = df[df["acc"] > min_acc]
        sns.boxenplot(
            my_df,
            x="exp_type",
            y="acc",
            hue="feature_type",
            ax=ax,
        )
        # ax.set_title(f"Val Acc > {min_acc:0.2f}")
        ax.set_xlabel(None)
        ax.set_ylim(bottom=0.40, top=0.72)
        ax.set_ylabel(None)
        # ax.tick_params(rotation=90)

    def resolve_log_tags(self, log_file: str):
        log_file_parts = log_file.split("/")
        exp_type = log_file_parts[5]
        log_tag = log_file_parts[7].removesuffix(".log")
        (
            feature_type,
            sample_rate,
            model_type,
            train_data,
            feature_name,
            model_conf,
        ) = log_tag.split(".", maxsplit=5)
        sample_rate = int(sample_rate)
        model_conf = model_conf.split("_")
        return [
            exp_type,
            feature_type,
            sample_rate,
            model_type,
            train_data,
            feature_name,
            model_conf,
        ]

    def filter_model_confs(self, df):
        def nn_filter(model_conf):
            epochs, lr, layers, latent_dim = model_conf
            epochs = int(epochs)
            lr = float(lr)
            layers = int(layers)
            latent_dim = int(latent_dim)
            return epochs == 1000 and lr == 1e-4 and layers > 1

        def svm_filter(model_conf):
            C, degree, kernel = model_conf
            C = float(C)
            degree = int(degree)
            return kernel == "rbf"

        def model_filter(model_type, model_conf):
            # return True
            if model_type == "nn":
                return nn_filter(model_conf)
            else:
                return svm_filter(model_conf)

        mask = df.apply(
            lambda x: model_filter(x["model_type"], x["model_conf"]), axis=1
        )
        return df[mask]


if __name__ == "__main__":
    Grapher().run(
        logs=[
            "/home/workspace/ssl_vdml/evaluators/crunched_cross_accuracies_svd_to_voiced.csv",
            "/home/workspace/ssl_vdml/evaluators/crunched_cross_accuracies_voiced_to_svd_a.csv",
            "/home/workspace/ssl_vdml/evaluators/crunched_cross_accuracies_voiced_to_svd_i.csv",
            "/home/workspace/ssl_vdml/evaluators/crunched_cross_accuracies_voiced_to_svd_u.csv",
            "/home/workspace/ssl_vdml/evaluators/crunched_cross_accuracies_voiced_to_svd_aiu.csv",
        ],
        sample_rate=16000,
    )
