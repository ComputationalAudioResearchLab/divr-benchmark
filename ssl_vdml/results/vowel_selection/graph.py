import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns


class Grapher:
    def __init__(self):
        pass

    def run(self, log_path: str, sample_rate):
        df = pd.read_csv(log_path)
        conf_df = df["log_file"].apply(self.resolve_log_tags).apply(pd.Series)
        conf_df.columns = [
            "exp_type",
            "sample_rate",
            "model_type",
            "model_tag",
            "dataset",
            "model_name",
            "model_conf",
        ]
        full_df = pd.concat((df, conf_df), axis=1)
        full_df = full_df[full_df["sample_rate"] == sample_rate]
        full_df = full_df[full_df["dataset"] != "voiced"]
        full_df = self.filter_model_confs(full_df)
        fig, ax = plt.subplots(2, 1, figsize=(7, 7), constrained_layout=True)
        self.my_plot(full_df, min_acc=0.5, ax=ax[0])
        # self.my_plot(full_df, min_acc=0.65, ax=ax[1])
        self.my_plot(full_df, min_acc=0.70, ax=ax[1])
        fig.suptitle("Validation Accuracy / Vowels")
        plt.savefig("vowels.png")

    def my_plot(self, df, min_acc, ax):
        my_df = df[df["val_acc"] > min_acc]
        sns.boxenplot(
            my_df,
            x="dataset",
            y="val_acc",
            ax=ax,
        )
        counts = my_df.groupby("dataset").count()["val_acc"].to_dict()
        print(counts)
        ax.set_title(f"Val Acc > {min_acc:0.2f}")
        ax.set_xlabel(None)
        # ax.set_ylim(bottom=0.50, top=0.77)
        ax.set_ylabel(None)

    def resolve_log_tags(self, log_file: str):
        log_file_parts = log_file.split("/")
        exp_type = log_file_parts[4]
        sample_rate = int(log_file_parts[5])
        model_type = log_file_parts[6]
        model_tag = log_file_parts[8].removesuffix(".log")
        dataset, model_name, model_conf = model_tag.split(".", maxsplit=2)
        model_conf = model_conf.split("_")
        return [
            exp_type,
            sample_rate,
            model_type,
            model_tag,
            dataset,
            model_name,
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
            return True
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
        log_path="/home/workspace/ssl_vdml/evaluators/crunched_accuracies.csv",
        sample_rate=16000,
    )
