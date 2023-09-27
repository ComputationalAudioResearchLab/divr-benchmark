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
        full_df = full_df[full_df["dataset"] == "svd_aiu"]
        full_df = full_df[full_df["val_acc"] > 0.5]
        # full_df = self.filter_model_confs(full_df)
        # full_df = full_df[full_df["dataset"] == "svd_aiu"]
        # full_df = full_df[full_df["exp_type"] == "nn_latents_full"]
        # full_df = full_df[full_df["exp_type"] == "nn_latents"]
        fig, ax = plt.subplots(1, 1, figsize=(13, 6), constrained_layout=True)
        self.my_plot(full_df, None, ax)
        # self.my_plot(full_df, dataset="svd_a", ax=ax[0])
        # self.my_plot(full_df, dataset="svd_i", ax=ax[1])
        # self.my_plot(full_df, dataset="svd_u", ax=ax[2])
        # self.my_plot(full_df, dataset="svd_aiu", ax=ax[3])
        # self.my_plot(full_df, min_acc=0.65, ax=ax[1])
        # self.my_plot(full_df, min_acc=0.70, ax=ax[2])
        fig.suptitle(
            "Validation accuracy for different features on vowel set a,i,u with validation accuracy > 0.5"
        )
        plt.savefig("feature_analysis.png")

    def my_plot(self, my_df, dataset, ax):
        # my_df = df[df["dataset"] == dataset]
        maxs = my_df.groupby("model_name")["val_acc"].max().to_dict()
        means = my_df.groupby("model_name")["val_acc"].mean().to_dict()
        stds = my_df.groupby("model_name")["val_acc"].std().to_dict()
        best_keys = dict(
            list(sorted(means.items(), key=lambda x: x[1], reverse=True))
        ).keys()

        for key in best_keys:
            mean = means[key] * 100
            std = stds[key] * 100
            max = maxs[key] * 100
            print(f"{key}: avg={mean:2.2f} +- {std:2.2f}, max={max:2.2f}")
        sns.violinplot(
            my_df,
            x="model_name",
            y="val_acc",
            inner="quartile",
            # hue="exp_type",
            order=best_keys,
            ax=ax,
        )
        # ax.set_title(f"Val Acc > {min_acc:0.2f}")
        ax.set_xlabel(None)
        # ax.set_ylim(bottom=0.50, top=0.77)
        ax.tick_params(rotation=90)
        ax.set_ylabel(dataset)

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
        log_path="/home/workspace/ssl_vdml/evaluators/crunched_accuracies.csv",
        sample_rate=16000,
    )
