import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
        full_df = full_df[
            (full_df["exp_type"] == "baseline_latents")
            | (full_df["exp_type"] == "nn_latents")
        ]
        svd_a_mask = full_df["dataset"] == "svd_a"
        svd_i_mask = full_df["dataset"] == "svd_i"
        svd_u_mask = full_df["dataset"] == "svd_u"
        svd_aiu_mask = full_df["dataset"] == "svd_aiu"
        df_a = self.filter_model_confs(full_df[svd_a_mask])
        df_i = self.filter_model_confs(full_df[svd_i_mask])
        df_u = self.filter_model_confs(full_df[svd_u_mask])
        df_aiu = self.filter_model_confs(full_df[svd_aiu_mask])
        fig, ax = plt.subplots(
            4, 1, figsize=(25, 20), constrained_layout=True, sharex="col"
        )
        # order = sorted(full_df["model_name"].unique())
        # print(order)
        order = [
            "mel_mu",
            "mel_mu_mel_std",
            "mel_mu_mel_std_mfcc_mu_mfcc_std",
            "mel_std",
            "mfcc_mu",
            "mfcc_mu_mfcc_std",
            "mfcc_std",
            "apc_960hr",
            "data2vec_base_960",
            "decoar2",
            "decoar_layers",
            "distilhubert_base",
            "hubert_base",
            "hubert_base_robust_mgr",
            "modified_cpc",
            "npc_960hr",
            "unispeech_sat_large",
            "vq_apc_960hr",
            "vq_wav2vec_gumbel",
            "vq_wav2vec_kmeans",
            "vq_wav2vec_kmeans_roberta",
            "wav2vec2_large_lv60_cv_swbd_fsh",
            "wav2vec_large",
            "wavlm_large",
            "xls_r_2b",
            "xlsr_53",
        ]
        for idx, df_ in enumerate([df_a, df_i, df_u, df_aiu]):
            # sns.swarmplot(df_, x="model_name", y="val_acc", ax=ax[idx], order=order)
            sns.boxplot(df_, x="model_name", y="val_acc", ax=ax[idx], order=order)
        for ax_ in ax:
            ax_.tick_params(rotation=90)
            ax_.set_ylim(bottom=0.5, top=0.8)
        plt.savefig("test.png")

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
            return epochs == 1000 and lr == 1e-4 and layers > 0

        def svm_filter(model_conf):
            C, degree, kernel = model_conf
            C = float(C)
            degree = int(degree)
            return kernel == "rbf"

        def model_filter(model_type, model_conf):
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
