import json
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.decomposition import PCA
from divr_benchmark.diagnosis import DiagnosisMap
import matplotlib.pyplot as plt


class Analysis:
    __sr = 16000
    __device = torch.device("cuda")
    __data_path = Path("/home/storage/divr-data/svd")
    __orig_results_path = Path("/home/workspace/icassp_2025/similarity/data")
    diagnosis_map = DiagnosisMap.v1()
    pca = PCA(whiten=True)

    def run(self):
        df: pd.DataFrame = pd.read_pickle(f"{self.__orig_results_path}/data.pkl")
        df = df.explode(column=["diagnosis"])
        df = df.sample(frac=1).reset_index(drop=True)  # randomize rows
        for level in range(4):
            df[f"diagnosis_level_{level}"] = df["diagnosis"].apply(
                lambda x: self.diagnosis_map.get(x).at_level(level).name
            )

        keys = {
            "pyannote_embedding": [0, 1],
            "level_0_model_embedding": [2, 3],
            "level_1_model_embedding": [1, 3],
            "level_2_model_embedding": [1, 3],
            "level_3_model_embedding": [1, 2],
        }
        for key in keys:
            self.__results_path = Path(f"{self.__orig_results_path}/{key}")
            self.__results_path.mkdir(exist_ok=True)
            embeddings = np.stack(arrays=df[key], axis=0)
            pca = self.pca.fit(X=embeddings)
            p = pca.transform(embeddings)
            self.__cos_plot(df, embeddings, "original")
            self.__pca_plot_gender(df, p, "original")
            self.__pca_plot_age(df, p, 50, "original")
            self.__pca_plot_age(df, p, 33, "original")
            self.__pca_plot_age(df, p, 25, "original")

            # removing first two PCA component that correspond to gender
            start, end = keys[key]
            pca.components_[start:end] = 0
            embeddings = pca.inverse_transform(pca.transform(embeddings))
            pca = self.pca.fit(X=embeddings)
            p = pca.transform(embeddings)
            self.__cos_plot(df, embeddings, "genderless")
            self.__pca_plot_gender(df, p, "genderless")
            self.__pca_plot_age(df, p, 50, "genderless")
            self.__pca_plot_age(df, p, 33, "genderless")
            self.__pca_plot_age(df, p, 25, "genderless")

    def __cos_plot(self, df: pd.DataFrame, embeddings: np.ndarray, key: str):
        random_indices = np.arange(len(embeddings))
        np.random.shuffle(random_indices)

        fig, ax = plt.subplots(2, 3, figsize=(18, 12), constrained_layout=True)
        ax[0, 0].imshow(
            X=self.__cosine_distance(embeddings[random_indices]),
            cmap="magma",
            aspect="auto",
            interpolation=None,
        )
        ax[0, 0].set_title("randomized")

        self.__plot_cos_distances(embeddings, df, "gender", ax[0, 1])
        self.__plot_cos_distances(embeddings, df, "diagnosis_level_0", ax[0, 2])
        self.__plot_cos_distances(embeddings, df, "diagnosis_level_1", ax[1, 0])
        self.__plot_cos_distances(embeddings, df, "diagnosis_level_2", ax[1, 1])
        self.__plot_cos_distances(embeddings, df, "diagnosis_level_3", ax[1, 2])

        fig.suptitle("Cosine distances")
        fig.savefig(f"{self.__results_path}/{key}_cosine_distance.png")

    def __plot_cos_distances(self, embeddings, df: pd.DataFrame, key: str, ax):
        indices = df[key].sort_values()
        labels = indices.to_numpy()
        boundaries = np.where(labels[:-1] != labels[1:])[0]
        boundary_labels = labels[np.concatenate([boundaries, [boundaries[-1] + 1]])]
        print(boundary_labels)
        idx = indices.index
        ax.imshow(
            X=self.__cosine_distance(embeddings[idx]),
            cmap="magma",
            aspect="auto",
            interpolation=None,
        )
        for boundary in boundaries:
            ax.axvline(x=boundary)
            ax.axhline(y=boundary)
        ax.set_title(key)

    def __cosine_distance(self, embeddings):
        eps = 1e-8
        norm = np.minimum(np.linalg.norm(embeddings, axis=1), eps)
        num = (embeddings[:, None, None, :] @ embeddings[None, :, :, None]).squeeze()
        den = norm[:, None] * norm[None, :]
        cdist = num / den
        return cdist

    def __pca_plot_age(self, df, p, age_bucket_size, key):
        color_key = "age"
        labels = df[color_key].drop_duplicates().to_list()
        max_age = 100
        color_map = dict([(i, i) for i in range(max_age // age_bucket_size)])
        labels = [f"{i * age_bucket_size} - {(i+1)*age_bucket_size}" for i in color_map]
        colors = df[color_key].apply(lambda x: color_map[int(x) // age_bucket_size])
        self.__plot_data_scatter(p, labels, colors, f"{key}_age_{age_bucket_size}_pca")

    def __pca_plot_gender(self, df, p, key):
        color_key = "gender"
        labels = df[color_key].drop_duplicates().to_list()
        color_map = dict([(l, i) for i, l in enumerate(labels)])
        colors = df[color_key].apply(lambda x: color_map[x])
        self.__plot_data_scatter(p, labels, colors, f"{key}_gender_pca")

    def __plot_data_scatter(self, p, labels, colors, fig_key):
        rows = 5
        cols = 5
        fig, ax = plt.subplots(
            rows,
            cols,
            figsize=(rows * 6, cols * 6),
            constrained_layout=True,
        )
        for i in range(rows):
            for j in range(cols):
                scatter = ax[i, j].scatter(p[:, i], p[:, j], c=colors)
                ax[i, j].legend(handles=scatter.legend_elements()[0], labels=labels)
                ax[i, j].set_title(f"{i} vs {j} component")
        fig.suptitle("PCA for age")
        fig.savefig(f"{self.__results_path}/{fig_key}.png")
