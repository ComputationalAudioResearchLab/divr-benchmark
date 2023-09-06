import json
import torch
import random
import numpy as np
from pathlib import Path
from typing import Dict
from src.complexity.data import Data
from src.logger import Logger
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


class Complexity:
    def __init__(self, config: Dict) -> None:
        key = config["key"]
        logger = Logger(log_path=config["log_path"], key=key)
        logger.info(f"key: {key}")
        logger.info(f"config: {json.dumps(config)}\n\n")
        self.seed(config["random_seed"])
        self.data = Data(
            **config["data"],
        )
        self.pca = PCA(n_components=2)
        self.tsne = TSNE(n_components=2, perplexity=42, learning_rate="auto")
        self.output_path = Path(config["output_path"])
        self.output_path.mkdir(parents=True, exist_ok=True)

    def seed(self, random_seed):
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

    def run(self):
        self.data.load()
        X, Y = self.data.X, self.data.Y
        legends = [self.data.diagnosis_map.from_int(y).name for y in np.unique(Y)]
        X_pca = self.pca.fit_transform(X)
        X_tsne = self.tsne.fit_transform(X)
        fig, ax = plt.subplots(
            2, 1, figsize=(20, 20), constrained_layout=True, sharex="col"
        )
        for y in np.unique(Y):
            mask = Y == y
            ax[0].scatter(X_pca[mask, 0], X_pca[mask, 1], label=y)
            ax[1].scatter(X_tsne[mask, 0], X_tsne[mask, 1], label=y)

        ax[0].set_title("PCA")
        ax[0].legend(legends)
        ax[1].set_title("TSNE")
        ax[1].legend(legends)
        fig.savefig(f"{self.output_path}/complexity.png", bbox_inches="tight")
