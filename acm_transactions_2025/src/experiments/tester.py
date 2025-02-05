import torch
import torch.optim
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from ..model import Normalized
from ..data_loader import BaseDataLoader

ConfusionData = Dict[int, Dict[int, int]]


class Tester:
    best_eval_accuracy = 0

    def __init__(
        self,
        cache_path: Path,
        results_path: Path,
        device: torch.device,
        data_loader: BaseDataLoader,
        exp_key: str,
        load_epoch: int,
    ) -> None:
        max_diag_level = max(data_loader.num_unique_diagnosis.keys())
        num_classes = data_loader.num_unique_diagnosis[max_diag_level]
        self.unique_diagnosis = data_loader.unique_diagnosis[max_diag_level]
        model = Normalized(
            input_size=data_loader.feature_size,
            num_classes=num_classes,
            checkpoint_path=Path(f"{cache_path}/checkpoints/{exp_key}"),
        )
        model.to(device=device)
        model.eval()
        model.load(load_epoch)
        self.__data_loader = data_loader
        self.__results_path = Path(f"{results_path}/{exp_key}/{load_epoch}")
        self.__results_path.mkdir(parents=True, exist_ok=True)
        self.model = model

    @torch.no_grad()
    def test(self) -> None:
        results = []
        all_ids = []
        for batch in tqdm(
            self.__data_loader.test(),
            desc="Testing",
        ):
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs, labels, ids = batch
            labels = labels.squeeze(1)
            probabilities, _, _ = self.model(inputs)
            predicted_labels = probabilities.argmax(dim=1)
            data = torch.cat(
                [labels[:, None], predicted_labels[:, None], probabilities],
                dim=1,
            )
            all_ids += ids
            results += [data]
        results = torch.cat(results, dim=0).round(decimals=2)
        results = pd.DataFrame(
            data=results.cpu().numpy(),
            columns=["actual", "predicted"] + self.unique_diagnosis,
        )
        results["id"] = all_ids
        results = results[["id", "actual", "predicted"] + self.unique_diagnosis]
        results["actual"] = results["actual"].apply(self.map_diag_name)
        results["predicted"] = results["predicted"].apply(self.map_diag_name)
        results.to_csv(f"{self.__results_path}/results.csv", index=False)
        self.__add_confusion(results=results)

    def map_diag_name(self, idx: float):
        return self.__data_loader.idx_to_diag_name(
            int(idx), self.__data_loader.max_diag_level
        )

    def __add_confusion(self, results: pd.DataFrame) -> None:
        confusion = confusion_matrix(
            y_pred=results["predicted"],
            y_true=results["actual"],
            labels=self.unique_diagnosis,
        )
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
        sns.heatmap(
            data=confusion,
            ax=ax,
            xticklabels=self.unique_diagnosis,
            yticklabels=self.unique_diagnosis,
            annot=True,
            cbar=False,
        )
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        fig.savefig(f"{self.__results_path}/confusion.png")
        plt.close(fig=fig)
