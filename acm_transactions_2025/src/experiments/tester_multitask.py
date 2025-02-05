import torch
import torch.optim
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from ..model import NormalizedMultitask
from ..data_loader import BaseDataLoader

ConfusionData = Dict[int, Dict[int, int]]


class TesterMultiTask:
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
        self.unique_diagnosis = data_loader.unique_diagnosis
        model = NormalizedMultitask(
            input_size=data_loader.feature_size,
            num_classes=data_loader.num_unique_diagnosis,
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
        all_results = []
        all_ids = []
        for batch in tqdm(
            self.__data_loader.test(),
            desc="Validating",
        ):
            if len(batch) == 2:
                inputs, labels = batch
            else:
                inputs, labels, ids = batch
            labels = labels.squeeze(1)
            results = self.model(inputs)
            data_at_level = []
            data = []
            for i, result in enumerate(results):
                probabilities, _, _ = result
                predicted_labels = probabilities.argmax(dim=1)
                data_at_level += [
                    labels[:, i : i + 1],
                    predicted_labels[:, None],
                    probabilities,
                ]
            all_ids += ids
            data = torch.cat(data_at_level, dim=1)
            all_results += [data]
        all_results = torch.cat(all_results, dim=0).round(decimals=2)
        column_names: List[str] = []
        for key, val in self.unique_diagnosis.items():
            column_names += [f"actual_{key}", f"predicted_{key}"] + val
        all_results = pd.DataFrame(
            data=all_results.cpu().numpy(),
            columns=column_names,
        )
        all_results["id"] = all_ids
        all_results = all_results[["id"] + column_names]
        for cname in column_names:
            if cname.startswith("actual") or cname.startswith("predicted"):
                prefix, suffix = cname.split("_")
                level = int(suffix)
                all_results[cname] = all_results[cname].apply(
                    lambda idx: self.__data_loader.idx_to_diag_name(int(idx), level)
                )
        all_results.to_csv(f"{self.__results_path}/results.csv", index=False)
        for i in self.unique_diagnosis:
            self.__add_confusion(
                pred=all_results[f"predicted_{i}"],
                actual=all_results[f"actual_{i}"],
                level=i,
            )

    def __add_confusion(self, pred: pd.Series, actual: pd.Series, level: int) -> None:
        confusion = confusion_matrix(
            y_pred=pred,
            y_true=actual,
            labels=self.unique_diagnosis[level],
        )
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
        sns.heatmap(
            data=confusion,
            ax=ax,
            xticklabels=self.unique_diagnosis[level],
            yticklabels=self.unique_diagnosis[level],
            annot=True,
            cbar=False,
        )
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        fig.savefig(f"{self.__results_path}/confusion_{level}.png")
        plt.close(fig=fig)
