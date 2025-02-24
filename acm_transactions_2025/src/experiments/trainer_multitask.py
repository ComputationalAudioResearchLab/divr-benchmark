import torch
import torch.optim
import numpy as np
from torch import nn
from tqdm import tqdm
from typing import Dict, List
from pathlib import Path
import matplotlib.pyplot as plt

from .tboard import TBoard, MockBoard
from ..model import NormalizedMultitask
from ..data_loader import BaseDataLoader

ConfusionData = Dict[int, Dict[int, int]]


class TrainerMultiTask:
    best_eval_accuracy = 0

    def __init__(
        self,
        cache_path: Path,
        device: torch.device,
        data_loader: BaseDataLoader,
        exp_key: str,
        num_epochs: int,
        tboard_enabled: bool,
        lr: float,
    ) -> None:
        model = NormalizedMultitask(
            input_size=data_loader.feature_size,
            num_classes=data_loader.num_unique_diagnosis,
            checkpoint_path=Path(f"{cache_path}/checkpoints/{exp_key}"),
        )
        model.to(device=device)
        self.criterions = [
            nn.CrossEntropyLoss(
                weight=torch.tensor(
                    class_weights,
                    dtype=torch.float32,
                    device=device,
                )
            )
            for class_weights in data_loader.train_class_weights.values()
        ]
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)
        save_epochs = list(range(0, num_epochs + 1, num_epochs // 10))
        confusion_epochs = list(range(0, num_epochs + 1, 10))
        tensorboard_path = Path(f"{cache_path}/tboard/{exp_key}")
        self.__data_loader = data_loader
        if tboard_enabled:
            self.tboard = TBoard(tensorboard_path=tensorboard_path)
        else:
            self.tboard = MockBoard()
        self.model = model
        self.num_epochs = num_epochs
        self.save_epochs = save_epochs
        self.confusion_epochs = confusion_epochs
        self.save_enabled = True
        self.optimizer = optimizer

    def run(self) -> None:
        for epoch in tqdm(range(self.num_epochs), desc="Epoch", position=0):
            train_loss = self.__train_loop()
            eval_loss, eval_accuracies = self.__eval_loop(epoch=epoch)
            self.tboard.add_scalars(
                "loss",
                {"train": train_loss, "eval": eval_loss},
                global_step=epoch,
            )
            for level, accuracy in eval_accuracies.items():
                self.tboard.add_scalar(
                    f"eval accuracy (top 1), level {level}",
                    accuracy,
                    global_step=epoch,
                )
            self.__save(epoch=epoch, eval_accuracies=eval_accuracies)

    def __train_loop(self):
        self.model.train()
        total_loss = 0
        total_batch_size = 0
        for batch in tqdm(
            self.__data_loader.train(random_cuts=False),
            desc="Training",
        ):
            inputs, labels, id_tensor, ids = batch
            labels = labels.squeeze(1)
            self.optimizer.zero_grad(set_to_none=True)
            results = self.model(inputs)
            loss = torch.scalar_tensor(0, device=labels.device)
            data = zip(results, self.criterions)
            for i, (result, criterion) in enumerate(data):
                predicted_labels, _, _ = result
                loss += criterion(predicted_labels, labels[:, i])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_batch_size += 1
        return total_loss / total_batch_size

    @torch.no_grad()
    def __eval_loop(self, epoch):
        self.model.eval()
        total_loss = 0
        total_batch_size = 0
        confusions = dict(
            [
                (level, np.zeros((num_unique_diagnosis, num_unique_diagnosis)))
                for (
                    level,
                    num_unique_diagnosis,
                ) in self.__data_loader.num_unique_diagnosis.items()
            ]
        )

        for batch in tqdm(
            self.__data_loader.test(),
            desc="Validating",
        ):
            inputs, labels, id_tensor, ids = batch
            labels = labels.squeeze(1)
            results = self.model(inputs)
            loss = torch.scalar_tensor(0, device=labels.device)
            predictions = []
            data = zip(results, self.criterions)
            for i, (result, criterion) in enumerate(data):
                predicted_labels, _, _ = result
                loss += criterion(predicted_labels, labels[:, i])
                predictions += [predicted_labels.argmax(dim=1)]
            total_loss += loss.item()
            total_batch_size += 1

            self.__process_result(
                confusion_ref=confusions,
                actual=labels,
                predicted=predictions,
            )

        self.__add_confusions(epoch=epoch, confusions=confusions)
        eval_accuracies = dict(
            [
                (level, self.__weighted_accuracy(confusion=confusion))
                for level, confusion in confusions.items()
            ]
        )
        return total_loss / total_batch_size, eval_accuracies

    def __weighted_accuracy(self, confusion: np.ndarray) -> float:
        total_per_class = np.maximum(1, confusion.sum(axis=1))
        corrects = confusion.diagonal()
        per_class_accuracy = corrects / total_per_class
        accuracy = per_class_accuracy.mean()
        return accuracy

    def __process_result(
        self,
        confusion_ref: Dict[int, np.ndarray],
        actual: torch.Tensor,
        predicted: List[torch.Tensor],
    ) -> None:
        for idx, key in enumerate(confusion_ref):
            for actual_label, predicted_label in zip(
                actual[:, idx].cpu().tolist(),
                predicted[idx].cpu().tolist(),
            ):
                confusion_ref[key][actual_label, predicted_label] += 1

    def __add_confusions(
        self,
        epoch: int,
        confusions: Dict[int, np.ndarray],
    ) -> None:
        if epoch not in self.confusion_epochs:
            return
        for level, confusion in confusions.items():
            unique_diagnosis = self.__data_loader.unique_diagnosis[level]
            total_items_per_class = confusion.sum(axis=1, keepdims=True)
            total_items_per_class = np.maximum(1, total_items_per_class)
            confusion_matrix = confusion / total_items_per_class
            fig, ax = plt.subplots(
                1,
                1,
                figsize=(4, 4),
                constrained_layout=True,
            )
            ax.imshow(
                confusion_matrix,
                cmap="magma",
                interpolation="none",
                aspect="auto",
            )
            ax.xaxis.set_ticks(list(range(len(unique_diagnosis))))
            ax.xaxis.set_tick_params(rotation=90)
            ax.yaxis.set_ticks(list(range(len(unique_diagnosis))))
            ax.set_yticklabels(unique_diagnosis)
            ax.set_xticklabels(unique_diagnosis)
            ax.set_ylabel("Actual")
            ax.set_xlabel("Predicted")
            self.tboard.add_figure(
                tag=f"[Level {level}] eval confusion",
                figure=fig,
                global_step=epoch,
            )
            plt.close(fig=fig)

    def __save(self, epoch: int, eval_accuracies: Dict[int, float]):
        if not self.save_enabled:
            return
        # Take the last accuracy as that should be
        # for the max level of diagnosis
        eval_accuracy = list(eval_accuracies.values())[-1]
        if eval_accuracy > self.best_eval_accuracy:
            self.best_eval_accuracy = eval_accuracy
            self.model.save(epoch=epoch)
        elif epoch in self.save_epochs:
            self.model.save(epoch=epoch)
