import torch
import torch.optim
import numpy as np
from torch import nn
from tqdm import tqdm
from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt

from .tboard import TBoard, MockBoard
from ..model import Normalized
from ..data_loader import BaseDataLoader

ConfusionData = Dict[int, Dict[int, int]]


class Trainer:
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
        alpha: float,
    ) -> None:
        max_diag_level = max(data_loader.num_unique_diagnosis.keys())
        num_classes = data_loader.num_unique_diagnosis[max_diag_level]
        self.unique_diagnosis = data_loader.unique_diagnosis[max_diag_level]
        class_weights = torch.tensor(
            data_loader.train_class_weights[max_diag_level],
            dtype=torch.float32,
            device=device,
        )
        model = Normalized(
            input_size=data_loader.feature_size,
            num_ids=data_loader.total_ids,
            num_classes=num_classes,
            checkpoint_path=Path(f"{cache_path}/checkpoints/{exp_key}"),
        )
        model.to(device=device)
        id_criterion = nn.CrossEntropyLoss()
        diag_criterion = nn.CrossEntropyLoss(weight=class_weights)
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
        self.id_criterion = id_criterion
        self.diag_criterion = diag_criterion
        self.num_epochs = num_epochs
        self.save_epochs = save_epochs
        self.confusion_epochs = confusion_epochs
        self.save_enabled = True
        self.optimizer = optimizer
        self.alpha = alpha

    def run(self) -> None:
        for epoch in tqdm(range(self.num_epochs), desc="Epoch", position=0):
            train_loss = self.__train_loop()
            eval_loss, eval_accuracy_diag, eval_accuracy_id = self.__eval_loop(
                epoch=epoch
            )
            self.tboard.add_scalars(
                "loss",
                {"train": train_loss, "eval": eval_loss},
                global_step=epoch,
            )
            self.tboard.add_scalar(
                "eval accuracy diag (top 1)", eval_accuracy_diag, global_step=epoch
            )
            self.tboard.add_scalar(
                "eval accuracy id (top 1)", eval_accuracy_id, global_step=epoch
            )
            self.__save(epoch=epoch, eval_accuracy=eval_accuracy_diag)

    def __train_loop(self):
        self.model.train()
        total_loss = 0
        total_batch_size = 0
        for batch in tqdm(
            self.__data_loader.train(),
            desc="Training",
        ):
            if len(batch) == 3:
                inputs, labels, id_tensor = batch
            else:
                inputs, labels, id_tensor, ids = batch
            labels = labels.squeeze(1)
            self.optimizer.zero_grad(set_to_none=True)
            predicted_ids, (predicted_labels, _, _) = self.model(inputs)
            loss_diag = self.diag_criterion(predicted_labels, labels)
            loss_id = self.id_criterion(predicted_ids, id_tensor)
            loss = loss_diag + self.alpha * loss_id
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
        num_unique_diagnosis = len(self.unique_diagnosis)
        num_ids = self.__data_loader.total_ids
        confusion_diag = np.zeros((num_unique_diagnosis, num_unique_diagnosis))
        confusion_id = np.zeros((num_ids, num_ids))

        for batch in tqdm(
            self.__data_loader.test(),
            desc="Validating",
        ):
            if len(batch) == 3:
                inputs, labels, id_tensor = batch
            else:
                inputs, labels, id_tensor, ids = batch
            labels = labels.squeeze(1)
            predicted_ids, (predicted_labels, _, _) = self.model(inputs)
            loss_diag = self.diag_criterion(predicted_labels, labels)
            loss_id = self.id_criterion(predicted_ids, id_tensor)
            loss = loss_diag + self.alpha * loss_id
            predicted_labels = predicted_labels.argmax(dim=1)
            predicted_ids = predicted_ids.argmax(dim=1)
            total_loss += loss.item()
            total_batch_size += 1

            self.__process_result(
                confusion_ref=confusion_diag,
                actual=labels,
                predicted=predicted_labels,
            )
            self.__process_result(
                confusion_ref=confusion_id,
                actual=id_tensor,
                predicted=predicted_ids,
            )

        self.__add_confusion(epoch=epoch, confusion=confusion_diag)
        eval_accuracy_diag = self.__weighted_accuracy(confusion=confusion_diag)
        eval_accuracy_id = self.__weighted_accuracy(confusion=confusion_id)
        return total_loss / total_batch_size, eval_accuracy_diag, eval_accuracy_id

    def __weighted_accuracy(self, confusion: np.ndarray) -> float:
        total_per_class = np.maximum(1, confusion.sum(axis=1))
        corrects = confusion.diagonal()
        per_class_accuracy = corrects / total_per_class
        accuracy = per_class_accuracy.mean()
        return accuracy

    def __process_result(
        self,
        confusion_ref: np.ndarray,
        actual: torch.Tensor,
        predicted: torch.Tensor,
    ) -> None:
        for actual_label, predicted_label in zip(
            actual.cpu().tolist(),
            predicted.cpu().tolist(),
        ):
            confusion_ref[actual_label, predicted_label] += 1

    def __add_confusion(self, epoch: int, confusion: np.ndarray) -> None:
        if epoch not in self.confusion_epochs:
            return
        total_items_per_class = confusion.sum(axis=1, keepdims=True)
        total_items_per_class = np.maximum(1, total_items_per_class)
        confusion_matrix = confusion / total_items_per_class
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
        ax.imshow(
            confusion_matrix,
            cmap="magma",
            interpolation="none",
            aspect="auto",
        )
        ax.xaxis.set_ticks(list(range(len(self.unique_diagnosis))))
        ax.xaxis.set_tick_params(rotation=90)
        ax.yaxis.set_ticks(list(range(len(self.unique_diagnosis))))
        ax.set_yticklabels(self.unique_diagnosis)
        ax.set_xticklabels(self.unique_diagnosis)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        self.tboard.add_figure(
            tag="eval confusion",
            figure=fig,
            global_step=epoch,
        )
        plt.close(fig=fig)

    def __save(self, epoch: int, eval_accuracy: float):
        if not self.save_enabled:
            return

        if eval_accuracy > self.best_eval_accuracy:
            self.best_eval_accuracy = eval_accuracy
            self.model.save(epoch=epoch)
        elif epoch in self.save_epochs:
            self.model.save(epoch=epoch)
