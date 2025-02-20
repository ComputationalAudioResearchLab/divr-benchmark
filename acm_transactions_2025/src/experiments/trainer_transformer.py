import torch
import torch.optim
import numpy as np
from torch import nn
from tqdm import tqdm
from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt

from .tboard import TBoard, MockBoard
from ..model import SimpleTransformer
from ..data_loader import BaseDataLoader

ConfusionData = Dict[int, Dict[int, int]]


class TrainerTransformer:
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
        model = SimpleTransformer(
            input_size=data_loader.feature_size,
            num_classes=num_classes,
            num_speakers=data_loader.total_speakers,
            checkpoint_path=Path(f"{cache_path}/checkpoints/{exp_key}"),
        )
        model.to(device=device)
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
        self.criterion_diag = nn.CrossEntropyLoss(weight=class_weights)
        self.criterion_spk = nn.CrossEntropyLoss()
        self.num_epochs = num_epochs
        self.save_epochs = save_epochs
        self.confusion_epochs = confusion_epochs
        self.save_enabled = True
        self.optimizer = optimizer
        self.alpha = alpha

    def run(self) -> None:
        for epoch in tqdm(range(self.num_epochs), desc="Epoch", position=0):
            train_loss, train_diag_accuracy, train_spk_accuracy = self.__train_loop()
            eval_loss, eval_diag_accuracy, eval_spk_accuracy = self.__eval_loop(
                epoch=epoch
            )
            self.tboard.add_scalars(
                "loss",
                {"train": train_loss, "eval": eval_loss},
                global_step=epoch,
            )
            self.tboard.add_scalars(
                "spk_accuracy",
                {"train": train_spk_accuracy, "eval": eval_spk_accuracy},
                global_step=epoch,
            )
            self.tboard.add_scalars(
                "diag_accuracy (top 1)",
                {"train": train_diag_accuracy, "eval": eval_diag_accuracy},
                global_step=epoch,
            )
            self.__save(epoch=epoch, eval_accuracy=eval_diag_accuracy)

    def __train_loop(self):
        self.model.train()
        total_loss = 0
        total_batch_size = 0
        correct_spks = 0
        total_spks = 0
        correct_diags = 0
        for batch in tqdm(
            self.__data_loader.train(random_cuts=True),
            desc="Training",
        ):
            inputs, labels, id_tensor, ids = batch
            assert id_tensor is not None
            labels = labels.squeeze(1)
            self.optimizer.zero_grad(set_to_none=True)
            predicted_speaker_ids, (predicted_labels, _, _) = self.model(inputs)
            diag_loss = self.criterion_diag(predicted_labels, labels)
            spk_loss = self.criterion_spk(predicted_speaker_ids, id_tensor)
            if torch.isnan(diag_loss).any():
                raise ValueError("NaNs at diag_loss")
            if torch.isnan(spk_loss).any():
                raise ValueError("NaNs at spk_loss")
            loss = diag_loss + self.alpha * spk_loss
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_batch_size += 1
            predicted_spks = predicted_speaker_ids.argmax(dim=1)
            predicted_diags = predicted_labels.argmax(dim=1)
            correct_spks += (predicted_spks == id_tensor).sum()
            correct_diags += (predicted_diags == labels).sum()
            total_spks += len(id_tensor)
        total_diags = total_spks
        return (
            total_loss / total_batch_size,
            correct_diags / total_diags,
            correct_spks / total_spks,
        )

    @torch.no_grad()
    def __eval_loop(self, epoch):
        self.model.eval()
        total_loss = 0
        total_batch_size = 0
        num_unique_diagnosis = len(self.unique_diagnosis)
        confusion_diag = np.zeros((num_unique_diagnosis, num_unique_diagnosis))
        correct_spks = 0
        total_spks = 0

        for batch in tqdm(
            self.__data_loader.eval(),
            desc="Validating",
        ):
            inputs, labels, id_tensor, ids = batch
            assert id_tensor is not None
            labels = labels.squeeze(1)
            predicted_speaker_ids, (predicted_labels, _, _) = self.model(inputs)
            diag_loss = self.criterion_diag(predicted_labels, labels)
            spk_loss = self.criterion_spk(predicted_speaker_ids, id_tensor)
            if torch.isnan(diag_loss).any():
                raise ValueError("NaNs at diag_loss")
            if torch.isnan(spk_loss).any():
                raise ValueError("NaNs at spk_loss")
            loss = diag_loss + self.alpha * spk_loss
            predicted_diags = predicted_labels.argmax(dim=1)
            total_loss += loss.item()
            total_batch_size += 1

            predicted_spks = predicted_speaker_ids.argmax(dim=1)
            correct_spks += (predicted_spks == id_tensor).sum()
            total_spks += len(id_tensor)

            self.__process_result(
                confusion_ref=confusion_diag,
                actual=labels,
                predicted=predicted_diags,
            )

        self.__add_confusion(epoch=epoch, confusion=confusion_diag)
        eval_accuracy = self.__weighted_accuracy(confusion=confusion_diag)
        return total_loss / total_batch_size, eval_accuracy, correct_spks / total_spks

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
