import torch
import torch.optim
import numpy as np
from torch import nn
from tqdm import tqdm
from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt

from .tboard import TBoard, MockBoard
from ..model import Feature, Normalized
from ..data_loader import DataLoaderWithFeature, RandomAudioDataLoaderWithFeature, Database

ConfusionData = Dict[int, Dict[int, int]]


class Trainer:
    best_eval_accuracy = 0

    def __init__(
        self,
        data_path: Path,
        cache_path: Path,
        device: torch.device,
        feature: Feature,
        database: Database,
        exp_key: str,
        num_epochs: int,
        sample_rate: int = 16000,
        random_seed: int = 42,
    ) -> None:
        data_loader = DataLoaderWithFeature(
            data_root=data_path,
            sample_rate=sample_rate,
            feature=feature,
            device=device,
            batch_size=32,
            random_seed=random_seed,
            shuffle_train=True,
            database=database,
        )
        num_classes = len(data_loader.unique_diagnosis)
        class_weights = data_loader.class_counts.sum() / data_loader.class_counts
        model = Normalized(
            input_size=feature.feature_size,
            num_classes=num_classes,
            checkpoint_path=Path(f"{cache_path}/checkpoints/{exp_key}"),
        )
        model.to(device=device)
        criterion=nn.CrossEntropyLoss(weight=class_weights)
        optimizer=torch.optim.Adam(params=model.parameters(), lr=1e-5)
        save_epochs=list(range(0, num_epochs + 1, num_epochs // 10))
        confusion_epochs=list(range(0, num_epochs + 1, 10))
        tensorboard_path = Path(
            f"{cache_path}/tboard/{exp_key}"
        )
        self.data_loader = data_loader
        tboard_enabled = True
        if tboard_enabled:
            self.tboard = TBoard(tensorboard_path=tensorboard_path)
        else:
            self.tboard = MockBoard()
        self.model = model
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.save_epochs = save_epochs
        self.confusion_epochs = confusion_epochs
        self.save_enabled = True
        self.optimizer = optimizer
        self.unique_diagnosis = self.data_loader.unique_diagnosis

    def run(self) -> None:
        for epoch in tqdm(range(self.num_epochs), desc="Epoch", position=0):
            train_loss = self.__train_loop()
            eval_loss, eval_accuracy = self.__eval_loop(epoch=epoch)
            self.tboard.add_scalars(
                "loss", {"train": train_loss, "eval": eval_loss}, global_step=epoch
            )
            self.tboard.add_scalar(
                "eval accuracy (top 1)", eval_accuracy, global_step=epoch
            )
            self.__save(epoch=epoch, eval_accuracy=eval_accuracy)

    def __train_loop(self):
        self.model.train()
        total_loss = 0
        total_batch_size = 0
        for inputs, labels in tqdm(self.data_loader.train(), desc="Training"):
            self.optimizer.zero_grad(set_to_none=True)
            predicted_labels, _ = self.model(inputs)
            loss = self.criterion(predicted_labels, labels)
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
        confusion = np.zeros((num_unique_diagnosis, num_unique_diagnosis))

        for inputs, labels in tqdm(self.data_loader.eval(), desc="Validating"):
            predicted_labels, _ = self.model(inputs)
            loss = self.criterion(predicted_labels, labels)
            predicted_labels = predicted_labels.argmax(dim=1)
            total_loss += loss.item()
            total_batch_size += 1

            self.__process_result(
                confusion_ref=confusion,
                actual=labels,
                predicted=predicted_labels,
            )

        self.__add_confusion(epoch=epoch, confusion=confusion)
        eval_accuracy = self.__weighted_accuracy(confusion=confusion)
        return total_loss / total_batch_size, eval_accuracy

    def __weighted_accuracy(self, confusion: np.ndarray) -> float:
        total_per_class = np.maximum(1, confusion.sum(axis=1))
        corrects = confusion.diagonal()
        per_class_accuracy = corrects / total_per_class
        accuracy = per_class_accuracy.mean()
        return accuracy

    def __process_result(
        self,
        confusion_ref: np.ndarray,
        actual: torch.LongTensor,
        predicted: torch.LongTensor,
    ) -> None:
        for actual_label, predicted_label in zip(
            actual.cpu().tolist(),
            predicted.cpu().tolist(),
        ):
            confusion_ref[actual_label, predicted_label] += 1

    def __add_confusion(self, epoch: int, confusion: np.ndarray) -> None:
        if epoch not in self.confusion_epochs:
            return
        total_items_per_class = np.maximum(1, confusion.sum(axis=1, keepdims=True))
        confusion_matrix = confusion / total_items_per_class
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
        ax.imshow(confusion_matrix, cmap="magma", interpolation="none", aspect="auto")
        ax.xaxis.set_ticks(list(range(len(self.unique_diagnosis))))
        ax.xaxis.set_tick_params(rotation=90)
        ax.yaxis.set_ticks(list(range(len(self.unique_diagnosis))))
        ax.set_yticklabels(self.unique_diagnosis)
        ax.set_xticklabels(self.unique_diagnosis)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        self.tboard.add_figure(tag="eval confusion", figure=fig, global_step=epoch)
        plt.close(fig=fig)

    def __save(self, epoch: int, eval_accuracy: float):
        if not self.save_enabled:
            return

        if eval_accuracy > self.best_eval_accuracy:
            self.best_eval_accuracy = eval_accuracy
            self.model.save(epoch=epoch)
        elif epoch in self.save_epochs:
            self.model.save(epoch=epoch)


class TrainerShort:
    best_eval_accuracy = 0

    def __init__(
        self,
        data_path: Path,
        cache_path: Path,
        device: torch.device,
        feature: Feature,
        database: Database,
        exp_key: str,
        max_audio_seconds: float,
        num_epochs: int,
        sample_rate: int = 16000,
        random_seed: int = 42,
    ) -> None:
        data_loader = RandomAudioDataLoaderWithFeature(
            data_root=data_path,
            sample_rate=sample_rate,
            feature=feature,
            device=device,
            batch_size=32,
            random_seed=random_seed,
            shuffle_train=True,
            database=database,
            max_audio_samples=int(sample_rate*max_audio_seconds),
        )
        num_classes = len(data_loader.unique_diagnosis)
        class_weights = data_loader.class_counts.sum() / data_loader.class_counts
        model = Normalized(
            input_size=feature.feature_size,
            num_classes=num_classes,
            checkpoint_path=Path(f"{cache_path}/checkpoints/{exp_key}"),
        )
        model.to(device=device)
        criterion=nn.CrossEntropyLoss(weight=class_weights)
        optimizer=torch.optim.Adam(params=model.parameters(), lr=1e-5)
        save_epochs=list(range(0, num_epochs + 1, num_epochs // 10))
        confusion_epochs=list(range(0, num_epochs + 1, 10))
        tensorboard_path = Path(
            f"{cache_path}/tboard/{exp_key}"
        )
        self.data_loader = data_loader
        tboard_enabled = True
        if tboard_enabled:
            self.tboard = TBoard(tensorboard_path=tensorboard_path)
        else:
            self.tboard = MockBoard()
        self.model = model
        self.criterion = criterion
        self.num_epochs = num_epochs
        self.save_epochs = save_epochs
        self.confusion_epochs = confusion_epochs
        self.save_enabled = True
        self.optimizer = optimizer
        self.unique_diagnosis = self.data_loader.unique_diagnosis

    def run(self) -> None:
        for epoch in tqdm(range(self.num_epochs), desc="Epoch", position=0):
            train_loss = self.__train_loop()
            eval_loss, eval_accuracy = self.__eval_loop(epoch=epoch)
            self.tboard.add_scalars(
                "loss", {"train": train_loss, "eval": eval_loss}, global_step=epoch
            )
            self.tboard.add_scalar(
                "eval accuracy (top 1)", eval_accuracy, global_step=epoch
            )
            self.__save(epoch=epoch, eval_accuracy=eval_accuracy)

    def __train_loop(self):
        self.model.train()
        total_loss = 0
        total_batch_size = 0
        for inputs, labels in tqdm(self.data_loader.train(), desc="Training"):
            self.optimizer.zero_grad(set_to_none=True)
            predicted_labels, _ = self.model(inputs)
            loss = self.criterion(predicted_labels, labels)
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
        confusion = np.zeros((num_unique_diagnosis, num_unique_diagnosis))

        for inputs, labels in tqdm(self.data_loader.eval(), desc="Validating"):
            predicted_labels, _ = self.model(inputs)
            loss = self.criterion(predicted_labels, labels)
            predicted_labels = predicted_labels.argmax(dim=1)
            total_loss += loss.item()
            total_batch_size += 1

            self.__process_result(
                confusion_ref=confusion,
                actual=labels,
                predicted=predicted_labels,
            )

        self.__add_confusion(epoch=epoch, confusion=confusion)
        eval_accuracy = self.__weighted_accuracy(confusion=confusion)
        return total_loss / total_batch_size, eval_accuracy

    def __weighted_accuracy(self, confusion: np.ndarray) -> float:
        total_per_class = np.maximum(1, confusion.sum(axis=1))
        corrects = confusion.diagonal()
        per_class_accuracy = corrects / total_per_class
        accuracy = per_class_accuracy.mean()
        return accuracy

    def __process_result(
        self,
        confusion_ref: np.ndarray,
        actual: torch.LongTensor,
        predicted: torch.LongTensor,
    ) -> None:
        for actual_label, predicted_label in zip(
            actual.cpu().tolist(),
            predicted.cpu().tolist(),
        ):
            confusion_ref[actual_label, predicted_label] += 1

    def __add_confusion(self, epoch: int, confusion: np.ndarray) -> None:
        if epoch not in self.confusion_epochs:
            return
        total_items_per_class = np.maximum(1, confusion.sum(axis=1, keepdims=True))
        confusion_matrix = confusion / total_items_per_class
        fig, ax = plt.subplots(1, 1, figsize=(4, 4), constrained_layout=True)
        ax.imshow(confusion_matrix, cmap="magma", interpolation="none", aspect="auto")
        ax.xaxis.set_ticks(list(range(len(self.unique_diagnosis))))
        ax.xaxis.set_tick_params(rotation=90)
        ax.yaxis.set_ticks(list(range(len(self.unique_diagnosis))))
        ax.set_yticklabels(self.unique_diagnosis)
        ax.set_xticklabels(self.unique_diagnosis)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        self.tboard.add_figure(tag="eval confusion", figure=fig, global_step=epoch)
        plt.close(fig=fig)

    def __save(self, epoch: int, eval_accuracy: float):
        if not self.save_enabled:
            return

        if eval_accuracy > self.best_eval_accuracy:
            self.best_eval_accuracy = eval_accuracy
            self.model.save(epoch=epoch)
        elif epoch in self.save_epochs:
            self.model.save(epoch=epoch)
