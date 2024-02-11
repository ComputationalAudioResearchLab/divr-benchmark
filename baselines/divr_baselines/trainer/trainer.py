import torch
import numpy as np
from tqdm import tqdm
from typing import Dict
from pathlib import Path
from .tboard import TBoard
from .hparams import HParams
import matplotlib.pyplot as plt
import torch.nn.functional as F
from divr_benchmark import Benchmark

ConfusionData = Dict[int, Dict[int, int]]


class Trainer:
    best_eval_accuracy = 0

    def __init__(self, hparams: HParams) -> None:
        storage_path = Path(f"{hparams.base_path}/storage")
        checkpoint_path = Path(
            f"{hparams.base_path}/checkpoints/{hparams.experiment_key}"
        )
        tensorboard_path = Path(f"{hparams.base_path}/tboard/{hparams.experiment_key}")
        storage_path.mkdir(parents=True, exist_ok=True)
        self.benchmark = Benchmark(
            storage_path=storage_path,
            version=hparams.benchmark_version,
        )
        task = self.benchmark.task(stream=hparams.stream, task=hparams.task)
        self.data_loader = hparams.DataLoaderClass(
            task=task,
            device=hparams.device,
            batch_size=hparams.batch_size,
            random_seed=hparams.random_seed,
            shuffle_train=hparams.shuffle_train,
        )
        self.model = hparams.ModelClass(
            input_size=self.data_loader.feature_size,
            num_classes=self.data_loader.num_unique_diagnosis,
            checkpoint_path=checkpoint_path,
        ).to(hparams.device)
        self.optimizer = hparams.OptimClass(
            params=self.model.parameters(), lr=hparams.lr
        )
        self.tboard = TBoard(tensorboard_path=tensorboard_path)
        self.criterion = hparams.criterion
        self.num_epochs = hparams.num_epochs
        self.save_epochs = hparams.save_epochs
        self.confusion_epochs = hparams.confusion_epochs
        self.save_enabled = hparams.save_enabled
        self.unique_diagnosis = task.unique_diagnosis

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
            self.optimizer.zero_grad()
            predicted_labels = self.model(inputs)
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
        total_corrects = 0
        total_elements = 0
        num_unique_diagnosis = len(self.unique_diagnosis)
        confusion = np.zeros((num_unique_diagnosis, num_unique_diagnosis))

        for inputs, labels in tqdm(self.data_loader.eval(), desc="Validating"):
            predicted_labels = self.model(inputs)
            loss = F.cross_entropy(predicted_labels, labels)
            predicted_labels = predicted_labels.argmax(dim=1)
            corrects = labels == predicted_labels
            total_elements += len(corrects)
            total_corrects += corrects.count_nonzero().item()
            total_loss += loss.item()
            total_batch_size += 1
            total_batch_size += 1

            self.__process_confusion(
                epoch=epoch,
                confusion_ref=confusion,
                actual=labels,
                predicted=predicted_labels,
            )

        self.__add_confusion(epoch=epoch, confusion=confusion)
        return total_loss / total_batch_size, total_corrects / total_elements

    def __process_confusion(
        self,
        epoch: int,
        confusion_ref: np.ndarray,
        actual: torch.LongTensor,
        predicted: torch.LongTensor,
    ) -> None:
        if epoch not in self.confusion_epochs:
            return
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
        ax.imshow(confusion_matrix, cmap="magma", interpolation=None, aspect="auto")
        ax.xaxis.set_ticks(range(len(self.unique_diagnosis)))
        ax.xaxis.set_tick_params(rotation=90)
        ax.yaxis.set_ticks(range(len(self.unique_diagnosis)))
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
