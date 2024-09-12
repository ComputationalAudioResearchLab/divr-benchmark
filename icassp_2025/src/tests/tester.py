import torch
import torch.optim
import numpy as np
from torch import nn
from tqdm import tqdm
from typing import Dict, List, Union
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from ..model import Feature, Normalized, SavableModule
from ..data_loader import DataLoaderWithFeature, RandomAudioDataLoaderWithFeature, Database

ConfusionData = Dict[int, Dict[int, int]]

class TesterBase:
    results_path: Path
    model: SavableModule
    data_loader: Union[DataLoaderWithFeature, RandomAudioDataLoaderWithFeature]
    unique_diagnosis: List[str]

    @torch.no_grad()
    def run(self) -> None:
        num_unique_diagnosis = len(self.unique_diagnosis)
        confusion = np.zeros((num_unique_diagnosis, num_unique_diagnosis))
        all_results = []
        for inputs, labels in tqdm(self.data_loader.test(), desc="Testing"):
            predicted_labels, _ = self.model(inputs)
            predicted_labels = predicted_labels.argmax(dim=1)
            for actual_label, predicted_label in zip(
                labels.cpu().tolist(),
                predicted_labels.cpu().tolist(),
            ):
                confusion[actual_label, predicted_label] += 1
                all_results += [(
                    self.unique_diagnosis[actual_label],
                    self.unique_diagnosis[predicted_label],
                )]
        accuracy = self.__weighted_accuracy(confusion=confusion)
        with open(f"{self.results_path}/result.log", "w") as result_file:
            result_file.write(f"Top 1 Accuracy: {accuracy}\n")
        self.__save_confusion(confusion=confusion)
        self._save_results(results=all_results)

    def __weighted_accuracy(self, confusion: np.ndarray) -> float:
        total_per_class = np.maximum(1, confusion.sum(axis=1))
        corrects = confusion.diagonal()
        per_class_accuracy = corrects / total_per_class
        accuracy = per_class_accuracy.mean()
        return accuracy
    
    def _save_results(self, results) -> None:
        df = pd.DataFrame.from_records(results, columns=["actual", "predicted"])
        df.to_csv(f"{self.results_path}/results.csv", index=False)

    def __save_confusion(self, confusion: np.ndarray) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
        ax.imshow(confusion, cmap="magma", interpolation="none", aspect="auto")
        for (j,i), label in np.ndenumerate(confusion):
            color = 'white' if label < confusion.mean() else 'black'
            ax.text(i,j,label,ha='center',va='center',color=color)
        ax.xaxis.set_ticks(list(range(len(self.unique_diagnosis))))
        ax.xaxis.set_tick_params(rotation=90)
        ax.yaxis.set_ticks(list(range(len(self.unique_diagnosis))))
        ax.set_yticklabels(self.unique_diagnosis)
        ax.set_xticklabels(self.unique_diagnosis)
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        fig.savefig(f"{self.results_path}/confusion.png")
        plt.close(fig=fig)

class Tester(TesterBase):

    def __init__(
        self, 
        data_path: Path,
        cache_path: Path,
        device: torch.device,
        feature: Feature,
        database: Database,
        exp_key: str,
        max_audio_seconds: float|None,
        best_checkpoint_epoch: int,
        sample_rate: int = 16000,
        random_seed: int = 42,
    ) -> None:
        if max_audio_seconds is None:
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
        else:
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
        results_path = Path(f"{cache_path}/results/{exp_key}")
        results_path.mkdir(parents=True, exist_ok=True)
        self.results_path = results_path
        num_classes = len(data_loader.unique_diagnosis)
        model = Normalized(
            input_size=feature.feature_size,
            num_classes=num_classes,
            checkpoint_path=Path(f"{cache_path}/checkpoints/{exp_key}"),
        )
        model.to(device=device)
        assert best_checkpoint_epoch is not None
        model.load(epoch=best_checkpoint_epoch)
        self.model = model.eval()
        self.data_loader = data_loader
        self.unique_diagnosis = data_loader.unique_diagnosis


class CrossTester(TesterBase):

    def __init__(
        self, 
        data_path: Path,
        cache_path: Path,
        device: torch.device,
        feature: Feature,
        train_database: Database,
        cross_database: Database,
        train_exp_key: str,
        test_exp_key: str,
        max_audio_seconds: float|None,
        best_checkpoint_epoch: int,
        sample_rate: int = 16000,
        random_seed: int = 42,
    ) -> None:
        if max_audio_seconds is None:
            train_data_loader = DataLoaderWithFeature(
                data_root=data_path,
                sample_rate=sample_rate,
                feature=feature,
                device=device,
                batch_size=32,
                random_seed=random_seed,
                shuffle_train=True,
                database=train_database,
            )
            # cross_data_loader = DataLoaderWithFeature(
            #     data_root=data_path,
            #     sample_rate=sample_rate,
            #     feature=feature,
            #     device=device,
            #     batch_size=32,
            #     random_seed=random_seed,
            #     shuffle_train=True,
            #     database=cross_database,
            # )
        else:
            train_data_loader = RandomAudioDataLoaderWithFeature(
                data_root=data_path,
                sample_rate=sample_rate,
                feature=feature,
                device=device,
                batch_size=32,
                random_seed=random_seed,
                shuffle_train=True,
                database=train_database,
                max_audio_samples=int(sample_rate*max_audio_seconds),
            )
            # cross_data_loader = RandomAudioDataLoaderWithFeature(
            #     data_root=data_path,
            #     sample_rate=sample_rate,
            #     feature=feature,
            #     device=device,
            #     batch_size=32,
            #     random_seed=random_seed,
            #     shuffle_train=True,
            #     database=cross_database,
            #     max_audio_samples=int(sample_rate*max_audio_seconds),
            # )
        
        cross_data_loader = DataLoaderWithFeature(
            data_root=data_path,
            sample_rate=sample_rate,
            feature=feature,
            device=device,
            batch_size=32,
            random_seed=random_seed,
            shuffle_train=True,
            database=cross_database,
        )
        results_path = Path(f"{cache_path}/results/{test_exp_key}/{train_exp_key}")
        results_path.mkdir(parents=True, exist_ok=True)
        self.results_path = results_path
        self.unique_diagnosis = train_data_loader.unique_diagnosis
        self.cross_unique_diagnosis = cross_data_loader.unique_diagnosis
        num_classes = len(self.unique_diagnosis)
        model = Normalized(
            input_size=feature.feature_size,
            num_classes=num_classes,
            checkpoint_path=Path(f"{cache_path}/checkpoints/{train_exp_key}"),
        )
        model.to(device=device)
        assert best_checkpoint_epoch is not None
        model.load(epoch=best_checkpoint_epoch)
        self.model = model.eval()
        self.data_loader = cross_data_loader

    @torch.no_grad()
    def run_without_confusion(self) -> None:
        all_results = []
        for inputs, labels in tqdm(self.data_loader.test(), desc="Testing"):
            predicted_labels, _ = self.model(inputs)
            predicted_labels = predicted_labels.argmax(dim=1)
            for actual_label, predicted_label in zip(
                labels.cpu().tolist(),
                predicted_labels.cpu().tolist(),
            ):
                all_results += [(
                    self.cross_unique_diagnosis[actual_label],
                    self.unique_diagnosis[predicted_label],
                )]
        self._save_results(results=all_results)
