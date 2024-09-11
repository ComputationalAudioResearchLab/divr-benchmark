import torch
import pandas as pd
from tqdm import tqdm
import seaborn as sns
from pathlib import Path
from .hparams import HParams
import matplotlib.pyplot as plt


class Tester:
    def __init__(self, hparams: HParams) -> None:
        checkpoint_path = Path(
            f"{hparams.base_path}/checkpoints/{hparams.checkpoint_key}"
        )
        results_path = Path(f"{hparams.base_path}/results/{hparams.results_key}")
        results_path.mkdir(parents=True, exist_ok=True)
        self.results_path = results_path
        self.data_loader = hparams.DataLoaderClass(
            benchmark_path=hparams.benchmark_path,
            benchmark_version=hparams.benchmark_version,
            stream=hparams.stream,
            task=hparams.task,
            device=hparams.device,
            batch_size=hparams.batch_size,
            random_seed=hparams.random_seed,
            shuffle_train=hparams.shuffle_train,
            cache_base_path=hparams.cache_base_path,
            cache_key=hparams.cache_key,
        )
        self.task = hparams.task
        model = hparams.ModelClass(
            input_size=self.data_loader.feature_size,
            num_classes=self.data_loader.num_unique_diagnosis,
            checkpoint_path=checkpoint_path,
        ).to(hparams.device)
        assert hparams.best_checkpoint_epoch is not None
        model.load(epoch=hparams.best_checkpoint_epoch)
        self.model = model.eval()

    @torch.no_grad()
    def run(self) -> None:
        predictions = {}
        for ids, inputs in tqdm(self.data_loader.test(), desc="Testing"):
            predicted_labels = self.model(inputs)
            predicted_labels = predicted_labels.argmax(dim=1)
            for id, label in zip(ids, predicted_labels.cpu().tolist()):
                predictions[id] = label
        results = self.data_loader.score(predictions)
        accuracy = results.top_1_accuracy
        with open(f"{self.results_path}/result.log", "w") as result_file:
            result_file.write(f"Top 1 Accuracy: {accuracy}\n")
        results.confusion.to_csv(f"{self.results_path}/confusion.csv")
        self.__save_confusion(confusion_frame=results.confusion)
        self.__save_results(results=results.results)

    def __save_results(self, results) -> None:
        df = pd.DataFrame.from_records(results, columns=["actual", "predicted"])
        df = df.map(lambda x: x.name)
        df.to_csv(f"{self.results_path}/results.csv", index=False)

    def __save_confusion(self, confusion_frame: pd.DataFrame) -> None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 8), constrained_layout=True)
        sns.heatmap(
            data=confusion_frame,
            cmap="magma",
            ax=ax,
            annot=True,
            fmt="g",
            cbar=False,
        )
        ax.set_ylabel("Actual")
        ax.set_xlabel("Predicted")
        fig.savefig(f"{self.results_path}/confusion.png")
        plt.close(fig=fig)
