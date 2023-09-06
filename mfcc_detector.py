import json
import librosa
from pathlib import Path
from tqdm import tqdm
import numpy as np
from typing import List, Dict
from src.diagnosis import Diagnosis, DiagnosisMap
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import hashlib
import matplotlib.pyplot as plt
import torch
import shutil
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard.writer import SummaryWriter


class DiarizedDetector(nn.Module):
    def __init__(self, input_shape, classes, device) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_shape, 1024),
            nn.ReLU(),
            nn.Linear(1024, classes),
        )
        self.to(device)

    def forward(self, X):
        return self.model(X)


class DiarizedDetectorTrainer:
    num_mfcc = 6

    def __init__(
        self, device: torch.device, diagnosis_level: int, output_folder: Path
    ) -> None:
        self.diagnosis_level = diagnosis_level
        self.diagnosis_map = DiagnosisMap()
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)
        self.device = device

    def run(
        self, epochs: int, train_files, val_files, allowed_labels: List[str] | None
    ):
        config = {
            "diagnosis_level": self.diagnosis_level,
            "allowed_labels": allowed_labels,
            "train_files": train_files,
            "val_files": val_files,
        }
        config_str = json.dumps(config)
        key = hashlib.sha1(config_str.encode("utf-8")).hexdigest()
        tb_dir = f"{self.output_folder}/{key}/tensorboard"
        shutil.rmtree(tb_dir, ignore_errors=True)
        self.tb = SummaryWriter(log_dir=tb_dir)
        self.classMap, trainX, trainY = self.generate_support_vectors(
            train_files=train_files, allowed_labels=allowed_labels
        )
        print(trainX.shape, len(trainY))
        diagnosis, embeddings = self.load_data(val_files, allowed_labels=allowed_labels)
        self.model = DiarizedDetector(
            input_shape=self.num_mfcc, classes=len(self.classMap), device=self.device
        )
        self.optimizer = torch.optim.Adam(params=self.model.parameters(), lr=1e-5)

        trainX = torch.tensor(trainX, device=self.device)
        trainY = torch.tensor(trainY, device=self.device)

        valX = torch.tensor(embeddings, device=self.device)
        valY = torch.tensor(
            [self.classMap[diag] for diag in diagnosis], device=self.device
        )

        pbar = tqdm(range(epochs), position=0)
        fig_epochs = list(range(0, epochs, 100))
        for epoch in pbar:
            trainX, trainY = self.shuffle_data(trainX, trainY)
            train_loss, train_predY = self.train(trainX, trainY)
            val_loss, val_predY = self.val(valX, valY)
            details = {"train": train_loss, "val": val_loss}
            self.tb.add_scalars("loss", details, epoch)
            if epoch in fig_epochs:
                self.add_confusion_image("confusion_val", epoch, valY, val_predY)
                self.add_confusion_image("confusion_train", epoch, trainY, train_predY)
            pbar.set_postfix(details)

    def add_confusion_image(self, key, epoch, tgt, pred):
        confusion = confusion_matrix(tgt.cpu(), pred.cpu())
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion, display_labels=list(self.classMap.keys())
        )
        fig, ax = plt.subplots(1, 1, figsize=(6, 6), constrained_layout=True)
        disp.plot(ax=ax)
        self.tb.add_figure(key, fig, epoch)
        plt.close()

    def train(self, trainX, trainY):
        self.model.train()
        self.optimizer.zero_grad()
        predY = self.model(trainX)
        loss = F.cross_entropy(predY, trainY, reduction="mean")
        loss.backward()
        self.optimizer.step()
        return loss.item(), predY.argmax(dim=1).detach()

    def shuffle_data(self, X, Y):
        idx = np.arange(X.shape[0])
        np.random.shuffle(idx)
        X = X[idx, :]
        Y = Y[idx]
        return X, Y

    @torch.no_grad()
    def val(self, valX, valY):
        self.model.eval()
        predY = self.model(valX)
        loss = F.cross_entropy(predY, valY, reduction="mean")
        return loss.item(), predY.argmax(dim=1)

        # scores = embeddings @ support_vectors.T
        # predictions = [classes[score] for score in scores.argmax(axis=1)]
        # confusion = confusion_matrix(diagnosis, predictions)
        # np.savetxt(
        #     f"{self.output_folder}/{key}.csv",
        #     confusion,
        #     header=config_str,
        # )
        # display_labels = None
        # if len(classes) == confusion.shape[0]:
        #     display_labels = classes
        # disp = ConfusionMatrixDisplay(
        #     confusion_matrix=confusion, display_labels=display_labels
        # )
        # fig, ax = plt.subplots(1, 1, figsize=(20, 20), constrained_layout=True)
        # disp.plot(ax=ax)
        # fig.savefig(f"{self.output_folder}/{key}.png", bbox_inches="tight")
        # plt.close()

    def generate_support_vectors(self, train_files, allowed_labels: List[str] | None):
        diagnosis, embeddings = self.load_data(
            data_files=train_files, allowed_labels=allowed_labels
        )
        all_classes = np.unique(diagnosis).tolist()
        support_vectors = []
        classes = []
        class_map = dict(zip(all_classes, range(len(all_classes))))
        for class_name in all_classes:
            mask = diagnosis == class_name
            new_vectors = embeddings[mask]
            support_vectors += [new_vectors]
            classes += [class_map[class_name]] * len(new_vectors)
        return (
            class_map,
            np.concatenate(support_vectors, axis=0),
            classes,
        )

    def load_data(self, data_files, allowed_labels: List[str] | None):
        dataset = []
        for data_file in data_files:
            with open(data_file, "r") as config_data:
                config = json.load(config_data)
                for i, session in tqdm(enumerate(config), total=len(config)):
                    dataset += [
                        self.load_session(
                            session=session, allowed_labels=allowed_labels
                        )
                    ]
        dataset = list(filter(None, dataset))
        diagnosis, audio_files = list(zip(*dataset))
        diagnosis = np.array(diagnosis)
        embeddings = self.generate_embeddings(audio_files)
        return diagnosis, embeddings

    def load_session(self, session: Dict, allowed_labels: List[str] | None):
        audio_file = session["files"][0]["path"]
        diagnosis_list = list(map(Diagnosis.from_json, session["diagnosis"]))
        if allowed_labels is None:
            diagnosis = self.diagnosis_map.from_int(
                self.diagnosis_map.most_severe(
                    diagnosis_list,
                    level=self.diagnosis_level,
                )
            ).name
            return (diagnosis, audio_file)
        all_diagnosis = []
        for diag in diagnosis_list:
            all_diagnosis += diag.to_list()
        for label in allowed_labels:
            if label in all_diagnosis:
                # sometimes people have multiple diagnosis,
                # we want to make sure we only keep the one we are interested in
                label = self.diagnosis_map.get(label).at_level(self.diagnosis_level)
                return (label, audio_file)
        return None

    def generate_embeddings(self, wav_paths: List[str]) -> np.ndarray:
        embeds = []
        for wav_fpath in tqdm(wav_paths, "Preprocessing wavs"):
            audio, sr = librosa.load(wav_fpath, sr=16000)
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=sr,
                n_mfcc=self.num_mfcc,
                hop_length=256,
                n_fft=1024,
                win_length=1024,
            )
            mean_mfcc = mfcc.mean(axis=1)
            embeds += [mean_mfcc]
        return np.stack(embeds)


if __name__ == "__main__":
    measure = DiarizedDetectorTrainer(
        diagnosis_level=0,
        output_folder=Path("/home/workspace/output_mfcc_detector"),
        device=torch.device("cuda"),
    )
    measure.run(
        epochs=10000000,
        train_files=[
            "/home/workspace/data/preprocessed/svd_a_n_train.json",
            "/home/workspace/data/preprocessed/svd_i_n_train.json",
            "/home/workspace/data/preprocessed/svd_u_n_train.json",
        ],
        val_files=[
            "/home/workspace/data/preprocessed/svd_a_n_val.json",
            "/home/workspace/data/preprocessed/svd_i_n_val.json",
            "/home/workspace/data/preprocessed/svd_u_n_val.json",
        ],
        allowed_labels=None,
        # allowed_labels=["dysphonie", "healthy"],
        # allowed_labels=["cyste", "healthy", "stimmlippenpolyp", "bulbärparalyse"],
    )
