import json
from resemblyzer import preprocess_wav, VoiceEncoder
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


class DiarizedDetector:
    def __init__(self, diagnosis_level: int, output_folder: Path) -> None:
        self.diagnosis_level = diagnosis_level
        self.diagnosis_map = DiagnosisMap()
        self.encoder = VoiceEncoder(device="cuda")
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def run(self, train_files, val_files, allowed_labels: List[str] | None):
        config = {
            "diagnosis_level": self.diagnosis_level,
            "allowed_labels": allowed_labels,
            "train_files": train_files,
            "val_files": val_files,
        }
        config_str = json.dumps(config)
        key = hashlib.sha1(config_str.encode("utf-8")).hexdigest()
        classes, support_vectors = self.generate_support_vectors(
            train_files=train_files, allowed_labels=allowed_labels
        )
        diagnosis, embeddings = self.load_data(val_files, allowed_labels=allowed_labels)
        scores = embeddings @ support_vectors.T
        predictions = [classes[score] for score in scores.argmax(axis=1)]
        confusion = confusion_matrix(diagnosis, predictions)
        np.savetxt(
            f"{self.output_folder}/{key}.csv",
            confusion,
            header=config_str,
        )
        display_labels = None
        if len(classes) == confusion.shape[0]:
            display_labels = classes
        disp = ConfusionMatrixDisplay(
            confusion_matrix=confusion, display_labels=display_labels
        )
        fig, ax = plt.subplots(1, 1, figsize=(20, 20), constrained_layout=True)
        disp.plot(ax=ax)
        fig.savefig(f"{self.output_folder}/{key}.png", bbox_inches="tight")
        plt.close()

    def generate_support_vectors(self, train_files, allowed_labels: List[str] | None):
        diagnosis, embeddings = self.load_data(
            data_files=train_files, allowed_labels=allowed_labels
        )
        all_classes = np.unique(diagnosis).tolist()
        support_vectors = []
        for class_name in all_classes:
            mask = diagnosis == class_name
            support_vectors += [embeddings[mask].mean(axis=0)]
        return all_classes, np.stack(support_vectors)

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
            embeds += [self.encoder.embed_utterance(preprocess_wav(wav_fpath))]
        return np.stack(embeds)


if __name__ == "__main__":
    measure = DiarizedDetector(
        diagnosis_level=0,
        output_folder=Path("/home/workspace/output_diarized_detection"),
    )
    measure.run(
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
