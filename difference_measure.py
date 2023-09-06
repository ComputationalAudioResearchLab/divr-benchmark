import json
import librosa
import itertools
import numpy as np
from tqdm import tqdm
from typing import Dict
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats
from multiprocessing import Pool
from src.diagnosis import Diagnosis, DiagnosisMap


def process_combiation_batch(data):
    x1, x2 = data
    similarity = similarity_measure(X1=x1["audio"], X2=x2["audio"])
    return (x1, x2, similarity)


class DifferenceMeasure:
    def __init__(
        self, sampling_rate: int, diagnosis_level: int, output_folder: Path
    ) -> None:
        self.sampling_rate = sampling_rate
        self.diagnosis_level = diagnosis_level
        self.diagnosis_map = DiagnosisMap()
        self.output_folder = output_folder
        self.output_folder.mkdir(parents=True, exist_ok=True)

    def run(self, config_file: str):
        dataset = []
        with open(config_file, "r") as config_data:
            config = json.load(config_data)
            for i, session in tqdm(enumerate(config), total=len(config)):
                dataset += [self.load_session(session=session, idx=i)]
        dataset = sorted(dataset, key=lambda x: x["diagnosis"])
        total_sessions = len(dataset)
        combinations = list(itertools.combinations_with_replacement(dataset, r=2))
        similarity_matrix = np.zeros((total_sessions, total_sessions))
        x1_labels = [""] * total_sessions
        x2_labels = [""] * total_sessions
        for data in tqdm(combinations):
            x1, x2 = data
            sim = self.similarity_measure(X1=x1["audio"], X2=x2["audio"])
            x1_idx = x1["idx"]
            x2_idx = x2["idx"]
            x1_labels[x1_idx] = x1["diagnosis"]
            x2_labels[x2_idx] = x2["diagnosis"]
            similarity_matrix[x1_idx, x2_idx] = sim
            similarity_matrix[x2_idx, x1_idx] = sim
        np.savetxt(f"{self.output_folder}/similarity.csv", similarity_matrix)
        plt.imshow(similarity_matrix, cmap="magma", interpolation=None, aspect="auto")
        plt.savefig(f"{self.output_folder}/similarity.png", bbox_inches="tight")
        plt.close()

    def similarity_measure(self, X1: np.ndarray, X2: np.ndarray) -> float:
        x1_len = X1.shape[0]
        x2_len = X2.shape[0]
        max_len = max(x1_len, x2_len)
        X1 = np.pad(X1, (0, max_len - x1_len))
        X2 = np.pad(X2, (0, max_len - x2_len))
        r, p = scipy.stats.pearsonr(X1, X2)
        return r

    def load_session(self, session: Dict, idx: int):
        audio_file = session["files"][0]["path"]
        audio, _ = librosa.load(audio_file, sr=self.sampling_rate)
        diagnosis = self.diagnosis_map.from_int(
            self.diagnosis_map.most_severe(
                list(map(Diagnosis.from_json, session["diagnosis"])),
                level=self.diagnosis_level,
            )
        ).name
        return {"diagnosis": diagnosis, "audio": audio, "idx": idx}


if __name__ == "__main__":
    measure = DifferenceMeasure(
        sampling_rate=16000,
        diagnosis_level=0,
        output_folder=Path("/home/workspace/output_difference_measures"),
    )
    measure.run("/home/workspace/data/preprocessed/svd_a_n_train.json")
