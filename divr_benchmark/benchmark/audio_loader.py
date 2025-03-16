from typing import List
import librosa
import numpy as np
from pathlib import Path


class AudioLoader:
    def __init__(self, version: str, data_path: Path, sample_rate: int) -> None:
        self.__data_path = data_path
        self.__version = version
        self.sample_rate = sample_rate

    def __call__(self, keys: List[str]) -> List[np.ndarray]:
        if self.__version == "v1":
            return list(map(self.__v1, keys))
        raise ValueError(f"invalid version {self.__version} selected in AudioLoader")

    def __v1(self, key: str) -> np.ndarray:
        audio_path = f"{self.__data_path}/{key}"
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)
        if len(audio) < 1:
            raise ValueError(f"0 length audio file: {audio_path}")
        return self.__normalize(audio)

    def __normalize(self, audio: np.ndarray) -> np.ndarray:
        mean = np.mean(audio, axis=0)
        stddev = np.std(audio, axis=0)
        normalized_audio = (audio - mean) / stddev
        return normalized_audio
