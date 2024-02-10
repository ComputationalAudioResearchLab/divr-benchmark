from typing import List
import librosa
import numpy as np
from pathlib import Path


class AudioLoader:
    def __init__(self, version: str, data_path: Path) -> None:
        self.__data_path = data_path
        self.__version = version

    def __call__(self, keys: List[str]) -> List[np.ndarray]:
        if self.__version == "v1":
            return list(map(self.__v1, keys))
        raise ValueError(f"invalid version {self.__version} selected in AudioLoader")

    def __v1(self, key: str) -> np.ndarray:
        audio_path = f"{self.__data_path}/{key}"
        audio, _ = librosa.load(audio_path, sr=16000)
        return self.__normalize(audio)

    def __normalize(self, audio: np.ndarray) -> np.ndarray:
        mean = np.mean(audio, axis=0)
        stddev = np.std(audio, axis=0)
        normalized_audio = (audio - mean) / stddev
        return normalized_audio
