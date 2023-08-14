import librosa
from pathlib import Path
from typing import List
import numpy as np
from tqdm import tqdm
from src.preprocess.processed import ProcessedSession
from .config import ExperimentConfig
import vdml_features as vf


class Data:
    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        train_data = self.load_data(
            key="train",
            files=self.collect_files(config.train_data),
        )
        val_data = self.load_data(
            key="val",
            files=self.collect_files(config.val_data),
        )
        print(val_data)

    def collect_files(self, data: List[ProcessedSession]) -> List[Path]:
        files = []
        for datum in data:
            files += [file.path for file in datum.files]
        return files

    def load_data(self, key: str, files: List[Path]):
        return [
            self.load_features(file_path)
            for file_path in tqdm(files, desc=f"Loading {key}")
        ]

    def load_features(self, file_path):
        sr = self.config.hyper_params.sampling_frequency
        audio = librosa.load(path=file_path, sr=sr)[0]
        end_time = librosa.get_duration(y=audio, sr=sr)
        jitter = self.get_jitter(audio=audio, sr=sr, end_time=end_time)
        mfcc = self.get_mfcc(audio=audio, sr=sr)
        shimmer = self.get_shimmer(audio=audio, sr=sr, end_time=end_time)
        return (jitter, shimmer, mfcc)

    def get_jitter(self, audio: np.ndarray, sr: int, end_time: float):
        if self.config.hyper_params.jitter is not None:
            return vf.jitter(
                audio=audio,
                sampling_frequency=sr,
                pitch_floor=self.config.hyper_params.jitter.pitch_floor,
                pitch_ceiling=self.config.hyper_params.jitter.pitch_ceiling,
                start_time=self.config.hyper_params.jitter.start_time,
                end_time=end_time,
                shortest_period=self.config.hyper_params.jitter.shortest_period,
                longest_period=self.config.hyper_params.jitter.longest_period,
                max_period_factor=self.config.hyper_params.jitter.max_period_factor,
            )

    def get_mfcc(self, audio: np.ndarray, sr: int):
        if self.config.hyper_params.mfcc is not None:
            return vf.mfcc(
                audio=audio,
                sampling_frequency=sr,
                number_of_coefficients=self.config.hyper_params.mfcc.number_of_coefficients,
                window_length=self.config.hyper_params.mfcc.window_length,
                time_step=self.config.hyper_params.mfcc.time_step,
                first_filter_frequency=self.config.hyper_params.mfcc.first_filter_frequency,
                max_filter_frequency=self.config.hyper_params.mfcc.max_filter_frequency,
                distance_between_filters=self.config.hyper_params.mfcc.distance_between_filters,
            )

    def get_shimmer(self, audio: np.ndarray, sr: int, end_time: float):
        if self.config.hyper_params.shimmer is not None:
            return vf.shimmer(
                audio=audio,
                sampling_frequency=sr,
                pitch_floor=self.config.hyper_params.shimmer.pitch_floor,
                pitch_ceiling=self.config.hyper_params.shimmer.pitch_ceiling,
                start_time=self.config.hyper_params.shimmer.start_time,
                end_time=end_time,
                shortest_period=self.config.hyper_params.shimmer.shortest_period,
                longest_period=self.config.hyper_params.shimmer.longest_period,
                max_period_factor=self.config.hyper_params.shimmer.max_period_factor,
                max_amplitude_factor=self.config.hyper_params.shimmer.max_amplitude_factor,
            )
