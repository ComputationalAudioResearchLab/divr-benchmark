import librosa
import numpy as np
from pathlib import Path
from typing import Tuple

# import vdml_features as vf
import src.experiment.praat_features as vf

# from .config import HyperParams
from src.experiment.config import HyperParams


def load_features(inputs: Tuple[Path, HyperParams]):
    file_path, hyper_params = inputs
    sr = hyper_params.sampling_frequency
    audio = librosa.load(path=file_path, sr=sr)[0]
    end_time = librosa.get_duration(y=audio, sr=sr)
    # print(file_path, audio.shape)
    jitter = get_jitter(audio=audio, sr=sr, end_time=end_time, params=hyper_params)
    mfcc = get_mfcc(audio=audio, sr=sr, params=hyper_params)
    shimmer = get_shimmer(audio=audio, sr=sr, end_time=end_time, params=hyper_params)
    return (jitter, shimmer, mfcc)


def get_jitter(audio: np.ndarray, sr: int, end_time: float, params: HyperParams):
    if params.jitter is not None:
        return vf.jitter(
            audio=audio,
            sampling_frequency=sr,
            pitch_floor=params.jitter.pitch_floor,
            pitch_ceiling=params.jitter.pitch_ceiling,
            start_time=params.jitter.start_time,
            end_time=end_time,
            shortest_period=params.jitter.shortest_period,
            longest_period=params.jitter.longest_period,
            max_period_factor=params.jitter.max_period_factor,
        )


def get_mfcc(audio: np.ndarray, sr: int, params: HyperParams):
    if params.mfcc is not None:
        return vf.mfcc(
            audio=audio,
            sampling_frequency=sr,
            number_of_coefficients=params.mfcc.number_of_coefficients,
            window_length=params.mfcc.window_length,
            time_step=params.mfcc.time_step,
            first_filter_frequency=params.mfcc.first_filter_frequency,
            max_filter_frequency=params.mfcc.max_filter_frequency,
            distance_between_filters=params.mfcc.distance_between_filters,
        )


def get_shimmer(audio: np.ndarray, sr: int, end_time: float, params: HyperParams):
    if params.shimmer is not None:
        return vf.shimmer(
            audio=audio,
            sampling_frequency=sr,
            pitch_floor=params.shimmer.pitch_floor,
            pitch_ceiling=params.shimmer.pitch_ceiling,
            start_time=params.shimmer.start_time,
            end_time=end_time,
            shortest_period=params.shimmer.shortest_period,
            longest_period=params.shimmer.longest_period,
            max_period_factor=params.shimmer.max_period_factor,
            max_amplitude_factor=params.shimmer.max_amplitude_factor,
        )
