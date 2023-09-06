import librosa
import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple
from .collate import CollateFunc

# import vdml_features as vf
import src.experiment.praat_features as vf


class Base:
    pass

    def __call__(self, *args: Any, **kwds: Any) -> np.ndarray:
        raise NotImplementedError()


class FeaturePraatJitter(Base):
    def __init__(
        self,
        pitch_floor: float,
        pitch_ceiling: float,
        start_time: float,
        shortest_period: float,
        longest_period: float,
        max_period_factor: float,
        **kwargs,
    ) -> None:
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
        self.start_time = start_time
        self.shortest_period = shortest_period
        self.longest_period = longest_period
        self.max_period_factor = max_period_factor

    def __call__(
        self, audio: np.ndarray, end_time: float, sampling_frequency: int
    ) -> np.ndarray:
        jitter = vf.jitter(
            audio=audio,
            sampling_frequency=sampling_frequency,
            pitch_floor=self.pitch_floor,
            pitch_ceiling=self.pitch_ceiling,
            start_time=self.start_time,
            end_time=end_time,
            shortest_period=self.shortest_period,
            longest_period=self.longest_period,
            max_period_factor=self.max_period_factor,
        )
        return np.array(
            [
                jitter.local,
                jitter.local_absolute,
                jitter.rap,
                jitter.ppq5,
                jitter.ddp,
            ]
        )


class FeaturePraatShimmer(Base):
    def __init__(
        self,
        pitch_floor: float,
        pitch_ceiling: float,
        start_time: float,
        shortest_period: float,
        longest_period: float,
        max_period_factor: float,
        max_amplitude_factor: float,
        **kwargs,
    ) -> None:
        self.pitch_floor = pitch_floor
        self.pitch_ceiling = pitch_ceiling
        self.start_time = start_time
        self.shortest_period = shortest_period
        self.longest_period = longest_period
        self.max_period_factor = max_period_factor
        self.max_amplitude_factor = max_amplitude_factor

    def __call__(
        self, audio: np.ndarray, end_time: float, sampling_frequency: int
    ) -> np.ndarray:
        shimmer = vf.shimmer(
            audio=audio,
            sampling_frequency=sampling_frequency,
            pitch_floor=self.pitch_floor,
            pitch_ceiling=self.pitch_ceiling,
            start_time=self.start_time,
            end_time=end_time,
            shortest_period=self.shortest_period,
            longest_period=self.longest_period,
            max_period_factor=self.max_period_factor,
            max_amplitude_factor=self.max_amplitude_factor,
        )
        return np.array(
            [
                shimmer.local,
                shimmer.local_db,
                shimmer.apq3,
                shimmer.apq5,
                shimmer.apq11,
                shimmer.dda,
            ]
        )


class FeaturePraatMeanMfcc(Base):
    def __init__(
        self,
        number_of_coefficients: int,
        window_length: float,
        time_step: float,
        first_filter_frequency: float,
        max_filter_frequency: float,
        distance_between_filters: float,
        **kwargs,
    ) -> None:
        self.number_of_coefficients = number_of_coefficients
        self.window_length = window_length
        self.time_step = time_step
        self.first_filter_frequency = first_filter_frequency
        self.max_filter_frequency = max_filter_frequency
        self.distance_between_filters = distance_between_filters

    def __call__(
        self, audio: np.ndarray, end_time: float, sampling_frequency: int
    ) -> np.ndarray:
        return vf.mfcc(
            audio=audio,
            sampling_frequency=sampling_frequency,
            number_of_coefficients=self.number_of_coefficients,
            window_length=self.window_length,
            time_step=self.time_step,
            first_filter_frequency=self.first_filter_frequency,
            max_filter_frequency=self.max_filter_frequency,
            distance_between_filters=self.distance_between_filters,
        ).mean(axis=1)


class FeaturePraatStdMfcc(Base):
    def __init__(
        self,
        number_of_coefficients: int,
        window_length: float,
        time_step: float,
        first_filter_frequency: float,
        max_filter_frequency: float,
        distance_between_filters: float,
        **kwargs,
    ) -> None:
        self.number_of_coefficients = number_of_coefficients
        self.window_length = window_length
        self.time_step = time_step
        self.first_filter_frequency = first_filter_frequency
        self.max_filter_frequency = max_filter_frequency
        self.distance_between_filters = distance_between_filters

    def __call__(
        self, audio: np.ndarray, end_time: float, sampling_frequency: int
    ) -> np.ndarray:
        return vf.mfcc(
            audio=audio,
            sampling_frequency=sampling_frequency,
            number_of_coefficients=self.number_of_coefficients,
            window_length=self.window_length,
            time_step=self.time_step,
            first_filter_frequency=self.first_filter_frequency,
            max_filter_frequency=self.max_filter_frequency,
            distance_between_filters=self.distance_between_filters,
        ).std(axis=1)


class FeatureLibrosaMeanMfcc(Base):
    def __init__(
        self,
        number_of_coefficients: int,
        win_length: float,
        hop_length: float,
        n_fft: float,
        window: str,
        fmin: float,
        fmax: float,
        **kwargs,
    ) -> None:
        self.number_of_coefficients = number_of_coefficients
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.window = window
        self.fmin = fmin
        self.fmax = fmax

    def __call__(
        self, audio: np.ndarray, end_time: float, sampling_frequency: int
    ) -> np.ndarray:
        return librosa.feature.mfcc(
            y=audio,
            sr=sampling_frequency,
            n_mfcc=self.number_of_coefficients,
            n_fft=self.n_fft,
            win_length=self.win_length,
            hop_length=self.hop_length,
            window=self.window,
            fmin=self.fmin,
            fmax=self.fmax,
        ).mean(axis=1)


features = {
    "mean_mfcc_librosa": FeatureLibrosaMeanMfcc,
    "mean_mfcc_praat": FeaturePraatMeanMfcc,
    "std_mfcc_praat": FeaturePraatStdMfcc,
    "jitter_praat": FeaturePraatJitter,
    "shimmer_praat": FeaturePraatShimmer,
}


def load_features(inputs: Tuple[Path, Dict[str, Any], CollateFunc, Dict]) -> np.ndarray:
    file_path, input_features, collate_fn, collate_fn_args = inputs
    sr: int = input_features["sampling_frequency"]
    audio = librosa.load(path=file_path, sr=sr)[0]
    end_time = librosa.get_duration(y=audio, sr=sr)
    output_features: Dict[str, np.ndarray] = {}
    for key in features.keys():
        if key in input_features:
            output_features[key] = input_features[key](
                audio=audio, sampling_frequency=sr, end_time=end_time
            )
            if np.isnan(output_features[key]).any():
                print(file_path, audio.shape, output_features[key])
    return collate_fn(input=output_features, **collate_fn_args)


FeatureMap = Dict[str, Base]
