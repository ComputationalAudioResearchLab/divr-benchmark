from __future__ import annotations
import parselmouth
import numpy as np
from dataclasses import dataclass
from parselmouth.praat import call


@dataclass
class Jitter:
    local: float
    local_absolute: float
    rap: float
    ppq5: float
    ddp: float


@dataclass
class Shimmer:
    local: float
    local_db: float
    apq3: float
    apq5: float
    apq11: float
    dda: float


def jitter(
    audio: np.ndarray,
    sampling_frequency: float,
    pitch_floor: float,
    pitch_ceiling: float,
    start_time: float,
    end_time: float,
    shortest_period: float,
    longest_period: float,
    max_period_factor: float,
) -> Jitter:
    sound = parselmouth.Sound(values=audio, sampling_frequency=sampling_frequency)
    point_process = call(
        sound, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling
    )

    def call_jitter_fun(key) -> float:
        return call(
            point_process,
            key,
            start_time,
            end_time,
            shortest_period,
            longest_period,
            max_period_factor,
        )

    return Jitter(
        local=call_jitter_fun("Get jitter (local)"),
        local_absolute=call_jitter_fun("Get jitter (local, absolute)"),
        rap=call_jitter_fun("Get jitter (rap)"),
        ppq5=call_jitter_fun("Get jitter (ppq5)"),
        ddp=call_jitter_fun("Get jitter (ddp)"),
    )


def shimmer(
    audio: np.ndarray,
    sampling_frequency: float,
    pitch_floor: float,
    pitch_ceiling: float,
    start_time: float,
    end_time: float,
    shortest_period: float,
    longest_period: float,
    max_period_factor: float,
    max_amplitude_factor: float,
) -> Shimmer:
    sound = parselmouth.Sound(values=audio, sampling_frequency=sampling_frequency)
    point_process = call(
        sound, "To PointProcess (periodic, cc)", pitch_floor, pitch_ceiling
    )

    def call_shimmer_fun(key) -> float:
        return call(
            [point_process, sound],
            key,
            start_time,
            end_time,
            shortest_period,
            longest_period,
            max_period_factor,
            max_amplitude_factor,
        )

    return Shimmer(
        local=call_shimmer_fun("Get shimmer (local)"),
        local_db=call_shimmer_fun("Get shimmer (local_dB)"),
        apq3=call_shimmer_fun("Get shimmer (apq3)"),
        apq5=call_shimmer_fun("Get shimmer (apq5)"),
        apq11=call_shimmer_fun("Get shimmer (apq11)"),
        dda=call_shimmer_fun("Get shimmer (dda)"),
    )


def melspectrogram(
    audio: np.ndarray,
    sampling_frequency: float,
    window_length: float,
    time_step: float,
    first_filter_frequency: float,
    max_filter_frequency: float,
    distance_between_filters: float,
) -> np.ndarray:
    sound = parselmouth.Sound(values=audio, sampling_frequency=sampling_frequency)
    return sound.to_mel(
        window_length,
        time_step,
        first_filter_frequency,
        distance_between_filters,
        max_filter_frequency,
    )


def spectrum(
    audio: np.ndarray,
    sampling_frequency: float,
) -> parselmouth.Spectrum:
    sound = parselmouth.Sound(values=audio, sampling_frequency=sampling_frequency)
    return sound.to_spectrum()


def mfcc(
    audio: np.ndarray,
    sampling_frequency: float,
    number_of_coefficients: int,
    window_length: float,
    time_step: float,
    first_filter_frequency: float,
    max_filter_frequency: float,
    distance_between_filters: float,
) -> np.ndarray:
    sound = parselmouth.Sound(values=audio, sampling_frequency=sampling_frequency)
    return sound.to_mfcc(
        number_of_coefficients,
        window_length,
        time_step,
        first_filter_frequency,
        distance_between_filters,
        max_filter_frequency,
    ).to_array()
