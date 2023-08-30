# include these 2 lines to mitigate no module name 'src' error
import sys

sys.path.append("/home/workspace")

import numpy as np
from features import *
from pathlib import Path


audio_file_path = "/home/databases/svd/healthy/male/4/4/4-a_n.wav"

hyper_params = {
    "common": {
        "sampling_frequency": 16000,
        "pitch_floor": 75.0,
        "pitch_ceiling": 500.0,
        "start_time": 0.0,
        "shortest_period": 0.0001,
        "longest_period": 0.02,
        "max_period_factor": 1.3,
    },
    "mfcc": {
        "number_of_coefficients": 12,
        "window_length": 0.015,
        "time_step": 0.005,
        "first_filter_frequency": 100.0,
        "max_filter_frequency": 8000,
        "distance_between_filters": 100.0,
    },
    "jitter": {},
    "shimmer": {"max_amplitude_factor": 1.6},
}

# mfcc_features = # Load the features
jitter, shimmer, mfcc = load_features((audio_file_path, hyper_params))

# Print the MFCC features
print("MFCC features:")
print(mfcc)

# Optionally print jitter and shimmer if required
print("Jitter:", jitter)
print("Shimmer:", shimmer)
