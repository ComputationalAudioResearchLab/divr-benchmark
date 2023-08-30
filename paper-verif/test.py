import sys
import yaml
from pathlib import Path

# Add the parent directory to the system path
sys.path.append(str(Path(__file__).parent.parent))
from experiment.features import load_features
from experiment.config import HyperParams, FeatureMFCC
from pathlib import Path


# 1. Load the experiment configuration
with open("experiment1.yml", "r") as f:
    config_data = yaml.safe_load(f)

# 2. Set up the hyperparameters based on the loaded configuration
hyper_params = HyperParams(
    sampling_frequency=config_data["hyper_params"]["common"]["sampling_frequency"],
    mfcc=FeatureMFCC(
        number_of_coefficients=config_data["hyper_params"]["mfcc"]["number_of_coefficients"],
        sampling_frequency=config_data["hyper_params"]["common"]["sampling_frequency"],
        window_length=config_data["hyper_params"]["mfcc"]["window_length"],
        time_step=config_data["hyper_params"]["mfcc"]["time_step"],
        first_filter_frequency=config_data["hyper_params"]["mfcc"]["first_filter_frequency"],
        max_filter_frequency=config_data["hyper_params"]["mfcc"]["max_filter_frequency"],
        distance_between_filters=config_data["hyper_params"]["mfcc"]["distance_between_filters"]
    ),
    jitter=None,  # Not using jitter for MFCC extraction
    shimmer=None  # Not using shimmer for MFCC extraction
)

def extract_mfcc_new(audio_path: Path, hyper_params: HyperParams):
    _, _, mfccs = load_features((audio_path, hyper_params))
    return mfccs

# 3. Extract MFCCs for a given audio path
audio_path = Path("/path/to/your/audio/file.wav")  # Replace this with your actual audio file path
mfcc_values = extract_mfcc_new(audio_path, hyper_params)

# 4. Print the extracted MFCCs
print(mfcc_values)
