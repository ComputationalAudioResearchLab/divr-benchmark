import torch
from tqdm import tqdm
from pathlib import Path
from divr_feature_toolkit import FeatureFactory

ignore_files = [
    "1405/713/713-iau.wav",  # invalid file
    "1405/713/713-i_n.wav",  # invalid file
]


def valid_files(file_name: Path):
    for ignore_file in ignore_files:
        if ignore_file in str(file_name):
            return False
    return True


if __name__ == "__main__":
    # audio_files = list(
    #     Path("/home/storage/PRJ-VDML/databases/svd2/pathological/male/1303/106/").glob("*.wav")
    # )
    audio_files = list(
        filter(valid_files, Path("/home/storage/PRJ-VDML/databases/svd2").rglob("*.wav"))
    )
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factory = FeatureFactory(
        audio_files=audio_files,
        sampling_rate=16000,
        device=torch_device,
    )
    print(factory.audio_data)
    print(factory.audio_lengths)
    for i, data in enumerate(tqdm(factory.apc_960hr(batch_size=2))):
        # print(i, len(data))
        pass
