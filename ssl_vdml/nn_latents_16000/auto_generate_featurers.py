import json
import numpy as np
import torch
import librosa
from s3prl.nn import S3PRLUpstream
from typing import Tuple, List
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool

class FeatureGenerator:
    ssl_models = ["data2vec_base_960"]

    def __init__(
        self, device: torch.device, output_folder: Path, sampling_rate: int, layer_idx: int = 13
    ) -> None:
        self.device = device
        self.output_folder = output_folder
        self.files_folder = Path(f"{output_folder}/files")
        self.files_folder.mkdir(parents=True, exist_ok=True)
        self.sampling_rate = sampling_rate
        self.layer_idx = layer_idx

    def load_audio(self, args):
        input_file, sampling_rate = args
        return librosa.load(path=input_file, sr=sampling_rate)[0]

    def load_all_audios(self, file_map: List[Tuple[Path, Path]]):
        with Pool(10) as pool:
            input_args = [(files[0], self.sampling_rate) for files in file_map]
            audios = list(
                tqdm(
                    pool.imap(self.load_audio, input_args),
                    total=len(file_map),
                    desc="loading files",
                )
            )
            lengths = list(map(len, audios))
            max_length = max(lengths)
            total_audios = len(lengths)
            X = np.zeros((total_audios, max_length))
            for i, (audio, length) in tqdm(
                enumerate(zip(audios, lengths)),
                desc="setting audios",
                total=total_audios,
            ):
                X[i, :length] = audio
            return X, np.array(lengths)

    def run(self, data_files: List[Path]):
        file_map: List[Tuple[Path, Path]] = []
        for data_file in data_files:
            with open(data_file, "r") as config_data:
                config = json.load(config_data)
                output_file_name = f"{self.output_folder}/{data_file.name}"
                for session in config:
                    for file_idx in range(len(session["files"])):
                        original_path = Path(session["files"][file_idx]["path"])
                        new_path = Path(f"{self.files_folder}/{original_path.stem}")
                        session["files"][file_idx]["path"] = str(new_path)
                        file_map += [(original_path, new_path)]
                with open(output_file_name, "w") as output_file:
                    json.dump(config, output_file, indent=2, default=vars)
        audio, lengths = self.load_all_audios(file_map)
        audio = torch.FloatTensor(audio).to(self.device)
        lengths = torch.LongTensor(lengths).to(self.device)

        for feature_name in tqdm(self.ssl_models, "generating features", position=0):
            model = S3PRLUpstream(name=feature_name).to(self.device)
            model.eval()
            with torch.no_grad():
                for i, (_, output_file) in tqdm(
                    enumerate(file_map), total=len(file_map), position=1, leave=False
                ):
                    all_hs, _ = model(audio[i][None], lengths[i][None])
                    feature = all_hs[self.layer_idx][0].clone().cpu()
                    self.save_or_retry(feature, f"{output_file}.{feature_name}.pt")

    def save_or_retry(self, feature, file_path):
        try:
            torch.save(feature, file_path)
        except Exception:
            print(f"Retry saving: {file_path}")
            self.save_or_retry(feature, file_path)


def get_max_layers(model_name: str, device: torch.device) -> int:
    model = S3PRLUpstream(name=model_name).to(device)
    dummy_input = torch.rand(1, 16000).to(device)  
    dummy_length = torch.LongTensor([16000]).to(device)
    
    with torch.no_grad():
        all_hs, _ = model(dummy_input[None], dummy_length[None])
    
    return len(all_hs)


if __name__ == "__main__":
    # Specify the layer index
    layer_idx = 12
    
    # Initialize the FeatureGenerator with the specified layer index
    feature_generator = FeatureGenerator(
        device=torch.device("cuda"),
        output_folder=Path("/home/workspace/data/nn_latents[12][0]_a_n/16000/features"),
        sampling_rate=16000,
        layer_idx=layer_idx,
    )
    
    # Example usage of get_max_layers
    model_name = "data2vec_base_960"
    max_layers = get_max_layers(model_name, torch.device("cuda"))
    print(f"The model {model_name} has a maximum of {max_layers} layers.")
    
    # Run the feature extraction with specified files
    feature_generator.run(
        data_files=[
            Path("/home/workspace/data/preprocessed/svd_a_n_train.json"),
            Path("/home/workspace/data/preprocessed/svd_a_n_val.json"),
            Path("/home/workspace/data/preprocessed/svd_a_n_test.json"),
        ],
    )
