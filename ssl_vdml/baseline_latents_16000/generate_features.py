import json
import numpy as np
import torch
import librosa
from typing import Tuple, List
from tqdm import tqdm
from pathlib import Path
from multiprocessing import Pool
import torchaudio


class FeatureModel:
    def __init__(
        self,
        features: List[str],
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        n_mfcc: int,
        device: torch.device,
    ):
        self.features = features
        melkwargs = {
            "n_fft": n_fft,
            "win_length": win_length,
            "hop_length": hop_length,
            "n_mels": n_mels,
        }
        self.MelSpect = torchaudio.transforms.MelSpectrogram(
            **melkwargs,
            sample_rate=sample_rate,
        ).to(device)
        self.MFCC = torchaudio.transforms.MFCC(
            sample_rate=sample_rate, n_mfcc=n_mfcc, melkwargs=melkwargs
        ).to(device)

    def __call__(self, audio):
        mel = self.MelSpect(audio)
        mfcc = self.MFCC(audio)
        mel_mu = mel.mean(dim=1)
        mel_std = mel.std(dim=1)
        mfcc_mu = mfcc.mean(dim=1)
        mfcc_std = mfcc.std(dim=1)
        features = {
            "mel_mu": mel_mu,
            "mel_std": mel_std,
            "mfcc_mu": mfcc_mu,
            "mfcc_std": mfcc_std,
        }
        X_feature = torch.cat([features[feature] for feature in self.features])
        return X_feature


class FeatureGenerator:
    base_models = [
        ["mel_mu"],
        ["mel_std"],
        ["mfcc_mu"],
        ["mfcc_std"],
        ["mfcc_mu", "mfcc_std"],
        ["mel_mu", "mel_std"],
        ["mel_mu", "mel_std", "mfcc_mu", "mfcc_std"],
    ]
    n_fft = 1024
    win_length = 1024
    hop_length = 1024
    n_mels = 80
    n_mfcc = 13

    def __init__(
        self, device: torch.device, output_folder: Path, sample_rate: int
    ) -> None:
        self.device = device
        self.output_folder = output_folder
        self.files_folder = Path(f"{output_folder}/files")
        self.files_folder.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate

    def load_audio(self, args):
        input_file, sample_rate = args
        return librosa.load(path=input_file, sr=sample_rate)[0]

    def load_all_audios(self, file_map: List[Tuple[Path, Path]]):
        with Pool(10) as pool:
            input_args = [(files[0], self.sample_rate) for files in file_map]
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
                        orginal_path = Path(session["files"][file_idx]["path"])
                        new_path = Path(f"{self.files_folder}/{orginal_path.stem}")
                        session["files"][file_idx]["path"] = str(new_path)
                        file_map += [(orginal_path, new_path)]
                with open(output_file_name, "w") as output_file:
                    json.dump(config, output_file, indent=2, default=vars)
        audio, lengths = self.load_all_audios(file_map)
        audio = torch.FloatTensor(audio).to(self.device)
        lengths = torch.LongTensor(lengths).to(self.device)

        for features in tqdm(self.base_models, "generating features", position=0):
            model = FeatureModel(
                features=features,
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                win_length=self.win_length,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                n_mfcc=self.n_mfcc,
                device=self.device,
            )
            feature_key = "_".join(features)
            for i, (_, output_file) in tqdm(
                enumerate(file_map), total=len(file_map), position=1, leave=False
            ):
                audio_length = lengths[i]
                all_hs = model(audio[i, :audio_length])
                feature = all_hs.clone().cpu()
                self.save_or_retry(feature, f"{output_file}.{feature_key}.pt")

    def save_or_retry(self, feature, file_path):
        try:
            torch.save(feature, file_path)
        except Exception:
            print(f"Retry saving: {file_path}")
            self.save_or_retry(feature, file_path)


if __name__ == "__main__":
    FeatureGenerator(
        device=torch.device("cuda"),
        output_folder=Path("/home/storage/data/baseline_latents/16000/features"),
        sample_rate=16000,
    ).run(
        data_files=[
            Path("/home/workspace/data/preprocessed/svd_a_n_train.json"),
            Path("/home/workspace/data/preprocessed/svd_i_n_train.json"),
            Path("/home/workspace/data/preprocessed/svd_u_n_train.json"),
            Path("/home/workspace/data/preprocessed/svd_a_n_val.json"),
            Path("/home/workspace/data/preprocessed/svd_i_n_val.json"),
            Path("/home/workspace/data/preprocessed/svd_u_n_val.json"),
            Path("/home/workspace/data/preprocessed/svd_a_n_test.json"),
            Path("/home/workspace/data/preprocessed/svd_i_n_test.json"),
            Path("/home/workspace/data/preprocessed/svd_u_n_test.json"),
            Path("/home/workspace/data/preprocessed/voiced_train.json"),
            Path("/home/workspace/data/preprocessed/voiced_val.json"),
            Path("/home/workspace/data/preprocessed/voiced_test.json"),
        ],
    )
