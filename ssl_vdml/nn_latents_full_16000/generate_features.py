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
    ssl_models = [
        # "apc_960hr",
        # "vq_apc_960hr",
        # "npc_960hr",
        # "modified_cpc",
        # "decoar_layers",
        # "decoar2",
        # "wav2vec_large",
        # "vq_wav2vec_gumbel",
        # "vq_wav2vec_kmeans",
        # "vq_wav2vec_kmeans_roberta",
        # "wav2vec2_large_lv60_cv_swbd_fsh",
        # "xlsr_53",
        # "xls_r_2b",
        # "hubert_base",
        # "distilhubert_base",
        # "hubert_base_robust_mgr",
        # "unispeech_sat_large",
        # "wavlm_large",
        "data2vec_base_960",
        # "vggish",
    ]

    def __init__(
        self, device: torch.device, output_folder: Path, sampling_rate: int
    ) -> None:
        self.device = device
        self.output_folder = output_folder
        self.files_folder = Path(f"{output_folder}/files")
        self.files_folder.mkdir(parents=True, exist_ok=True)
        self.sampling_rate = sampling_rate

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
                        orginal_path = Path(session["files"][file_idx]["path"])
                        new_path = Path(f"{self.files_folder}/{orginal_path.stem}")
                        session["files"][file_idx]["path"] = str(new_path)
                        file_map += [(orginal_path, new_path)]
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
                    all_hs, all_hs_len = model(audio[i][None], lengths[i][None])
                    feature = torch.cat(all_hs, dim=2)[0].clone().cpu()
                    self.save_or_retry(feature, f"{output_file}.{feature_name}.pt")

    def save_or_retry(self, feature, file_path):
        try:
            torch.save(feature, file_path)
        except Exception:
            print(f"Retry saving: {file_path}")
            self.save_or_retry(feature, file_path)


if __name__ == "__main__":
    FeatureGenerator(
        device=torch.device("cuda"),
        output_folder=Path("/home/workspace/data/nn_latents_full/16000/features"),
        sampling_rate=16000,
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
