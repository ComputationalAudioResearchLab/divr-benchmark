import json
import torch
import librosa
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from divr_benchmark.diagnosis import DiagnosisMap
from pyannote.audio import Model, Inference
from src.model.feature import UnispeechSAT
from src.model.output import Normalized


class TrainedModel:
    __checkpoint_root = Path("/home/workspace/icassp_2025/.cache/checkpoints")

    def __init__(
        self,
        checkpoint_key: str,
        num_classes: int,
        epoch: int,
        device: torch.device,
        sampling_rate: int,
    ) -> None:
        self.__device = device
        self.__sample_rate = sampling_rate
        self.__feature = UnispeechSAT(device=device)
        model = Normalized(
            input_size=self.__feature.feature_size,
            num_classes=num_classes,
            checkpoint_path=Path(f"{self.__checkpoint_root}/{checkpoint_key}"),
        )
        model.to(device=device)
        model.load(epoch=epoch)
        model.eval()
        self.__model = model

    @torch.no_grad()
    def __call__(self, audio_file: Path) -> np.ndarray:
        audio = self.__load_audio(audio_file_path=audio_file)
        audio_tensor = torch.tensor(
            audio,
            device=self.__device,
            dtype=torch.float32,
        )[None]
        audio_lens = torch.tensor(
            len(audio),
            device=self.__device,
            dtype=torch.long,
        )[None]
        feature, feature_lens = self.__feature((audio_tensor, audio_lens))
        first_layer = self.__model.model[0]
        per_frame_embedding = first_layer(feature)
        per_frame_embedding = self.__mask(per_frame_embedding, feature_lens)
        per_frame_embedding = per_frame_embedding.sum(dim=1) / feature_lens.unsqueeze(1)
        per_frame_embedding = per_frame_embedding[0].cpu().numpy()
        return per_frame_embedding

    def __load_audio(self, audio_file_path: Path) -> np.ndarray:
        audio, _ = librosa.load(path=audio_file_path, sr=self.__sample_rate)
        return self.__normalize(audio)

    def __normalize(self, audio: np.ndarray) -> np.ndarray:
        mean = np.mean(audio, axis=0)
        stddev = np.std(audio, axis=0)
        normalized_audio = (audio - mean) / stddev
        return normalized_audio

    def __mask(
        self, per_frame_labels: torch.Tensor, input_lens: torch.Tensor
    ) -> torch.Tensor:
        max_len = int(input_lens.max().item())
        (batch_size,) = input_lens.shape
        mask = torch.arange(max_len, device=input_lens.device).expand(
            batch_size, max_len
        ) < input_lens.unsqueeze(1)
        return per_frame_labels * mask.unsqueeze(2)


class Data:
    __sr = 16000
    __device = torch.device("cuda")
    __data_path = Path("/home/storage/divr-data/svd")
    __results_path = Path("/home/workspace/icassp_2025/similarity/data")
    __checkpoint_root = Path("/home/workspace/icassp_2025/.cache/checkpoints")

    def __init__(self) -> None:
        self.diagnosis_map = DiagnosisMap.v1()
        self.level_0_model = TrainedModel(
            num_classes=2,
            epoch=200,
            device=self.__device,
            checkpoint_key="svd_speech_0_unispeechSAT",
            sampling_rate=self.__sr,
        )
        self.level_1_model = TrainedModel(
            num_classes=4,
            epoch=200,
            device=self.__device,
            checkpoint_key="svd_speech_1_unispeechSAT",
            sampling_rate=self.__sr,
        )
        self.level_2_model = TrainedModel(
            num_classes=6,
            epoch=200,
            device=self.__device,
            checkpoint_key="svd_speech_2_unispeechSAT",
            sampling_rate=self.__sr,
        )
        self.level_3_model = TrainedModel(
            num_classes=9,
            epoch=200,
            device=self.__device,
            checkpoint_key="svd_speech_3_unispeechSAT",
            sampling_rate=self.__sr,
        )
        self.pyannote_model = Inference(
            Model.from_pretrained("pyannote/embedding"),
            window="whole",
        )

    def save_data_pickle(self):
        audio_files = list(self.__data_path.rglob("*-phrase.wav"))
        with open(f"{self.__data_path}/data.json", "r") as data_file:
            data = json.load(fp=data_file)
        records = [
            self.__read_session(data=data, audio_file=audio_file)
            for audio_file in audio_files
        ]
        df = pd.DataFrame.from_records(records)
        tqdm.pandas(desc="Generating embeddings")
        df["level_0_model_embedding"] = df["audio_file"].progress_apply(
            lambda x: self.level_0_model(x)
        )
        df["level_1_model_embedding"] = df["audio_file"].progress_apply(
            lambda x: self.level_1_model(x)
        )
        df["level_2_model_embedding"] = df["audio_file"].progress_apply(
            lambda x: self.level_2_model(x)
        )
        df["level_3_model_embedding"] = df["audio_file"].progress_apply(
            lambda x: self.level_3_model(x)
        )
        df["pyannote_embedding"] = df["audio_file"].progress_apply(
            lambda x: self.pyannote_model(x)
        )
        df.to_pickle(f"{self.__results_path}/data.pkl")

    def __read_session(self, data, audio_file):
        session_id = audio_file.parent.stem
        patient_id = audio_file.parent.parent.stem
        gender = data[patient_id]["gender"]
        session = self.__find_session(data[patient_id]["sessions"], session_id)
        classification = session["classification"]
        pathologies = session["pathologies"]
        diagnosis = pathologies if pathologies != "" else classification
        diagnosis = [
            self.diagnosis_map.get(p.strip().lower()).name for p in diagnosis.split(",")
        ]
        age = session["age"]
        return {
            "gender": gender,
            "classification": classification,
            "diagnosis": diagnosis,
            "age": age,
            "audio_file": audio_file,
        }

    def __find_session(self, sessions, session_id):
        for session in sessions:
            if session["session_id"] == session_id:
                return session
        raise ValueError("Unknown session id")
