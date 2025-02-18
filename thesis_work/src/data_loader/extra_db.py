import librosa
import numpy as np
from tqdm import tqdm
from pathlib import Path


class ExtraDB:
    name: str
    audio_file_filter: str
    audios: list[np.ndarray]

    def __init__(
        self,
        research_data_path: Path,
        max_audio_length: int,
        sample_rate: int,
        random_seed: int,
        extra_audios_bucket_size: int,
    ) -> None:
        self.data_path = Path(f"{research_data_path}/{self.name}")
        self.max_audio_samples = sample_rate * max_audio_length
        self.sample_rate = sample_rate
        self.audios = self.load_audios(
            extra_audios_bucket_size=extra_audios_bucket_size
        )
        self.total_data = len(self.audios)
        self.rng = np.random.RandomState(seed=random_seed)
        if self.total_data < 1:
            raise ValueError(
                f"No data loaded. Extra db config {{name={self.name}, file_filter={self.audio_file_filter}}}"
            )

    def get_audios(self, num_audios: int) -> list[list[np.ndarray]]:
        keys = self.rng.randint(low=0, high=self.total_data, size=num_audios)
        return [[self.audios[k]] for k in keys]

    def load_audios(self, extra_audios_bucket_size: int):
        wav_paths = sorted(list(self.data_path.rglob(self.audio_file_filter)))
        return [
            self.load_audio(wav_path)
            for wav_path in tqdm(
                wav_paths[:extra_audios_bucket_size], desc=f"Loading {self.name}"
            )
        ]

    def load_audio(self, wav_file: Path) -> np.ndarray:
        data, sr = librosa.load(path=wav_file, sr=self.sample_rate, mono=True)
        return data[: self.max_audio_samples]


class CommonVoiceDeltaSegment20(ExtraDB):
    name = "common_voice_delta_segment_20.0"
    audio_file_filter = "cv-corpus-20.0-delta-2024-12-06/en/clips/*.mp3"


class EmoDB(ExtraDB):
    name = "emodb"
    audio_file_filter = "wav/*.wav"


class LibrispeechDevClean(ExtraDB):
    name = "librispeech_dev_clean"
    audio_file_filter = "LibriSpeech/dev-clean/**/*.flac"
