from __future__ import annotations
import wfdb
import nspfile
import soundfile
import random
from typing import List, Tuple
from pathlib import Path
from dataclasses import dataclass
from ..diagnosis import Diagnosis


@dataclass
class ProcessedFile:
    path: Path

    @property
    def __dict__(self):
        return {
            "path": str(self.path),
        }

    @staticmethod
    async def from_wfdb(dat_path: Path, extraction_path: Path) -> ProcessedFile:
        extracted_path = Path(f"{extraction_path}/{dat_path.name}.wav")
        record = wfdb.rdrecord(dat_path)
        sample_rate = record.fs
        audio = record.p_signal
        soundfile.write(extracted_path, audio, sample_rate)
        return ProcessedFile(path=extracted_path)

    @staticmethod
    async def from_nsp(nsp_path: Path, extraction_path: Path) -> ProcessedFile:
        extracted_path = Path(f"{extraction_path}/{nsp_path.name}.wav")
        sample_rate, audio = nspfile.read(nsp_path)
        soundfile.write(extracted_path, audio, sample_rate)
        return ProcessedFile(path=extracted_path)

    @staticmethod
    def from_json(json_data):
        return ProcessedFile(**json_data)


@dataclass
class ProcessedSession:
    id: str
    age: int | None
    gender: str
    diagnosis: List[Diagnosis]
    files: List[ProcessedFile]

    @property
    def __dict__(self):
        return {
            "id": self.id,
            "age": self.age,
            "gender": self.gender,
            "diagnosis": [diagnosis.name for diagnosis in self.diagnosis],
            "files": self.files,
        }


@dataclass
class ProcessedDataset:
    db: str
    train_sessions: List[ProcessedSession]
    val_sessions: List[ProcessedSession]
    test_sessions: List[ProcessedSession]

    @staticmethod
    def generate_dataset(
        db: str,
        sessions: List[ProcessedSession],
        split: Tuple[float, float] = (0.7, 0.1),
        seed: int = 42,
    ):
        total_data = len(sessions)
        train_start = 0
        train_end = int(train_start + split[0] * total_data)
        val_start = train_end
        val_end = int(val_start + split[1] * total_data)
        test_start = val_end
        test_end = total_data

        random.Random(seed).shuffle(sessions)

        return ProcessedDataset(
            db=db,
            train_sessions=sessions[train_start:train_end],
            val_sessions=sessions[val_start:val_end],
            test_sessions=sessions[test_start:test_end],
        )

    @property
    def __dict__(self):
        return {
            "db": self.db,
            "train_sessions": self.train_sessions,
            "val_sessions": self.val_sessions,
            "test_sessions": self.test_sessions,
        }
