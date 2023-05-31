from __future__ import annotations
import wfdb
import nspfile
import soundfile
from typing import List
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
            "diagnosis": self.diagnosis,
            "files": self.files,
        }


@dataclass
class ProcessedDataset:
    db: str
    sessions: List[ProcessedSession]

    @property
    def __dict__(self):
        return {
            "db": self.db,
            "sessions": self.sessions,
        }
