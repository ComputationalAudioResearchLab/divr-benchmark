from __future__ import annotations
import wfdb
import nspfile
import soundfile
from typing import List, Set
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

    def has_diagnosis(self, diagnosis_name) -> bool:
        for diag in self.diagnosis:
            if diag.satisfies(diagnosis_name):
                return True
        return False

    def diagnosis_names_at_level(self, level: int) -> Set[str]:
        diag_names = set()
        for diagnosis in self.diagnosis:
            diag_names.add(diagnosis.at_level(level).name)
        return diag_names


@dataclass
class ProcessedDataset:
    db_name: str
    train_sessions: List[ProcessedSession]
    val_sessions: List[ProcessedSession]
    test_sessions: List[ProcessedSession]

    @property
    def __dict__(self):
        return {
            "db_name": self.db_name,
            "train_sessions": self.train_sessions,
            "val_sessions": self.val_sessions,
            "test_sessions": self.test_sessions,
        }
