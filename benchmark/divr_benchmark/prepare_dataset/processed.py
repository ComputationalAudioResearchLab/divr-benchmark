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
    num_files: int

    @property
    def __dict__(self):
        return {
            "id": self.id,
            "age": self.age,
            "gender": self.gender,
            "diagnosis": [diagnosis.name for diagnosis in self.diagnosis],
            "files": self.files,
            "num_files": self.num_files,
        }

    @property
    def best_diagnosis(self) -> Diagnosis:
        sorted_diagnosis = sorted(self.diagnosis, reverse=True)
        complete_diagnosis = list(
            filter(lambda x: not x.incompletely_classified, sorted_diagnosis)
        )
        if len(complete_diagnosis) > 0:
            return complete_diagnosis[0]
        return sorted_diagnosis[0]

    def diagnosis_names_at_level(self, level: int) -> Set[str]:
        diag_names = set()
        for diagnosis in self.diagnosis:
            diag_names.add(diagnosis.at_level(level).name)
        return diag_names

    def diagnosis_at_level(self, level: int) -> List[Diagnosis]:
        diags = {}
        for diagnosis in self.diagnosis:
            diag = diagnosis.at_level(level)
            if diag.name not in diags:
                diags[diag.name] = diag
        return sorted(list(diags.values()), reverse=True)


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
