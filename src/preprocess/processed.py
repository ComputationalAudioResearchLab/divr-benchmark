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
