from dataclasses import dataclass
from ..diagnosis import Diagnosis


@dataclass
class Task:
    id: str
    age: int | None
    gender: str
    label: Diagnosis
    audio_key: str
