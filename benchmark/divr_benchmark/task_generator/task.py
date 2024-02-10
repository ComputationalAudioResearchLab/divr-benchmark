from dataclasses import dataclass
from typing import List
from ..diagnosis import Diagnosis


@dataclass
class Task:
    id: str
    age: int | None
    gender: str
    label: Diagnosis
    audio_keys: List[str]
