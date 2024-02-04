from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class DiagnosisLink:
    parent: Diagnosis
    weight: float


@dataclass
class Diagnosis:
    name: str
    level: int
    alias: List[str]
    parents: List[DiagnosisLink]

    def root(self) -> str:
        return self.name
