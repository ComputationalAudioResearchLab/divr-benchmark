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

    def satisfies(self, name: str) -> bool:
        if name == self.name or name in self.alias:
            return True
        for parent in self.parents:
            if parent.parent.satisfies(name):
                return True
        return False

    def at_level(self, level: int) -> Diagnosis:
        if level >= self.level:
            return self

        return self.best_parent_link.parent.at_level(level)

    @property
    def best_parent_link(self) -> DiagnosisLink:
        return sorted(self.parents, key=lambda x: x.weight, reverse=True)[0]
