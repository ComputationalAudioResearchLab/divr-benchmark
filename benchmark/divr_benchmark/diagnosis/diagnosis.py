from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

classification_weights = {
    "pathological": 3,
    "normal": 2,
    "unclassified": 1,
}


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
    def root(self) -> Diagnosis:
        return self.at_level(0)

    @property
    def best_parent_link(self) -> DiagnosisLink:
        return sorted(self.parents, key=self.__parent_sort_key, reverse=True)[0]

    def __parent_sort_key(self, x: DiagnosisLink) -> Tuple[float, int]:
        return (x.weight, classification_weights[x.parent.root.name])
