from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple

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
    votes: Dict[str, str]

    def satisfies(self, name: str) -> bool:
        if name == self.name or name in self.alias:
            return True
        best_parent = self.best_parent_link
        if best_parent and best_parent.parent.satisfies(name):
            return True
        return False

    def at_level(self, level: int) -> Diagnosis:
        if level >= self.level:
            return self
        best_parent = self.best_parent_link
        if best_parent:
            return best_parent.parent.at_level(level)
        return self

    @property
    def root(self) -> Diagnosis:
        return self.at_level(0)

    @property
    def best_parent_link(self) -> DiagnosisLink | None:
        if len(self.parents) > 0:
            return sorted(self.parents, key=self.__parent_sort_key, reverse=True)[0]
        return None

    def __lt__(self, other: Diagnosis) -> bool:
        self_weight = self.__max_parent_weight()
        other_weight = other.__max_parent_weight()

        if self_weight is None:
            # current class has no parents so can't be less than other
            return False

        if other_weight is None:
            # other class has no parents, and current class must have parents
            # hence the current class is less than other
            return True

        return self_weight < other_weight

    def __parent_sort_key(self, x: DiagnosisLink) -> Tuple[float, int]:
        return (x.weight, classification_weights[x.parent.root.name])

    def __max_parent_weight(self) -> float | None:
        if len(self.parents) > 0:
            return max([parent.weight for parent in self.parents])
        return None
