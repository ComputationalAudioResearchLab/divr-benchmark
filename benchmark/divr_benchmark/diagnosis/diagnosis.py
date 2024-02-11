from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

root_classification_weights = {
    "pathological": 3,
    "normal": 2,
    "unclassified": 1,
}

incomplete_classifications = ["unclassified", "unclassified_pathology"]


@dataclass
class DiagnosisLink:
    parent: Diagnosis
    weight: float

    def __lt__(self, other: DiagnosisLink) -> bool:
        if self.weight == other.weight:
            self_root_weight = root_classification_weights[self.parent.root.name]
            other_root_weight = root_classification_weights[other.parent.root.name]
            return self_root_weight < other_root_weight
        return self.weight < other.weight


@dataclass
class Diagnosis:
    name: str
    level: int
    alias: List[str]
    parents: List[DiagnosisLink]
    votes: Dict[str, str]

    def __hash__(self) -> int:
        return hash(self.name)

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
    def incompletely_classified(self) -> bool:
        for key in incomplete_classifications:
            if self.satisfies(key):
                return True
        return False

    @property
    def root(self) -> Diagnosis:
        return self.at_level(0)

    @property
    def best_parent_link(self) -> DiagnosisLink | None:
        if len(self.parents) <= 0:
            return None

        diags_at_levels: Dict[int, Dict[str, DiagnosisLink]] = {
            self.level: {self.name: DiagnosisLink(parent=self, weight=1)}
        }
        for level in range(self.level - 1, -1, -1):
            if (level + 1) in diags_at_levels:
                diags_at_prev_level = diags_at_levels[level + 1].values()
                for diag_link in diags_at_prev_level:
                    for parent_link in diag_link.parent.parents:
                        parent = parent_link.parent
                        parent_level = parent.level
                        weight = parent_link.weight * diag_link.weight
                        parent_name = parent.name
                        if parent_level not in diags_at_levels:
                            diags_at_levels[parent_level] = {}
                        if parent_name not in diags_at_levels[parent_level]:
                            diags_at_levels[parent_level][parent_name] = DiagnosisLink(
                                parent=parent, weight=weight
                            )
                        else:
                            diags_at_levels[parent_level][parent_name].weight += weight

        best_diag = max(diags_at_levels[0].values(), key=lambda x: x.weight)
        for level in range(1, self.level):
            if level in diags_at_levels:
                best_diag = self.__matching_parent(
                    best_diag, list(diags_at_levels[level].values())
                )

        return best_diag

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

        if self_weight == other_weight:
            self_root_weight = root_classification_weights[self.root.name]
            other_root_weight = root_classification_weights[other.root.name]
            return self_root_weight < other_root_weight

        return self_weight < other_weight

    def __max_parent_weight(self) -> float | None:
        if len(self.parents) > 0:
            return max([parent.weight for parent in self.parents])
        return None

    def __matching_parent(
        self, to_match: DiagnosisLink, diags: List[DiagnosisLink]
    ) -> DiagnosisLink:
        sorted_diags = sorted(diags, reverse=True)
        for diag in sorted_diags:
            for parent in diag.parent.parents:
                if parent.parent == to_match.parent:
                    return diag
        # none of the parents from current level match
        # so pass the current best match to next level
        return to_match
