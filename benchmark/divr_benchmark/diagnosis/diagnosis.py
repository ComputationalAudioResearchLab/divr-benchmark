from __future__ import annotations
from dataclasses import dataclass
from typing import Optional


@dataclass
class Diagnosis:
    name: str
    parent: Optional[Diagnosis] = None

    def root(self) -> str:
        return self.name

    @staticmethod
    def from_json(json_data) -> Diagnosis:
        parent = None
        if json_data["parent"] is not None:
            parent = Diagnosis.from_json(json_data["parent"])
        return Diagnosis(
            name=json_data["name"],
            parent=parent,
        )
