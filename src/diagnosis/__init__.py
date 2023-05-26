from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import json
from pathlib import Path


@dataclass
class Diagnosis:
    name: str
    parent: Optional[Diagnosis] = None

    def __str__(self) -> str:
        if self.parent is not None:
            return f"{self.parent} > {self.name}"
        else:
            return self.name


class DiagnosisMap:
    curdir = Path(__file__).resolve().parent
    map_path = f"{curdir}/diagnosis_map.json"
    diagnosis_map: Dict[str, Diagnosis] = {}

    def __init__(self) -> None:
        def process_json_map(json_data: dict, parent=None):
            for key, value in json_data.items():
                current = Diagnosis(name=key, parent=parent)
                self.diagnosis_map[key] = current
                process_json_map(json_data=value, parent=current)

        with open(self.map_path, "r") as map_file:
            process_json_map(json.load(map_file))

    def find(self, name: str) -> Diagnosis:
        return self.diagnosis_map[name]
