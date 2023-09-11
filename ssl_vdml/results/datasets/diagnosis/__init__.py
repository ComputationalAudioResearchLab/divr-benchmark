from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, List
import json
import pandas as pd
from pathlib import Path


@dataclass
class Diagnosis:
    name: str
    parent: Optional[Diagnosis] = None

    def to_list(self) -> List[str]:
        if self.parent is not None:
            return self.parent.to_list() + [self.name]
        else:
            return [self.name]

    def __str__(self) -> str:
        return " > ".join(self.to_list())

    def root(self) -> str:
        return self.at_level(0)

    def at_level(self, level: int) -> str:
        return self.to_list()[level]

    @staticmethod
    def from_string(diagnosis: str) -> Diagnosis:
        names = map(str.strip, diagnosis.split(">"))
        current = None
        for name in names:
            current = Diagnosis(name=name, parent=current)
        assert current is not None
        return current

    @staticmethod
    def from_json(json_data) -> Diagnosis:
        parent = None
        if json_data["parent"] is not None:
            parent = Diagnosis.from_json(json_data["parent"])
        return Diagnosis(
            name=json_data["name"],
            parent=parent,
        )


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

        self.diagnosis_keys = list(self.diagnosis_map.keys())

    def get(self, name: str) -> Diagnosis:
        return self.diagnosis_map[name]

    def from_tsv(self, input_tsv: Path) -> dict:
        df = pd.read_csv(input_tsv, sep="\t")
        df.columns = ["diagnosis", "classification"]
        possible_classes = {
            "others": {},
            "unknown": {},
            "normal": {},
            "cancer": {},
            "inflammatory": {},
            "neuro_muscular": {},
            "structural": {},
            "traumatic": {},
            "hyperfunction": {},
        }
        for _, row in df.iterrows():
            diagnosis = row["diagnosis"].strip().lower()
            classification = row["classification"]
            try:
                possible_classes[classification][diagnosis] = {}
            except Exception:
                print(diagnosis, classification)
                raise KeyError()

        return {
            "normal": possible_classes["normal"],
            "pathological": {
                "organic": {
                    "cancer": possible_classes["cancer"],
                    "inflammatory": possible_classes["inflammatory"],
                    "neuro_muscular": possible_classes["neuro_muscular"],
                    "structural": possible_classes["structural"],
                    "traumatic": possible_classes["traumatic"],
                },
                "others": possible_classes["others"],
                "muscle_tension_dysphonia": {
                    "hyperfunction": possible_classes["hyperfunction"]
                },
                "psychogenic": {},
            },
            "unknown": possible_classes["unknown"],
        }

    def to_int(self, name: str) -> int:
        return self.diagnosis_keys.index(name)

    def from_int(self, index: int) -> Diagnosis:
        return self.get(self.diagnosis_keys[index])

    def most_severe(self, diagnosis: List[Diagnosis], level: int) -> int:
        diagnosis_at_level = [diag.at_level(level) for diag in diagnosis]
        severest_diagnosis = diagnosis_at_level[0]
        return self.to_int(severest_diagnosis)
