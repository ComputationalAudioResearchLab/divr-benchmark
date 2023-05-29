from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict
import json
import pandas as pd
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

    @staticmethod
    def from_string(diagnosis: str) -> Diagnosis:
        names = map(str.strip, diagnosis.split(">"))
        current = None
        for name in names:
            current = Diagnosis(name=name, parent=current)
        assert current is not None
        return current


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
