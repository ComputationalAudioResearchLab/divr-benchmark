from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, List
from .diagnosis import Diagnosis, DiagnosisLink


class DiagnosisMap:

    __index: Dict[str, Diagnosis] = {}

    def __init__(self, map_file: Path) -> None:
        self.__load_map(map_file)

    @staticmethod
    def v1() -> DiagnosisMap:
        version = "v1"
        curdir = Path(__file__).parent
        map_file = Path(f"{curdir}/diagnosis_map_{version}.yml")
        return DiagnosisMap(map_file=map_file)

    def get(self, name: str) -> Diagnosis:
        return self.__index[name.lower()]

    def find(self, name: str) -> List[Diagnosis]:
        return list(filter(lambda diag: diag.satisfies(name), self.__index.values()))

    def __load_map(self, diagnosis_map_file_path: Path) -> None:
        with open(diagnosis_map_file_path, "r") as diagnosis_map_file:
            data = yaml.load(diagnosis_map_file, Loader=yaml.FullLoader)
        for key, value in data.items():
            diagnosis = self.__parse_yaml_diagnosis(key, value)
            self.__index_data(diagnosis)

    def __index_data(self, diagnosis: Diagnosis) -> None:
        index_keys = [diagnosis.name] + diagnosis.alias
        for key in index_keys:
            self.__index[key] = diagnosis

    def __parse_yaml_diagnosis(self, key, data) -> Diagnosis:
        level = data["level"]

        alias = []
        if "alias" in data and data["alias"] is not None:
            alias = data["alias"]

        parents = []
        if "parents" in data and data["parents"] is not None:
            parents = [
                DiagnosisLink(parent=self.get(parent_name), weight=parent_weight)
                for parent_name, parent_weight in data["parents"].items()
            ]
        votes = {}
        if "votes" in data and data["votes"] is not None:
            votes = data["votes"]

        return Diagnosis(
            name=key,
            alias=alias,
            level=level,
            parents=parents,
            votes=votes,
        )
