from pathlib import Path
from typing import Dict

import yaml
from .diagnosis import Diagnosis, DiagnosisLink


class DiagnosisMap:

    __index: Dict[str, Diagnosis] = {}

    def __init__(self) -> None:
        curdir = Path(__file__).parent
        self.__load_map(Path(f"{curdir}/diagnosis_map.yml"))

    def get(self, name: str) -> Diagnosis:
        return self.__index[name.lower()]

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

        return Diagnosis(
            name=key,
            alias=alias,
            level=level,
            parents=parents,
        )
