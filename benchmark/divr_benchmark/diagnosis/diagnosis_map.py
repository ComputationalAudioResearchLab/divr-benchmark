from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, List
from .diagnosis import Diagnosis, DiagnosisLink


class DiagnosisMap:
    __index: Dict[str, Diagnosis] = {}
    __curdir = Path(__file__).parent
    __diagnosis_maps_dir = Path(f"{__curdir}/diagnosis_maps")
    __max_level = 0
    __unclassified_link: DiagnosisLink

    def __init__(self, allow_unmapped: bool = False) -> None:
        map_name = self.__class__.__name__
        map_file = Path(f"{self.__diagnosis_maps_dir}/{map_name}.yml")
        self.__allow_unmapped = allow_unmapped
        self.__load_map(map_file)

    def get(self, name: str) -> Diagnosis:
        name = name.lower()
        try:
            return self.__index[name]
        except KeyError as err:
            if self.__allow_unmapped:
                return Diagnosis(
                    name=name,
                    level=self.__max_level,
                    alias=[],
                    parents=[self.__unclassified_link],
                    votes={},
                )
                pass
            else:
                raise err

    def find(self, name: str) -> List[Diagnosis]:
        return list(filter(lambda diag: diag.satisfies(name), self.__index.values()))

    def __load_map(self, diagnosis_map_file_path: Path) -> None:
        with open(diagnosis_map_file_path, "r") as diagnosis_map_file:
            data = yaml.load(diagnosis_map_file, Loader=yaml.FullLoader)
        for key, value in data.items():
            diagnosis = self.__parse_yaml_diagnosis(key, value)
            self.__index_data(diagnosis)
        if "unclassified" not in self.__index:
            raise ValueError(
                "Invalid map: class 'unclassified' must be present in diagnosis map"
            )
        self.__unclassified_link = DiagnosisLink(
            parent=self.__index["unclassified"],
            weight=1.00,
        )

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

        if level > self.__max_level:
            self.__max_level = level

        return Diagnosis(
            name=key,
            alias=alias,
            level=level,
            parents=parents,
            votes=votes,
        )
