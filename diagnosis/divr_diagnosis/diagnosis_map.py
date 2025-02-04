from __future__ import annotations
import yaml
from pathlib import Path
from typing import Dict, Set
from .diagnosis import Diagnosis, DiagnosisLink


class DiagnosisMap:
    __index: Dict[str, Diagnosis]
    __curdir = Path(__file__).parent
    __diagnosis_maps_dir = Path(f"{__curdir}/diagnosis_maps")
    __max_level: int
    __unclassified_link: DiagnosisLink

    def __init__(self, allow_unmapped: bool = False) -> None:
        self.__index = {}
        self.__max_level = 0
        map_name = self.__class__.__name__
        map_file = Path(f"{self.__diagnosis_maps_dir}/{map_name}.yml")
        self.__allow_unmapped = allow_unmapped
        self.__load_map(map_file)

    def __contains__(self, name: str) -> bool:
        return name.lower() in self.__index

    def __getitem__(self, name: str) -> Diagnosis:
        return self.get(name)

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
            else:
                raise err

    def find(self, name: str) -> Set[Diagnosis]:
        return set(filter(lambda diag: diag.satisfies(name), self.__index.values()))

    def unique_diags(self) -> Set[Diagnosis]:
        return set([diag for diag in self.__index.values()])

    @property
    def tree(self):
        tree = {}
        for diag in self.__index.values():
            branch = tree
            for level in range(diag.level + 1):
                diag_at_level = diag.at_level(level)
                if diag_at_level.level == level:
                    # because values from the root of the tree propage to leaves
                    if diag_at_level not in branch:
                        branch[diag_at_level] = {}
                    branch = branch[diag_at_level]
        return tree

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def unclassified(self) -> Diagnosis:
        return self.__index["unclassified"]

    @property
    def max_diag_level(self) -> int:
        return self.__max_level

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
