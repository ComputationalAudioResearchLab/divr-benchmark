import yaml
from pathlib import Path
from divr_diagnosis import DiagnosisMap


class AliasDict:

    __curdir = Path(__file__).parent.resolve()
    __alias_file = f"{__curdir}/aliases.yml"

    __index_forward = {}
    __index_reverse = {}

    def __init__(self, diagnosis_map: DiagnosisMap) -> None:
        with open(self.__alias_file, "r") as alias_file:
            alias_terms = yaml.full_load(alias_file)
        for aliases in alias_terms:
            for alias in aliases:
                if alias in diagnosis_map:
                    self.__index_forward[alias] = aliases
        for key, aliases in self.__index_forward.items():
            for alias in aliases:
                self.__index_reverse[alias] = key

    def __getitem__(self, term: str) -> str:
        return self.__index_reverse[term]

    def __contains__(self, term: str) -> bool:
        return term in self.__index_reverse


class ValidateTermsOthers:
    __curdir = Path(__file__).parent.resolve()
    __terms_file = f"{__curdir}/diag_terms.yml"

    def __init__(self, diagnosis_map: DiagnosisMap) -> None:
        self.__diagnosis_map = diagnosis_map
        self.__alias = AliasDict(diagnosis_map)
        with open(self.__terms_file, "r") as terms_file:
            self.__terms = yaml.full_load(terms_file)

    def run(self):
        present_terms = []
        absent_terms = {}
        terms_to_alias = {}
        for term, dbs in self.__terms.items():
            if term in self.__diagnosis_map:
                present_terms += [term]
            elif term in self.__alias:
                terms_to_alias[term] = self.__alias[term]
            else:
                absent_terms[term] = dbs
        print(f"{len(present_terms)} Terms found: ", present_terms)
        print(f"{len(terms_to_alias)} Terms to alias: ", terms_to_alias)
        print(f"{len(absent_terms)} Terms absent: ", absent_terms.keys())


class ValidateTermsUSVAC:
    __curdir = Path(__file__).parent.resolve()
    __terms_file = f"{__curdir}/diag_terms.yml"

    def __init__(self, diagnosis_map: DiagnosisMap) -> None:
        self.__diagnosis_map = diagnosis_map
        with open(self.__terms_file, "r") as terms_file:
            self.__terms = yaml.full_load(terms_file)

    def run(self):
        present_terms = []
        absent_terms = {}
        for term, dbs in self.__terms.items():
            if term in self.__diagnosis_map:
                present_terms += [term]
            else:
                absent_terms[term] = dbs
        print(f"{len(present_terms)} Terms found")
        print(f"{len(absent_terms)} Terms absent: ", absent_terms)
