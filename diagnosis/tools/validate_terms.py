import yaml
from pathlib import Path
from divr_diagnosis import Diagnosis, DiagnosisMap


class AliasDict:

    __curdir = Path(__file__).parent.resolve()
    __alias_file = f"{__curdir}/aliases.yml"

    __index_forward = {}
    __index_reverse = {}

    def __init__(self, diagnosis_map: DiagnosisMap) -> None:
        with open(self.__alias_file, "r") as alias_file:
            alias_terms = yaml.full_load(alias_file)
        self.__all_terms = {}
        for aliases in alias_terms:
            for alias in aliases:
                self.__all_terms[alias] = aliases
                if alias in diagnosis_map:
                    alias_key = diagnosis_map[alias].name
                    self.__index_forward[alias_key] = aliases
        for key, aliases in self.__index_forward.items():
            for alias in aliases:
                self.__index_reverse[alias] = key

    def __getitem__(self, term: str) -> str:
        return self.__index_reverse[term]

    def __contains__(self, term: str) -> bool:
        return term in self.__index_reverse

    @property
    def all_terms(self) -> dict[str, list[str]]:
        return self.__all_terms


class ValidateTermsOthers:
    __curdir = Path(__file__).parent.resolve()
    __terms_file = f"{__curdir}/diag_terms.yml"

    def __init__(self, diagnosis_map: DiagnosisMap) -> None:
        self.__diagnosis_map = diagnosis_map
        self.__alias = AliasDict(diagnosis_map)
        with open(self.__terms_file, "r") as terms_file:
            terms = yaml.full_load(terms_file)
            self.__all_terms = set(terms.keys())
            self.__terms = self.__coalesce_by_alias(terms)

    def run(self):
        # self.from_db_to_diag_map()
        self.from_diag_map_to_db()

    def from_diag_map_to_db(self):
        present_terms = []
        absent_terms = []
        for diag in self.__diagnosis_map.unique_diags():
            if diag.level == self.__diagnosis_map.max_diag_level:
                if self.__diag_in_terms(diag):
                    present_terms += [diag.name]
                else:
                    absent_terms += [diag.name]
        print(f"{len(present_terms)} Terms found: ", present_terms)
        print(f"{len(absent_terms)} Terms absent: ", absent_terms)

    def __diag_in_terms(self, diag: Diagnosis) -> bool:
        possible_aliases = diag.alias + [diag.name]
        additional_aliases = []
        for al in possible_aliases:
            if al in self.__alias.all_terms:
                additional_aliases += self.__alias.all_terms[al]
        diag_aliases = possible_aliases + additional_aliases
        for al in diag_aliases:
            if al in self.__terms:
                return True
        return False

    def from_db_to_diag_map(self):
        present_terms = []
        absent_terms = {}
        terms_to_alias = {}
        for term, aliases in self.__terms.items():
            if term in self.__diagnosis_map:
                present_terms += [term]
            elif term in self.__alias:
                alias_key = self.__alias[term]
                terms_to_alias[term] = alias_key
                for alias in aliases:
                    terms_to_alias[alias] = alias_key
            else:
                absent_terms[term] = aliases
        print(f"{len(present_terms)} Terms found: ", present_terms)
        print(f"{len(terms_to_alias)} Terms to alias: ", terms_to_alias)
        print(f"{len(absent_terms)} Terms absent: ", absent_terms.keys())

    def __coalesce_by_alias(self, terms: dict[str, list[str]]) -> dict[str, list[str]]:
        new_terms = {}
        old_terms = list(terms.keys())
        while len(old_terms) > 0:
            term = old_terms.pop(0)
            new_terms[term] = []
            if term in self.__alias.all_terms:
                for alias in self.__alias.all_terms[term]:
                    if alias in old_terms:
                        new_terms[term] += [alias]
                        old_terms.remove(alias)

        return new_terms


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
