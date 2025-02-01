import yaml
from pathlib import Path
from divr_diagnosis import DiagnosisMap


class ValidateTerms:
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
