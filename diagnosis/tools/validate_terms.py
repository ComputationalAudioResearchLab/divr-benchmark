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

        for term in self.__terms:
            print(term)
            print(self.__diagnosis_map[term])
