from .diagnosis import Diagnosis


class DiagnosisMap:

    def get(self, name: str) -> Diagnosis:
        return Diagnosis(name=name)
