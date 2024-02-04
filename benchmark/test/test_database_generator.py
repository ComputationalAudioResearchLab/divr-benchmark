from divr_benchmark.diagnosis import DiagnosisMap
from divr_benchmark.prepare_dataset.database_generator import DatabaseGenerator
from divr_benchmark.prepare_dataset.processed import ProcessedSession


class TestDatabaseGenerator:

    def setup_method(self) -> None:
        self.database_generator = DatabaseGenerator()
        self.diagnosis_map = DiagnosisMap()

    def test_same_diagnosis_gets_equally_split(self):
        assert True
