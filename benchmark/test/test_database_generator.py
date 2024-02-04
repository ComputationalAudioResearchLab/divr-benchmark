import pytest
from uuid import uuid4
from divr_benchmark.diagnosis import DiagnosisMap
from divr_benchmark.prepare_dataset.database_generator import DatabaseGenerator
from divr_benchmark.prepare_dataset.processed import ProcessedSession


class TestDatabaseGenerator:

    def setup_method(self) -> None:
        self.train_split = 0.7
        self.test_split = 0.2
        self.random_seed = 42
        self.database_generator = DatabaseGenerator(
            train_split=self.train_split,
            test_split=self.test_split,
            random_seed=self.random_seed,
        )
        self.diagnosis_map = DiagnosisMap()

    @pytest.mark.parametrize("total_data", [5, 10, 100])
    def test_data_is_split_in_expected_splits(self, total_data: int):
        db_name = str(uuid4())
        sessions = [
            ProcessedSession(
                id=str(uuid4()), age=None, gender="", diagnosis=[], files=[]
            )
            for _ in range(total_data)
        ]
        expected_train_data_len = int(total_data * self.train_split)
        expected_test_data_len = int(total_data * self.test_split)
        expected_val_data_len = (
            total_data - expected_train_data_len - expected_test_data_len
        )
        dataset = self.database_generator.generate(db_name=db_name, sessions=sessions)
        assert len(dataset.train_sessions) == expected_train_data_len
        assert len(dataset.test_sessions) == expected_test_data_len
        assert len(dataset.val_sessions) == expected_val_data_len
