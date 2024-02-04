from typing import List
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
        diagnosis = self.diagnosis_map.get("unclassified")
        sessions = [
            ProcessedSession(
                id=str(uuid4()), age=None, gender="", diagnosis=[diagnosis], files=[]
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

    @pytest.mark.parametrize("sessions_per_diagnosis", [10, 100])
    @pytest.mark.parametrize(
        "diagnosis_keys",
        [
            ["normal"],
            ["normal", "pathological"],
            ["normal", "pathological", "unclassified"],
            ["normal", "pathological", "unclassified", "organic"],
        ],
    )
    def test_evenly_split_diagnosis_get_split_evently(
        self,
        sessions_per_diagnosis: int,
        diagnosis_keys: List[str],
    ):
        db_name = str(uuid4())
        sessions = []
        for diagnosis_key in diagnosis_keys:
            diagnosis = self.diagnosis_map.get(diagnosis_key)
            sessions += [
                ProcessedSession(
                    id=str(uuid4()),
                    age=None,
                    gender="",
                    diagnosis=[diagnosis],
                    files=[],
                )
                for _ in range(sessions_per_diagnosis)
            ]
        expected_train_data_count = int(sessions_per_diagnosis * self.train_split)
        expected_test_data_count = int(sessions_per_diagnosis * self.test_split)
        expected_val_data_count = (
            sessions_per_diagnosis
            - expected_train_data_count
            - expected_test_data_count
        )
        dataset = self.database_generator.generate(db_name=db_name, sessions=sessions)
        for diagnosis_key in diagnosis_keys:
            train_count = self.__count_diagnosis(dataset.train_sessions, diagnosis_key)
            test_count = self.__count_diagnosis(dataset.test_sessions, diagnosis_key)
            val_count = self.__count_diagnosis(dataset.val_sessions, diagnosis_key)
            assert expected_train_data_count == train_count
            assert expected_test_data_count == test_count
            assert expected_val_data_count == val_count

    def __count_diagnosis(
        self, sessions: List[ProcessedSession], diagnosis_key: str
    ) -> int:
        count = 0
        for session in sessions:
            for diagnosis in session.diagnosis:
                if diagnosis.name == diagnosis_key:
                    count += 1
        return count
