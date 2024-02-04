import random
from typing import List, Tuple
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

    @pytest.mark.parametrize("sessions_count", [5, 10, 100])
    @pytest.mark.parametrize(
        "diagnosis_keys",
        [
            ["unclassified"],
            ["normal", "pathological"],
            ["normal", "pathological", "unclassified"],
            ["normal", "pathological", "unclassified", "organic"],
        ],
    )
    @pytest.mark.parametrize(
        "genders",
        [
            ["male"],
            ["male", "female"],
            ["male", "female", "others"],
        ],
    )
    @pytest.mark.parametrize(
        "age_ranges",
        [
            [(0, 10)],
            [(0, 10), (10, 20)],
            [(0, 10), (10, 20), None],
        ],
    )
    def test_evenly_split_data_gets_split_evenly(
        self,
        sessions_count: int,
        diagnosis_keys: List[str],
        genders: List[str],
        age_ranges: List[Tuple[int, int] | None],
    ):
        db_name = str(uuid4())
        sessions = []
        for diagnosis_key in diagnosis_keys:
            diagnosis = self.diagnosis_map.get(diagnosis_key)
            for gender in genders:
                for age_range in age_ranges:
                    if age_range is not None:
                        age = age_range[0] + random.randint(
                            0, age_range[1] - age_range[0] - 1
                        )
                    else:
                        age = None
                    sessions += [
                        ProcessedSession(
                            id=str(uuid4()),
                            age=age,
                            gender=gender,
                            diagnosis=[diagnosis],
                            files=[],
                        )
                        for _ in range(sessions_count)
                    ]
        expected_train_data_count = int(sessions_count * self.train_split)
        expected_test_data_count = int(sessions_count * self.test_split)
        expected_val_data_count = (
            sessions_count - expected_train_data_count - expected_test_data_count
        )
        dataset = self.database_generator.generate(db_name=db_name, sessions=sessions)
        for diagnosis_key in diagnosis_keys:
            for gender in genders:
                for age_range in age_ranges:
                    train_count = self.__count_diagnosis(
                        dataset.train_sessions, diagnosis_key, gender, age_range
                    )
                    test_count = self.__count_diagnosis(
                        dataset.test_sessions, diagnosis_key, gender, age_range
                    )
                    val_count = self.__count_diagnosis(
                        dataset.val_sessions, diagnosis_key, gender, age_range
                    )
                    assert expected_train_data_count == train_count
                    assert expected_test_data_count == test_count
                    assert expected_val_data_count == val_count

    def __count_diagnosis(
        self,
        sessions: List[ProcessedSession],
        diagnosis_key: str,
        gender: str,
        age_range: Tuple[int, int] | None,
    ) -> int:
        count = 0
        for session in sessions:
            if session.gender == gender:
                for diagnosis in session.diagnosis:
                    if diagnosis.name == diagnosis_key:
                        age_bracket = self.__age_to_bracket(session.age)
                        if age_bracket == age_range:
                            count += 1
        return count

    def __age_to_bracket(self, age: int | None) -> Tuple[int, int] | None:
        if age is None:
            return None
        lower = (age // 10) * 10
        upper = lower + 10
        return (lower, upper)
