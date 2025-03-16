import random
import numpy as np
from typing import List, Tuple
import pytest
from uuid import uuid4
from divr_diagnosis import diagnosis_maps
from divr_benchmark.prepare_dataset.database_generator import DatabaseGenerator
from divr_benchmark.prepare_dataset.processed import ProcessedSession
from test.database_generator.count_sessions import count_sessions
from test.database_generator.assert_all_sessions_allocated import (
    assert_all_sessions_allocated,
)

train_split = 0.7
test_split = 0.2
random_seed = 42
diagnosis_map = diagnosis_maps.USVAC_2025()
database_generator = DatabaseGenerator(
    train_split=train_split,
    test_split=test_split,
    random_seed=random_seed,
)


@pytest.mark.parametrize(
    "sessions_count",
    [
        10,
        25,
        50,
        100,
    ],
)
@pytest.mark.parametrize(
    "diagnosis_keys",
    [
        ["unclassified"],
        ["normal", "pathological"],
        ["normal", "pathological", "unclassified"],
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
def test(
    sessions_count: int,
    diagnosis_keys: List[str],
    genders: List[str],
    age_ranges: List[Tuple[int, int] | None],
):
    db_name = str(uuid4())
    sessions = []
    for diagnosis_key in diagnosis_keys:
        diagnosis = diagnosis_map.get(diagnosis_key)
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
                        speaker_id=str(uuid4()),
                        age=age,
                        gender=gender,
                        diagnosis=[diagnosis],
                        files=[],
                        num_files=0,
                    )
                    for _ in range(sessions_count)
                ]
    expected_ratio = np.array([train_split, test_split, 1 - train_split - test_split])
    dataset = database_generator.generate(db_name=db_name, sessions=sessions)
    assert_all_sessions_allocated(sessions, dataset)
    for diagnosis_key in diagnosis_keys:
        for gender in genders:
            for age_range in age_ranges:
                train_count = count_sessions(
                    dataset.train_sessions, diagnosis_key, gender, age_range
                )
                test_count = count_sessions(
                    dataset.test_sessions, diagnosis_key, gender, age_range
                )
                val_count = count_sessions(
                    dataset.val_sessions, diagnosis_key, gender, age_range
                )
                counts = np.array([train_count, test_count, val_count])
                ratios = counts / counts.sum()
                ratio_diff = expected_ratio - ratios
                mean_l1_error = np.abs(ratio_diff).mean()
                assert mean_l1_error < 0.1
