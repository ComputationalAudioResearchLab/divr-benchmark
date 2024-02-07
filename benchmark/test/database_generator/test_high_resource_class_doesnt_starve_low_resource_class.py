from uuid import uuid4
from divr_benchmark.diagnosis import DiagnosisMap
from divr_benchmark.prepare_dataset.database_generator import DatabaseGenerator
from divr_benchmark.prepare_dataset.processed import ProcessedSession
from test.database_generator.count_sessions import count_sessions
from test.database_generator.assert_all_sessions_allocated import (
    assert_all_sessions_allocated,
)

train_split = 0.7
test_split = 0.2
random_seed = 42
diagnosis_map = DiagnosisMap.v1()
database_generator = DatabaseGenerator(
    diagnosis_map=diagnosis_map,
    train_split=train_split,
    test_split=test_split,
    random_seed=random_seed,
)


def test_5():
    db_name = str(uuid4())
    diagnosis_keys = [
        "organic_structural",
        "organic",
        "muscle_tension",
        "muscle_tension",
        "muscle_tension",
    ]
    age = None
    gender = ""
    sessions = [
        ProcessedSession(
            id=str(uuid4()),
            age=age,
            gender=gender,
            diagnosis=[diagnosis_map.get(diagnosis_key)],
            files=[],
        )
        for diagnosis_key in diagnosis_keys
    ]
    dataset = database_generator.generate(
        db_name=db_name,
        sessions=sessions,
    )
    assert_all_sessions_allocated(sessions, dataset)

    ## Actual expected
    expected_organic = [
        (dataset.train_sessions, "organic", 1),
        (dataset.test_sessions, "organic", 1),
        (dataset.val_sessions, "organic", 0),
        (dataset.train_sessions, "muscle_tension", 2),
        (dataset.test_sessions, "muscle_tension", 1),
        (dataset.val_sessions, "muscle_tension", 0),
    ]
    for bucket, diagnosis_key, expected_count in expected_organic:
        assert expected_count == count_sessions(
            sessions=bucket,
            diagnosis_key=diagnosis_key,
            gender=gender,
            age_range=age,
        )


def test_10():
    db_name = str(uuid4())
    diagnosis_keys = [
        "organic_inflammatory",
        "organic_neuro_muscular",
        "organic_structural",
        "organic_trauma",
        "organic_neuro_muscular",
        "organic_structural",
        "organic",
        "muscle_tension",
        "muscle_tension",
        "muscle_tension",
    ]
    age = None
    gender = ""
    sessions = [
        ProcessedSession(
            id=str(uuid4()),
            age=age,
            gender=gender,
            diagnosis=[diagnosis_map.get(diagnosis_key)],
            files=[],
        )
        for diagnosis_key in diagnosis_keys
    ]
    dataset = database_generator.generate(
        db_name=db_name,
        sessions=sessions,
    )
    assert_all_sessions_allocated(sessions, dataset)

    ## Actual expected
    expected_organic = [
        (dataset.train_sessions, "organic", 5),
        (dataset.test_sessions, "organic", 1),
        (dataset.val_sessions, "organic", 1),
        (dataset.train_sessions, "muscle_tension", 2),
        (dataset.test_sessions, "muscle_tension", 1),
        (dataset.val_sessions, "muscle_tension", 0),
    ]
    for bucket, diagnosis_key, expected_count in expected_organic:
        assert expected_count == count_sessions(
            sessions=bucket,
            diagnosis_key=diagnosis_key,
            gender=gender,
            age_range=age,
        )


def test_20():
    db_name = str(uuid4())
    diagnosis_keys = [
        "organic",
        "organic",
        "organic",
        "organic",
        "organic",
        "organic",
        "organic",
        "organic",
        "organic",
        "organic",
        "organic",
        "organic",
        "organic",
        "organic",
        "muscle_tension",
        "muscle_tension",
        "muscle_tension",
        "muscle_tension",
        "muscle_tension",
        "muscle_tension",
    ]
    age = None
    gender = ""
    sessions = [
        ProcessedSession(
            id=str(uuid4()),
            age=age,
            gender=gender,
            diagnosis=[diagnosis_map.get(diagnosis_key)],
            files=[],
        )
        for diagnosis_key in diagnosis_keys
    ]
    dataset = database_generator.generate(
        db_name=db_name,
        sessions=sessions,
    )
    assert_all_sessions_allocated(sessions, dataset)

    ## Actual expected
    expected_organic = [
        (dataset.train_sessions, "organic", 10),
        (dataset.test_sessions, "organic", 3),
        (dataset.val_sessions, "organic", 1),
        (dataset.train_sessions, "muscle_tension", 4),
        (dataset.test_sessions, "muscle_tension", 1),
        (dataset.val_sessions, "muscle_tension", 1),
    ]
    for bucket, diagnosis_key, expected_count in expected_organic:
        assert expected_count == count_sessions(
            sessions=bucket,
            diagnosis_key=diagnosis_key,
            gender=gender,
            age_range=age,
        )
