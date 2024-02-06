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
diagnosis_map = DiagnosisMap()
database_generator = DatabaseGenerator(
    diagnosis_map=diagnosis_map,
    train_split=train_split,
    test_split=test_split,
    random_seed=random_seed,
)


def test_1():
    db_name = str(uuid4())
    diagnosis_keys = ["organic"] * 10 + ["muscle_tension"] * 3
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
    expected_counts = [
        (dataset.train_sessions, "muscle_tension", 1),
        (dataset.test_sessions, "muscle_tension", 1),
        (dataset.val_sessions, "muscle_tension", 1),
        (dataset.train_sessions, "organic", 8),  # 7
        (dataset.test_sessions, "organic", 1),  # 2
        (dataset.val_sessions, "organic", 1),  # 1
    ]
    for bucket, diagnosis_key, expected_count in expected_counts:
        assert expected_count == count_sessions(
            sessions=bucket,
            diagnosis_key=diagnosis_key,
            gender=gender,
            age_range=age,
        )


def test_2():
    db_name = str(uuid4())
    diagnosis_keys = ["organic"] * 100 + ["muscle_tension"] * 30
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
    expected_counts = [
        (dataset.train_sessions, "muscle_tension", 12),  # 21
        (dataset.test_sessions, "muscle_tension", 12),  # 6
        (dataset.val_sessions, "muscle_tension", 6),  # 3
        (dataset.train_sessions, "organic", 79),  # 70
        (dataset.test_sessions, "organic", 14),  # 20
        (dataset.val_sessions, "organic", 7),  # 10
    ]
    for bucket, diagnosis_key, expected_count in expected_counts:
        assert expected_count == count_sessions(
            sessions=bucket,
            diagnosis_key=diagnosis_key,
            gender=gender,
            age_range=age,
        )


def test_3():
    """
    seems balanced because we have 10 of each organic, muscle_tension and healthy
    but since organic and muscle_tension fall under pathological and healthy is its own class
    it is imbalaced with 20 for pathological and 10 for healthy.
    """
    db_name = str(uuid4())
    diagnosis_keys = []
    for diagnosis in ["organic", "muscle_tension", "healthy"]:
        diagnosis_keys += [diagnosis] * 10
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
    expected_counts = [
        (dataset.train_sessions, "muscle_tension", 7),  # 7
        (dataset.test_sessions, "muscle_tension", 2),  # 2
        (dataset.val_sessions, "muscle_tension", 1),  # 1
        (dataset.train_sessions, "organic", 8),  # 7
        (dataset.test_sessions, "organic", 1),  # 2
        (dataset.val_sessions, "organic", 1),  # 1
        (dataset.train_sessions, "healthy", 6),  # 7
        (dataset.test_sessions, "healthy", 3),  # 2
        (dataset.val_sessions, "healthy", 1),  # 1
    ]
    for bucket, diagnosis_key, expected_count in expected_counts:
        assert expected_count == count_sessions(
            sessions=bucket,
            diagnosis_key=diagnosis_key,
            gender=gender,
            age_range=age,
        )


def test_4():
    """
    seems balanced because we have 10 of each organic, muscle_tension, functional and healthy
    but since organic and muscle_tension fall under pathological and healthy is its own class
    it is imbalaced with 30 for pathological and 10 for healthy.
    """
    db_name = str(uuid4())
    diagnosis_keys = []
    for diagnosis in ["organic", "muscle_tension", "functional", "healthy"]:
        diagnosis_keys += [diagnosis] * 10
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
    expected_counts = [
        (dataset.train_sessions, "functional", 9),  # 7
        (dataset.test_sessions, "functional", 1),  # 2
        (dataset.val_sessions, "functional", 0),  # 1
        (dataset.train_sessions, "muscle_tension", 8),  # 7
        (dataset.test_sessions, "muscle_tension", 1),  # 2
        (dataset.val_sessions, "muscle_tension", 1),  # 1
        (dataset.train_sessions, "organic", 7),  # 7
        (dataset.test_sessions, "organic", 2),  # 2
        (dataset.val_sessions, "organic", 1),  # 1
        (dataset.train_sessions, "healthy", 4),  # 7
        (dataset.test_sessions, "healthy", 4),  # 2
        (dataset.val_sessions, "healthy", 2),  # 1
    ]
    for bucket, diagnosis_key, expected_count in expected_counts:
        assert expected_count == count_sessions(
            sessions=bucket,
            diagnosis_key=diagnosis_key,
            gender=gender,
            age_range=age,
        )
