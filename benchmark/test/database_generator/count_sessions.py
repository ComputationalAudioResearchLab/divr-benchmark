from typing import List, Tuple
from test.database_generator.age_to_bracket import age_to_bracket
from divr_benchmark.prepare_dataset.processed import ProcessedSession


def count_sessions(
    sessions: List[ProcessedSession],
    diagnosis_key: str,
    gender: str,
    age_range: Tuple[int, int] | None,
) -> int:
    count = 0
    for session in sessions:
        if session.gender == gender:
            for diagnosis in session.diagnosis:
                if diagnosis.satisfies(diagnosis_key):
                    age_bracket = age_to_bracket(session.age)
                    if age_bracket == age_range:
                        count += 1
    return count
