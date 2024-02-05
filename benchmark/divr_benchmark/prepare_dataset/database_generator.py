import random
from typing import List, Tuple
from ..diagnosis import DiagnosisMap, Diagnosis
from .processed import ProcessedDataset, ProcessedSession


class DatabaseGenerator:
    """
    This class does takes a best effort approach to put a proportional amount
    of diagnosis, age and gender in the three different datasets i.e. train, val and test.

    First priority goes to diagnosis, then gender and then age.
    Age is considered in buckets of 10 i.e. 0-10, 11-20, 21-30, ...

    As we don't have a lot of diagnosis at every level of diagnosis, if a diagnosis can not
    be distributed equitably at a given level it would be resolved later at a parent level
    along with other unresolved diagnosis.

    Since a diagnosis can have multiple parents, the parent that it gets grouped with for
    distribution is decided with majority vote across the parent weight. In case of a tie
    classes are chosen in the order of pathological, normal and then unclassified.

    Since a session can have multiple diagnosis, and we don't have a diagnostic confidence
    metric as of now, the diagnosis that best balances out the dataset is chosen. This is
    achieved by chosing the diagnosis with most occurences in the input sessions so that
    it can appear in all train, test and val sets.

    We only consider balance of the dataset in terms of sessions and not in terms of files,
    this can result in slight imbalances in data if more sessions are recorded for a given
    pathology.
    """

    def __init__(
        self,
        diagnosis_map: DiagnosisMap,
        train_split: float,
        test_split: float,
        random_seed: float,
    ) -> None:
        self.diagnosis_map = diagnosis_map
        self.train_split = train_split
        self.test_split = test_split
        self.random_seed = random_seed

    def generate(
        self,
        db_name: str,
        sessions: List[ProcessedSession],
    ):
        random.Random(self.random_seed).shuffle(sessions)

        dataset = ProcessedDataset(
            db_name=db_name,
            train_sessions=[],
            test_sessions=[],
            val_sessions=[],
        )

        total_sessions = len(sessions)
        expected_train_len = int(total_sessions * self.train_split)
        expected_test_len = int(total_sessions * self.test_split)
        expected_val_len = total_sessions - expected_train_len - expected_test_len
        assignments = [
            (dataset.test_sessions, expected_test_len),
            (dataset.train_sessions, expected_train_len),
            (dataset.val_sessions, expected_val_len),
        ]

        all_diagnosis: List[Diagnosis] = []
        for session in sessions:
            all_diagnosis += session.diagnosis
        level = max([diagnosis.level for diagnosis in all_diagnosis])

        while len(sessions) > 0:
            selected_sessions = self.select_at_level(sessions, level)
            selection_count = len(selected_sessions)
            if selection_count > 2 or level == 0:
                if selection_count < 1:
                    raise ValueError("called fill dataset with 0 sessions")
                j = 0
                for assignment in assignments:
                    if (j < selection_count) and assignment[1] > len(assignment[0]):
                        session = selected_sessions[j]
                        assignment[0].append(session)
                        sessions.remove(session)
                        j += 1
            else:
                level -= 1

        return dataset

    def select_at_level(
        self, sessions: List[ProcessedSession], level: int
    ) -> List[ProcessedSession]:
        grouped_diag_sessions = {}
        for session in sessions:
            for diag_name in session.diagnosis_names_at_level(level):
                if diag_name not in grouped_diag_sessions:
                    grouped_diag_sessions[diag_name] = []
                grouped_diag_sessions[diag_name].append(session)

        max_diag_sessions = max(
            grouped_diag_sessions.items(),
            key=lambda x: len(x[1]),
        )[1]

        grouped_sessions = {}
        for session in max_diag_sessions:
            gender = session.gender
            age_bracket = self.__age_to_bracket(session.age)
            key = (gender, age_bracket)
            if key not in grouped_sessions:
                grouped_sessions[key] = []
            grouped_sessions[key].append(session)

        sorted_sessions_by_count = sorted(
            grouped_sessions.values(), key=lambda x: len(x), reverse=True
        )

        selected_sessions = []
        for sorted_sessions in sorted_sessions_by_count:
            selected_sessions += sorted_sessions
            if len(selected_sessions) > 2:
                return selected_sessions
        return selected_sessions

    def __age_to_bracket(self, age: int | None) -> Tuple[int, int]:
        if age is None:
            return (-1, 0)
        lower = (age // 10) * 10
        upper = lower + 10
        return (lower, upper)
