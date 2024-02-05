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
    Further leaf level diagnosis that can not be distributed equitably are distributed in
    test, train and validation set in that order so that they definitely appear in test set.

    Since a diagnosis can have multiple parents, the parent that it gets grouped with for
    distribution is decided with majority vote across the parent weight. In case of a tie
    classes are chosen in the order of pathological, normal and then unclassified. In case
    ties are within a subgroup then choice is made via random selection. This randomness
    is achieved by the presorting logic of the sessions list.

    Since a session can have multiple diagnosis, and we don't have a diagnostic confidence
    metric as of now, the diagnosis that best balances out the dataset is chosen.

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
        all_diag_names: List[str] = []
        for session in sessions:
            for diagnosis in session.diagnosis:
                all_diag_names.append(diagnosis.at_level(level).name)
        max_occurence_diag = max(all_diag_names, key=all_diag_names.count)

        grouped_sessions = {}
        for session in sessions:
            for diag in session.diagnosis:
                if diag.satisfies(max_occurence_diag):
                    gender = session.gender
                    if gender not in grouped_sessions:
                        grouped_sessions[gender] = {}
                    age_bracket = self.__age_to_bracket(session.age)
                    if age_bracket not in grouped_sessions[gender]:
                        grouped_sessions[gender][age_bracket] = []
                    grouped_sessions[gender][age_bracket].append(session)

        keys_and_counts = []
        for gender_key, gender_val in grouped_sessions.items():
            for age_key, age_val in gender_val.items():
                keys_and_counts.append(((gender_key, age_key), len(age_val)))
        sorted_keys_and_counts = sorted(
            keys_and_counts,
            key=lambda x: x[1],
            reverse=True,
        )

        selected_sessions = []
        total_sessions = 0
        for (gender_key, age_key), session_count in sorted_keys_and_counts:
            selected_sessions += grouped_sessions[gender_key][age_key]
            total_sessions += session_count
            if total_sessions > 2:
                return selected_sessions
        return selected_sessions

    def __age_to_bracket(self, age: int | None) -> Tuple[int, int]:
        if age is None:
            return (-1, 0)
        lower = (age // 10) * 10
        upper = lower + 10
        return (lower, upper)
