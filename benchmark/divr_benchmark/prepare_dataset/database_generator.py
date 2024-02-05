import random
from typing import List, Tuple
from dataclasses import dataclass
from ..diagnosis import DiagnosisMap, Diagnosis
from .processed import ProcessedDataset, ProcessedSession


@dataclass
class BucketLimit:
    train: int
    test: int
    val: int


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
        bucket_limit = BucketLimit(
            train=expected_train_len, test=expected_test_len, val=expected_val_len
        )
        return self.distribute(dataset, sessions, bucket_limit)

    def distribute(
        self,
        dataset: ProcessedDataset,
        sessions: List[ProcessedSession],
        bucket_limit: BucketLimit,
        level: int | None = None,
    ):
        if len(sessions) < 1:
            return dataset

        if level is not None and level < 0:
            raise RuntimeError("level should never drop below 0")

        all_diagnosis: List[Diagnosis] = []
        for session in sessions:
            all_diagnosis += session.diagnosis

        if level is None:
            level = max([diagnosis.level for diagnosis in all_diagnosis])

        all_names: List[str] = []
        for diagnosis in all_diagnosis:
            all_names.append(diagnosis.at_level(level).name)

        max_occurence_diag = max(all_names, key=all_names.count)
        max_occurence_diag_count = all_names.count(max_occurence_diag)
        if max_occurence_diag_count > 2:
            # can be allocated to all 3 datasets
            selected_sessions = self.select_from_diagnosis(
                sessions, max_occurence_diag
            )[:3]
            for session in selected_sessions:
                sessions.remove(session)
            dataset = self.fill_dataset(
                dataset=dataset,
                sessions=selected_sessions,
                bucket_limit=bucket_limit,
            )
        elif level == 0:
            # we have reached base level and still data is very fragmented
            selected_sessions = self.select_from_diagnosis(
                sessions, max_occurence_diag
            )[:3]
            for session in selected_sessions:
                sessions.remove(session)
            dataset = self.fill_dataset(
                dataset=dataset,
                sessions=selected_sessions,
                bucket_limit=bucket_limit,
            )
        else:
            level -= 1

        return self.distribute(
            dataset=dataset, sessions=sessions, bucket_limit=bucket_limit, level=level
        )

    def fill_dataset(
        self,
        dataset: ProcessedDataset,
        sessions: List[ProcessedSession],
        bucket_limit: BucketLimit,
    ) -> ProcessedDataset:
        train_len = len(dataset.train_sessions)
        test_len = len(dataset.test_sessions)
        val_len = len(dataset.val_sessions)

        train_space = bucket_limit.train - train_len
        test_space = bucket_limit.test - test_len
        val_space = bucket_limit.val - val_len

        total_sessions = len(sessions)
        if total_sessions < 1:
            raise ValueError("called fill dataset with 0 sessions")

        if total_sessions == 1:
            if test_space > 0:
                dataset.test_sessions.append(sessions[0])
            elif train_space > 0:
                dataset.train_sessions.append(sessions[0])
            elif val_space > 0:
                dataset.val_sessions.append(sessions[0])
            else:
                # all buckets full, putting it in test set anyway
                dataset.test_sessions.append(sessions[0])
            return dataset

        if total_sessions == 2:
            if test_space > 0:
                dataset.test_sessions.append(sessions[0])
                if train_space > 0:
                    dataset.train_sessions.append(sessions[1])
                elif val_space > 0:
                    dataset.val_sessions.append(sessions[1])
                else:
                    dataset.test_sessions.append(sessions[1])
            elif train_space > 0:
                dataset.train_sessions.append(sessions[0])
                if val_space > 0:
                    dataset.val_sessions.append(sessions[1])
                elif train_space > 1:
                    dataset.train_sessions.append(sessions[1])
                else:
                    dataset.test_sessions.append(sessions[1])
            elif val_space > 0:
                dataset.val_sessions.append(sessions[0])
                if val_space > 1:
                    dataset.val_sessions.append(sessions[1])
                else:
                    dataset.test_sessions.append(sessions[1])
            else:
                dataset.test_sessions.append(sessions[0])
                dataset.test_sessions.append(sessions[2])
            return dataset

        if train_space > 0:
            if test_space > 0:
                if val_space > 0:
                    dataset.test_sessions.append(sessions[0])
                    dataset.train_sessions.append(sessions[1])
                    dataset.val_sessions.append(sessions[2])
                else:
                    if train_space > test_space:
                        dataset.test_sessions.append(sessions[0])
                        dataset.train_sessions.append(sessions[1])
                        dataset.train_sessions.append(sessions[2])
                    else:
                        dataset.test_sessions.append(sessions[0])
                        dataset.test_sessions.append(sessions[1])
                        dataset.train_sessions.append(sessions[2])
            else:
                if val_space > 0:
                    if train_space > val_space:
                        dataset.train_sessions.append(sessions[0])
                        dataset.train_sessions.append(sessions[1])
                        dataset.val_sessions.append(sessions[2])
                    elif train_space == 1 and val_space == 1:
                        dataset.train_sessions.append(sessions[0])
                        dataset.val_sessions.append(sessions[1])
                        dataset.test_sessions.append(sessions[2])
                    else:
                        dataset.train_sessions.append(sessions[0])
                        dataset.val_sessions.append(sessions[1])
                        dataset.val_sessions.append(sessions[2])
                else:
                    if train_space > 2:
                        dataset.train_sessions.append(sessions[0])
                        dataset.train_sessions.append(sessions[1])
                        dataset.train_sessions.append(sessions[2])
                    elif train_space > 1:
                        dataset.train_sessions.append(sessions[0])
                        dataset.train_sessions.append(sessions[1])
                        dataset.test_sessions.append(sessions[2])
        else:
            if test_space > 0:
                if val_space > 0:
                    if test_space > val_space:
                        dataset.test_sessions.append(sessions[0])
                        dataset.test_sessions.append(sessions[1])
                        dataset.val_sessions.append(sessions[2])
                    else:
                        dataset.test_sessions.append(sessions[0])
                        dataset.val_sessions.append(sessions[1])
                        dataset.val_sessions.append(sessions[2])
                else:
                    dataset.test_sessions.append(sessions[0])
                    dataset.test_sessions.append(sessions[1])
                    dataset.test_sessions.append(sessions[2])
            else:
                if val_space > 2:
                    dataset.val_sessions.append(sessions[0])
                    dataset.val_sessions.append(sessions[1])
                    dataset.val_sessions.append(sessions[2])
                elif val_space == 2:
                    dataset.val_sessions.append(sessions[0])
                    dataset.val_sessions.append(sessions[1])
                    dataset.test_sessions.append(sessions[2])
                elif val_space == 1:
                    dataset.val_sessions.append(sessions[0])
                    dataset.test_sessions.append(sessions[1])
                    dataset.test_sessions.append(sessions[2])
                else:
                    raise RuntimeError("all buckets full, this should be impossible")

        return dataset

    def select_from_diagnosis(
        self, sessions: List[ProcessedSession], max_occurence_diag: str
    ) -> List[ProcessedSession]:
        filtered_sessions: List[ProcessedSession] = []
        for session in sessions:
            for diag in session.diagnosis:
                if diag.satisfies(max_occurence_diag):
                    filtered_sessions.append(session)
        return self.select_from_gender(filtered_sessions)

    def select_from_gender(
        self, sessions: List[ProcessedSession]
    ) -> List[ProcessedSession]:
        selected_sessions: List[ProcessedSession] = []
        unselected_sessions: List[ProcessedSession] = []
        all_genders = []
        for session in sessions:
            all_genders.append(session.gender)
        max_occurence_gender = max(all_genders, key=all_genders.count)
        for session in sessions:
            if session.gender == max_occurence_gender:
                selected_sessions.append(session)
            else:
                unselected_sessions.append(session)
        selected_sessions_count = len(selected_sessions)
        if selected_sessions_count < 3:
            sessions_to_add = 3 - selected_sessions_count
            selected_sessions += unselected_sessions[:sessions_to_add]
        return self.select_from_age_bracket(sessions=selected_sessions)

    def select_from_age_bracket(
        self, sessions: List[ProcessedSession]
    ) -> List[ProcessedSession]:
        selected_sessions: List[ProcessedSession] = []
        unselected_sessions: List[ProcessedSession] = []
        all_age_brackets = []
        for session in sessions:
            age_bracket = self.__age_to_bracket(session.age)
            all_age_brackets.append(age_bracket)
        max_occurence_bracket = max(all_age_brackets, key=all_age_brackets.count)
        for session in sessions:
            age_bracket = self.__age_to_bracket(session.age)
            if age_bracket == max_occurence_bracket:
                selected_sessions.append(session)
            else:
                unselected_sessions.append(session)
        selected_sessions_count = len(selected_sessions)
        if selected_sessions_count < 3:
            sessions_to_add = 3 - selected_sessions_count
            selected_sessions += unselected_sessions[:sessions_to_add]
        return selected_sessions

    def __age_to_bracket(self, age: int | None) -> Tuple[int, int]:
        if age is None:
            return (-1, 0)
        lower = (age // 10) * 10
        upper = lower + 10
        return (lower, upper)
