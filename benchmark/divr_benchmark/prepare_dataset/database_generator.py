from __future__ import annotations
import random
from typing import Dict, List, Tuple
from ..diagnosis import DiagnosisMap, Diagnosis
from .processed import ProcessedSession
from .database_plan import DatabasePlan


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

        all_levels = []
        for session in sessions:
            for diag in session.diagnosis:
                all_levels.append(diag.level)
        max_level = max(all_levels)
        database_plan = DatabasePlan(
            total_sessions=len(sessions),
            train_split=self.train_split,
            test_split=self.test_split,
        )
        max_level_diag_counts = self.__exhaust_names_at_level(sessions, max_level)
        for level in range(max_level):
            diag_counts = self.__to_specific_level(max_level_diag_counts, level)
            while len(diag_counts) > 0:
                for diag in diag_counts.values():
                    count_to_add = min(diag["count"], 3)
                    diag["count"] -= database_plan.add([diag["diag"]] * count_to_add)
                for diag_name, diag in list(diag_counts.items()):
                    if diag["count"] == 0:
                        del diag_counts[diag_name]

        while len(max_level_diag_counts) > 0:
            for diag in max_level_diag_counts.values():
                count_to_add = min(diag["count"], 3)
                selected_sessions = self.select_gender_and_age(diag["sessions"])
                data = [
                    (diag["diag"], session)
                    for session in selected_sessions[:count_to_add]
                ]
                count_added, sessions_added = database_plan.add_with_sessions(data)
                diag["count"] -= count_added
                for session_id in sessions_added:
                    del diag["sessions"][session_id]
            for diag_name, max_diag in list(max_level_diag_counts.items()):
                if max_diag["count"] == 0:
                    del max_level_diag_counts[diag_name]

        return database_plan.to_dataset(db_name=db_name)

    def __to_specific_level(self, max_level_diag_counts: Dict, level: int):
        counts: Dict = {}
        for val in max_level_diag_counts.values():
            count = val["count"]
            diag = val["diag"].at_level(level)
            if diag.level == level:
                if diag.name not in counts:
                    counts[diag.name] = {"count": count, "diag": diag}
                else:
                    counts[diag.name]["count"] += count
        sorted_counts = dict(
            sorted(
                counts.items(),
                key=lambda x: x[1]["count"],
                reverse=True,
            )
        )
        return sorted_counts

    def __exhaust_names_at_level(self, sessions: List[ProcessedSession], level: int):
        working_sessions = sessions.copy()
        counts = {}
        while len(working_sessions) > 0:
            best_diag = self.__most_popular_diag(working_sessions, level)
            if best_diag.name not in counts:
                counts[best_diag.name] = {"diag": best_diag, "count": 0, "sessions": {}}
            for session in working_sessions:
                if best_diag.name in session.diagnosis_names_at_level(level):
                    counts[best_diag.name]["count"] += 1
                    counts[best_diag.name]["sessions"][session.id] = session
                    working_sessions.remove(session)
        sorted_counts = dict(
            sorted(
                counts.items(),
                key=lambda x: x[1]["count"],
                reverse=True,
            )
        )
        return sorted_counts

    def __most_popular_diag(
        self, sessions: List[ProcessedSession], level: int
    ) -> Diagnosis:
        best_diag = {}
        for session in sessions:
            for diag in session.diagnosis_at_level(level):
                if diag.name not in best_diag:
                    best_diag[diag.name] = {"diag": diag, "count": 1}
                else:
                    best_diag[diag.name]["count"] += 1
        return max(best_diag.items(), key=lambda x: x[1]["count"])[1]["diag"]

    def select_gender_and_age(
        self, sessions: Dict[str, ProcessedSession]
    ) -> List[ProcessedSession]:
        grouped_sessions = {}
        for session in sessions.values():
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
