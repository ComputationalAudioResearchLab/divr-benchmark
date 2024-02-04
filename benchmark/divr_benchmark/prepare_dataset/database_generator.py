from enum import Enum
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from ..diagnosis import DiagnosisMap
from .processed import ProcessedDataset, ProcessedSession

AgeBracket = Tuple[int, int]
AgePlan = Dict[AgeBracket, int]
GenderPlan = Dict[str, AgePlan]
DiagnosisPlan = Dict[str, GenderPlan]
CountDiagnosisPlan = Dict[str, Tuple[str, GenderPlan]]


@dataclass
class DatabasePlan:
    train: DiagnosisPlan
    test: DiagnosisPlan
    val: DiagnosisPlan


class DatasetType(Enum):
    TRAIN = "TRAIN"
    TEST = "TEST"
    VAL = "VAL"


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
        plan = DatabasePlan(
            train={},
            test={},
            val={},
        )
        self.__hydrate_plan(plan=plan, sessions=sessions)
        print(plan)
        return self.__execute_plan(db_name=db_name, plan=plan, sessions=sessions)

    def __execute_plan(
        self, db_name: str, plan: DatabasePlan, sessions: List[ProcessedSession]
    ) -> ProcessedDataset:
        buckets: Dict[DatasetType, List[ProcessedSession]] = {
            DatasetType.TRAIN: [],
            DatasetType.TEST: [],
            DatasetType.VAL: [],
        }
        current_plan = plan
        for session in sessions:
            current_plan, bucket = self.__where_to_put(
                session=session, current_plan=current_plan
            )
            buckets[bucket] += [session]
        return ProcessedDataset(
            db_name=db_name,
            train_sessions=buckets[DatasetType.TRAIN],
            test_sessions=buckets[DatasetType.TEST],
            val_sessions=buckets[DatasetType.VAL],
        )

    def __where_to_put(
        self, session: ProcessedSession, current_plan: DatabasePlan
    ) -> Tuple[DatabasePlan, DatasetType]:
        """
        This mutates the current plan
        """
        buckets = {
            DatasetType.TRAIN: current_plan.train,
            DatasetType.TEST: current_plan.test,
            DatasetType.VAL: current_plan.val,
        }
        diagnosis = session.diagnosis[0].name
        gender = session.gender
        age_bracket = self.__age_to_bracket(session.age)
        for key, val in buckets.items():
            if val[diagnosis][gender][age_bracket] > 0:
                val[diagnosis][gender][age_bracket] -= 1
                return (current_plan, key)

        raise RuntimeError(
            f"the plan does not allow accommodating this session, plan must be invalid. [diagnosis: {diagnosis}, gender: {gender}, age_bracket: {age_bracket}]"
        )

    def __hydrate_plan(
        self,
        plan: DatabasePlan,
        sessions: List[ProcessedSession],
        level: int | None = None,
        unresolved_keys: List[str] | None = None,
    ) -> None:
        if level is None:
            level = self.__get_max_diagnosis_level(sessions)
        if level < 0:
            raise RuntimeError("level should never be less than 0, failed planning")
        diagnosis_counts = self.__count_diagnosis(sessions=sessions, level=level)
        print(diagnosis_counts)
        print(unresolved_keys)

        if unresolved_keys is not None:
            for diag_key, diag_val in diagnosis_counts.items():
                for gender_key, gender_val in diag_val.items():
                    for age_key, age_val in gender_val.items():
                        train_len, test_len, val_len = self.__calculate_split(age_val)
                        if min(train_len, test_len, val_len) > 0 or level == 0:
                            resolved_keys = []
                            for unresolved_key in unresolved_keys:
                                if self.diagnosis_map.get(unresolved_key).satisfies(
                                    diag_key
                                ):
                                    resolved_keys.append(unresolved_key)
                                    unresolved_keys.remove(unresolved_key)
                            while train_len > 0:
                                train_len -= 1

        if unresolved_keys is None:
            unresolved_keys = []

            for diag_key, diag_val in diagnosis_counts.items():
                plan.train[diag_key] = {}
                plan.test[diag_key] = {}
                plan.val[diag_key] = {}
                for gender_key, gender_val in diag_val.items():
                    plan.train[diag_key][gender_key] = {}
                    plan.test[diag_key][gender_key] = {}
                    plan.val[diag_key][gender_key] = {}
                    for age_key, age_val in gender_val.items():
                        train_len, test_len, val_len = self.__calculate_split(age_val)
                        print(diag_key, train_len, test_len, val_len)
                        if min(train_len, test_len, val_len) > 0 or level == 0:
                            # either all values are available or we are at root level
                            plan.train[diag_key][gender_key][age_key] = train_len
                            plan.test[diag_key][gender_key][age_key] = test_len
                            plan.val[diag_key][gender_key][age_key] = val_len
                        else:
                            # need to try later
                            plan.train[diag_key][gender_key][age_key] = 0
                            plan.test[diag_key][gender_key][age_key] = 0
                            plan.val[diag_key][gender_key][age_key] = 0
                            unresolved_keys.append(diag_key)

        if len(unresolved_keys) > 0:
            filtered_sessions = []
            for session in sessions:
                for diagnosis in session.diagnosis:
                    if diagnosis.name in unresolved_keys:
                        filtered_sessions += [session]
            self.__hydrate_plan(
                plan=plan,
                sessions=filtered_sessions,
                level=level - 1,
                unresolved_keys=unresolved_keys,
            )

    def __count_diagnosis(
        self, sessions: List[ProcessedSession], level: int
    ) -> DiagnosisPlan:
        counter: DiagnosisPlan = {}
        for session in sessions:
            for diagnosis in session.diagnosis:
                diag_key = diagnosis.name
                count_key = diagnosis.at_level(level).name
                if count_key not in counter:
                    counter[count_key] = {}
                diagnosis_ref = counter[count_key]
                if session.gender not in diagnosis_ref:
                    diagnosis_ref[session.gender] = {}
                gender_ref = diagnosis_ref[session.gender]
                age_bracket = self.__age_to_bracket(session.age)
                if age_bracket not in gender_ref:
                    gender_ref[age_bracket] = 0
                gender_ref[age_bracket] += 1
        return counter

    def __calculate_split(self, total_data: int) -> Tuple[int, int, int]:
        train_len = int(self.train_split * total_data)
        test_len = int(self.test_split * total_data)
        val_len = total_data - train_len - test_len
        return (train_len, test_len, val_len)

    def __age_to_bracket(self, age: int | None) -> Tuple[int, int]:
        if age is None:
            return (-1, 0)
        lower = (age // 10) * 10
        upper = lower + 10
        return (lower, upper)

    def __get_max_diagnosis_level2(self, plan: DiagnosisPlan) -> int:
        levels = []
        for key in plan.keys():
            levels.append(self.diagnosis_map.get(key).level)
        return max(levels)

    def __get_max_diagnosis_level(self, sessions: List[ProcessedSession]) -> int:
        levels = []
        for session in sessions:
            for diagnosis in session.diagnosis:
                levels.append(diagnosis.level)
        return max(levels)
