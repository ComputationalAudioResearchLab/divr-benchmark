from enum import Enum
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple
from .processed import ProcessedDataset, ProcessedSession

AgePlan = Dict[Tuple[int, int], int]
GenderPlan = Dict[str, AgePlan]
DiagnosisPlan = Dict[str, GenderPlan]


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
        train_split: float,
        test_split: float,
        random_seed: float,
    ) -> None:
        self.train_split = train_split
        self.test_split = test_split
        self.random_seed = random_seed

    def generate(
        self,
        db_name: str,
        sessions: List[ProcessedSession],
    ):
        random.Random(self.random_seed).shuffle(sessions)
        plan = self.__plan_dataset(sessions=sessions)
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
        session_diagnosis = session.diagnosis[0].name
        session_gender = session.gender
        session_age_bracket = self.__age_to_bracket(session.age)
        if (
            current_plan.train[session_diagnosis][session_gender][session_age_bracket]
            > 0
        ):
            current_plan.train[session_diagnosis][session_gender][
                session_age_bracket
            ] -= 1
            bucket = DatasetType.TRAIN
        elif (
            current_plan.test[session_diagnosis][session_gender][session_age_bracket]
            > 0
        ):
            current_plan.test[session_diagnosis][session_gender][
                session_age_bracket
            ] -= 1
            bucket = DatasetType.TEST
        elif (
            current_plan.val[session_diagnosis][session_gender][session_age_bracket] > 0
        ):
            current_plan.val[session_diagnosis][session_gender][
                session_age_bracket
            ] -= 1
            bucket = DatasetType.VAL
        else:
            raise RuntimeError(
                "the plan does not allow accommodating this session, plan must be invalid"
            )
        return (current_plan, bucket)

    def __plan_dataset(self, sessions: List[ProcessedSession]) -> DatabasePlan:
        diagnosis_counts = self.__count_diagnosis(sessions=sessions)
        plan = DatabasePlan(
            train={},
            test={},
            val={},
        )
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
                    plan.train[diag_key][gender_key][age_key] = train_len
                    plan.test[diag_key][gender_key][age_key] = test_len
                    plan.val[diag_key][gender_key][age_key] = val_len
        return plan

    def __count_diagnosis(self, sessions: List[ProcessedSession]) -> DiagnosisPlan:
        counter: DiagnosisPlan = {}
        for session in sessions:
            for diagnosis in session.diagnosis:
                if diagnosis.name not in counter:
                    counter[diagnosis.name] = {}
                diagnosis_ref = counter[diagnosis.name]
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
