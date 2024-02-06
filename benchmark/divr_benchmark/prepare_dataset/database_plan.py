from __future__ import annotations
from typing import Dict, List, Literal, Tuple
from dataclasses import dataclass

from .processed import ProcessedSession, ProcessedDataset
from ..diagnosis import Diagnosis


@dataclass
class DatabaseBucketPlan:
    name: str
    limit: int
    level: int
    children: DatabaseBucketLink
    sessions: List[ProcessedSession]
    occupation: int = 0

    @property
    def child_level(self) -> int:
        return self.level + 1

    @property
    def space(self) -> int:
        return self.limit - self.occupation

    def all_sessions(self) -> List[ProcessedSession]:
        all = self.sessions
        for child in self.children.values():
            all += child.all_sessions()
        return all

    def add(self, diag: Diagnosis) -> bool:
        if diag.level <= self.level:
            raise ValueError(
                f"Can't add sibling or parent from here. self.{{name: {self.name}, level: {self.level}}} diag.{{name: {diag.name}, level: {diag.level}}}"
            )

        if diag.level > self.child_level:
            # grandchildren
            child_link = diag.at_level(self.child_level)
            return self.children[child_link.name].add(diag)

        # direct child
        if self.space < 1:
            return False

        if diag.name not in self.children:
            self.children[diag.name] = DatabaseBucketPlan(
                name=diag.name,
                limit=1,
                level=self.child_level,
                children={},
                sessions=[],
            )
        else:
            self.children[diag.name].limit += 1

        self.occupation += 1
        return True

    def add_with_session(self, data: Tuple[Diagnosis, ProcessedSession]) -> bool:
        diag = data[0]
        session = data[1]
        if diag.level <= self.level:
            raise ValueError(
                f"Can't add sibling or parent from here. self.{{name: {self.name}, level: {self.level}}} diag.{{name: {diag.name}, level: {diag.level}}}"
            )

        if diag.level > self.child_level:
            # grandchildren
            child_link = diag.at_level(self.child_level)
            if child_link.name in self.children:
                return self.children[child_link.name].add_with_session(data)
            else:
                return False

        # direct child
        if self.space < 1:
            if diag.name in self.children:
                child = self.children[diag.name]
                if child.space < 1:
                    return False
                else:
                    child.sessions.append(session)
                    child.occupation += 1
                    return True
            else:
                return False

        if diag.name not in self.children:
            self.children[diag.name] = DatabaseBucketPlan(
                name=diag.name,
                limit=1,
                level=self.child_level,
                children={},
                sessions=[session],
                occupation=1,
            )
        else:
            self.children[diag.name].limit += 1
            self.children[diag.name].occupation += 1
            self.children[diag.name].sessions.append(session)

        self.occupation += 1
        return True


DatabaseBucketLink = Dict[str, DatabaseBucketPlan]


class DatabasePlan:

    def __init__(
        self,
        total_sessions: int,
        train_split: float,
        test_split: float,
    ) -> None:
        train_len = int(total_sessions * train_split)
        test_len = int(total_sessions * test_split)
        val_len = total_sessions - train_len - test_len
        self.train = DatabaseBucketPlan(
            name="train", limit=train_len, level=-1, children={}, sessions=[]
        )
        self.test = DatabaseBucketPlan(
            name="test", limit=test_len, level=-1, children={}, sessions=[]
        )
        self.val = DatabaseBucketPlan(
            name="val", limit=val_len, level=-1, children={}, sessions=[]
        )

    def add(self, diags: List[Diagnosis]) -> int:
        total = len(diags)
        added = 0
        if total > 3:
            raise ValueError(
                "only 3 or less diags should be added at a time to ensure every class appears in final datasets"
            )
        if (added < total) and self.test.add(diags[added]):
            added += 1
        if (added < total) and self.train.add(diags[added]):
            added += 1
        if (added < total) and self.val.add(diags[added]):
            added += 1
        return added

    def add_with_sessions(
        self, data: List[Tuple[Diagnosis, ProcessedSession]]
    ) -> Tuple[int, List[str]]:
        total = len(data)
        added = 0
        added_session_ids = []
        if total > 3:
            raise ValueError(
                "only 3 or less items should be added at a time to ensure every class appears in final datasets"
            )
        if (added < total) and self.test.add_with_session(data[added]):
            added_session_ids.append(data[added][1].id)
            added += 1
        if (added < total) and self.train.add_with_session(data[added]):
            added_session_ids.append(data[added][1].id)
            added += 1
        if (added < total) and self.val.add_with_session(data[added]):
            added_session_ids.append(data[added][1].id)
            added += 1
        return added, added_session_ids

    def to_dataset(self, db_name: str) -> ProcessedDataset:
        return ProcessedDataset(
            db_name=db_name,
            train_sessions=self.train.all_sessions(),
            test_sessions=self.test.all_sessions(),
            val_sessions=self.val.all_sessions(),
        )


@dataclass
class Cup:
    occupancy: int
    sessions: List[ProcessedSession]


@dataclass
class Bucket:
    capacity: int
    train: Cup
    test: Cup
    val: Cup

    @property
    def occupancy(self) -> int:
        return self.train.occupancy + self.test.occupancy + self.val.occupancy

    @property
    def has_space(self) -> bool:
        return self.capacity > self.occupancy

    @property
    def has_zeros(self) -> bool:
        return min([self.train.occupancy, self.test.occupancy, self.val.occupancy]) == 0

    def allocate_sessions(self, sessions: List[ProcessedSession]) -> None:
        test_start = 0
        test_end = test_start + self.test.occupancy
        train_start = test_end
        train_end = train_start + self.train.occupancy
        val_start = train_end
        val_end = val_start + self.val.occupancy
        if val_end != len(sessions):
            raise RuntimeError("Unable to allocate session")
        self.train.sessions += sessions[train_start:train_end]
        self.test.sessions += sessions[test_start:test_end]
        self.val.sessions += sessions[val_start:val_end]


class BucketCollection(Dict[str, Bucket]):

    def setup(
        self,
        total_train_len: int,
        total_test_len: int,
        total_val_len: int,
        train_split: float,
        test_split: float,
        val_split: float,
        values: List[Tuple[str, int]],
    ) -> BucketCollection:
        self.total_train_len = total_train_len
        self.total_test_len = total_test_len
        self.total_val_len = total_val_len
        self.train_split = train_split
        self.test_split = test_split
        self.val_split = val_split

        for diag_name, count in values:
            self.__add_diagnosis(diag_name, count)

        self.__fill_zeros()
        self.__fill_remaining()
        return self

    def allocate_sessions(self, vals: Dict[str, List[ProcessedSession]]) -> None:
        for diag_name, sessions in vals.items():
            bucket = self[diag_name]
            bucket.allocate_sessions(sessions)

    def to_dataset(self, db_name) -> ProcessedDataset:
        train_sessions = []
        test_sessions = []
        val_sessions = []
        for bucket in self.values():
            train_sessions += bucket.train.sessions
            test_sessions += bucket.test.sessions
            val_sessions += bucket.val.sessions
        return ProcessedDataset(
            db_name=db_name,
            train_sessions=train_sessions,
            test_sessions=test_sessions,
            val_sessions=val_sessions,
        )

    def __add_diagnosis(self, diag_name: str, count: int) -> None:
        train_len = int(self.train_split * count)
        test_len = int(self.test_split * count)
        val_len = int(self.val_split * count)
        self[diag_name] = Bucket(
            capacity=count,
            train=Cup(occupancy=train_len, sessions=[]),
            test=Cup(occupancy=test_len, sessions=[]),
            val=Cup(occupancy=val_len, sessions=[]),
        )

    def __fill_zeros(self) -> None:
        for key, bucket in self.items():
            if bucket.has_space and bucket.has_zeros:
                (current_train_len, current_test_len, current_val_len) = (
                    self.__count_sets()
                )
                bucket = self[key]
                if (
                    bucket.test.occupancy == 0
                    and current_test_len < self.total_test_len
                    and bucket.has_space
                ):
                    bucket.test.occupancy = 1
                if (
                    bucket.train.occupancy == 0
                    and current_train_len < self.total_train_len
                    and bucket.has_space
                ):
                    bucket.train.occupancy = 1
                if (
                    bucket.val.occupancy == 0
                    and current_val_len < self.total_val_len
                    and bucket.has_space
                ):
                    bucket.val.occupancy = 1

    def __fill_remaining(self) -> None:
        while key := self.__has_remaining():
            (current_train_len, current_test_len, current_val_len) = self.__count_sets()
            bucket = self[key]
            keyed_counts = [
                [current_train_len - self.total_train_len, bucket.train],
                [current_test_len - self.total_test_len, bucket.test],
                [current_val_len - self.total_val_len, bucket.val],
            ]
            filtered_key_counts = filter(lambda x: x[0] < 0, keyed_counts)
            sorted_key_counts = sorted(
                filtered_key_counts,
                key=lambda x: x[0],
            )
            best_option = sorted_key_counts[0]
            best_option[1].occupancy += 1

    def __has_remaining(self) -> str | Literal[False]:
        for key, bucket in self.items():
            if bucket.has_space:
                return key
        return False

    def __count_sets(self) -> Tuple[int, int, int]:
        train_len = 0
        test_len = 0
        val_len = 0
        for bucket in self.values():
            train_len += bucket.train.occupancy
            test_len += bucket.test.occupancy
            val_len += bucket.val.occupancy
        return (train_len, test_len, val_len)
