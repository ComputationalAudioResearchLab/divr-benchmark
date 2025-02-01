from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

from .processed import ProcessedSession, ProcessedDataset


@dataclass
class Cup:
    occupancy: int
    sessions: List[ProcessedSession]

    def allocate_session(self, session: ProcessedSession) -> bool:
        if len(self.sessions) < self.occupancy:
            self.sessions.append(session)
            return True
        return False


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

    def allocate_sessions(self, data: List[ProcessedSession]) -> List[str]:
        total = len(data)
        added = 0
        added_session_ids = []
        if total > 3:
            raise ValueError(
                "only 3 or less items should be added at a time to ensure every class appears in final datasets"
            )
        if (added < total) and self.test.allocate_session(data[added]):
            added_session_ids.append(data[added].id)
            added += 1
        if (added < total) and self.train.allocate_session(data[added]):
            added_session_ids.append(data[added].id)
            added += 1
        if (added < total) and self.val.allocate_session(data[added]):
            added_session_ids.append(data[added].id)
            added += 1
        return added_session_ids


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
        val_len = int(self.val_split * count)
        test_len = count - train_len - val_len
        self[diag_name] = Bucket(
            capacity=count,
            train=Cup(occupancy=train_len, sessions=[]),
            test=Cup(occupancy=test_len, sessions=[]),
            val=Cup(occupancy=val_len, sessions=[]),
        )

    def __fill_zeros(self) -> None:
        for key, bucket in self.items():
            if bucket.has_space and bucket.has_zeros:
                (
                    current_train_len,
                    current_test_len,
                    current_val_len,
                ) = self.__count_sets()
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
