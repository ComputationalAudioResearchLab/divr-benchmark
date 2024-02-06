from __future__ import annotations
from typing import Dict, List, Tuple
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
