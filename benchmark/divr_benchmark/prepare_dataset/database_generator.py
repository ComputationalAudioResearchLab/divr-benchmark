import random
from typing import List
from .processed import ProcessedDataset, ProcessedSession


class DatabaseGenerator:
    """
    This class does takes a best effort approach to put a proportional amount
    of diagnosis, age and gender in the three different datasets i.e. train, val and test.

    First priority goes to diagnosis, then gender and then age.

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
        total_data = len(sessions)
        train_start = 0
        train_end = int(train_start + self.train_split * total_data)
        test_start = train_end
        test_end = int(test_start + self.test_split * total_data)
        val_start = test_end
        val_end = total_data

        random.Random(self.random_seed).shuffle(sessions)

        return ProcessedDataset(
            db_name=db_name,
            train_sessions=sessions[train_start:train_end],
            val_sessions=sessions[val_start:val_end],
            test_sessions=sessions[test_start:test_end],
        )
